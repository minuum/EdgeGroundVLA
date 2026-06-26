"""
CH54 perception model ablation — basket 탐지 모델 비교

대상 모델:
  1. PG2 (oracle)         — 현재 사용 중, 기준선
  2. Pure Kosmos-2        — 로컬 보유, Exp10 IoU 0.87
  3. CLIP                 — openai/clip-vit-large-patch14, 방향 분류
  4. OWL-v2               — google/owlv2-base-patch16-ensemble
  5. GroundingDINO        — IDEA-Research/grounding-dino-base
  6. Florence-2           — microsoft/Florence-2-base

측정 (6/26 세션 frame 0 이미지 기준):
  - detection rate: 적어도 한 bbox 찾음
  - cx match:  |model_cx - pg2_cx| < 0.2  (PG2 oracle과 비교)
  - dir match: L/C/R 방향 일치
  - latency:   첫 추론(워밍업 포함), 이후 평균

Usage:
  .venv/bin/python3 scripts/ablate_perception_models.py [--models all|clip|owlv2|gdino|florence|kosmos]
"""

import argparse, io, json, time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

SESSIONS_DIR = Path("docs/inference_sessions")
OUT_FILE     = Path("docs/v5/ablate_perception_models.json")
DEVICE       = "cuda"
PHRASE       = "gray basket"

# 6/26 세션 frame 0 이미지 + PG2 oracle 수집
def load_eval_frames():
    frames = []
    for sp in sorted(SESSIONS_DIR.glob("session_20260626*.h5")):
        with h5py.File(sp, "r") as f:
            if "grounding/bbox" not in f: continue
            bbox = f["grounding/bbox"][:]
            # oracle: frame 1 (warmup 후 frame 0 ≈ frame 1)
            if len(bbox) < 2: continue
            pg2_cx   = float(bbox[1, 0])
            pg2_has  = bool(bbox[1, 3] > 0.5)
            pg2_area = float(bbox[1, 2])
            # 이미지: frame 0 (출발 시점 원본)
            if "observations" in f:
                raw = f["observations"]["images"][0]
            elif "images" in f:
                raw = f["images"][0]
            else:
                continue
            try:
                if raw.ndim == 1:
                    img = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
                else:
                    img = Image.fromarray(raw.astype(np.uint8)).convert("RGB")
            except Exception:
                continue
            frames.append({
                "name": sp.stem,
                "img": img,
                "pg2_cx": pg2_cx,
                "pg2_has": pg2_has,
                "pg2_area": pg2_area,
            })
    print(f"평가 프레임: {len(frames)}개 (6/26 세션)")
    return frames


def cx_bucket(cx):
    if cx < 0.4: return "L"
    if cx > 0.6: return "R"
    return "C"


def eval_metrics(preds, frames, label):
    """
    preds: list of {"detected": bool, "cx": float|None}
    frames: oracle list
    """
    valid = [(p, f) for p, f in zip(preds, frames) if f["pg2_has"]]
    n = len(valid)
    if n == 0:
        print(f"  {label:<30} | no valid oracle frames")
        return {}

    det    = sum(1 for p, _ in valid if p["detected"])
    cx_ok  = sum(1 for p, f in valid if p["detected"] and abs(p["cx"] - f["pg2_cx"]) < 0.2)
    dir_ok = sum(1 for p, f in valid if p["detected"] and cx_bucket(p["cx"]) == cx_bucket(f["pg2_cx"]))

    det_r  = det / n
    cx_r   = cx_ok / n
    dir_r  = dir_ok / n
    print(f"  {label:<30} | det={det_r:.0%}  cx={cx_r:.0%}  dir={dir_r:.0%}  (n={n})")
    return {"label": label, "det": det_r, "cx_match": cx_r, "dir_match": dir_r, "n": n}


# ── 모델별 추론 함수 ─────────────────────────────────────────────────────────

def run_clip(frames):
    from transformers import CLIPProcessor, CLIPModel
    import torch
    print("  [CLIP] 로딩...")
    t0 = time.time()
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print(f"  [CLIP] 로딩 완료 {time.time()-t0:.1f}s")

    # 방향 분류: 3 텍스트 프롬프트
    texts = [
        "a gray basket on the left side of the image",
        "a gray basket in the center of the image",
        "a gray basket on the right side of the image",
        "no basket visible",
    ]
    preds = []
    lats = []
    for fr in frames:
        t0 = time.time()
        inputs = proc(text=texts, images=fr["img"], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image[0]
            probs  = logits.softmax(dim=0).cpu().numpy()
        lat = (time.time() - t0) * 1000
        lats.append(lat)
        best = int(np.argmax(probs))
        if best == 3 or probs[best] < 0.35:  # "no basket" 또는 확신 낮음
            preds.append({"detected": False, "cx": None})
        else:
            cx_map = [0.2, 0.5, 0.8]
            preds.append({"detected": True, "cx": cx_map[best]})
    print(f"  [CLIP] latency mean={np.mean(lats):.0f}ms")
    return preds


def run_owlv2(frames):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    import torch
    print("  [OWL-v2] 로딩...")
    t0 = time.time()
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble",
    ).to(DEVICE)
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    print(f"  [OWL-v2] 로딩 완료 {time.time()-t0:.1f}s")

    preds = []
    lats = []
    for fr in frames:
        t0 = time.time()
        inputs = proc(text=[[PHRASE]], images=fr["img"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
        W = fr["img"].width
        results = proc.post_process_object_detection(
            out, threshold=0.1, target_sizes=[(fr["img"].height, W)]
        )[0]
        lat = (time.time() - t0) * 1000
        lats.append(lat)
        boxes = results["boxes"]
        if len(boxes) == 0:
            preds.append({"detected": False, "cx": None})
        else:
            scores = results["scores"].cpu().numpy()
            best_i = int(np.argmax(scores))
            x1, _, x2, _ = boxes[best_i].cpu().tolist()
            cx = (x1 + x2) / 2.0 / W
            preds.append({"detected": True, "cx": cx})
    print(f"  [OWL-v2] latency mean={np.mean(lats):.0f}ms")
    return preds


def run_gdino(frames):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import torch
    print("  [GroundingDINO] 로딩...")
    t0 = time.time()
    model_id = "IDEA-Research/grounding-dino-base"
    proc  = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id
    ).to(DEVICE)
    print(f"  [GroundingDINO] 로딩 완료 {time.time()-t0:.1f}s")

    preds = []
    lats = []
    for fr in frames:
        t0 = time.time()
        inputs = proc(images=fr["img"], text=PHRASE + ".", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
        W = fr["img"].width
        results = proc.post_process_grounded_object_detection(
            out, inputs["input_ids"],
            box_threshold=0.3, text_threshold=0.25,
            target_sizes=[(fr["img"].height, W)]
        )[0]
        lat = (time.time() - t0) * 1000
        lats.append(lat)
        boxes = results["boxes"]
        if len(boxes) == 0:
            preds.append({"detected": False, "cx": None})
        else:
            scores = results["scores"].cpu().numpy()
            best_i = int(np.argmax(scores))
            x1, _, x2, _ = boxes[best_i].cpu().tolist()
            cx = (x1 + x2) / 2.0 / W
            preds.append({"detected": True, "cx": cx})
    print(f"  [GroundingDINO] latency mean={np.mean(lats):.0f}ms")
    return preds


def run_florence2(frames):
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch
    print("  [Florence-2] 로딩...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True
    ).to(DEVICE)
    proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    print(f"  [Florence-2] 로딩 완료 {time.time()-t0:.1f}s")

    preds = []
    lats = []
    task = "<OPEN_VOCABULARY_DETECTION>"
    for fr in frames:
        t0 = time.time()
        inputs = proc(text=task + PHRASE, images=fr["img"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=128)
        out_text = proc.decode(ids[0], skip_special_tokens=False)
        result = proc.post_process_generation(
            out_text, task=task, image_size=(fr["img"].height, fr["img"].width)
        )
        lat = (time.time() - t0) * 1000
        lats.append(lat)
        bboxes = result.get(task, {}).get("bboxes", [])
        if not bboxes:
            preds.append({"detected": False, "cx": None})
        else:
            W = fr["img"].width
            x1, _, x2, _ = bboxes[0]
            cx = (x1 + x2) / 2.0 / W
            preds.append({"detected": True, "cx": cx})
    print(f"  [Florence-2] latency mean={np.mean(lats):.0f}ms")
    return preds


def run_kosmos2(frames):
    """Pure HF Kosmos-2 (.vlms/kosmos-2-patch14-224) — 로컬 보유."""
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq

    local_path = Path(".vlms/kosmos-2-patch14-224")
    if not local_path.exists():
        print("  [Kosmos-2] 로컬 없음, 스킵")
        return [{"detected": False, "cx": None}] * len(frames)

    print("  [Kosmos-2] 로딩...")
    t0 = time.time()
    model = AutoModelForVision2Seq.from_pretrained(
        str(local_path), torch_dtype=torch.float16
    ).to(DEVICE)
    proc = AutoProcessor.from_pretrained(str(local_path))
    print(f"  [Kosmos-2] 로딩 완료 {time.time()-t0:.1f}s")

    preds = []
    lats = []
    prompt = "<grounding><phrase>gray basket</phrase>"
    for fr in frames:
        t0 = time.time()
        inputs = proc(text=prompt, images=fr["img"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=64)
        out = proc.decode(ids[0], skip_special_tokens=False)
        result = proc.post_process_generation(out, cleanup_and_extract=True)
        lat = (time.time() - t0) * 1000
        lats.append(lat)
        # result: (caption, entities)
        # entities = [("phrase", (tok_start, tok_end), [(x1,y1,x2,y2),...]), ...]
        entities = result[1] if isinstance(result, tuple) and len(result) > 1 else []
        found = False
        for ent in entities:
            ent_bboxes = ent[2] if len(ent) >= 3 else []
            if ent_bboxes:
                bbox = ent_bboxes[0]  # (x1,y1,x2,y2) normalized 0~1
                cx = (bbox[0] + bbox[2]) / 2.0
                preds.append({"detected": True, "cx": cx})
                found = True
                break
        if not found:
            preds.append({"detected": False, "cx": None})
    print(f"  [Kosmos-2] latency mean={np.mean(lats):.0f}ms")
    return preds


# ── PG2 oracle 자체 측정 (기준선) ────────────────────────────────────────────

def run_pg2_oracle(frames):
    """PG2 결과를 직접 h5에서 읽어 기준선으로 사용."""
    preds = []
    for fr in frames:
        if fr["pg2_has"]:
            preds.append({"detected": True, "cx": fr["pg2_cx"]})
        else:
            preds.append({"detected": False, "cx": None})
    return preds


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["clip", "owlv2", "gdino", "florence", "kosmos"],
                        help="모델 목록: clip owlv2 gdino florence kosmos")
    args = parser.parse_args()

    frames = load_eval_frames()
    print(f"\n{'모델':<30} | det    cx_match  dir_match")
    print("─" * 65)

    all_results = {}

    # PG2 oracle 기준선
    preds_pg2 = run_pg2_oracle(frames)
    all_results["PG2(oracle)"] = eval_metrics(preds_pg2, frames, "PG2(oracle)")

    runners = {
        "clip":    run_clip,
        "owlv2":   run_owlv2,
        "gdino":   run_gdino,
        "florence":run_florence2,
        "kosmos":  run_kosmos2,
    }

    for name in args.models:
        if name not in runners:
            print(f"  {name} 알 수 없는 모델, 스킵")
            continue
        try:
            preds = runners[name](frames)
            all_results[name] = eval_metrics(preds, frames, name)
        except Exception as e:
            print(f"  [{name}] 오류: {e}")
            all_results[name] = {"error": str(e)}

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\n결과 저장: {OUT_FILE}")


if __name__ == "__main__":
    main()
