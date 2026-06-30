"""
Preview Model Self-Labeling Thumbnail Generator (v2)

5개 모델 × V5(path_type별 3ep, frame 0~2) + 세션(frame 0~2) 썸네일 생성.
각 썸네일에 5개 모델 bbox를 색깔별로 오버레이.

Usage:
  .venv/bin/python3 -u scripts/gen_preview_label_thumbs_v2.py
  .venv/bin/python3 -u scripts/gen_preview_label_thumbs_v2.py --n-per-type 2
  .venv/bin/python3 -u scripts/gen_preview_label_thumbs_v2.py --models kosmos owlv2
"""
import argparse, io, json, re, time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT     = Path(__file__).parent.parent
V5_DIR   = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
SESS_DIR = ROOT / "docs" / "inference_sessions"
ANN_FILE = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_pg2_cx_freehsv.json"
OUT_DIR  = ROOT / "docs" / "v5" / "preview_label_thumbs_v2"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

THUMB_W, THUMB_H = 640, 360

MODEL_COLORS = {
    "K_current":        (255,  80,  80),   # 빨강
    "K_refexp_laundry": ( 60, 220,  60),   # 초록
    "O_laundry":        (240, 200,   0),   # 노랑
    "PG":               ( 80, 150, 255),   # 파랑
    "Florence2":        (255, 140,   0),   # 주황
}
MODEL_SHORT = {
    "K_current":"Kc", "K_refexp_laundry":"Kr",
    "O_laundry":"Ow", "PG":"PG", "Florence2":"F2",
}


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def raw_to_pil(raw):
    if raw.ndim == 1:
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    return Image.fromarray(raw.astype(np.uint8)).convert("RGB")


def draw_thumb(img, model_results):
    thumb = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
    draw  = ImageDraw.Draw(thumb)
    # 중앙선
    draw.line([(THUMB_W//2, 0), (THUMB_W//2, THUMB_H)], fill=(160,160,160), width=1)
    for mname, r in model_results.items():
        if r is None:
            continue
        color = MODEL_COLORS[mname]
        cx = r["cx"]
        x1 = int(r.get("x1", cx-0.07) * THUMB_W)
        y1 = int(r.get("y1", 0.30)    * THUMB_H)
        x2 = int(r.get("x2", cx+0.07) * THUMB_W)
        y2 = int(r.get("y2", 0.80)    * THUMB_H)
        draw.rectangle([x1,y1,x2,y2], outline=color, width=2)
        draw.text((x1+2, y1+1), MODEL_SHORT[mname], fill=color)
    return thumb


# ── 데이터 선택 ───────────────────────────────────────────────────────────────

def select_v5_episodes(n_per_type=3):
    ann = json.load(open(ANN_FILE))
    by_type = defaultdict(list)
    for ep in ann:
        pt = ep.get("path_type","?")
        if pt in ("free","?"): continue
        name = Path(ep["episode"]).name
        if name not in by_type[pt]:
            by_type[pt].append(name)
    selected = []
    for pt in sorted(by_type):
        for name in by_type[pt][:n_per_type]:
            p = V5_DIR / name
            if p.exists():
                selected.append({"path": p, "path_type": pt, "source": "v5"})
    print(f"[V5] {len(selected)}개 에피소드 (path_type별 최대 {n_per_type}개)")
    return selected


def select_sessions():
    eps = [{"path": sp, "path_type": "session", "source": "session"}
           for sp in sorted(SESS_DIR.glob("session_20260626*.h5"))]
    print(f"[Session] {len(eps)}개")
    return eps


def load_frames(ep_info, max_frames=3):
    frames = []
    with h5py.File(ep_info["path"]) as f:
        imgs = f["observations"]["images"]
        n = min(max_frames, imgs.shape[0])
        for i in range(n):
            frames.append({
                "img":       raw_to_pil(imgs[i]),
                "frame_idx": i,
                "path_type": ep_info["path_type"],
                "source":    ep_info["source"],
                "episode":   ep_info["path"].name,
                "key":       f"{ep_info['path'].stem}_f{i}",
            })
    return frames


# ── Kosmos-2 ──────────────────────────────────────────────────────────────────

_CAPTION_DIR = [
    (["far left","leftmost","bottom left","left side","left corner"], 0.15),
    (["left"], 0.25),
    (["far right","rightmost","bottom right","right side","right corner"], 0.85),
    (["right"], 0.75),
    (["center","middle","straight","front"], 0.50),
]
_TARGET_KW = {"basket","container","bin","laundry","gray box","pot"}


def _kosmos_parse(caption, entities):
    for ename, _span, boxes in entities:
        for box in boxes:
            x1,y1,x2,y2 = [float(v) for v in box]
            if max(x1,y1,x2,y2) > 1.5:
                x1,y1,x2,y2 = x1/1000,y1/1000,x2/1000,y2/1000
            area = (x2-x1)*(y2-y1)
            if area > 0.85: continue
            if any(k in ename.lower() for k in _TARGET_KW):
                return {"cx":(x1+x2)/2,"x1":x1,"y1":y1,"x2":x2,"y2":y2}
    for phrases, cx in _CAPTION_DIR:
        if any(p in caption.lower() for p in phrases):
            return {"cx":cx,"x1":cx-0.06,"y1":0.35,"x2":cx+0.06,"y2":0.75}
    return None


def run_kosmos(frames):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    print("\n[Kosmos-2] 로딩...")
    t0 = time.time()
    proc  = AutoProcessor.from_pretrained(str(ROOT/".vlms"/"kosmos-2-patch14-224"))
    model = AutoModelForVision2Seq.from_pretrained(
                str(ROOT/".vlms"/"kosmos-2-patch14-224")).to(DEVICE).eval()
    print(f"  로드 {time.time()-t0:.1f}s")

    def _run(img, prompt):
        inp = proc(text=prompt, images=img, return_tensors="pt")
        inp = {k: v.to(DEVICE) for k,v in inp.items()}
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=64, use_cache=True)
        new_ids = gen[:, inp["input_ids"].shape[1]:]
        raw = proc.batch_decode(new_ids, skip_special_tokens=False)[0]
        caption, entities = proc.post_process_generation(raw)
        return _kosmos_parse(caption, entities)

    K_current = {}; K_ref = {}
    for i, fr in enumerate(frames):
        K_current[fr["key"]] = _run(fr["img"], "<grounding>The basket is at")
        K_ref[fr["key"]]     = _run(fr["img"], "<grounding><phrase>gray laundry basket</phrase>")
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(frames)} ({(time.time()-t0):.0f}s)")

    del model; torch.cuda.empty_cache()
    return {"K_current": K_current, "K_refexp_laundry": K_ref}


# ── OWL-v2 ───────────────────────────────────────────────────────────────────

def run_owlv2(frames):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    print("\n[OWL-v2] 로딩...")
    t0 = time.time()
    proc  = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
                "google/owlv2-base-patch16-ensemble").to(DEVICE).eval()
    print(f"  로드 {time.time()-t0:.1f}s")

    results = {}
    for i, fr in enumerate(frames):
        img = fr["img"]; W,H = img.width, img.height
        inp = proc(text=[["gray laundry basket"]], images=img,
                   return_tensors="pt").to(DEVICE)
        with torch.no_grad(): out = model(**inp)
        res  = proc.post_process_object_detection(
                   out, threshold=0.1, target_sizes=[(H,W)])[0]
        boxes = res["boxes"]
        if len(boxes) == 0:
            results[fr["key"]] = None
        else:
            best = int(res["scores"].argmax())
            x1,y1,x2,y2 = boxes[best].cpu().tolist()
            results[fr["key"]] = {
                "cx":(x1+x2)/2/W, "x1":x1/W,"y1":y1/H,"x2":x2/W,"y2":y2/H}
        if (i+1) % 30 == 0:
            print(f"  {i+1}/{len(frames)}")

    del model; torch.cuda.empty_cache()
    return results


# ── PaliGemma2 ────────────────────────────────────────────────────────────────

def run_paligemma(frames):
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    print("\n[PaliGemma2] 로딩...")
    t0 = time.time()
    proc  = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-448")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
                "google/paligemma2-3b-mix-448",
                torch_dtype=torch.bfloat16).to(DEVICE).eval()
    print(f"  로드 {time.time()-t0:.1f}s")

    results = {}
    for i, fr in enumerate(frames):
        img = fr["img"].resize((448,448))
        inp = proc(text="<image>detect basket", images=img,
                   return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=100)
        raw = proc.decode(out[0], skip_special_tokens=False)
        locs = re.findall(r'<loc(\d+)>', raw)
        if len(locs) >= 4:
            y1,x1,y2,x2 = [int(l)/1024 for l in locs[:4]]
            results[fr["key"]] = {
                "cx":(x1+x2)/2,"x1":x1,"y1":y1,"x2":x2,"y2":y2}
        else:
            results[fr["key"]] = None
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(frames)}")

    del model; torch.cuda.empty_cache()
    return results


# ── Florence-2 ────────────────────────────────────────────────────────────────

def run_florence(frames):
    from transformers import AutoProcessor, AutoModelForCausalLM
    print("\n[Florence-2] 로딩...")
    t0 = time.time()
    proc  = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-base",
                trust_remote_code=True).to(DEVICE).eval()
    print(f"  로드 {time.time()-t0:.1f}s")
    TASK = "<OPEN_VOCABULARY_DETECTION>"

    results = {}
    for i, fr in enumerate(frames):
        img = fr["img"]; W,H = img.width, img.height
        inp = proc(text=TASK+"basket", images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): ids = model.generate(**inp, max_new_tokens=128)
        txt = proc.decode(ids[0], skip_special_tokens=False)
        res = proc.post_process_generation(txt, task=TASK, image_size=(H,W))
        bbs = res.get(TASK,{}).get("bboxes",[])
        if not bbs:
            results[fr["key"]] = None
        else:
            x1,y1,x2,y2 = bbs[0]
            results[fr["key"]] = {
                "cx":(x1+x2)/2/W,"x1":x1/W,"y1":y1/H,"x2":x2/W,"y2":y2/H}
        if (i+1) % 30 == 0:
            print(f"  {i+1}/{len(frames)}")

    del model; torch.cuda.empty_cache()
    return results


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-type", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=3)
    p.add_argument("--models", nargs="+",
                   default=["kosmos","owlv2","paligemma","florence"],
                   choices=["kosmos","owlv2","paligemma","florence"])
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 프레임 수집
    v5_eps   = select_v5_episodes(args.n_per_type)
    sess_eps = select_sessions()
    all_frames = []
    for ep in v5_eps + sess_eps:
        all_frames.extend(load_frames(ep, args.max_frames))
    print(f"총 {len(all_frames)}프레임")

    # 추론
    all_res = {m: {} for m in MODEL_COLORS}

    if "kosmos" in args.models:
        k = run_kosmos(all_frames)
        all_res["K_current"]        = k["K_current"]
        all_res["K_refexp_laundry"] = k["K_refexp_laundry"]

    if "owlv2" in args.models:
        all_res["O_laundry"] = run_owlv2(all_frames)

    if "paligemma" in args.models:
        all_res["PG"] = run_paligemma(all_frames)

    if "florence" in args.models:
        all_res["Florence2"] = run_florence(all_frames)

    # 썸네일 생성 + 메타
    print("\n[썸네일] 생성 중...")
    meta = []
    for fr in all_frames:
        key = fr["key"]
        model_results = {mn: all_res[mn].get(key) for mn in MODEL_COLORS}
        thumb = draw_thumb(fr["img"], model_results)
        fname = f"thumb_{key}.jpg"
        thumb.save(OUT_DIR / fname, quality=83)
        meta.append({
            "key":       key,
            "fname":     fname,
            "episode":   fr["episode"],
            "frame_idx": fr["frame_idx"],
            "path_type": fr["path_type"],
            "source":    fr["source"],
            "models": {
                mn: ({"cx": round(r["cx"],4),
                      "x1": round(r.get("x1",0),4),
                      "y1": round(r.get("y1",0),4),
                      "x2": round(r.get("x2",1),4),
                      "y2": round(r.get("y2",1),4)} if r else None)
                for mn, r in model_results.items()
            },
        })

    json.dump(meta, open(OUT_DIR/"meta.json","w"), indent=2, ensure_ascii=False)
    print(f"완료: {len(meta)}개 → {OUT_DIR}")
    det_stats = {mn: sum(1 for fr in meta if fr["models"][mn]) for mn in MODEL_COLORS}
    print("검출률:", {mn: f"{n}/{len(meta)}" for mn,n in det_stats.items()})


if __name__ == "__main__":
    main()
