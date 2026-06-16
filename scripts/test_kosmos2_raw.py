#!/usr/bin/env python3
"""
Pure Kosmos-2 종합 테스트 스위트

Option C 설계를 위한 베이스라인 측정.
학습 전후 모두 사용 가능.

테스트 모드:
  --mode grounding     : basket grounding IoU (path_type별)
  --mode caption       : 복도 이미지 free-form 캡션 생성
  --mode zeroshot      : 학습 없이 action 예측 (zero-shot)
  --mode prompt        : 프롬프트 변형 비교 (P1/P2/P3)
  --mode object        : 다른 객체 프롬프트 → action 변화 확인
  --mode all           : 전체 실행

Usage:
  .venv/bin/python3 scripts/test_kosmos2_raw.py --mode all
  .venv/bin/python3 scripts/test_kosmos2_raw.py --mode grounding --n-episodes 20
  .venv/bin/python3 scripts/test_kosmos2_raw.py --mode zeroshot --lora
  .venv/bin/python3 scripts/test_kosmos2_raw.py --mode prompt --n-frames 10
  .venv/bin/python3 scripts/test_kosmos2_raw.py --mode object --n-frames 10
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
LORA_DIR  = ROOT / "runs" / "v5_nav" / "mlp" / "exp53" / "clip_lora_adapter"
DATA_FULL = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
DATA_FL   = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"

CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
IOU_THR = 0.3


# ─── 유틸 ────────────────────────────────────────────────────────────────────

def load_frame(ep_entry, frame_idx, data_dir=None):
    ep_path = Path(ep_entry.get("episode", ep_entry.get("h5_path", "")))
    if ep_path.is_absolute() and ep_path.exists():
        h5_path = ep_path
    else:
        base = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
        candidates = list(base.glob(f"{ep_path.stem}.h5"))
        if not candidates:
            candidates = list(base.glob(f"**/{ep_path.stem}.h5"))
        if not candidates:
            return None
        h5_path = candidates[0]
    try:
        with h5py.File(str(h5_path), "r") as f:
            arr = f["observations"]["images"][frame_idx]
        return Image.fromarray(arr)
    except Exception:
        return None


def compute_iou(pred_box, cx, cy, area):
    side = area ** 0.5
    gx1, gy1 = cx - side / 2, cy - side / 2
    gx2, gy2 = cx + side / 2, cy + side / 2
    px1, py1, px2, py2 = pred_box
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
    return inter / union if union > 0 else 0.0


def extract_grounding_boxes(proc, model, img, phrase="gray basket", device="cuda"):
    prompt = f"<grounding> An image of {phrase}."
    inputs = proc(text=prompt, images=img, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen_text = proc.decode(out[0], skip_special_tokens=False)
    # Kosmos-2 박스 파싱 — post_process_generation returns (clean_text, entities)
    # entities: [(phrase_str, (start, end), [(x1,y1,x2,y2), ...]), ...]
    # boxes are already normalized [0,1]
    clean_text, entities = proc.post_process_generation(gen_text, cleanup_and_extract=True)
    boxes = []
    for ent_phrase, _span, ent_boxes in entities:
        if phrase.lower() in ent_phrase.lower() or "basket" in ent_phrase.lower():
            boxes.extend(ent_boxes)
    return boxes, clean_text


def generate_text(proc, model, img, prompt, max_new_tokens=30, device="cuda"):
    """generate() 호출 후 prompt 토큰 이후의 생성 텍스트만 반환."""
    inputs = proc(text=prompt, images=img, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # prompt 이후 토큰만 디코딩
    new_tokens = out[0][input_len:]
    return proc.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_model(use_lora=False, device="cuda"):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    proc = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16).to(device)
    if use_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(LORA_DIR))
        print("[MODEL] Stage1 LoRA (exp53) 적용됨")
    model.eval()
    return proc, model


def sample_episodes(data, n_episodes, frames_per_ep=3, seed=42):
    rng = np.random.RandomState(seed)
    eps = rng.choice(len(data), min(n_episodes, len(data)), replace=False)
    samples = []
    for i in eps:
        ep = data[i]
        frames = ep.get("frames", [])
        if not frames:
            continue
        idxs = rng.choice(len(frames), min(frames_per_ep, len(frames)), replace=False)
        for fi in idxs:
            samples.append((ep, frames[fi], fi))
    return samples


# ─── 테스트 1: Grounding ────────────────────────────────────────────────────

def test_grounding(proc, model, data, n_episodes, device):
    print("\n" + "="*60)
    print("TEST 1: Basket Grounding (IoU ≥ 0.3)")
    print("="*60)

    samples = sample_episodes(data, n_episodes, frames_per_ep=4)
    by_path = defaultdict(lambda: {"hit": 0, "miss": 0, "fp": 0, "total": 0})
    overall = {"hit": 0, "miss": 0, "fp": 0, "total_present": 0, "total_absent": 0}

    # cx/cy 컬럼명 자동 감지
    fr0 = data[0]["frames"][0] if data[0].get("frames") else {}
    cx_key  = "cx_det" if "cx_det" in fr0 else "cx"
    cy_key  = "cy_det" if "cy_det" in fr0 else "cy"
    ar_key  = "area_det" if "area_det" in fr0 else "area"
    has_key = "detected" if "detected" in fr0 else "has_bbox"

    for ep, fr, fi in samples:
        img = load_frame(ep, fr.get("frame_idx", fi))
        if img is None:
            continue
        pt = ep.get("path_type", "unknown")
        cx   = fr.get(cx_key, 0.5)
        cy   = fr.get(cy_key, 0.5)
        area = fr.get(ar_key, 0.05)
        has  = bool(fr.get(has_key, fr.get("has_bbox", False)))

        boxes, _ = extract_grounding_boxes(proc, model, img, "gray basket", device)

        by_path[pt]["total"] += 1
        if has:
            overall["total_present"] += 1
            if boxes:
                best_iou = max(compute_iou(b, cx, cy, area) for b in boxes)
                if best_iou >= IOU_THR:
                    by_path[pt]["hit"] += 1
                    overall["hit"] += 1
                else:
                    by_path[pt]["miss"] += 1
                    overall["miss"] += 1
            else:
                by_path[pt]["miss"] += 1
                overall["miss"] += 1
        else:
            overall["total_absent"] += 1
            if boxes:
                by_path[pt]["fp"] += 1
                overall["fp"] += 1

    # 출력
    n_present = overall["total_present"]
    hit_rate  = overall["hit"] / n_present if n_present > 0 else 0
    fp_rate   = overall["fp"]  / max(1, overall["total_absent"])
    print(f"\n  basket_present 프레임: hit_rate = {hit_rate:.1%}  ({overall['hit']}/{n_present})")
    print(f"  basket_absent  프레임: fp_rate  = {fp_rate:.1%}   ({overall['fp']}/{overall['total_absent']})")
    print(f"\n  path_type별 grounding 성공률 (basket present만):")
    print(f"  {'path_type':<22} {'hit':>5} {'miss':>5} {'rate':>7}")
    print(f"  {'-'*45}")
    for pt, v in sorted(by_path.items()):
        total_pres = v["hit"] + v["miss"]
        if total_pres == 0:
            continue
        rate = v["hit"] / total_pres
        print(f"  {pt:<22} {v['hit']:>5} {v['miss']:>5} {rate:>7.1%}")

    return {"hit_rate": hit_rate, "fp_rate": fp_rate, "by_path": dict(by_path)}


# ─── 테스트 2: Caption ───────────────────────────────────────────────────────

def test_caption(proc, model, data, n_frames, device):
    print("\n" + "="*60)
    print("TEST 2: Free-Form Scene Caption")
    print("="*60)
    print("  Pure Kosmos-2가 복도 이미지를 어떻게 묘사하는지 확인\n")

    samples = sample_episodes(data, n_frames, frames_per_ep=1)[:n_frames]
    prompt = "<image> Describe what you see in this image:"

    for i, (ep, fr, fi) in enumerate(samples[:10]):
        img = load_frame(ep, fr.get("frame_idx", fi))
        if img is None:
            continue
        pt   = ep.get("path_type", "?")
        has  = bool(fr.get("has_bbox", fr.get("detected", False)))
        text = generate_text(proc, model, img, prompt, max_new_tokens=40, device=device)
        print(f"  [{i+1:2d}] {pt:<22} bbox={'Y' if has else 'N'}  →  {text[:80]}")

    return {}


# ─── 테스트 3: Zero-shot Action Prediction ───────────────────────────────────

def test_zeroshot(proc, model, data, n_episodes, device):
    print("\n" + "="*60)
    print("TEST 3: Zero-Shot Action Prediction (학습 없이)")
    print("="*60)

    action_str = "/".join(CLASS_NAMES[1:])  # STOP 제외
    prompt_tmpl = (
        "<image> A mobile robot is navigating a corridor following a gray basket. "
        f"Choose the next navigation action from: {action_str}. "
        "Answer with exactly one action word:"
    )

    samples = sample_episodes(data, n_episodes, frames_per_ep=3)
    correct = 0
    total   = 0
    by_class = defaultdict(lambda: {"hit": 0, "total": 0})
    outputs  = []

    for ep, fr, fi in samples:
        img = load_frame(ep, fr.get("frame_idx", fi))
        if img is None:
            continue
        gt_class = fr.get("gt_class", 0)
        gt_name  = CLASS_NAMES[gt_class]

        text = generate_text(proc, model, img, prompt_tmpl, max_new_tokens=5, device=device)
        pred_word = text.split()[0].upper() if text.split() else "?"
        pred_class = next((n for n in CLASS_NAMES if n.upper() in pred_word or pred_word in n.upper()), None)

        hit = (pred_class == gt_name)
        correct += int(hit)
        total   += 1
        by_class[gt_name]["hit"]   += int(hit)
        by_class[gt_name]["total"] += 1
        outputs.append((gt_name, pred_class or pred_word, hit))

    acc = correct / total if total > 0 else 0
    print(f"\n  Zero-shot accuracy: {acc:.1%}  ({correct}/{total})")
    print(f"\n  클래스별:")
    print(f"  {'class':<12} {'hit':>5} {'total':>7} {'rate':>7}")
    print(f"  {'-'*35}")
    for cls in CLASS_NAMES:
        v = by_class[cls]
        if v["total"] == 0:
            continue
        rate = v["hit"] / v["total"]
        print(f"  {cls:<12} {v['hit']:>5} {v['total']:>7} {rate:>7.1%}")

    print(f"\n  예시 출력 (GT → 예측):")
    for gt, pred, hit in outputs[:10]:
        mark = "✅" if hit else "❌"
        print(f"    {mark} {gt:<12} → {pred}")

    return {"accuracy": acc, "n": total}


# ─── 테스트 4: Prompt Ablation (P1 vs P2 vs P3) ──────────────────────────────

def test_prompt(proc, model, data, n_frames, device):
    print("\n" + "="*60)
    print("TEST 4: Prompt Strategy Comparison (P1 vs P2 vs P3)")
    print("="*60)

    # cx_key 감지
    fr0 = data[0]["frames"][0] if data[0].get("frames") else {}
    cx_key = "cx_det" if "cx_det" in fr0 else "cx"
    cy_key = "cy_det" if "cy_det" in fr0 else "cy"
    ar_key = "area_det" if "area_det" in fr0 else "area"

    action_str = "/".join(CLASS_NAMES[1:])
    samples = sample_episodes(data, n_frames, frames_per_ep=1)[:n_frames]

    results = defaultdict(lambda: {"correct": 0, "total": 0, "outputs": []})

    for ep, fr, fi in samples:
        img = load_frame(ep, fr.get("frame_idx", fi))
        if img is None:
            continue
        gt   = CLASS_NAMES[fr.get("gt_class", 0)]
        cx   = fr.get(cx_key, 0.5)
        cy   = fr.get(cy_key, 0.5)
        area = fr.get(ar_key, 0.05)

        prompts = {
            "P1-blind":   (f"<image> Follow the gray basket in the corridor. "
                          f"Navigation action ({action_str}):"),
            "P2-bbox":    (f"<image> Gray basket at ({cx:.2f}, {cy:.2f}), area={area:.3f}. "
                          f"Navigation action ({action_str}):"),
            "P3-grounding": None,  # 별도 처리
        }

        for pname, prompt in prompts.items():
            if pname == "P3-grounding":
                # Step 1: grounding
                boxes, _ = extract_grounding_boxes(proc, model, img, "gray basket", device)
                if boxes:
                    # 가장 큰 box 사용
                    box = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                    gcx = (box[0]+box[2])/2
                    gcy = (box[1]+box[3])/2
                    prompt = (f"<image> Basket detected at ({gcx:.2f}, {gcy:.2f}). "
                              f"Navigation action ({action_str}):")
                else:
                    prompt = (f"<image> Basket not detected. "
                              f"Navigation action ({action_str}):")

            text = generate_text(proc, model, img, prompt, max_new_tokens=5, device=device)
            pred_w = text.split()[0].upper() if text.split() else "?"
            pred = next((n for n in CLASS_NAMES if n.upper() in pred_w or pred_w in n.upper()), pred_w)
            hit = (pred == gt)
            results[pname]["correct"] += int(hit)
            results[pname]["total"]   += 1
            results[pname]["outputs"].append((gt, pred or pred_w, hit))

    print(f"\n  {'프롬프트':<15} {'acc':>7}  {'correct':>8} {'total':>6}")
    print(f"  {'-'*45}")
    for pname, v in results.items():
        acc = v["correct"] / v["total"] if v["total"] > 0 else 0
        print(f"  {pname:<15} {acc:>7.1%}  {v['correct']:>8} {v['total']:>6}")

    # 일치율 분석: 3가지 프롬프트가 동일 예측 내리는 비율
    print(f"\n  [참고] 모든 프롬프트가 동일 답을 내는 비율: 학습 전이면 높을수록 prompt-insensitive")

    return {k: {"acc": v["correct"]/max(1,v["total"])} for k, v in results.items()}


# ─── 테스트 5: Object Substitution ──────────────────────────────────────────

def test_object(proc, model, data, n_frames, device):
    print("\n" + "="*60)
    print("TEST 5: Object Substitution")
    print("  '다른 물체'를 프롬프트에 넣으면 action이 달라지는가?")
    print("="*60)

    action_str = "/".join(CLASS_NAMES[1:])
    samples = sample_episodes(data, n_frames, frames_per_ep=1)[:n_frames]

    objects = [
        ("gray basket", "우리 목표"),
        ("red ball",    "다른 물체"),
        ("person",      "사람"),
        ("door",        "문"),
    ]

    by_obj = defaultdict(list)

    for ep, fr, fi in samples[:n_frames]:
        img = load_frame(ep, fr.get("frame_idx", fi))
        if img is None:
            continue
        gt = CLASS_NAMES[fr.get("gt_class", 0)]

        preds = {}
        for obj, label in objects:
            prompt = (f"<image> The robot must follow the {obj}. "
                      f"Navigation action ({action_str}):")
            text = generate_text(proc, model, img, prompt, max_new_tokens=5, device=device)
            suffix = text[len(prompt.replace("<image> ", "")):].strip()
            pred_w = suffix.split()[0].upper() if suffix.split() else "?"
            pred = next((n for n in CLASS_NAMES if n.upper() in pred_w or pred_w in n.upper()), pred_w)
            preds[obj] = pred
            by_obj[obj].append(pred)

        gt_pred = preds.get("gray basket", "?")
        print(f"  GT={gt:<10}  basket→{gt_pred:<10}  "
              + "  ".join(f"{o}→{preds.get(o,'?')}" for o, _ in objects[1:]))

    # 요약: basket과 다른 물체간 일치율
    print(f"\n  [일치율] 다른 물체 프롬프트가 basket 프롬프트와 같은 action 내는 비율:")
    basket_preds = by_obj.get("gray basket", [])
    for obj, label in objects[1:]:
        other_preds = by_obj.get(obj, [])
        n_same = sum(1 for a, b in zip(basket_preds, other_preds) if a == b)
        rate = n_same / max(1, len(basket_preds))
        flag = "⚠️ 구분 불가" if rate > 0.7 else "✅ 구분됨"
        print(f"    {label:<8} ({obj:<12}): {rate:.1%} 일치  {flag}")

    return {}


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["grounding","caption","zeroshot","prompt","object","all"],
                   default="all")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--n-frames",   type=int, default=15)
    p.add_argument("--lora",       action="store_true", help="Stage1 LoRA (exp53) 적용")
    p.add_argument("--ckpt",       type=str, default=None, help="Option C fine-tuned ckpt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    # 데이터 로드 (frame-level 우선, fallback full)
    if DATA_FL.exists():
        data = json.loads(DATA_FL.read_text())
        print(f"[DATA] {len(data)} episodes (frame-level)")
    else:
        data = json.loads(DATA_FULL.read_text())
        print(f"[DATA] {len(data)} episodes (full)")

    # 모델 로드
    tag = "Stage1 LoRA (exp53)" if args.lora else "Base Kosmos-2"
    if args.ckpt:
        tag = f"Option C ({args.ckpt})"
    print(f"[MODEL] {tag} 로드 중...")
    proc, model = load_model(use_lora=args.lora, device=device)
    if args.ckpt:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.ckpt)
        model.eval()
    print(f"[MODEL] 완료\n")

    results = {}
    modes = ["grounding","caption","zeroshot","prompt","object"] if args.mode == "all" else [args.mode]

    for m in modes:
        if m == "grounding":
            results["grounding"] = test_grounding(proc, model, data, args.n_episodes, device)
        elif m == "caption":
            results["caption"] = test_caption(proc, model, data, args.n_frames, device)
        elif m == "zeroshot":
            results["zeroshot"] = test_zeroshot(proc, model, data, args.n_episodes, device)
        elif m == "prompt":
            results["prompt"] = test_prompt(proc, model, data, args.n_frames, device)
        elif m == "object":
            results["object"] = test_object(proc, model, data, args.n_frames, device)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  모델: {tag}")
    if "grounding" in results:
        g = results["grounding"]
        print(f"  Grounding hit_rate:   {g['hit_rate']:.1%}")
    if "zeroshot" in results:
        z = results["zeroshot"]
        print(f"  Zero-shot accuracy:   {z['accuracy']:.1%}  (n={z['n']})")
    if "prompt" in results:
        for pname, v in results["prompt"].items():
            print(f"  Prompt {pname:<12}: {v['acc']:.1%}")
    print("\n[참고]")
    print("  학습 후 재실행 → --ckpt runs/v5_nav/optionC/ckpt_best")
    print("  LoRA 비교      → --lora")
    print("  특정 모드만    → --mode grounding|caption|zeroshot|prompt|object")


if __name__ == "__main__":
    main()
