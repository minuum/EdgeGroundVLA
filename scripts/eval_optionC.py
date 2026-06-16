#!/usr/bin/env python3
"""
Option C 평가 스크립트 — generate() 기반

학습된 LoRA adapter를 로드해 validation set 평가.
Exp54 (92.6%) 대비 성능 비교.

Usage:
  .venv/bin/python3 scripts/eval_optionC.py
  .venv/bin/python3 scripts/eval_optionC.py --ckpt runs/v5_nav/optionC/optionC_p2_r8/adapter_best
  .venv/bin/python3 scripts/eval_optionC.py --mode object   # 다른 물체 테스트
  .venv/bin/python3 scripts/eval_optionC.py --mode prompt   # 프롬프트 민감도
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
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
DATA_DIR  = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
CKPT_DIR  = ROOT / "runs" / "v5_nav" / "optionC"

CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]


def load_frame(ep_entry):
    ep_path = Path(ep_entry["episode"])
    if ep_path.is_absolute() and ep_path.exists():
        h5_path = ep_path
    else:
        cands = list(DATA_DIR.glob(f"{ep_path.stem}.h5"))
        if not cands:
            cands = list(DATA_DIR.glob(f"**/{ep_path.stem}.h5"))
        if not cands:
            return None, None
        h5_path = cands[0]
    frames_data = []
    try:
        with h5py.File(str(h5_path), "r") as f:
            for fr in ep_entry["frames"]:
                arr = f["observations"]["images"][fr.get("frame_idx", 0)]
                frames_data.append(Image.fromarray(arr))
    except Exception as e:
        print(f"  [SKIP] {h5_path}: {e}")
        return None, None
    return frames_data, ep_entry["frames"]


def build_prompt_p2(fr):
    action_str = "/".join(CLASS_NAMES[1:])
    cx, cy, area = fr.get("cx", 0.5), fr.get("cy", 0.5), fr.get("area", 0.05)
    has = bool(fr.get("has_bbox", False))
    if has:
        loc = "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")
        return (f"<image> Gray basket at ({cx:.2f}, {cy:.2f}), area={area:.3f} ({loc}). "
                f"Navigation action ({action_str}):")
    return f"<image> No basket visible. Navigation action ({action_str}):"


def predict_action(proc, model, img, prompt, action_token_ids, device):
    enc = proc(text=prompt, images=img, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    att_mask  = enc["attention_mask"].to(device)
    pv = enc.get("pixel_values")
    if pv is not None:
        pv = pv.to(device, dtype=torch.float16)
    iep_mask = enc.get("image_embeds_position_mask")
    if iep_mask is not None:
        iep_mask = iep_mask.to(device)

    kw = dict(input_ids=input_ids, attention_mask=att_mask, max_new_tokens=3, do_sample=False)
    if pv is not None:
        kw["pixel_values"] = pv
    if iep_mask is not None:
        kw["image_embeds_position_mask"] = iep_mask

    with torch.no_grad(), torch.cuda.amp.autocast():
        gen = model.generate(**kw)
    new_tok = gen[0][input_ids.shape[1]]
    pred_class = (action_token_ids == new_tok).nonzero(as_tuple=True)
    pred = pred_class[0][0].item() if len(pred_class[0]) > 0 else -1
    pred_text = proc.tokenizer.decode([new_tok.item()], skip_special_tokens=True).strip()
    return pred, pred_text


# ─── 메인 평가 ───────────────────────────────────────────────────────────────

def eval_accuracy(proc, model, val_eps, action_token_ids, device):
    print("\n[기본 정확도 평가 — generate() 기반]")
    correct = 0
    total   = 0
    by_path = defaultdict(lambda: {"c": 0, "t": 0})
    by_class = defaultdict(lambda: {"c": 0, "t": 0})
    wrong_examples = []

    for ep in val_eps:
        imgs, frames = load_frame(ep)
        if imgs is None:
            continue
        pt = ep.get("path_type", "unknown")
        for img, fr in zip(imgs, frames):
            gt  = fr.get("gt_class", 1)
            prompt = build_prompt_p2(fr)
            pred, pred_text = predict_action(proc, model, img, prompt, action_token_ids, device)

            hit = (pred == gt)
            correct += int(hit)
            total   += 1
            by_path[pt]["c"]  += int(hit)
            by_path[pt]["t"]  += 1
            by_class[CLASS_NAMES[gt]]["c"] += int(hit)
            by_class[CLASS_NAMES[gt]]["t"] += 1
            if not hit and len(wrong_examples) < 5:
                wrong_examples.append((CLASS_NAMES[gt], pred_text))

    acc = correct / total if total > 0 else 0
    print(f"\n  전체 accuracy: {acc:.1%}  ({correct}/{total})")
    print(f"  [참고] Exp54 Stage2 v2: 92.6%  |  zero-shot: ??%")

    print(f"\n  path_type별:")
    print(f"  {'path_type':<22} {'acc':>7}  {'n':>5}")
    for pt in sorted(by_path.keys()):
        v = by_path[pt]
        print(f"  {pt:<22} {v['c']/max(1,v['t']):>7.1%}  {v['t']:>5}")

    print(f"\n  action별:")
    for cls in CLASS_NAMES:
        v = by_class.get(cls, {"c": 0, "t": 0})
        if v["t"] == 0:
            continue
        print(f"  {cls:<12} {v['c']/v['t']:.1%}  ({v['c']}/{v['t']})")

    if wrong_examples:
        print(f"\n  오분류 예시 (GT → 예측):")
        for gt_name, pred_text in wrong_examples:
            print(f"    {gt_name} → '{pred_text}'")

    return acc


def eval_object_substitution(proc, model, val_eps, action_token_ids, device):
    print("\n[객체 대체 테스트 — 교수님 핵심 요구]")
    print("  '다른 물체를 넣으면 이상한 행동을 해야 한다'")

    action_str = "/".join(CLASS_NAMES[1:])
    objects = [
        ("gray basket", "원래 목표"),
        ("red ball",    "다른 물체 1"),
        ("person",      "다른 물체 2"),
        ("nothing",     "목표 없음"),
    ]

    results = {o: {"same_as_basket": 0, "total": 0} for o, _ in objects[1:]}
    basket_preds = []
    other_preds  = {o: [] for o, _ in objects[1:]}

    sample_eps = val_eps[:10]
    for ep in sample_eps:
        imgs, frames = load_frame(ep)
        if imgs is None:
            continue
        for img, fr in zip(imgs[:3], frames[:3]):
            preds_this = {}
            for obj, label in objects:
                cx = fr.get("cx", 0.5)
                cy = fr.get("cy", 0.5)
                area = fr.get("area", 0.05)
                if obj == "gray basket":
                    prompt = (f"<image> Gray basket at ({cx:.2f}, {cy:.2f}), area={area:.3f}. "
                              f"Navigation action ({action_str}):")
                elif obj == "nothing":
                    prompt = (f"<image> No target visible in the corridor. "
                              f"Navigation action ({action_str}):")
                else:
                    prompt = (f"<image> The robot must follow the {obj} in the corridor. "
                              f"Navigation action ({action_str}):")
                pred, _ = predict_action(proc, model, img, prompt, action_token_ids, device)
                preds_this[obj] = pred

            basket_preds.append(preds_this.get("gray basket", -1))
            for obj, label in objects[1:]:
                other_preds[obj].append(preds_this.get(obj, -1))

    # 일치율 계산
    print(f"\n  {'객체':<14} {'basket과 같은 action 비율':>25}  {'판정':>12}")
    print(f"  {'-'*55}")
    for obj, label in objects[1:]:
        n_same = sum(1 for a, b in zip(basket_preds, other_preds[obj]) if a == b)
        rate = n_same / max(1, len(basket_preds))
        if rate < 0.4:
            verdict = "✅ 구분됨 (good)"
        elif rate < 0.7:
            verdict = "⚠️  부분 구분"
        else:
            verdict = "❌ 구분 불가"
        print(f"  {label:<14} {rate:>25.1%}  {verdict}")

    print(f"\n  [해석]")
    print(f"  basket과 다른 물체 → action이 달라지면: text path가 살아있고 객체를 구분함")
    print(f"  basket과 다른 물체 → action이 같으면:  visual(장면)만 보고 결정 (text 무시)")


def eval_prompt_sensitivity(proc, model, val_eps, action_token_ids, device):
    print("\n[프롬프트 민감도 — P1 vs P2 vs no-bbox]")

    action_str = "/".join(CLASS_NAMES[1:])
    sample_eps = val_eps[:5]
    n_agree = 0
    n_total = 0

    for ep in sample_eps:
        imgs, frames = load_frame(ep)
        if imgs is None:
            continue
        for img, fr in zip(imgs[:3], frames[:3]):
            gt = fr.get("gt_class", 1)
            cx, cy, area = fr.get("cx", 0.5), fr.get("cy", 0.5), fr.get("area", 0.05)
            prompts = {
                "P1-blind": (f"<image> Follow the gray basket. "
                             f"Navigation action ({action_str}):"),
                "P2-bbox":  (f"<image> Gray basket at ({cx:.2f}, {cy:.2f}), area={area:.3f}. "
                             f"Navigation action ({action_str}):"),
                "P-empty":  (f"<image> Navigation action ({action_str}):"),
            }
            preds = {}
            for pname, prompt in prompts.items():
                pred, _ = predict_action(proc, model, img, prompt, action_token_ids, device)
                preds[pname] = pred

            all_same = len(set(preds.values())) == 1
            n_agree += int(all_same)
            n_total += 1
            mark = "=" if all_same else "≠"
            print(f"  GT={CLASS_NAMES[gt]:<10}  "
                  + "  ".join(f"{p}={CLASS_NAMES[v] if 0<=v<8 else '?'}" for p, v in preds.items())
                  + f"  {mark}")

    agree_rate = n_agree / max(1, n_total)
    print(f"\n  전체 일치율: {agree_rate:.1%}")
    print(f"  [해석] 높으면 prompt-insensitive (text 무시), 낮으면 text 활용 중")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None,
                   help="LoRA adapter 경로 (없으면 best 자동 탐색)")
    p.add_argument("--mode", choices=["accuracy", "object", "prompt", "all"],
                   default="all")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # 체크포인트 탐색
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        candidates = sorted(CKPT_DIR.glob("*/adapter_best"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("[ERROR] 학습된 체크포인트가 없습니다.")
            print(f"  먼저 실행: .venv/bin/python3 scripts/train_optionC_lora.py")
            return
        ckpt_path = candidates[0]
    print(f"[CKPT] {ckpt_path}")

    # 모델 로드
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import PeftModel

    proc  = AutoProcessor.from_pretrained(str(VLM_PATH))
    base  = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(ckpt_path)).to(device).eval()

    # action token ids
    action_token_ids = torch.tensor(
        [proc.tokenizer(n, add_special_tokens=False)["input_ids"][0] for n in CLASS_NAMES],
        dtype=torch.long, device=device
    )
    print(f"[TOKEN] {dict(zip(CLASS_NAMES, action_token_ids.tolist()))}")

    # 데이터 (validation split)
    data = json.loads(DATA_PATH.read_text())
    labels = [ep["path_type"] for ep in data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, va_idx = next(sss.split(np.zeros(len(data)), labels))
    val_eps = [data[i] for i in va_idx]
    print(f"[DATA] {len(val_eps)} validation episodes")

    modes = ["accuracy", "object", "prompt"] if args.mode == "all" else [args.mode]
    for m in modes:
        if m == "accuracy":
            eval_accuracy(proc, model, val_eps, action_token_ids, device)
        elif m == "object":
            eval_object_substitution(proc, model, val_eps, action_token_ids, device)
        elif m == "prompt":
            eval_prompt_sensitivity(proc, model, val_eps, action_token_ids, device)

    print("\n" + "="*55)
    print("평가 완료. 교수님 보고 포인트:")
    print("  1. accuracy: Exp54(92.6%) 대비 Option C 정확도")
    print("  2. object:   다른 물체 → 다른 action (text 활용 증명)")
    print("  3. prompt:   프롬프트 바꾸면 결과 바뀜 (text path 살아있음)")


if __name__ == "__main__":
    main()
