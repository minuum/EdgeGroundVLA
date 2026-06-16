#!/usr/bin/env python3
"""
실험 A: 에피소드 내 프레임 위치별 Stage 1 Retrieval 정확도

가설:
  모델이 basket 위치를 보고 방향을 판단한다면:
    → 에피소드 후반(basket이 크고 명확)에서 정확도가 높아야 함
  모델이 복도 전체 패턴을 암기했다면:
    → 에피소드 초반/후반 무관하게 정확도 비슷함

에피소드 내 구간:
  early : 앞 33%  — basket 멀리, 작음
  mid   : 중간 34%
  late  : 뒤 33%  — basket 가까이, 큼

추가 분석:
  cx 기반 frame-level 레이블 vs. path_type 레이블 일치 여부
  → 어느 쪽을 모델이 따르는지 교차 확인

Usage:
  .venv/bin/python3 scripts/exp54_exp_a_episode_position.py
"""

import json, sys, warnings
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH   = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH  = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
STAGE1_DIR = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage1"
CKPT_PATH  = STAGE1_DIR / "stage1_projs.pt"
LORA_DIR   = STAGE1_DIR / "clip_lora_adapter"

PROJ_DIM = 256
LM_DIM   = 2048
VIS_DIM  = 1024

PATH_TO_DIR = {
    "left_straight":"left",  "left_left":"left",   "left_right":"left",
    "center_straight":"center","center_left":"center","center_right":"center",
    "right_straight":"right", "right_left":"right", "right_right":"right",
}
DIR_IDX = {"left": 0, "center": 1, "right": 2}
IDX_DIR = {0: "left", 1: "center", 2: "right"}


# ─────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────

def load_stage1(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    print(f"[MODEL] Stage1 val_acc={ckpt['val_acc']:.4f}")

    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    base = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    vision_model = PeftModel.from_pretrained(base.vision_model, str(LORA_DIR)).to(device).eval()

    image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
    proj_state = {k[len("proj."):]: v for k, v in ckpt["clip_lora"].items() if k.startswith("proj.")}
    image_proj.load_state_dict(proj_state)
    image_proj.eval()

    text_proj = nn.Linear(LM_DIM, PROJ_DIM).to(device)
    text_proj.load_state_dict(ckpt["text_proj"])
    text_proj.eval()

    anchor_raw = ckpt["anchor_raw"].to(device)  # (3, 2048) — left/center/right
    anchor_feats = F.normalize(text_proj(anchor_raw), dim=-1)  # (3, 256)

    return processor, vision_model, image_proj, anchor_feats


@torch.no_grad()
def encode_images(vision_model, image_proj, processor, images, device):
    inputs = processor(images=images, return_tensors="pt")
    pv = inputs["pixel_values"].to(
        device, dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    out = vision_model(pixel_values=pv)
    feat = out.last_hidden_state.mean(dim=1).float()
    return F.normalize(image_proj(feat), dim=-1)  # (N, 256)


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}\n")

    processor, vision_model, image_proj, anchor_feats = load_stage1(device)

    data = json.loads(DATA_PATH.read_text())
    ep_labels_all = [PATH_TO_DIR.get(ep["path_type"], "unk") for ep in data]
    valid_idx = [i for i, l in enumerate(ep_labels_all) if l != "unk"]
    data_valid = [data[i] for i in valid_idx]
    labels_valid = [ep_labels_all[i] for i in valid_idx]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(sss.split(np.zeros(len(data_valid)), labels_valid))
    val_eps = [data_valid[i] for i in te_idx]
    print(f"Val 에피소드: {len(val_eps)}개\n")

    # ─── 구간 정의 ───────────────────────────────────────
    #  early: 앞 33% / mid: 33~67% / late: 뒤 33%
    def split_frames(frames):
        n = len(frames)
        e = max(1, n // 3)
        l = n - max(1, n // 3)
        return {
            "early": frames[:e],
            "mid":   frames[e:l],
            "late":  frames[l:],
        }

    # ─── 평가 ────────────────────────────────────────────
    # segment별 correct/total
    seg_correct  = defaultdict(int)
    seg_total    = defaultdict(int)
    # path_type별 + segment별
    dir_seg_correct = defaultdict(lambda: defaultdict(int))
    dir_seg_total   = defaultdict(lambda: defaultdict(int))

    # cx 기반 레이블 분석: path_type vs cx 방향이 일치하는 비율
    cx_agree_by_seg = defaultdict(list)  # seg → list of bool (path_type dir == cx dir)

    print("평가 중...")
    for ep in val_eps:
        pt = ep["path_type"]
        true_dir = PATH_TO_DIR[pt]
        true_idx = DIR_IDX[true_dir]
        frames    = ep["frames"]
        segs      = split_frames(frames)

        for seg_name, seg_frames in segs.items():
            if not seg_frames:
                continue

            # 이미지 로드
            images = []
            valid_frames = []
            for fr in seg_frames:
                try:
                    with h5py.File(ep["episode"], "r") as f:
                        img = Image.fromarray(f["observations"]["images"][fr["frame_idx"]])
                    images.append(img)
                    valid_frames.append(fr)
                except:
                    pass

            if not images:
                continue

            # retrieval
            feats = encode_images(vision_model, image_proj, processor, images, device)
            preds = (feats @ anchor_feats.T).argmax(dim=1).cpu().numpy()

            for pred, fr in zip(preds, valid_frames):
                correct = int(pred == true_idx)
                seg_correct[seg_name]  += correct
                seg_total[seg_name]    += 1
                dir_seg_correct[true_dir][seg_name] += correct
                dir_seg_total[true_dir][seg_name]   += 1

                # cx 기반 방향과 path_type 방향이 일치하는지
                if fr["has_bbox"]:
                    cx = fr["cx"]
                    if cx < 0.43:
                        cx_dir = "left"
                    elif cx > 0.57:
                        cx_dir = "right"
                    else:
                        cx_dir = "center"
                    cx_agree_by_seg[seg_name].append(cx_dir == true_dir)

    # ─── 결과 출력 ────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  실험 A: 에피소드 내 위치별 Retrieval 정확도")
    print("=" * 62)
    print(f"\n  {'구간':<8} {'정답':>6} {'전체':>6} {'정확도':>9}  {'해석'}")
    print("  " + "-" * 52)
    segs_order = ["early", "mid", "late"]
    accs = {}
    for seg in segs_order:
        tot = seg_total[seg]
        cor = seg_correct[seg]
        acc = cor / tot * 100 if tot > 0 else 0.0
        accs[seg] = acc
        note = "(basket 멀리)" if seg == "early" else "(접근 중)" if seg == "mid" else "(basket 가까이)"
        print(f"  {seg:<8} {cor:>6} {tot:>6} {acc:>8.1f}%  {note}")

    gap = accs.get("late", 0) - accs.get("early", 0)
    print(f"\n  late - early 격차: {gap:+.1f}%p")
    if gap >= 10:
        verdict = "basket 위치를 보고 있다 ✅ (격차 10%p 이상)"
    elif gap >= 4:
        verdict = "약한 basket 의존성 (격차 4~10%p)"
    else:
        verdict = "복도 전체 패턴 암기 가능성 ⚠️ (격차 4%p 미만)"
    print(f"  → 판정: {verdict}")

    print(f"\n{'방향별 early vs late':}")
    print(f"  {'방향':<8} {'early':>8} {'mid':>8} {'late':>8}  {'격차':>8}")
    print("  " + "-" * 48)
    for d in ["left", "center", "right"]:
        row = []
        for seg in segs_order:
            tot = dir_seg_total[d][seg]
            cor = dir_seg_correct[d][seg]
            a = cor / tot * 100 if tot > 0 else 0.0
            row.append(a)
        gap_d = row[2] - row[0]
        print(f"  {d:<8} {row[0]:>7.1f}% {row[1]:>7.1f}% {row[2]:>7.1f}%  {gap_d:>+7.1f}%p")

    print(f"\n{'cx 기반 방향 ↔ path_type 방향 일치율 (has_bbox 프레임만)':}")
    print(f"  {'구간':<8} {'일치':>6} {'전체':>6} {'일치율':>9}")
    print("  " + "-" * 40)
    for seg in segs_order:
        agrees = cx_agree_by_seg[seg]
        if agrees:
            rate = sum(agrees) / len(agrees) * 100
            print(f"  {seg:<8} {sum(agrees):>6} {len(agrees):>6} {rate:>8.1f}%")
        else:
            print(f"  {seg:<8}      -      -         -")

    print(f"\n  [참고] cx 일치율이 낮으면 bbox 데이터 자체가 노이즈임을 의미")
    print(f"         → retrieval 정확도가 cx 일치율보다 높으면 모델이 cx보다 더 잘 본다")
    print("=" * 62)


if __name__ == "__main__":
    main()
