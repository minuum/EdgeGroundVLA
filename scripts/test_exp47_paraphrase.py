#!/usr/bin/env python3
"""
Exp47 Paraphrase Generalization Test

원래 instruction과 완전히 다른 표현(paraphrase)으로 교체해도
PM이 유지되는지 확인 → 의미 일반화 증명

3가지 테스트:
  1. Paraphrase test  — 같은 의미, 다른 문장
  2. Shuffle test     — 무작위 path_type 할당 (sanity check)
  3. Null test        — 아무 의미 없는 문장
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_KOSMOS  = ROOT / ".vlms" / "kosmos-2-patch14-224"
EXP46_DIR  = ROOT / "docs" / "v5" / "bbox_nav_exp46"
EXP47_DIR  = ROOT / "docs" / "v5" / "bbox_nav_exp47"
MLP47_DIR  = ROOT / "runs" / "v5_nav" / "mlp" / "exp47"
OUT_PATH   = ROOT / "docs" / "v5" / "bbox_nav_exp47" / "paraphrase_test_results.json"

WINDOW      = 8
VIS_DIM     = 1024
INSTR_DIM   = 2048
NUM_CLASSES = 8
CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]

# ── 원본 instructions ──
ORIGINAL_MAP = {
    "center_straight": "Drive forward along the center path to the basket ahead.",
    "center_left":     "Move toward the basket by swinging left from the center.",
    "center_right":    "Approach the basket by gradually turning right while moving forward.",
    "left_straight":   "Approach the basket on your left by first turning to face it, then going straight.",
    "left_left":       "Navigate to the gray basket on the left side with a left-curving path.",
    "left_right":      "Navigate to the left-side basket by curving to the right.",
    "right_straight":  "Approach the basket on your right by first turning to face it, then going straight.",
    "right_left":      "Navigate to the right-side basket by curving to the left.",
    "right_right":     "Navigate to the gray basket on the right side with a right-curving path.",
}

# ── Paraphrase instructions (완전히 다른 표현, 같은 의미) ──
PARAPHRASE_MAP = {
    "center_straight": "Go straight ahead toward the target in front of you.",
    "center_left":     "Head left while moving forward to reach the goal.",
    "center_right":    "Bear right as you advance toward the destination.",
    "left_straight":   "Rotate to align with the left target, then proceed straight.",
    "left_left":       "Follow a curved leftward route to the basket on your left.",
    "left_right":      "Take a rightward arc to reach the basket that is on the left.",
    "right_straight":  "Turn to face the basket on your right side, then go forward.",
    "right_left":      "Curve leftward to arrive at the basket positioned to the right.",
    "right_right":     "Travel along a right-curving path to the basket on the right.",
}

# ── Null instructions (의미 없는 문장) ──
NULL_MAP = {pt: "The weather is nice today." for pt in ORIGINAL_MAP}


def build_mlp(d_in):
    return nn.Sequential(
        nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64),   nn.ReLU(),
        nn.Linear(64, NUM_CLASSES),
    )


def load_model():
    ckpt = torch.load(str(MLP47_DIR / "exp47_mlp.pt"), map_location="cpu", weights_only=False)
    net  = build_mlp(ckpt["d_in"])
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


def extract_embeddings(text_map, proc, model, device, label):
    print(f"\n[EMBED] {label} 임베딩 추출...")
    embs = {}
    for pt, text in text_map.items():
        inputs   = proc(text=text, images=None, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            out = model.text_model(input_ids=input_ids, output_hidden_states=True)
            emb = out.hidden_states[-1][0].mean(0).cpu().float().numpy()
        embs[pt] = emb
        print(f"  [{pt}] '{text[:60]}'")
    return embs


def compute_pm(bbox_data, vis_cache, instr_embs, net, device, val_idx):
    net.eval()
    correct, total = 0, 0
    per_class = {i: {"correct": 0, "total": 0} for i in range(NUM_CLASSES)}

    for i in val_idx:
        ep_data  = bbox_data[i]
        ep_path  = ep_data["episode"]
        pt       = ep_data["path_type"]
        frames   = ep_data["frames"]
        vis_feats = vis_cache.get(ep_path)
        instr_emb = instr_embs.get(pt)
        if vis_feats is None or instr_emb is None:
            continue

        for t in range(len(frames)):
            bbox_feat = []
            for k in range(WINDOW):
                idx = max(0, t - (WINDOW - 1 - k))
                fr  = frames[idx]
                bbox_feat.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])

            feat = np.concatenate([
                np.array(bbox_feat, dtype=np.float32),
                vis_feats[t],
                instr_emb,
            ])
            x   = torch.tensor([feat], dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = int(net(x).argmax(1).item())
            gt = frames[t]["gt_class"]
            per_class[gt]["total"] += 1
            per_class[gt]["correct"] += int(pred == gt)
            correct += int(pred == gt)
            total   += 1

    pm = correct / total if total > 0 else 0.0
    return pm, correct, total, per_class


def main():
    print("=" * 60)
    print("Exp47 Paraphrase Generalization Test")
    print("=" * 60)

    # ── 데이터 로드 ──
    bbox_data = json.loads((EXP46_DIR / "bbox_dataset_full.json").read_text())
    vis_npz   = np.load(str(EXP46_DIR / "vision_features.npz"))
    vis_idx   = json.loads((EXP46_DIR / "vision_features_index.json").read_text())
    vis_cache = {ep: vis_npz[f"ep_{i}"] for ep, i in vis_idx.items()}

    # val split (Exp47 학습과 동일한 seed)
    path_labels = [ep["path_type"] for ep in bbox_data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(sss.split(np.zeros(len(bbox_data)), path_labels))
    print(f"\nVal episodes: {len(val_idx)}")

    # ── 모델 로드 ──
    net    = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = net.to(device)
    print(f"Device: {device}")

    # ── text encoder 로드 ──
    print("\n[MODEL] Pure HF Kosmos-2 text encoder 로드...")
    proc  = AutoProcessor.from_pretrained(str(HF_KOSMOS), trust_remote_code=True)
    vlm   = AutoModelForVision2Seq.from_pretrained(
        str(HF_KOSMOS), trust_remote_code=True
    ).to(device).eval()

    # ── 임베딩 추출 ──
    orig_embs  = extract_embeddings(ORIGINAL_MAP,  proc, vlm, device, "Original")
    para_embs  = extract_embeddings(PARAPHRASE_MAP, proc, vlm, device, "Paraphrase")
    null_embs  = extract_embeddings(NULL_MAP,       proc, vlm, device, "Null")

    # shuffle: 모든 path_type에 무작위 다른 path_type의 embedding 할당
    pts = list(ORIGINAL_MAP.keys())
    rng = np.random.RandomState(0)
    shuffled_pts = rng.permutation(pts).tolist()
    # 자기 자신이 배정되지 않도록 보장
    for i, pt in enumerate(pts):
        if shuffled_pts[i] == pt:
            swap = (i + 1) % len(pts)
            shuffled_pts[i], shuffled_pts[swap] = shuffled_pts[swap], shuffled_pts[i]
    shuffle_embs = {pts[i]: orig_embs[shuffled_pts[i]] for i in range(len(pts))}
    print(f"\n[EMBED] Shuffle 매핑:")
    for i, pt in enumerate(pts):
        print(f"  {pt} → {shuffled_pts[i]}")

    del vlm
    torch.cuda.empty_cache()

    # ── PM 계산 ──
    print("\n" + "=" * 60)
    results = {}
    for label, embs in [
        ("original",   orig_embs),
        ("paraphrase", para_embs),
        ("shuffle",    shuffle_embs),
        ("null",       null_embs),
    ]:
        pm, correct, total, per_class = compute_pm(
            bbox_data, vis_cache, embs, net, device, val_idx
        )
        results[label] = {
            "pm": pm, "correct": correct, "total": total,
            "per_class": {str(k): v for k, v in per_class.items()},
        }
        print(f"[{label:>10}]  PM = {pm*100:.1f}%  ({correct}/{total})")

    # ── 요약 출력 ──
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    orig_pm  = results["original"]["pm"]
    para_pm  = results["paraphrase"]["pm"]
    shuf_pm  = results["shuffle"]["pm"]
    null_pm  = results["null"]["pm"]

    print(f"  Original   : {orig_pm*100:.1f}%  (baseline)")
    print(f"  Paraphrase : {para_pm*100:.1f}%  (delta={para_pm-orig_pm:+.1%})")
    print(f"  Shuffle    : {shuf_pm*100:.1f}%  (delta={shuf_pm-orig_pm:+.1%})")
    print(f"  Null       : {null_pm*100:.1f}%  (delta={null_pm-orig_pm:+.1%})")

    verdict = "PASS" if (para_pm >= orig_pm - 0.05 and shuf_pm < orig_pm - 0.05) else "INCONCLUSIVE"
    print(f"\n  Generalization verdict: {verdict}")
    print("  (Paraphrase ≈ Original AND Shuffle << Original → 의미 이해 증명)")

    OUT_PATH.write_text(json.dumps({
        "shuffle_map": {pts[i]: shuffled_pts[i] for i in range(len(pts))},
        "results": results,
        "verdict": verdict,
    }, indent=2))
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
