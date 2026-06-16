#!/usr/bin/env python3
"""
Counterfactual Action Test — Stage 2 v2 모델

교수님 5/22 미팅 핵심 요구:
  "basket 대신 다른 걸 집어넣었더니 이상한 행동을 해야 되잖아."
  "bbox를 반전시키면 액션도 반전된다 → 모델이 bbox 위치를 사용한다"

4가지 조건으로 Stage 2 v2 모델 입력을 조작해 인과성 측정:
  A (baseline):   정상 입력
  B (bbox=zeros): bbox 제거 (basket 없는 척)
  C (bbox flip):  cx를 1-cx로 반전 (LEFT basket → RIGHT로 속임)
  D (vis swap):   다른 방향 에피소드의 시각 특징으로 교체

Usage:
  .venv/bin/python3 scripts/test_action_counterfactual.py
  .venv/bin/python3 scripts/test_action_counterfactual.py --n-episodes 20
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH   = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH  = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
STAGE1_V2  = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
STAGE2_V2  = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage2_v2" / "stage2_v2_mlp.pt"

CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
PROJ_DIM    = 256
D_IN        = WINDOW * 4 + PROJ_DIM  # 288


# ─── 모델 ────────────────────────────────────────────────

class FrozenCLIPV2(nn.Module):
    def __init__(self, vlm_path, ckpt_path, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor

        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        print(f"[MODEL] Stage1 v2 val_acc={ckpt['val_acc']:.4f}", flush=True)

        self.processor = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vision_model = base.vision_model.to(device)
        self.image_proj   = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.image_proj.load_state_dict(ckpt["image_proj"])

        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.image_proj.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, pil_images, device, batch=32):
        all_feats = []
        for i in range(0, len(pil_images), batch):
            imgs = pil_images[i:i+batch]
            inputs = self.processor(images=imgs, return_tensors="pt")
            pv = inputs["pixel_values"].to(device, dtype=torch.float16)
            out = self.vision_model(pixel_values=pv)
            feat = out.last_hidden_state.mean(dim=1).float()
            all_feats.append(F.normalize(self.image_proj(feat), dim=-1))
        return torch.cat(all_feats, dim=0)


class ActionMLP(nn.Module):
    def __init__(self, d_in=D_IN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


# ─── 데이터 유틸 ─────────────────────────────────────────

def load_images(h5_path, indices):
    with h5py.File(h5_path, "r") as f:
        return [Image.fromarray(f["observations"]["images"][i]) for i in indices]


def bbox_feat(frames, t):
    arr = []
    for k in range(WINDOW):
        fr = frames[max(0, t - (WINDOW - 1 - k))]
        arr.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])
    return np.array(arr, dtype=np.float32)


def bbox_feat_zeros(frames, t):
    """Condition B: bbox를 전부 0으로 (basket 없는 척)."""
    return np.zeros(WINDOW * 4, dtype=np.float32)


def bbox_feat_cx_flipped(frames, t):
    """Condition C: cx를 1-cx로 반전 (LEFT basket → RIGHT로 속임)."""
    arr = []
    for k in range(WINDOW):
        fr = frames[max(0, t - (WINDOW - 1 - k))]
        cx_flipped = 1.0 - fr["cx"]  # 좌우 반전
        arr.extend([cx_flipped, fr["cy"], fr["area"], float(fr["has_bbox"])])
    return np.array(arr, dtype=np.float32)


# ─── 캐싱 ─────────────────────────────────────────────────

def precompute_features(enc, eps, device, label=""):
    cache = {}
    n = len(eps)
    print(f"[CACHE] {label} 특징 추출 중 ({n} episodes)...", flush=True)
    t0 = time.time()
    for i, ep in enumerate(eps):
        try:
            imgs = load_images(ep["episode"], list(range(len(ep["frames"]))))
        except Exception as e:
            print(f"  skip {ep['episode']}: {e}", flush=True)
            cache[ep["episode"]] = None
            continue
        feats = enc.encode_batch(imgs, device)
        cache[ep["episode"]] = feats.cpu()
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[CACHE] 완료 ({time.time()-t0:.1f}s)", flush=True)
    return cache


# ─── 추론 ─────────────────────────────────────────────────

@torch.no_grad()
def run_condition(mlp, cache, ep, vis_cache_override, bbox_fn, device):
    """단일 조건으로 에피소드 추론. Returns list of (gt, pred)."""
    ep_path = ep["episode"]
    vis_feats = vis_cache_override if vis_cache_override is not None else cache.get(ep_path)
    if vis_feats is None:
        return []

    results = []
    for t, fr in enumerate(ep["frames"]):
        bf = torch.tensor(bbox_fn(ep["frames"], t), dtype=torch.float32)
        vf = vis_feats[t]
        x  = torch.cat([bf, vf]).unsqueeze(0).to(device)
        pred = mlp(x).argmax(1).item()
        results.append((fr["gt_class"], pred))
    return results


# ─── 결과 집계 ───────────────────────────────────────────

def summarize(results, label):
    """results: list of (gt, pred)"""
    correct = sum(1 for g, p in results if g == p)
    total   = len(results)
    acc     = correct / total if total > 0 else 0.0

    per_class = defaultdict(lambda: [0, 0])
    for g, p in results:
        per_class[g][0] += int(g == p)
        per_class[g][1] += 1

    print(f"\n[{label}]  acc={acc*100:.1f}%  ({correct}/{total})")
    for cls_id in sorted(per_class.keys()):
        c, n = per_class[cls_id]
        print(f"  {CLASS_NAMES[cls_id]:8s} : {c}/{n} = {c/n*100:.1f}%" if n > 0 else f"  {CLASS_NAMES[cls_id]:8s} : —")
    return acc, dict(per_class)


def flip_rate(a_results, b_results):
    """Condition A vs B에서 prediction이 달라진 비율."""
    assert len(a_results) == len(b_results)
    flipped = sum(1 for (_, pa), (_, pb) in zip(a_results, b_results) if pa != pb)
    return flipped / len(a_results) if a_results else 0.0


# ─── 메인 ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="테스트에 사용할 에피소드 수 (None=전체)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}", flush=True)

    # 데이터 로드
    data = json.loads(DATA_PATH.read_text())
    print(f"[DATA] {len(data)} episodes loaded", flush=True)

    # 에피소드 서브셋 (--n-episodes 옵션)
    if args.n_episodes is not None:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(data), size=min(args.n_episodes, len(data)), replace=False)
        episodes = [data[i] for i in sorted(idx)]
        print(f"[DATA] {len(episodes)} episodes 선택 (seed={args.seed})", flush=True)
    else:
        episodes = data

    # LEFT/RIGHT 에피소드를 방향별로 분리 (Condition D용)
    left_eps  = [ep for ep in episodes if "left"  in ep["path_type"] and "right" not in ep["path_type"]]
    right_eps = [ep for ep in episodes if "right" in ep["path_type"] and "left"  not in ep["path_type"]]
    print(f"[DATA] left={len(left_eps)}, right={len(right_eps)} episodes (for condition D)", flush=True)

    # 모델 로드
    print("\n[MODEL] 로딩 중...", flush=True)
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_V2, device)
    enc.eval()

    mlp_ckpt = torch.load(str(STAGE2_V2), map_location=device, weights_only=False)
    mlp = ActionMLP().to(device)
    # stage2_v2 ckpt 형식: {'mlp': state_dict, 'val_acc': float, 'd_in': int}
    state = mlp_ckpt["mlp"] if "mlp" in mlp_ckpt else mlp_ckpt
    mlp.load_state_dict(state)
    mlp.eval()
    val_acc = mlp_ckpt.get("val_acc", "?")
    print(f"[MODEL] Stage2 v2 val_acc={val_acc}", flush=True)
    print("[MODEL] 로딩 완료", flush=True)

    # Feature Pre-caching
    cache = precompute_features(enc, episodes, device, label="main")

    # Condition D를 위한 반대 방향 캐시 (LEFT ep의 visual → RIGHT로 쓸 때)
    # 실제 구현: LEFT 에피소드 프레임을 오른쪽 방향 에피소드에 주입
    # 간단화: LEFT ep 중 하나의 feature를 다른 LEFT ep에 교차 주입
    # (full cross-swap은 구현 복잡하므로: "반대 cx 에피소드" 매핑)

    # ─── 조건별 추론 ────────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("COUNTERFACTUAL ACTION TEST", flush=True)
    print("="*60, flush=True)

    all_A, all_B, all_C, all_D = [], [], [], []

    for ep in episodes:
        ep_key = ep["episode"]
        if cache.get(ep_key) is None:
            continue

        # Condition A: 정상
        rA = run_condition(mlp, cache, ep, None, bbox_feat, device)

        # Condition B: bbox zeros
        rB = run_condition(mlp, cache, ep, None, bbox_feat_zeros, device)

        # Condition C: bbox cx flipped
        rC = run_condition(mlp, cache, ep, None, bbox_feat_cx_flipped, device)

        # Condition D: visual 교체 (반대 방향 에피소드의 feature)
        # 이 에피소드의 path_type을 보고 반대 방향 에피소드의 feature를 주입
        rD = []
        pt = ep["path_type"]
        if "left" in pt and "right" not in pt:
            # LEFT ep → RIGHT ep의 feature로 교체
            candidates = [e for e in right_eps if e["episode"] != ep_key and
                          cache.get(e["episode"]) is not None and
                          len(cache[e["episode"]]) >= len(ep["frames"])]
            if candidates:
                swap_ep = candidates[0]
                swap_vis = cache[swap_ep["episode"]][:len(ep["frames"])]
                rD = run_condition(mlp, cache, ep, swap_vis, bbox_feat, device)
        elif "right" in pt and "left" not in pt:
            # RIGHT ep → LEFT ep의 feature로 교체
            candidates = [e for e in left_eps if e["episode"] != ep_key and
                          cache.get(e["episode"]) is not None and
                          len(cache[e["episode"]]) >= len(ep["frames"])]
            if candidates:
                swap_ep = candidates[0]
                swap_vis = cache[swap_ep["episode"]][:len(ep["frames"])]
                rD = run_condition(mlp, cache, ep, swap_vis, bbox_feat, device)

        all_A.extend(rA)
        all_B.extend(rB)
        all_C.extend(rC)
        all_D.extend(rD)

    # 결과 출력
    acc_A, _ = summarize(all_A, "Condition A — baseline (정상 입력)")
    acc_B, _ = summarize(all_B, "Condition B — bbox=zeros (basket 제거)")
    acc_C, _ = summarize(all_C, "Condition C — bbox cx flipped (좌우 반전)")
    if all_D:
        acc_D, _ = summarize(all_D, "Condition D — visual swapped (반대 방향 visual)")

    # flip rate 계산
    fr_B = flip_rate(all_A, all_B)
    fr_C = flip_rate(all_A, all_C)
    # Condition D는 좌/우 에피소드만 포함 → A 전체와 길이 다름, D 자체 기준으로 flip 계산
    # all_D에 대응하는 all_A 서브셋을 별도로 수집했어야 하므로 단순 비율로 계산
    fr_D = None  # 아래에서 별도 계산
    if all_D:
        # summarize에서 나온 acc로 역산: flip = 1 - relative accuracy (근사)
        acc_D_val = sum(1 for g, p in all_D if g == p) / len(all_D)
        acc_A_comparable = sum(1 for g, p in all_A[:len(all_D)] if g == p) / len(all_D) if len(all_A) >= len(all_D) else None
        fr_D = "별도 집계 (Condition D는 left/right ep만 포함)"

    print("\n" + "="*60)
    print("FLIP RATE SUMMARY (교수님 보고용)")
    print("="*60)
    print(f"  A vs B (bbox 제거):     flip_rate={fr_B*100:.1f}%  ← bbox 의존도")
    print(f"  A vs C (bbox 반전):     flip_rate={fr_C*100:.1f}%  ← bbox 방향 결정력")
    if all_D:
        acc_D = sum(1 for g, p in all_D if g == p) / len(all_D)
        print(f"  A vs D (visual 교체):  acc_D={acc_D*100:.1f}%  ← visual 교체 시 accuracy (left/right ep 대상)")

    print("\n[해석 기준]")
    print("  flip_rate(B) ≥ 50%: bbox가 방향 결정에 필수적임")
    print("  flip_rate(C) ≥ 70%: bbox cx가 실제 방향 판단에 사용됨 → basket 위치 추적 증거")
    print("  flip_rate(D) ≥ 50%: visual도 방향에 기여함")
    print()

    # 결론
    print("[결론]")
    if fr_C >= 0.7:
        print("  ✅ bbox cx를 반전시키면 action도 반전 → 모델이 basket 위치를 추적한다")
    elif fr_C >= 0.4:
        print("  ⚠️  bbox cx 반전 시 부분적 action 변화 → basket 위치 부분 활용")
    else:
        print("  ❌ bbox cx 반전 무관 → 모델이 bbox cx를 무시함")

    if fr_B >= 0.5:
        print("  ✅ bbox 제거 시 action 변화 → bbox가 필수적")
    else:
        print("  ⚠️  bbox 제거해도 action 유지 → visual만으로 충분한 정보")

    if all_D:
        acc_D = sum(1 for g, p in all_D if g == p) / len(all_D)
        if acc_D < 0.5:
            print("  ✅ visual 교체 시 acc 급락 → visual feature도 방향 결정에 핵심")
        else:
            print("  ⚠️  visual 교체해도 acc 유지 → visual 기여 낮음")


if __name__ == "__main__":
    main()
