#!/usr/bin/env python3
"""
CH52 — LSTM hidden state 효과를 path_type별로 분리 평가.

6/22 미팅 TODO: "val 29개로는 SR 변별 안 됨 — 어려운 path_type만 분리 평가 필요"
에 대응. 좌/우 전환이 필요한 harder path(left_left, right_right, center_left/right)에서
hidden state add/replace가 none(기준)보다 유리한지 확인.

체크포인트:
  none    : stage2_v2_lstm_areadelta/stage2_lstm_none.pt  (val_acc 95.08%, area_delta 없음)
  add     : stage2_v2_lstm/stage2_lstm_add.pt             (CH43 원본, area_delta 없음)
  replace : stage2_v2_lstm/stage2_lstm_replace.pt         (CH43 원본, area_delta 없음)
  → 셋 모두 bbox_dataset_full.json(Kosmos2 주석)으로 학습, 동일 데이터로 평가

Usage:
  .venv/bin/python3 scripts/eval/closed_loop_eval_pathtype_breakdown.py
"""
import sys
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.sim.rollout_core import build_trajectory, DT_DEFAULT  # noqa: E402
from scripts.train_hidden_state_lstm import (  # noqa: E402
    LSTMHidden, build_seq, load_hidden_cache, filter_episodes_with_hidden,
    PROJ_DIM, WINDOW,
)

VLM_PATH = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_CKPT = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
DATA_PATH = ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full.json"
SUCCESS_FPE = 0.5

CKPT_PATHS = {
    "none":    ROOT / "runs/v5_nav/mlp/exp_hidden_state/stage2_v2_lstm/stage2_lstm_none.pt",
    "add":     ROOT / "runs/v5_nav/mlp/exp_hidden_state/stage2_v2_lstm/stage2_lstm_add.pt",
    "replace": ROOT / "runs/v5_nav/mlp/exp_hidden_state/stage2_v2_lstm/stage2_lstm_replace.pt",
}

# path_type 난이도 그룹 (직진 계열 vs 방향 전환 계열)
STRAIGHT = {"center_straight", "left_straight", "right_straight"}
TURNING  = {"center_left", "center_right", "left_left", "left_right", "right_left", "right_right"}


class FrozenCLIPV2(nn.Module):
    def __init__(self, vlm_path, ckpt_path, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        self.processor = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vision_model = base.vision_model.to(device)
        self.image_proj = nn.Linear(1024, PROJ_DIM).to(device)
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


def load_images(h5_path, indices):
    with h5py.File(h5_path, "r") as f:
        imgs_ds = f["observations"]["images"]
        return [Image.fromarray(imgs_ds[i].astype(np.uint8)).convert("RGB") for i in indices]


def aggregate(records):
    if not records:
        return {"n": 0, "sr": None, "fpe_mean": None}
    srs = [r["success"] for r in records]
    fpes = [r["fpe"] for r in records]
    return {
        "n": len(records),
        "sr": float(sum(srs) / len(srs)),
        "fpe_mean": float(np.mean(fpes)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = json.loads(DATA_PATH.read_text())
    ep_labels = [ep["path_type"] for ep in data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
    te_eps_all = [data[i] for i in te_idx]
    print(f"[DATA] val 에피소드: {len(te_eps_all)}개")

    hidden_cache = load_hidden_cache()
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_CKPT, device).to(device).eval()

    # by_mode[mode][path_type] = list of {fpe, success}
    by_mode = {}

    for mode, ckpt_path in CKPT_PATHS.items():
        print(f"\n[MODE={mode}] 체크포인트: {ckpt_path.name}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        proj_dim = ckpt.get("proj_dim", 32)

        te_eps = filter_episodes_with_hidden(te_eps_all, hidden_cache) if mode != "none" else te_eps_all

        seq_dim = PROJ_DIM
        if mode != "replace":
            seq_dim += 4
        if mode != "none":
            seq_dim += proj_dim
        model = LSTMHidden(seq_dim, proj_dim, mode).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        records_by_pt = defaultdict(list)
        for ep in te_eps:
            pt = ep["path_type"]
            try:
                imgs = load_images(ep["episode"], [fr["frame_idx"] for fr in ep["frames"]])
            except Exception as e:
                print(f"  [SKIP] {ep['episode']}: {e}")
                continue
            feats = enc.encode_batch(imgs, device).cpu()
            expert = [fr["gt_class"] for fr in ep["frames"]]
            pred = []
            with torch.no_grad():
                for t, fr in enumerate(ep["frames"]):
                    img_seq, bbox_seq, hidden_seq = build_seq(ep["frames"], feats, t, hidden_cache, ep["episode"], mode, WINDOW)
                    img_seq = img_seq.unsqueeze(0).to(device)
                    bbox_seq = bbox_seq.unsqueeze(0).to(device)
                    hidden_seq = hidden_seq.unsqueeze(0).to(device) if hidden_seq is not None else None
                    pred.append(model(img_seq, bbox_seq, hidden_seq).argmax(1).item())

            pred_traj  = build_trajectory(pred,   DT_DEFAULT)
            expert_traj = build_trajectory(expert, DT_DEFAULT)
            pf, ef = pred_traj.final_pos(), expert_traj.final_pos()
            fpe = float(np.sqrt((pf[0]-ef[0])**2 + (pf[1]-ef[1])**2))
            tld = pred_traj.total_length() / max(expert_traj.total_length(), 1e-6)
            success = bool((fpe < SUCCESS_FPE) and (0.7 <= tld <= 1.5))
            records_by_pt[pt].append({"fpe": fpe, "tld": tld, "success": success})

        by_mode[mode] = {pt: aggregate(recs) for pt, recs in records_by_pt.items()}

        # 전체 + 그룹별 요약 출력
        all_recs = [r for recs in records_by_pt.values() for r in recs]
        straight_recs = [r for pt, recs in records_by_pt.items() if pt in STRAIGHT for r in recs]
        turning_recs  = [r for pt, recs in records_by_pt.items() if pt in TURNING  for r in recs]
        agg_all = aggregate(all_recs)
        agg_str = aggregate(straight_recs)
        agg_trn = aggregate(turning_recs)
        print(f"  전체     n={agg_all['n']}  SR={agg_all['sr']*100:.1f}%  FPE={agg_all['fpe_mean']:.3f}m")
        print(f"  직진계열  n={agg_str['n']}  SR={agg_str['sr']*100:.1f}%  FPE={agg_str['fpe_mean']:.3f}m")
        print(f"  전환계열  n={agg_trn['n']}  SR={agg_trn['sr']*100:.1f}%  FPE={agg_trn['fpe_mean']:.3f}m")
        for pt in sorted(records_by_pt):
            a = by_mode[mode][pt]
            print(f"    {pt:<20} n={a['n']}  SR={a['sr']*100:.0f}%  FPE={a['fpe_mean']:.3f}m")

    out = {
        "by_mode": by_mode,
        "meta": {
            "data": str(DATA_PATH.name),
            "checkpoints": {m: str(p.name) for m, p in CKPT_PATHS.items()},
            "success_fpe_threshold": SUCCESS_FPE,
            "straight_types": sorted(STRAIGHT),
            "turning_types": sorted(TURNING),
        },
    }
    out_path = ROOT / "docs/v5/closed_loop_eval/lstm_pathtype_breakdown.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    main()
