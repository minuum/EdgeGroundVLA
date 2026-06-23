#!/usr/bin/env python3
"""
plan_20260623_bbox_pg2_reannotation.md §3 — closed_loop_eval_lstm.py를
bbox_dataset_full_pg2.json(현재 PG2 모델로 재주석된 데이터)로 재실행하는
복제본. 원본은 그대로 두고 DATA_PATH/출력 경로만 바꿈.

Usage:
  .venv/bin/python3 scripts/eval/closed_loop_eval_lstm_pg2.py
"""
import sys
import json
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
    PROJ_DIM, NUM_CLASSES, WINDOW,
)

VLM_PATH = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_CKPT = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
DATA_PATH = ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2.json"
LSTM_DIR = ROOT / "runs/v5_nav/mlp/exp_hidden_state/stage2_v2_lstm"
SUCCESS_FPE = 0.5


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = json.loads(DATA_PATH.read_text())
    ep_labels = [ep["path_type"] for ep in data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
    te_eps_all = [data[i] for i in te_idx]
    print(f"[DATA] val episodes(전체)={len(te_eps_all)}")

    hidden_cache = load_hidden_cache()
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_CKPT, device).to(device).eval()

    results = {}
    for mode in ["none", "add", "replace"]:
        ckpt_path = LSTM_DIR / f"stage2_lstm_{mode}.pt"
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

        fpes, tlds, successes = [], [], []
        for ep in te_eps:
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

            pred_traj = build_trajectory(pred, DT_DEFAULT)
            expert_traj = build_trajectory(expert, DT_DEFAULT)
            pf, ef = pred_traj.final_pos(), expert_traj.final_pos()
            fpe = float(np.sqrt((pf[0]-ef[0])**2 + (pf[1]-ef[1])**2))
            tld = pred_traj.total_length() / max(expert_traj.total_length(), 1e-6)
            success = (fpe < SUCCESS_FPE) and (0.7 <= tld <= 1.5)
            fpes.append(fpe); tlds.append(tld); successes.append(success)

        sr = sum(successes) / len(successes) if successes else 0.0
        results[mode] = {"n": len(successes), "sr": sr,
                          "fpe_mean": float(np.mean(fpes)) if fpes else None,
                          "tld_mean": float(np.mean(tlds)) if tlds else None,
                          "val_acc_pm": ckpt.get("val_acc")}
        print(f"\n[LSTM-{mode}] n={len(successes)}  SR={sr*100:.1f}%  FPE={np.mean(fpes):.3f}m  TLD={np.mean(tlds):.3f}  (PM={ckpt.get('val_acc',0)*100:.2f}%)")

    out_path = ROOT / "docs/v5/closed_loop_eval/lstm_comparison_pg2.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    main()
