#!/usr/bin/env python3
"""
plan_20260622_hidden_state_action_head.md §4 — baseline(Exp54 Step2, bbox+image)
vs hidden-state add/replace 변형의 closed-loop 성능(SR/FPE) 비교.

CH37의 결론(override 없는 게 최선, V0_no_override)에 맞춰 STOP override 없이
모델 argmax 그대로 trajectory를 만든다 — ablate_stop_proximity.py의 eval_episode/
rollout_core 패턴 재사용, 새 학습 없음(이미 끝난 3개 ckpt만 평가).

같은 val split(seed=42, train_hidden_state_action.py와 동일)을 모든 변형에 동일
적용 — baseline은 hidden state가 없는 episode도 포함(150 split 그대로),
add/replace는 hidden state 있는 episode만(147 중 val 29개).

Usage:
  .venv/bin/python3 scripts/eval/closed_loop_eval_hidden_state.py
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

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

VLM_PATH   = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_CKPT = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
DATA_PATH  = ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full.json"
HIDDEN_CACHE = ROOT / "docs/v5/hidden_state_cache/v5_hidden_states.npz"

CKPTS = {
    "baseline(bbox+image)": (ROOT / "runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt", "none"),
    "add(bbox+image+hidden)": (ROOT / "runs/v5_nav/mlp/exp_hidden_state/stage2_v2/stage2_hidden_add.pt", "add"),
    "replace(image+hidden)": (ROOT / "runs/v5_nav/mlp/exp_hidden_state/stage2_v2/stage2_hidden_replace.pt", "replace"),
}

PROJ_DIM = 256
HIDDEN_DIM = 2304
WINDOW = 8
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


class ActionMLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 8),
        )
    def forward(self, x): return self.net(x)


def bbox_feat(frames, t, window=WINDOW):
    arr = []
    for k in range(window):
        fr = frames[max(0, t - (window - 1 - k))]
        cx = fr.get("cx", 0.5); cy = fr.get("cy", 0.5)
        area = fr.get("area", 0.05); has = float(fr.get("has_bbox", False))
        arr.extend([cx, cy, area, has])
    return np.array(arr, dtype=np.float32)


def load_images(h5_path, indices):
    with h5py.File(h5_path, "r") as f:
        imgs_ds = f["observations"]["images"]
        return [Image.fromarray(imgs_ds[i].astype(np.uint8)).convert("RGB") for i in indices]


def build_x(mode, bf, img_feat, hv):
    if mode == "none":
        return torch.cat([bf, img_feat])
    h = torch.from_numpy(hv.astype(np.float32)) if hv is not None else torch.zeros(HIDDEN_DIM, dtype=torch.float32)
    if mode == "add":
        return torch.cat([bf, img_feat, h])
    return torch.cat([img_feat, h])  # replace


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = json.loads(DATA_PATH.read_text())
    ep_labels = [ep["path_type"] for ep in data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
    te_eps_all = [data[i] for i in te_idx]
    print(f"[DATA] val episodes(전체)={len(te_eps_all)}")

    hidden_cache = {k: v for k, v in np.load(HIDDEN_CACHE).items()}

    enc = FrozenCLIPV2(VLM_PATH, STAGE1_CKPT, device).to(device).eval()

    results = {}
    for name, (ckpt_path, mode) in CKPTS.items():
        if mode == "none":
            te_eps = te_eps_all
        else:
            te_eps = [ep for ep in te_eps_all
                      if any(f"{Path(ep['episode']).stem}__f{fr['frame_idx']}" in hidden_cache for fr in ep["frames"])]

        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        d_in = ckpt["d_in"]
        mlp = ActionMLP(d_in=d_in).to(device)
        mlp.load_state_dict(ckpt["mlp"])
        mlp.eval()

        fpes, tlds, successes = [], [], []
        for ep in te_eps:
            try:
                imgs = load_images(ep["episode"], [fr["frame_idx"] for fr in ep["frames"]])
            except Exception as e:
                print(f"  [SKIP] {ep['episode']}: {e}")
                continue
            feats = enc.encode_batch(imgs, device)
            expert = [fr["gt_class"] for fr in ep["frames"]]
            pred = []
            with torch.no_grad():
                for t, fr in enumerate(ep["frames"]):
                    bf = torch.tensor(bbox_feat(ep["frames"], t), dtype=torch.float32)
                    hv = hidden_cache.get(f"{Path(ep['episode']).stem}__f{fr['frame_idx']}") if mode != "none" else None
                    x = build_x(mode, bf, feats[t].cpu(), hv).unsqueeze(0).to(device)
                    pred.append(mlp(x).argmax(1).item())

            pred_traj = build_trajectory(pred, DT_DEFAULT)
            expert_traj = build_trajectory(expert, DT_DEFAULT)
            pf, ef = pred_traj.final_pos(), expert_traj.final_pos()
            fpe = float(np.sqrt((pf[0]-ef[0])**2 + (pf[1]-ef[1])**2))
            tld = pred_traj.total_length() / max(expert_traj.total_length(), 1e-6)
            success = (fpe < SUCCESS_FPE) and (0.7 <= tld <= 1.5)
            fpes.append(fpe); tlds.append(tld); successes.append(success)

        sr = sum(successes) / len(successes) if successes else 0.0
        results[name] = {"n": len(successes), "sr": sr, "fpe_mean": float(np.mean(fpes)) if fpes else None,
                          "tld_mean": float(np.mean(tlds)) if tlds else None}
        print(f"\n[{name}] n={len(successes)}  SR={sr*100:.1f}%  FPE={np.mean(fpes):.3f}m  TLD={np.mean(tlds):.3f}")

    out_path = ROOT / "docs/v5/closed_loop_eval/hidden_state_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    main()
