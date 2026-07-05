#!/usr/bin/env python3
"""CH60-d: exp71(운영 기본, Transformer w8) held-out 21ep에 OWL bbox drop-in 스왑.

eval_exp71_holdout_cl.py와 동일 split/헤드에서 bbox 입력(cx_det/cy_det/area_det/has_bbox)만
OWL-v2(th0.25, bbox_dataset_owl_150ep.json)로 교체. PG2 원본과 비교.

Usage: .venv/bin/python3 scripts/eval_exp71_owl_swap.py
출력: docs/v5/closed_loop_eval/exp71_owl_swap.json
"""
import copy
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.sim.rollout_core import build_trajectory, continuous_to_class, compute_metrics

spec = importlib.util.spec_from_file_location("clv5", ROOT / "scripts" / "sim" / "evaluate_closed_loop_v5.py")
clv5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clv5)

ANN_PATH = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json"
OWL_DS = ROOT / "docs/v5/bbox_nav_owl/bbox_dataset_owl_150ep.json"
CKPT = ROOT / "runs/v5_nav/mlp/exp71/action_transformer.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/exp71_owl_swap.json"
DT = 0.1
SUCCESS_FPE = 0.5


def load_head(device):
    import torch.nn as nn
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)

    class _TransHead(nn.Module):
        def __init__(self, fd, w, nh=4, nl=2):
            super().__init__()
            self.cls_token = nn.Parameter(torch.randn(1, 1, fd))
            self.pos_emb = nn.Embedding(w + 1, fd)
            el = nn.TransformerEncoderLayer(d_model=fd, nhead=nh, dim_feedforward=512,
                                            dropout=0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(el, num_layers=nl)
            self.head = nn.Sequential(nn.LayerNorm(fd), nn.Linear(fd, 128), nn.ReLU(),
                                      nn.Dropout(0.1), nn.Linear(128, clv5.NUM_CLASSES))
        def forward(self, x):
            B = x.size(0)
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
            pos = torch.arange(x.size(1), device=x.device)
            x = x + self.pos_emb(pos)
            x = self.encoder(x)
            return self.head(x[:, 0])

    head = _TransHead(fd=clv5.CLIP_PROJ + 4, w=clv5.CLIP_WINDOW).to(device)
    head.load_state_dict(ckpt["model"])
    head.eval()
    return head


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ann = json.loads(ANN_PATH.read_text())
    random.seed(42)
    np.random.seed(42)
    ann_s = ann[:]
    random.shuffle(ann_s)
    val_eps = ann_s[:max(1, int(len(ann_s) * 0.15))]

    owl = json.loads(OWL_DS.read_text())
    owl_by_ep = {Path(e["episode"]).stem: {f["frame_idx"]: f for f in e["frames"]} for e in owl}

    proc, vm, proj = clv5._load_frozen_clip(device)
    head = load_head(device)

    results = {}
    for variant in ["pg2", "owl"]:
        ep_results = defaultdict(list)
        for ep in val_eps:
            stem = Path(ep["episode"]).stem
            if not list(clv5.DATA_DIR.glob(f"{stem}.h5")):
                continue
            e = ep
            if variant == "owl":
                if stem not in owl_by_ep:
                    continue
                e = copy.deepcopy(ep)
                om = owl_by_ep[stem]
                for fr in e["frames"]:
                    ob = om.get(fr["frame_idx"])
                    if ob is None:
                        continue
                    fr["cx_det"], fr["cy_det"] = ob["cx_det"], ob["cy_det"]
                    fr["area_det"], fr["has_bbox"] = ob["area_det"], ob["has_bbox"]
            preds, expert = clv5.eval_exp71_episode(e, head, proc, vm, proj, device)
            expert_cls = [continuous_to_class(*a[:3]) for a in expert]
            m = compute_metrics(build_trajectory(expert_cls, DT), build_trajectory(preds, DT), SUCCESS_FPE)
            ep_results[ep.get("path_type", "unknown")].append(m)
        all_m = [m for ms in ep_results.values() for m in ms]
        results[variant] = {
            "n": len(all_m),
            "sr": sum(m["success"] for m in all_m) / len(all_m),
            "fpe": float(np.mean([m["fpe"] for m in all_m])),
            "tld": float(np.mean([m["tld"] for m in all_m])),
        }
        r = results[variant]
        print(f"[exp71+{variant}] SR {100*r['sr']:.1f}%  FPE {r['fpe']:.2f}m  (n={r['n']})")

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
