#!/usr/bin/env python3
"""CH60-c: exp71(운영 기본 모델) 오염 없는 CL 재평가.

발견: rollout_metrics.json의 exp71/72 SR 100%/FPE 0.00은 train/test 오염
(CL test 9 에피소드 전부가 exp71 학습 split에 포함 — seed 42/val_ratio 0.15 재구성으로 확정).

이 스크립트는 exp71 학습 당시의 진짜 val 22 에피소드로 CL 리플레이 재평가.
evaluate_closed_loop_v5.py의 eval_exp71_episode/_load_frozen_clip 재사용.

Usage: .venv/bin/python3 scripts/eval_exp71_holdout_cl.py
출력: docs/v5/closed_loop_eval/exp71_holdout.json
"""
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
CKPT = ROOT / "runs/v5_nav/mlp/exp71/action_transformer.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/exp71_holdout.json"
DT = 0.1
SUCCESS_FPE = 0.5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 학습 split 재구성 (train_exp71_stage2_transformer.py 기본값: seed 42, val 0.15)
    ann = json.loads(ANN_PATH.read_text())
    random.seed(42)
    np.random.seed(42)
    ann_s = ann[:]
    random.shuffle(ann_s)
    n_val = max(1, int(len(ann_s) * 0.15))
    val_eps = ann_s[:n_val]
    print(f"held-out val: {len(val_eps)} eps")

    proc, vm, proj = clv5._load_frozen_clip(device)
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)

    import torch.nn as nn
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
    print(f"exp71 ckpt val_acc(기록)={ckpt.get('val_acc', 0):.3f}")

    ep_results = defaultdict(list)
    for ep in val_eps:
        stem = Path(ep["episode"]).stem
        if not list(clv5.DATA_DIR.glob(f"{stem}.h5")):
            continue
        preds, expert = clv5.eval_exp71_episode(ep, head, proc, vm, proj, device)
        expert_cls = [continuous_to_class(*a[:3]) for a in expert]
        m = compute_metrics(build_trajectory(expert_cls, DT), build_trajectory(preds, DT), SUCCESS_FPE)
        pt = ep.get("path_type", "unknown")
        m["episode"] = stem
        ep_results[pt].append(m)
        print(f"  {pt:20s} FPE={m['fpe']:.2f}m {'✅' if m['success'] else '❌'}")

    all_m = [m for ms in ep_results.values() for m in ms]
    summary = {
        "n": len(all_m),
        "sr": sum(m["success"] for m in all_m) / len(all_m),
        "fpe": float(np.mean([m["fpe"] for m in all_m])),
        "tld": float(np.mean([m["tld"] for m in all_m])),
        "fpe_exact_zero": sum(1 for m in all_m if m["fpe"] == 0.0),
    }
    OUT.write_text(json.dumps({"summary": summary,
                                "per_path": {k: v for k, v in ep_results.items()}},
                               indent=2, ensure_ascii=False))
    print(f"\nexp71 held-out: SR {100*summary['sr']:.1f}%  FPE {summary['fpe']:.2f}m  "
          f"TLD {summary['tld']:.2f}  (n={summary['n']}, FPE=0인 ep {summary['fpe_exact_zero']}개)")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
