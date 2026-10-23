#!/usr/bin/env python3
"""타깃 서보 라벨(λ) 학습 체크포인트 저장: 300 에피소드 전체, 서브골 드롭아웃 30%, 120 epoch, seed 0. 배포 서버가 그대로 읽는 형식(arch=C_pi0_adaln).
출력: runs/egvla2_fm/deploy/C_pi0_servolabel{λ}_drop30_e120.pt, MANIFEST.json 항목 추가. 실행: .venv_lerobot/bin/python3 scripts/save_egvla2_servolabel_ckpt.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import eval_target_cond as TC  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T = S1.T
OUT = ROOT / "docs/../runs/egvla2_fm/deploy"


def main():
    OUT_ = (ROOT / "runs/egvla2_fm/deploy"); OUT_.mkdir(parents=True, exist_ok=True)
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); tr = T.to_t(E, range(len(E)))
    meta = json.loads((ROOT / "docs/v5/egvla2_fm/meta/egvla2_v3_main.json").read_text())
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    mp = OUT_ / "MANIFEST.json"; man = json.loads(mp.read_text())
    for lam in (0.5, 1.0):
        m = TC.fit(tr, lam=lam)
        ck = dict(arch="C_pi0_adaln", state=m.state_dict(), bdim=tr[0].shape[1], fdim=tr[1].shape[1], window=T.WINDOW, H=T.H, epochs=TC.EPOCHS, n_episodes=len(eps),
                  frames=int(len(tr[0])), azmask=False, subgoal_dropout=0.3, seed=0, git_commit=commit, servo_label_lambda=lam,
                  note=f"target-seeking labels (lambda={lam}, k=2.0, max=0.7); missing subgoal = all-zero bbox window", files=[d["name"] for d in meta["files"]])
        p = OUT_ / f"C_pi0_servolabel{lam}_drop30_e120.pt"; torch.save(ck, p)
        man[p.name] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size, arch="C_pi0_adaln", epochs=TC.EPOCHS, subgoal_dropout=0.3, n_episodes=len(eps), seed=0,
                           git_commit=commit, servo_label_lambda=lam)
        mp.write_text(json.dumps(man, indent=1, ensure_ascii=False)); print("saved", p.name, man[p.name]["sha256"][:8], flush=True)


if __name__ == "__main__":
    main()
