#!/usr/bin/env python3
"""실기 비교군 체크포인트: MLP 회귀 헤드(결정적), 300건 전체, 서브골 드롭아웃 30%(배포본 C와 동일 조건), epoch 60.
출력: runs/egvla2_fm/deploy/MLP_regression_drop30_e60.pt, MANIFEST.json에 항목 추가. 실행: .venv_lerobot/bin/python3 scripts/save_egvla2_reg_ckpt.py"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T = S1.T
DEV, BS = AB.DEV, AB.BS
OUT = ROOT / "runs/egvla2_fm/deploy"
EPOCHS, SEED, DROP = 60, 0, 0.3


def main():
    eps = S1.load_main(); tr = T.to_t(eps, range(len(eps)))
    meta = json.loads((ROOT / "docs/v5/egvla2_fm/meta/egvla2_v3_main.json").read_text())
    torch.manual_seed(SEED)
    m = T.MLPReg(tr[0].shape[1], tr[1].shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(tr[0])
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; bb = tr[0][i] * (torch.rand(len(i), 1, device=DEV) > DROP).float()
            l = F.mse_loss(m(bb, tr[1][i]), tr[2][i]); opt.zero_grad(); l.backward(); opt.step()
    m.eval()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    ck = dict(arch="MLP_regression", state=m.state_dict(), bdim=tr[0].shape[1], fdim=tr[1].shape[1], window=T.WINDOW, H=T.H, epochs=EPOCHS,
              n_episodes=len(eps), frames=int(n), azmask=False, subgoal_dropout=DROP, seed=SEED, git_commit=commit,
              note="egvla2 real-robot comparison arm: deterministic MLP regression; missing subgoal = all-zero bbox window",
              files=[d["name"] for d in meta["files"]])
    p = OUT / "MLP_regression_drop30_e60.pt"; torch.save(ck, p)
    mp = OUT / "MANIFEST.json"; man = json.loads(mp.read_text())
    man[p.name] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size, arch="MLP_regression", epochs=EPOCHS,
                       subgoal_dropout=DROP, n_episodes=len(eps), seed=SEED, git_commit=commit)
    mp.write_text(json.dumps(man, indent=1, ensure_ascii=False)); print(json.dumps(man[p.name], indent=1))


if __name__ == "__main__":
    main()
