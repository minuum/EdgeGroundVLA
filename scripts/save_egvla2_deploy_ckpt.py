#!/usr/bin/env python3
"""실기 배포용 체크포인트 저장: 300건 전체로 고정 epoch 학습.
- C_pi0_adaln_drop30_e120.pt  (권장: 서브골 드롭아웃 30%, epoch 120)
- C_pi0_adaln_e120.pt         (대조: 드롭아웃 없음, epoch 120)
기존 runs/egvla2_fm/*.pt는 250 epoch 학습이라 C는 과적합 구간(최적 약 80~160)이어서 배포용으로 쓰지 않는다.
출력: runs/egvla2_fm/deploy/*.pt, MANIFEST.json(sha256·크기·학습 설정·git 커밋). 실행: .venv_lerobot/bin/python3 scripts/save_egvla2_deploy_ckpt.py
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T = S1.T
OUT = ROOT / "runs/egvla2_fm/deploy"; OUT.mkdir(parents=True, exist_ok=True)
EPOCHS, SEED = 120, 0
VARIANTS = [("C_pi0_adaln_drop30_e120", 0.30), ("C_pi0_adaln_e120", 0.0)]


def sha(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes()); return h.hexdigest()


def main():
    eps = S1.load_main(); allt = T.to_t(eps, range(len(eps)))
    meta = json.loads((ROOT / "docs/v5/egvla2_fm/meta/egvla2_v3_main.json").read_text())
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    man = {}
    for name, drop in VARIANTS:
        t0 = time.time()
        m = AB.fit(S1.ARCHS["C_pi0_adaln"], allt, EPOCHS, SEED, drop=drop)
        ck = dict(arch="C_pi0_adaln", state=m.state_dict(), bdim=allt[0].shape[1], fdim=allt[1].shape[1], window=T.WINDOW, H=T.H,
                  epochs=EPOCHS, n_episodes=len(eps), frames=int(len(allt[0])), azmask=False, subgoal_dropout=drop, seed=SEED,
                  git_commit=commit, note="egvla2 deploy ckpt 2026-10-19; missing subgoal = all-zero bbox window",
                  files=[d["name"] for d in meta["files"]])
        p = OUT / f"{name}.pt"; torch.save(ck, p)
        man[p.name] = dict(sha256=sha(p), bytes=p.stat().st_size, arch="C_pi0_adaln", epochs=EPOCHS, subgoal_dropout=drop, n_episodes=len(eps), seed=SEED, git_commit=commit)
        print("saved", p, f"{(time.time() - t0) / 60:.1f}분", flush=True)
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
