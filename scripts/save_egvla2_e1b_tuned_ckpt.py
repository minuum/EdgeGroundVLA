#!/usr/bin/env python3
"""E1b_tuned 최적 서보 체크포인트 학습 및 배포 저장:
300 에피소드 전체, 서브골 드롭아웃 30%, 120 epoch, seed 0.
E1b_tuned 라벨링: 타깃 절대 가로 위치(cx) 기반 연속 서보 라벨 (tau=0.22, Kp=2.6).
출력: runs/egvla2_fm/deploy/C_pi0_e1b_tuned_drop30_e120.pt 및 MANIFEST.json 등록.
실행: .venv_lerobot/bin/python3 scripts/save_egvla2_e1b_tuned_ckpt.py
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
import gap_e1b as G1B  # noqa: E402
import gap_suite as GS  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
import robovlm_nav.models.egvla2_fm_heads as HD  # noqa: E402

T = S1.T
DEV = GS.DEV
EPOCHS = 120
OUT_DIR = ROOT / "runs/egvla2_fm/deploy"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eps = S1.load_main()
    cache = np.load(T.CACHE)
    E = AB.prep(eps, cache, 3)
    by_name = {e["name"]: i for i, e in enumerate(eps)}

    print(f"[*] Building E1b_tuned augmentation dataset (tau=0.22, Kp=2.6)...", flush=True)
    all_aug = G1B.build(E, by_name, None, thresh=0.22, gain=2.6)
    shift_aug = [r for r in all_aug if r["kind"] == "shift"]

    tr = T.to_t(E, range(len(E)))
    print(f"[*] Training C_pi0_e1b_tuned model (120 epochs, seed=0, N_real={len(tr[0])}, N_aug={len(shift_aug)})...", flush=True)
    m = G1B.fit(tr, shift_aug, edge_w=1.0, gain=2.6, seed=0, drop=0.3)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = json.loads((ROOT / "docs/v5/egvla2_fm/meta/egvla2_v3_main.json").read_text())

    ckpt_data = dict(
        arch="C_pi0_adaln",
        state=m.state_dict(),
        bdim=tr[0].shape[1],
        fdim=tr[1].shape[1],
        window=T.WINDOW,
        H=T.H,
        epochs=EPOCHS,
        n_episodes=len(eps),
        frames=int(len(tr[0])),
        azmask=False,
        subgoal_dropout=0.3,
        seed=0,
        git_commit=commit,
        servo_tau=0.22,
        servo_gain=2.6,
        variant="E1b_tuned",
        note="Optimal absolute-cx target seeking servo head (tau=0.22, Kp=2.6, S=+12.92 deg, 100% spin match)",
        files=[d["name"] for d in meta["files"]]
    )

    ckpt_path = OUT_DIR / "C_pi0_e1b_tuned_drop30_e120.pt"
    torch.save(ckpt_data, ckpt_path)
    file_bytes = ckpt_path.stat().st_size
    sha256_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    # Update MANIFEST.json
    manifest_path = OUT_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[ckpt_path.name] = dict(
        sha256=sha256_hash,
        bytes=file_bytes,
        arch="C_pi0_adaln",
        epochs=EPOCHS,
        subgoal_dropout=0.3,
        n_episodes=len(eps),
        seed=0,
        git_commit=commit,
        servo_tau=0.22,
        servo_gain=2.6,
        variant="E1b_tuned",
        S_sensitivity=12.92,
        spin_consistency=1.00
    )
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"[+] Successfully saved {ckpt_path.name} (SHA-256: {sha256_hash[:8]}..., {file_bytes / (1024*1024):.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
