#!/usr/bin/env python3
"""EGVLA2 300ep 본수집이 끝나면 한 번에 실행하는 파이프라인(plan_20261015 §3).

순서: 준비 확인 → F2 캐시(증분) → 배치 검증 → 본학습 세트 생성 → S1(5-fold) → S1(leave-one-position-out) → 서빙용 체크포인트(A/B2/C).
준비 기준: 위치(center, weak_left, strong_left, weak_right, strong_right) × 경로(left_curve, straight, right_curve) 15칸이 각각 --per-cell(기본 20) 이상이고 rsync가 돌고 있지 않을 때.
실행: .venv/bin/python3 -I scripts/run_egvla2_pipeline.py [--wait] [--force] [--epochs 250] [--seeds 0,1,2,3,4] [--dry]
  --wait  준비될 때까지 60초 간격으로 대기   --force  준비 기준 무시   --dry  단계만 출력
  --smoke 빠른 점검용(EPOCHS=2, SEEDS=0, 폴드 1개분만 아님 — 전체를 아주 짧게)
로그: /tmp/egvla2_pipeline/*.log, 요약: docs/v5/egvla2_fm/pipeline_status.json
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ROS_action/mobile_vla_dataset_egvla2"
LOG = Path("/tmp/egvla2_pipeline"); LOG.mkdir(exist_ok=True)
PY, PYL = str(ROOT / ".venv/bin/python3"), str(ROOT / ".venv_lerobot/bin/python3")
POS = ["center", "weak_left", "strong_left", "weak_right", "strong_right"]
PATH = ["left_curve", "straight", "right_curve"]


def cell_counts():
    c = collections.Counter()
    for f in DATA.glob("*.h5"):
        try:
            with h5py.File(f, "r") as h:
                if h.attrs.get("action_space") == "egvla2_continuous_v3" and h.attrs.get("cx_position"):
                    c[(str(h.attrs["cx_position"]), str(h.attrs["cx_path"]))] += 1
        except OSError:
            pass  # 전송 중인 파일
    return c


def ready(per_cell):
    c = cell_counts()
    missing = {f"{p}|{q}": per_cell - c[(p, q)] for p in POS for q in PATH if c[(p, q)] < per_cell}
    rsync = subprocess.run(["pgrep", "-f", "rsync.*mobile_vla_dataset_egvla2"], capture_output=True).returncode == 0
    return not missing and not rsync, missing, rsync, sum(c.values())


def step(name, cmd, env=None, dry=False):
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {' '.join(cmd)}", flush=True)
    if dry:
        return 0
    t0 = time.time()
    with open(LOG / f"{name}.log", "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **(env or {})}, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"   → rc={rc} ({(time.time() - t0) / 60:.1f}분)", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true"); ap.add_argument("--force", action="store_true"); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=250); ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    while not a.force:
        ok, missing, rsync, n = ready(a.per_cell)
        if ok:
            break
        print(f"[대기] v3 라벨 파일 {n}개, 미달 셀 {len(missing)}개 {dict(list(missing.items())[:4])}..., rsync={rsync}", flush=True)
        if not a.wait:
            sys.exit(2)
        time.sleep(60)
    ep, seeds = (2, "0") if a.smoke else (a.epochs, a.seeds)
    tag = "_smoke" if a.smoke else "_full300"
    steps = [
        ("1_precompute", [PY, "scripts/precompute_egvla2_continuous_features.py"], dict(HF_HUB_OFFLINE="1")),
        ("2_validate", [PY, "-I", "scripts/validate_egvla2_batch.py"], None),
        ("3_build_set", [PY, "-I", "scripts/build_egvla2_main_set.py"], None),
        ("4_s1_stratified", [PYL, "scripts/train_egvla2_s1.py"], dict(EPOCHS=str(ep), SEEDS=seeds, TAG=tag)),
        ("5_s1_lopo", [PYL, "scripts/train_egvla2_s1.py"], dict(EPOCHS=str(ep), SEEDS=seeds.split(",")[0] if a.smoke else "0,1,2", FOLDS="lopo", TAG=tag + "_lopo")),
        ("6_final_ckpt", [PYL, "scripts/train_egvla2_s1.py"], dict(SAVE_ONLY="1", FINAL_EPOCHS=str(ep), TAG=tag)),
    ]
    status = dict(started=time.strftime("%F %T"), epochs=ep, seeds=seeds, steps={})
    for name, cmd, env in steps:
        rc = step(name, cmd, env, a.dry)
        status["steps"][name] = rc
        (ROOT / "docs/v5/egvla2_fm/pipeline_status.json").write_text(json.dumps(status, indent=1))
        if rc:
            print("실패 — 로그:", LOG / f"{name}.log"); sys.exit(rc)
    print("완료. 결과: docs/v5/egvla2_fm/summary_s1" + tag + "*.json, 체크포인트 runs/egvla2_fm/")


if __name__ == "__main__":
    main()
