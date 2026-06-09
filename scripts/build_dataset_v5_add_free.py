#!/usr/bin/env python3
"""
V5_add_free 데이터셋 빌더

structured 에피소드 서브샘플 + free 에피소드 STOP 프레임 제거 후
ROS_action/mobile_vla_dataset_V5_add_free/ 에 새 데이터셋 생성.

Usage:
    python3 scripts/build_dataset_v5_add_free.py [--dry-run]
"""
import argparse
import os
import random
import shutil
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
DST  = ROOT / "ROS_action" / "mobile_vla_dataset_V5_add_free"

RANDOM_SEED = 42

# right_* 상한선 (나머지는 전량 유지)
SUBSAMPLE_CAPS = {
    "right_straight": 25,
    "right_left":     25,
    "right_right":    25,
}


def classify_path_type(name: str) -> str:
    """파일명에서 경로 타입 추출."""
    for pt in [
        "right_straight", "right_left", "right_right",
        "left_left", "left_straight", "left_right",
        "center_straight", "center_right", "center_left",
    ]:
        if f"_target_{pt}_path__" in name:
            return pt
    if "free_" in name:
        return "free"
    return "unknown"


def is_stop_frame(action: np.ndarray) -> bool:
    x, y, az = float(action[0]), float(action[1]), float(action[2])
    return abs(x) < 0.3 and abs(y) < 0.3 and abs(az) < 0.1


def copy_h5_without_mid_stops(src: Path, dst: Path) -> tuple[int, int]:
    """
    free 에피소드: 에피소드 중간 STOP 프레임 제거 후 새 H5 저장.
    마지막 프레임이 STOP이면 유지 (navigation 완료 신호).
    Returns (original_frames, kept_frames)
    """
    with h5py.File(src, "r") as f_in:
        actions = f_in["actions"][:]
        images  = f_in["observations"]["images"][:]
        instr   = f_in["language_instruction"][:]

    n = len(actions)
    keep = []
    for i, a in enumerate(actions):
        is_last = (i == n - 1)
        if is_stop_frame(a) and not is_last:
            continue
        keep.append(i)

    kept = np.array(keep)
    with h5py.File(dst, "w") as f_out:
        f_out.create_dataset("actions",              data=actions[kept])
        obs = f_out.create_group("observations")
        obs.create_dataset("images", data=images[kept])
        f_out.create_dataset("language_instruction", data=instr)

    return n, len(kept)


def copy_h5_relabel_mid_stops(src: Path, dst: Path) -> tuple[int, int]:
    """
    free 에피소드: 중간 STOP 프레임을 **삭제하지 않고** 직전 모션 액션으로 재지정.
    프레임(이미지)은 전부 유지 — STOP은 마지막 프레임에만 남는다.

    재지정 규칙(forward-fill): 비-STOP 직전 액션을 carry. 앞에 모션이 없으면 다음 모션 사용.
    마지막 프레임이 STOP이면 그대로 유지(navigation 완료 신호).
    Returns (total_frames, relabeled_count)
    """
    with h5py.File(src, "r") as f_in:
        actions = f_in["actions"][:].copy()
        images  = f_in["observations"]["images"][:]
        instr   = f_in["language_instruction"][:]

    n = len(actions)
    last_motion = None
    relabeled = 0
    for i in range(n):
        is_last = (i == n - 1)
        if is_stop_frame(actions[i]) and not is_last:
            # 직전 모션 carry, 없으면 다음 모션 backfill
            repl = last_motion
            if repl is None:
                repl = next((actions[j] for j in range(i + 1, n)
                             if not is_stop_frame(actions[j])), None)
            if repl is not None:
                actions[i] = repl
                relabeled += 1
        elif not is_stop_frame(actions[i]):
            last_motion = actions[i].copy()

    with h5py.File(dst, "w") as f_out:
        f_out.create_dataset("actions",              data=actions)   # 전 프레임 유지
        obs = f_out.create_group("observations")
        obs.create_dataset("images", data=images)
        f_out.create_dataset("language_instruction", data=instr)

    return n, relabeled


def main(dry_run: bool = False, stop_mode: str = "relabel") -> None:
    random.seed(RANDOM_SEED)

    all_h5 = sorted(SRC.glob("*.h5"))
    structured = [f for f in all_h5 if "free_" not in f.name]
    free_files  = [f for f in all_h5 if "free_" in f.name]

    # ── structured 서브샘플 ──────────────────────────────────────────
    by_type: dict[str, list[Path]] = {}
    for f in structured:
        pt = classify_path_type(f.name)
        by_type.setdefault(pt, []).append(f)

    selected_structured: list[Path] = []
    for pt, files in sorted(by_type.items()):
        cap = SUBSAMPLE_CAPS.get(pt, len(files))
        chosen = sorted(random.sample(files, min(cap, len(files))), key=lambda x: x.name)
        selected_structured.extend(chosen)
        print(f"  {pt:20s}: {len(files):3d} → {len(chosen):3d}개 선택")

    print(f"\nstructured 합계: {len(structured)} → {len(selected_structured)}개")
    mode_desc = "중간 STOP 재지정(프레임 유지)" if stop_mode == "relabel" else "중간 STOP 제거"
    print(f"free       합계: {len(free_files)}개 ({mode_desc})")

    if dry_run:
        print("\n[dry-run] 실제 파일 생성 없음.")
        return

    # ── 출력 디렉토리 생성 ────────────────────────────────────────────
    DST.mkdir(parents=True, exist_ok=True)

    # structured: 하드링크 (같은 파티션) 또는 복사
    copied_s = 0
    for src_f in selected_structured:
        dst_f = DST / src_f.name
        if dst_f.exists():
            dst_f.unlink()
        try:
            os.link(src_f, dst_f)
        except OSError:
            shutil.copy2(src_f, dst_f)
        copied_s += 1

    print(f"\nstructured {copied_s}개 링크/복사 완료")

    # free: relabel(프레임 유지) 또는 remove(삭제)
    total_before = total_after = total_relabel = 0
    for src_f in free_files:
        dst_f = DST / src_f.name
        if stop_mode == "relabel":
            orig, relabeled = copy_h5_relabel_mid_stops(src_f, dst_f)
            total_before += orig
            total_after  += orig          # 프레임 수 불변
            total_relabel += relabeled
            print(f"  {src_f.name[-60:]:60s}  {orig}프레임 유지 (mid-STOP {relabeled}개 재지정)")
        else:
            orig, kept = copy_h5_without_mid_stops(src_f, dst_f)
            total_before += orig
            total_after  += kept
            print(f"  {src_f.name[-60:]:60s}  {orig}→{kept} (-{orig-kept})")

    if stop_mode == "relabel":
        print(f"\nfree 합계: {total_before} 프레임 전부 유지 (mid-STOP {total_relabel}개 → 직전 모션 재지정, 마지막 STOP만 유지)")
    else:
        print(f"\nfree 합계: {total_before} → {total_after} 프레임 ({total_before-total_after} STOP 제거)")

    # ── 요약 ─────────────────────────────────────────────────────────
    total_ep = len(list(DST.glob("*.h5")))
    print(f"\n완료: {DST}")
    print(f"  총 에피소드: {total_ep}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-mode", choices=["relabel", "remove"], default="relabel",
                        help="relabel(기본): 중간 STOP을 직전 모션으로 재지정·프레임 유지 / "
                             "remove: 중간 STOP 프레임 삭제(구버전)")
    args = parser.parse_args()
    main(dry_run=args.dry_run, stop_mode=args.stop_mode)
