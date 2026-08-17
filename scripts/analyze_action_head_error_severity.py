#!/usr/bin/env python3
"""배포 행동 헤드의 val 오류를 '심각도'로 분해한다.

val 정확도(74.13%)가 실기 성공률(95%)보다 훨씬 낮은 이유를 설명하기 위한 분석.
프레임 단위 정확도는 사람이 조작한 단 하나의 행동과의 일치를 요구하지만, 실제로는
같은 시점에서 목표에 접근하는 행동이 여러 개일 수 있다. 따라서 오류를 다음으로 나눈다.

  좌우 반전  : 좌 계열 ↔ 우 계열 — 폐루프에서도 회복이 어려운 치명적 오류
  경미       : 같은 좌우 성향 내 혼동(예: FORWARD ↔ FWD+LEFT) — 대체로 목표 접근에 유효
  STOP 관련  : 정지 판정 시점 차이

입력 : docs/v5/detector/confusion_matrix_stage1v3_correct.json
출력 : docs/v5/detector/action_head_error_severity.json
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs/v5/detector/confusion_matrix_stage1v3_correct.json"
OUT = ROOT / "docs/v5/detector/action_head_error_severity.json"

LEFTISH  = {2, 4, 6}   # LEFT, FWD+L, ROT_L
RIGHTISH = {3, 5, 7}   # RIGHT, FWD+R, ROT_R
NEUTRAL  = {0, 1}      # STOP, FWD


def side(i: int) -> str:
    return "L" if i in LEFTISH else ("R" if i in RIGHTISH else "N")


def main():
    d = json.loads(SRC.read_text())
    cm = np.array(d["confusion_matrix_sum"])
    names = d["class_names"]
    total = int(cm.sum())
    correct = int(np.trace(cm))
    errors = total - correct

    flip = benign = stopmix = 0
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            v = int(cm[i][j])
            if not v:
                continue
            si, sj = side(i), side(j)
            if (si, sj) in (("L", "R"), ("R", "L")):
                flip += v
            elif i == 0 or j == 0:
                stopmix += v
            else:
                benign += v

    # FORWARD ↔ 대각(FWD+L/FWD+R) 상호 혼동만
    fwd_diag = sum(int(cm[i][j]) for i, j in [(1, 4), (4, 1), (1, 5), (5, 1)])

    # 좌/우/중립 3-way로 축약한 정확도
    idx = {"L": 0, "R": 1, "N": 2}
    agg = np.zeros((3, 3), dtype=int)
    for i in range(len(names)):
        for j in range(len(names)):
            agg[idx[side(i)]][idx[side(j)]] += cm[i][j]
    acc3 = float(np.trace(agg) / total)

    rep = {
        "source": str(SRC.relative_to(ROOT)),
        "n_val_decisions": total,
        "val_acc": correct / total,
        "errors": errors,
        "severity": {
            "direction_flip": {"n": flip, "of_total": flip / total, "of_errors": flip / errors},
            "benign_same_side": {"n": benign, "of_total": benign / total, "of_errors": benign / errors},
            "stop_related": {"n": stopmix, "of_total": stopmix / total, "of_errors": stopmix / errors},
        },
        "fwd_vs_diagonal_only": {"n": fwd_diag, "of_errors": fwd_diag / errors,
                                 "acc_if_treated_correct": (correct + fwd_diag) / total},
        "acc_collapsed_to_left_right_neutral": acc3,
    }
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    print(f"val {correct}/{total} = {correct/total*100:.2f}%  오류 {errors}건")
    print(f"  좌우 반전   {flip:4d}  전체 {flip/total*100:5.2f}%  오류중 {flip/errors*100:5.1f}%")
    print(f"  경미        {benign:4d}  전체 {benign/total*100:5.2f}%  오류중 {benign/errors*100:5.1f}%")
    print(f"  STOP 관련   {stopmix:4d}  전체 {stopmix/total*100:5.2f}%  오류중 {stopmix/errors*100:5.1f}%")
    print(f"  FWD↔대각만  {fwd_diag:4d}  오류중 {fwd_diag/errors*100:.1f}%  "
          f"→ 정답 처리 시 {(correct+fwd_diag)/total*100:.2f}%")
    print(f"  좌/우/중립 3-way 축약 정확도 {acc3*100:.2f}%")
    print(f"저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
