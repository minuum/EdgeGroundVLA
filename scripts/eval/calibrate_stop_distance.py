#!/usr/bin/env python3
"""
plan_20260622_train_inference_image_pipeline_unify.md §1 항목4 — STOP 거리를
"area>0.25"(정규화 면적) 대신 cm 단위로 설정하기 위한 캘리브레이션 계산기.

물리 원리: 카메라 핀홀 모델에서 물체의 화면상 면적은 거리의 제곱에 반비례한다
  area = k / distance^2   (k는 카메라·렌즈·물체 크기에 따른 상수)

실측 (distance_cm, area) 쌍이 **2개 이상**만 있으면 k를 구하고, 원하는 거리
(40cm, 50cm 등)에서의 area 임계값을 역산할 수 있다.

⚠️ 아직 실측 데이터 없음 — 이 스크립트는 실측 전까지 TODO 상태.
실측 방법: 로봇/카메라를 바스켓에서 정확히 알려진 거리(예: 30/40/60/80cm)에
놓고 grounding을 돌려서 area를 기록 → --measurements로 입력.

Usage (실측 후):
  .venv/bin/python3 scripts/eval/calibrate_stop_distance.py \
      --measurements 40:0.18 80:0.045 \
      --targets 40 50

Usage (지금, 실측 전 — 더미값으로 사용법만 확인):
  .venv/bin/python3 scripts/eval/calibrate_stop_distance.py --demo
"""
import argparse
import numpy as np


def fit_k(measurements: list[tuple[float, float]]) -> tuple[float, float]:
    """area = k / distance^2 최소제곱 적합. 반환: (k, residual_std)."""
    d = np.array([m[0] for m in measurements], dtype=float)
    a = np.array([m[1] for m in measurements], dtype=float)
    k_estimates = a * d**2
    k = float(np.mean(k_estimates))
    residual_std = float(np.std(k_estimates))
    return k, residual_std


def area_at_distance(k: float, distance_cm: float) -> float:
    return k / (distance_cm ** 2)


def distance_at_area(k: float, area: float) -> float:
    return float(np.sqrt(k / area))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--measurements", nargs="*", default=[],
                    help="distance_cm:area 쌍, 예: 40:0.18 80:0.045 (2개 이상)")
    p.add_argument("--targets", nargs="*", type=float, default=[40.0, 50.0],
                    help="threshold를 역산할 목표 거리(cm), 기본 40/50")
    p.add_argument("--demo", action="store_true",
                    help="실측 전 사용법만 확인 — 더미값(실제 측정 아님, 그래프 모양 확인용)")
    args = p.parse_args()

    if args.demo:
        print("⚠️  DEMO 모드 — 아래는 실측이 아니라 더미값입니다. 사용법 확인용.\n")
        measurements = [(40.0, 0.18), (80.0, 0.045)]
    elif args.measurements:
        measurements = []
        for m in args.measurements:
            d_str, a_str = m.split(":")
            measurements.append((float(d_str), float(a_str)))
    else:
        print("측정값이 없습니다. --measurements distance:area ... 형식으로 입력하거나, "
              "--demo로 사용법만 먼저 확인하세요.")
        print("\n실측 방법:")
        print("  1. 바스켓 실제 크기 한 번 측정(참고용, 필수 아님)")
        print("  2. 로봇/카메라를 바스켓에서 정확히 알려진 거리(예: 30/40/60/80cm)에 놓고")
        print("     grounding 1회씩 실행해 area 기록 (2곳 이상)")
        print("  3. 이 스크립트에 --measurements로 입력")
        return

    if len(measurements) < 2:
        print("측정값이 2개 미만이면 k를 신뢰성 있게 못 구합니다 — 최소 2곳 이상 측정 필요.")
        return

    k, residual_std = fit_k(measurements)
    print(f"입력 측정값: {measurements}")
    print(f"피팅 결과: area = {k:.4f} / distance_cm^2  (residual_std={residual_std:.4f}, 작을수록 핀홀모델에 잘 맞음)\n")

    print(f"{'목표 거리(cm)':>14} {'area threshold':>16}")
    for t in args.targets:
        a = area_at_distance(k, t)
        print(f"{t:>14.1f} {a:>16.4f}")

    print(f"\n→ '{args.targets[0]:.0f}~{args.targets[-1]:.0f}cm 이전에 정지' 구현 시,")
    print(f"  GOAL_AREA_THRESHOLD를 위 area 중 더 큰 값(=더 가까운 목표 거리)으로 설정하면 됩니다.")
    print(f"  (area가 그 값 이상이면 STOP — 현재 stage2_v2_inference_server.py의 GOAL_AREA_THRESHOLD와 동일한 의미)")


if __name__ == "__main__":
    main()
