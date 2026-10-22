#!/usr/bin/env python3
"""구동 방식 3종(정지-이동 S / 직전 명령 유지 H / 청크 스트림 C)의 도착 시간·이동 구성 시뮬레이션(분석적, 로봇·모델 불필요).
입력: 학습 300 에피소드의 사람 경로 길이(조이스틱 속도를 8fps 프레임 간격 0.128초로 적분), 추론 지연 L, 이동 구간 0.4초, 대기 0.45초, 속도 0.7(20.5cm/s).
S: 판단마다 0.4초 이동 후 정지(주기 = L + 0.45). H·C: 이동을 멈추지 않음(첫 추론 L초 후 출발, 이후 연속).
출력: docs/v5/egvla2_fm/drive_modes_sim.json. 실행: .venv_lerobot/bin/python3 scripts/sim_drive_modes.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import train_egvla2_s1 as S1  # noqa: E402

V = 6.59 + 19.85 * 0.7            # cm/s (규칙 속도 0.7)
DT, MOVE, WAIT = 0.128, 0.4, 0.45


def main():
    eps = S1.load_main()
    # 사람 경로 길이: 프레임별 속도(보정식) × dt 적분
    L_cm = []
    for e in eps:
        mag = np.linalg.norm(e["chunk"][:, 0, :2], axis=1)
        v = np.where(mag > 0.05, 6.59 + 19.85 * np.clip(mag, 0, 1.0), 0)
        L_cm.append(float((v * DT).sum()))
    L_cm = np.array(L_cm)
    out = {"human_path_cm": dict(p10=float(np.percentile(L_cm, 10)), median=float(np.median(L_cm)), p90=float(np.percentile(L_cm, 90)))}
    for lat in (0.5, 0.9, 1.3):
        rows = {}
        # S: 판단 n회 = ceil(L / (V×MOVE)), 시간 = n×(lat+WAIT), 이동 중 시간 비율 = MOVE/(lat+WAIT)
        n = np.ceil(L_cm / (V * MOVE)); t_s = n * (lat + WAIT)
        # H/C: 추론이 끝나는 lat초 후 출발, 이후 L/V초 연속 이동 (정지 판정은 응답 단위라 최대 lat초 늦음 → +lat/2 평균 가정)
        t_c = lat + L_cm / V + lat / 2
        # 낡은 명령으로 이동하는 거리: 연속 모드는 명령이 평균 lat/2 + 응답 간격 lat/2 만큼 낡음(평균 가정 lat) × V
        rows["S"] = dict(time_s=float(np.median(t_s)), decisions=float(np.median(n)), moving_ratio=MOVE / (lat + WAIT), stale_move_cm=0.0)
        rows["H_C"] = dict(time_s=float(np.median(t_c)), decisions=float(np.median(np.ceil(t_c / lat))), moving_ratio=float(np.median((L_cm / V) / t_c)), stale_move_cm=float(lat * V))
        rows["speedup"] = float(np.median(t_s) / np.median(t_c))
        out[f"latency_{lat}"] = rows
    (ROOT / "docs/v5/egvla2_fm/drive_modes_sim.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
