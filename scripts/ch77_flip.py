"""CH77 ⑥-15: 좌우 반전 증강 시험(기각). 데이터: flip_aug.json (FM-π0, 드롭아웃30%, 120ep, seed 0·1)."""
import json

import numpy as np

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R, B = V.A, V.G, V.R, V.B


def block(tbl, H, st):
    f = FM / "flip_aug.json"
    if not f.exists():
        return ""
    d = json.load(open(f))
    sp = sorted({k.split("|")[0] for k in d}, key=lambda x: (x != "fold0", x))
    rows = []
    for s in sp:
        ks = [k for k in d if k.startswith(s + "|")]
        b = np.mean([d[k]["base"]["angle_deg"] for k in ks]); fl = np.mean([d[k]["flip"]["angle_deg"] for k in ks])
        rows.append((s.replace("lopo_", "안 본 목표 위치: ").replace("fold0", "같은 목표 위치(fold0)"), [[b, fl, fl - b]], None))
    lo = [k for k in d if k.startswith("lopo")]
    worse = sum(d[k]["flip"]["angle_deg"] > d[k]["base"]["angle_deg"] for k in lo)
    mb = np.mean([d[k]["base"]["angle_deg"] for k in lo]); mf = np.mean([d[k]["flip"]["angle_deg"] for k in lo])
    t = Z.heat_table([("방향 오차(°, seed 2개 평균)", ["반전 없음", "반전 증강", "차이"], [B, A, R])], rows, fmt="{:+.2f}", corner="분할")
    return (H("⑥-15 좌우 반전 증강 시험 — 기각")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">안 본 목표 위치(특히 strong_right)의 체계적 오류를 줄이려고, 학습 에피소드를 수평 반전(이미지 반전 후 Florence-2 특징·bbox 재계산, 라벨 ly·az 부호 반전, 목표 위치·경로 좌우 교환)해 학습 데이터를 2배로 늘렸다. '
              '평가는 실제 hold-out 에피소드, hold-out의 반전본은 학습에서 제외(누설 없음), 같은 seed로 쌍 비교.</div>'
            + t.replace("+", "") if False else (H("⑥-15 좌우 반전 증강 시험 — 기각")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">안 본 목표 위치(특히 strong_right)의 체계적 오류를 줄이려고, 학습 에피소드를 수평 반전(이미지 반전 후 Florence-2 특징·bbox 재계산, 라벨 ly·az 부호 반전, 목표 위치·경로 좌우 교환)해 학습 데이터를 2배로 늘렸다. '
              '평가는 실제 hold-out 에피소드, hold-out의 반전본은 학습에서 제외(누설 없음), 같은 seed로 쌍 비교.</div>'
            + Z.heat_table([("방향 오차(°, seed 2개 평균)", ["반전 없음", "반전 증강", "차이"], [B, A, R])], rows, fmt="{:.2f}", corner="분할")
            + f'{st("결론", R)} <b>기각.</b> 안 본 목표 위치 평균 {mb:.2f}° → {mf:.2f}°(10쌍 중 {worse}쌍 악화), 같은 목표 위치도 소폭 악화. strong_right만 seed에 따라 엇갈렸다. '
              '가능한 이유(검증 안 함): 실험장이 좌우 대칭이 아니거나(배치·장애물·카메라 위치), 사람의 좌우 조작 습관이 달라 반전 장면이 실제 분포와 어긋남. EGVLA1에서 미러 증강은 <b>헤드의 좌우 편향 제거</b>에는 효과가 있었지만, 이번 목적(새 위치 일반화)에는 맞지 않았다. 배포 체크포인트는 그대로 둔다.'))
