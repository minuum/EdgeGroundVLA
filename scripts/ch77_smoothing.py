"""CH77 ⑥-9: 서빙 평활 파라미터 스윕(K 샘플 수 × EMA). 데이터: smoothing_sweep.json. 서버 VLA_EGVLA2_EMA = 새 값 가중 = 1 − (스윕의 α)."""
import json

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R = V.A, V.G, V.R
KS, AL = (1, 4, 8), (0.0, 0.3, 0.5, 0.7, 0.85)
SPN = {"fold0": "같은 목표 위치 (fold0)", "lopo_strong_right": "안 본 목표 위치 (strong_right)"}


def block(tbl, H, st):
    f = FM / "smoothing_sweep.json"
    if not f.exists():
        return ""
    d = json.load(open(f)); HC = ["#94a3b8", "#38bdf8", "#4ade80"]
    grp = [(SPN[s], [f"K={k}" for k in KS], HC) for s in d]
    mk = lambda met, fmt, better: Z.heat_table(grp, [(f"EMA 과거 가중 α={a}" + (" (끔)" if a == 0 else ""), [[d[s][f"K{k}|a{a}"][met] for k in KS] for s in d], better) for a in AL], fmt=fmt, corner="행: 평활 세기", by="col")
    # 권장: 두 분할 평균에서 저크 최소, 단 방향 오차 증가 ≤0.3° 이고 반응 지연 ≤1프레임
    cand = []
    for k in KS:
        for a in AL:
            m = lambda met: sum(d[s][f"K{k}|a{a}"][met] for s in d) / len(d)
            cand.append((k, a, m("angle"), m("jitter"), m("delay_frames"), m("go_f1")))
    best_ang = min(c[2] for c in cand)
    ok = [c for c in cand if c[2] <= best_ang + 0.3 and c[4] <= 1.0]
    rec = min(ok, key=lambda c: c[3]) if ok else None
    box = lambda t, b: f'<div style="margin:10px 0 4px;font-size:0.82rem;font-weight:800;color:#fcd34d">{t}</div>{b}'
    out = (H("⑥-9 서빙 평활 파라미터 스윕 — K 샘플 수 × EMA (오프라인, seed 0)")
           + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">FM 헤드의 프레임 간 흔들림(저크)을 줄이려고 서빙에서 쓸 K 샘플 평균과 EMA를 정하기 위한 스윕. C(드롭아웃30%, 120 epoch)의 안 본 에피소드 예측을 시간 순서대로 평활하고 '
             '방향 오차·저크·GO/STOP F1·반응 지연(프레임)을 비교했다. 서버 옵션 <code>VLA_EGVLA2_EMA</code>는 <b>새 값 가중</b>이라 스윕의 α와 1−α 관계다.</div>' + Z.LEG
           + box("방향 오차 (°, 낮을수록 좋음)", mk("angle", "{:.2f}", "low")) + box("프레임 저크 (낮을수록 부드러움, 사람 0.045~0.063)", mk("jitter", "{:.3f}", "low"))
           + box("반응 지연 (프레임, 사람 조작을 따라가는 지연. 1프레임≈0.13초)", mk("delay_frames", "{:.0f}", "low")) + box("GO/STOP F1", mk("go_f1", "{:.3f}", "high")))
    if rec:
        out += (f'{st("권장(자동 선정)", G)} 두 분할 평균에서 방향 오차가 최저+0.3° 이내이고 지연 ≤1프레임인 조합 중 저크 최소 → <b>K={rec[0]}, EMA 과거 가중 α={rec[1]}</b> '
                f'(방향 오차 {rec[2]:.2f}°, 저크 {rec[3]:.3f}, 지연 {rec[4]:.1f}프레임, F1 {rec[5]:.3f}). 서버 설정: <code>VLA_EGVLA2_EMA={1 - rec[1]:.2f}</code>(α=0이면 0=끔), K={rec[0]}.')
    else:
        out += f'{st("권장", R)} 조건(방향 오차 +0.3° 이내, 지연 ≤1프레임)을 만족하는 조합이 없다. 드라이런 로그로 재조정.'
    return out + f'<div style="font-size:0.74rem;color:#64748b;margin-top:6px">한계: seed 하나, 분할 2개, 오프라인 재생(폐루프 반응 없음). 실제 최적값은 드라이런 로그로 확정한다.</div>'
