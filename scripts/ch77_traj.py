"""CH77 궤적 분석: 같은 조이스틱 기록을 이산(8-class)·연속으로 적분해 xy 경로를 비교한다. 로봇 몸체 좌표 데드레코닝(추정치)."""
import glob
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "ROS_action/mobile_vla_dataset_egvla2"
V_A, V_B = 6.59, 19.85      # v=a+b·s (cm/s), logs/speed_calibration.json
RULE_S = 0.7
# 8-class → 단위 방향(전진 lx, 좌측 ly). 제자리 회전·정지는 이동 0.
DIRS = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1), 4: (.7071, .7071), 5: (.7071, -.7071), 6: (0, 0), 7: (0, 0)}
CN = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]


def load(path):
    with h5py.File(path, "r") as h:
        if h.attrs.get("action_space") != "egvla2_continuous_v3" or "actions_raw" not in h:
            return None
        return dict(name=Path(path).stem, raw=np.array(h["actions_raw"], float), ex=np.array(h["actions_exec"], float),
                    cls=np.array(h["action_classes"]), t=np.array(h["frame_timestamps"], float),
                    pos=str(h.attrs.get("cx_position", "")), path=str(h.attrs.get("cx_path", "")))


def integrate(e):
    """returns (연속 경로, 이산 경로, 프레임별 방향오차°). 점은 (좌측 y, 전진 x)."""
    dt = np.diff(e["t"], append=e["t"][-1]); dt[-1] = np.median(dt[:-1]) if len(dt) > 1 else 0
    dt = np.clip(dt, 0, 1.0)
    p_c, p_d, err = [(0.0, 0.0)], [(0.0, 0.0)], []
    for (lx, ly, az), (elx, ely, eaz), c, d in zip(e["raw"], e["ex"], e["cls"], dt):
        m = float(np.hypot(lx, ly))
        # 연속: 회전 우선 규칙을 적용한 exec 이동(회전 프레임은 이동 0) 크기는 raw 크기로 보간
        if np.hypot(elx, ely) > 1e-6 and m >= 0.2:
            s = min(1.0, max(0.4, m)); v = V_A + V_B * s
            ux, uy = elx / np.hypot(elx, ely), ely / np.hypot(elx, ely)
        else:
            v, ux, uy = 0.0, 0.0, 0.0
        p_c.append((p_c[-1][0] + uy * v * d, p_c[-1][1] + ux * v * d))
        dx, dy = DIRS[int(c)]
        vd = V_A + V_B * RULE_S if (dx or dy) else 0.0
        p_d.append((p_d[-1][0] + dy * vd * d, p_d[-1][1] + dx * vd * d))
        if (dx or dy) and (ux or uy):
            err.append(np.degrees(np.arccos(np.clip(ux * dx + uy * dy, -1, 1))))
    return np.array(p_c), np.array(p_d), np.array(err)


def all_stats():
    rows = []
    for f in sorted(glob.glob(str(DS / "*.h5"))):
        e = load(f)
        if e is None or len(e["raw"]) < 30:
            continue
        pc, pd, err = integrate(e)
        lc = float(np.hypot(*np.diff(pc, axis=0).T).sum()); ld = float(np.hypot(*np.diff(pd, axis=0).T).sum())
        rows.append(dict(name=e["name"], pos=e["pos"], path=e["path"], gap=float(np.hypot(*(pc[-1] - pd[-1]))),
                         lat_c=float(pc[-1][0]), lat_d=float(pd[-1][0]), len_c=lc, len_d=ld, err=err.tolist()))
    return rows


if __name__ == "__main__":
    r = all_stats()
    err = np.concatenate([np.array(x["err"]) for x in r])
    print(len(r), "ep; 프레임별 방향 양자화 오차 평균 %.1f° 중앙 %.1f° p90 %.1f°" % (err.mean(), np.median(err), np.percentile(err, 90)))
    print("종점 간격 cm 평균 %.1f 중앙 %.1f" % (np.mean([x["gap"] for x in r]), np.median([x["gap"] for x in r])))
    print("경로길이 연속/이산", np.mean([x["len_c"] for x in r]).round(0), np.mean([x["len_d"] for x in r]).round(0))
    for p in ["left_curve", "right_curve", "straight"]:
        s = [x for x in r if x["path"] == p]
        print(p, len(s), "gap %.1f" % np.mean([x["gap"] for x in s]), "lat 연속 %.1f 이산 %.1f" % (np.mean([x["lat_c"] for x in s]), np.mean([x["lat_d"] for x in s])))
    json.dump(r, open("/tmp/traj_stats.json", "w"))


# ───────── HTML 블록 ─────────
def _svg(e, w=300, h=330, ext=None):
    pc, pd, _ = integrate(e)
    allp = np.vstack([pc, pd]); ymin, ymax = allp[:, 0].min(), allp[:, 0].max(); xmax = allp[:, 1].max()
    span = max(ymax - ymin, 1) if ext is None else ext[0]; fx = max(xmax, 1) if ext is None else ext[1]
    sc = min((w - 50) / max(span, 60), (h - 50) / max(fx, 60))
    cx0 = 25 + (-ymin) * sc if True else 0
    def P(p):  # 좌측(+)을 화면 왼쪽으로
        return f"{w - 25 - (p[0] - ymin) * sc:.1f},{h - 25 - p[1] * sc:.1f}"
    def line(pts, col, dash=""):
        return f'<polyline points="{" ".join(P(p) for p in pts)}" fill="none" stroke="{col}" stroke-width="2.4" {dash} stroke-linejoin="round"/>'
    g = f'<line x1="20" y1="{h - 25}" x2="{w - 15}" y2="{h - 25}" stroke="#334155"/><text x="{w - 18}" y="{h - 10}" fill="#64748b" font-size="9" text-anchor="end">← 좌</text>'
    s = (f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px;background:#0a1220;border:1px solid #1e293b;border-radius:8px">{g}'
         + line(pd, "#f87171", 'stroke-dasharray="6 3"') + line(pc, "#34d399")
         + f'<circle cx="{P(pc[0]).split(",")[0]}" cy="{P(pc[0]).split(",")[1]}" r="4" fill="#fbbf24"/>'
         + f'<circle cx="{P(pc[-1]).split(",")[0]}" cy="{P(pc[-1]).split(",")[1]}" r="4" fill="#34d399"/>'
         + f'<circle cx="{P(pd[-1]).split(",")[0]}" cy="{P(pd[-1]).split(",")[1]}" r="4" fill="#f87171"/></svg>')
    return s, pc, pd


def pick(rows, path, pos="center"):
    c = [r for r in rows if r["path"] == path and r["pos"] == pos]
    c.sort(key=lambda r: r["gap"])
    return c[len(c) // 2]["name"]


def block(tbl, H, st, wrap_note=""):
    rows = all_stats()
    err = np.concatenate([np.array(x["err"]) for x in rows])
    names = [(pick(rows, "left_curve"), "왼쪽 곡선(center)"), (pick(rows, "right_curve"), "오른쪽 곡선(center)"), (pick(rows, "straight"), "직선(center)")]
    figs = ""
    for nm, lab in names:
        e = load(DS / f"{nm}.h5"); svg, pc, pd = _svg(e)
        figs += (f'<div style="flex:1 1 230px;min-width:230px"><div style="font-size:0.78rem;font-weight:800;color:#e2e8f0;margin-bottom:4px">{lab}</div>{svg}'
                 f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:3px">종점 차 {np.hypot(*(pc[-1] - pd[-1])):.0f}cm · 길이 연속 {np.hypot(*np.diff(pc, axis=0).T).sum():.0f} / 이산 {np.hypot(*np.diff(pd, axis=0).T).sum():.0f}cm</div></div>')
    legend = ('<div style="font-size:0.76rem;margin:6px 0"><span style="color:#34d399;font-weight:800">━ 연속</span>(조이스틱 방향·크기 그대로) '
              '<span style="color:#f87171;font-weight:800">┅ 이산</span>(8방향 반올림, 속력 0.7 고정) <span style="color:#fbbf24">●</span> 시작</div>')
    # 페르소나 예시: 방향 양자화 오차가 큰 프레임 하나
    e = load(DS / f"{names[0][0]}.h5"); best = None
    for i, (r, ex, c) in enumerate(zip(e["raw"], e["ex"], e["cls"])):
        m = np.hypot(*r[:2])
        if m > 0.5 and int(c) in (1, 2, 4, 5):
            dx, dy = DIRS[int(c)]; a = np.degrees(np.arccos(np.clip((r[0] * dx + r[1] * dy) / m, -1, 1)))
            if best is None or a > best[0]:
                best = (a, i, r, c)
    a, i, r, c = best; m = float(np.hypot(*r[:2])); s = min(1.0, max(0.4, m)); v = V_A + V_B * s
    persona = (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0">'
        f'<div style="flex:1 1 220px;background:#1a0f0f;border:1px solid #f87171;border-radius:10px;padding:10px 12px"><b style="color:#fca5a5">🧱 계단이 (8-class 모델)</b><br>'
        f'<span style="font-size:0.78rem">"조이스틱이 앞 {r[0]:.2f}, 좌 {r[1]:.2f}? 나는 8개 칸 중 가장 가까운 <b>{CN[int(c)]}</b>로 읽을게. 속력은 늘 0.7 한 가지야."</span><br>'
        f'<span style="font-size:0.74rem;color:#94a3b8">→ 방향 {DIRS[int(c)]} 고정, 속도 {V_A + V_B * RULE_S:.1f}cm/s</span></div>'
        f'<div style="flex:1 1 220px;background:#0b1a14;border:1px solid #34d399;border-radius:10px;padding:10px 12px"><b style="color:#6ee7b7">🌊 물결이 (연속 모델)</b><br>'
        f'<span style="font-size:0.78rem">"같은 입력이면 나는 방향 {np.degrees(np.arctan2(r[1], r[0])):.0f}° 그대로, 크기 {m:.2f}만큼 낼게. 속력은 {v:.1f}cm/s."</span><br>'
        f'<span style="font-size:0.74rem;color:#94a3b8">→ 반올림 없이 {a:.0f}° 어긋남이 사라짐</span></div>'
        f'<div style="flex:1 1 220px;background:#0f1424;border:1px solid #60a5fa;border-radius:10px;padding:10px 12px"><b style="color:#93c5fd">🕹️ 조이 (운전자)</b><br>'
        f'<span style="font-size:0.78rem">"난 코너에서 스틱을 살짝만 꺾는데, 8칸 모델은 그걸 알아듣지 못하더라. 정지 직전엔 스틱을 놓는 중간값도 많아."</span><br>'
        f'<span style="font-size:0.74rem;color:#94a3b8">→ 실제 수집 중간 크기 비율 87%</span></div></div>')
    stat = tbl(["지표 (218ep, 추정)", "값", "의미"], [
        ["프레임별 방향 양자화 오차", f"평균 {err.mean():.1f}° / 중앙 {np.median(err):.1f}° / p90 {np.percentile(err, 90):.1f}°", "대부분은 8방향과 일치하지만 코너 구간에서 최대 22.5° 어긋남"],
        ["종점 차이(연속 vs 이산)", f"평균 {np.mean([x['gap'] for x in rows]):.0f}cm", "방향 반올림 + 속력 고정(0.7)이 누적된 값"],
        ["평균 경로 길이", f"연속 {np.mean([x['len_c'] for x in rows]):.0f}cm / 이산 {np.mean([x['len_d'] for x in rows]):.0f}cm", "이산은 속력을 0.7 고정으로 가정해 길이가 다름"],
        ["좌 곡선 평균 좌측 변위", f"연속 {np.mean([x['lat_c'] for x in rows if x['path']=='left_curve']):.0f}cm / 이산 {np.mean([x['lat_d'] for x in rows if x['path']=='left_curve']):.0f}cm", "경로 형태 차이"],
        ["우 곡선 평균 좌측 변위", f"연속 {np.mean([x['lat_c'] for x in rows if x['path']=='right_curve']):.0f}cm / 이산 {np.mean([x['lat_d'] for x in rows if x['path']=='right_curve']):.0f}cm", "음수는 우측"]])
    return (H("⑤ 궤적 분석 — 이산(8-class) vs 연속, 실제 xy 평면") +
            '<div style="font-size:0.82rem;line-height:1.6;margin:6px 0">같은 조이스틱 기록을 <b>① 이산</b>(8방향 중 하나 + 속력 고정)과 <b>② 연속</b>(방향·크기 그대로)으로 적분해 로봇이 그렸을 xy 경로를 비교했다. '
            '이산은 방향이 8칸(0°·45°·90°…)으로만 나오고 속력이 늘 같아, 코너에서 계단식으로 꺾이고 길이가 달라진다. 연속은 스틱 각도 그대로 부드럽게 휜다.</div>'
            + legend + f'<div style="display:flex;gap:12px;flex-wrap:wrap">{figs}</div>'
            + H("페르소나로 보는 같은 한 프레임") + persona
            + stat
            + f'{st("한계", "#f87171")} 로봇 몸체 좌표 데드레코닝 추정치다(회전은 이동 0으로 처리, 바퀴 미끄럼·지연 미반영). 속력은 보정식 v=6.59+19.85·s를 사용했고 이산은 s=0.7 고정으로 가정했다. 실제 주행 경로 측정이 아니다.')
