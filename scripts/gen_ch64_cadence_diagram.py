#!/usr/bin/env python3
"""baseline / stride-입력만 / HELD 세 조건의 제어루프 구조도 (CH64 보완)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from pathlib import Path

_kf = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
fm.fontManager.addfont(_kf)
plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight",
                     "savefig.facecolor": "white", "figure.facecolor": "white"})

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "verm": "#D55E00", "grey": "#999999", "black": "#222222"}

OUT = Path("/home/minum/26CS/MoNaVLA/docs/v5/ch64_figs")

fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=True)
T = 5.0  # 5초 타임라인
dt_fast = 1/6   # baseline 6Hz
dt_slow = 5/6   # HELD ~1.3Hz (stride5*dt_fast)

def draw_row(ax, mode, title):
    ax.set_xlim(0, T); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", color=C["black"])
    if mode == "baseline":
        ticks = np.arange(0, T, dt_fast)
        for t in ticks:
            ax.axvline(t, color=C["blue"], lw=1.2, alpha=0.35, ymin=0.15, ymax=0.55)
        for t in ticks:
            ax.plot(t, 0.75, marker="v", color=C["blue"], ms=7)
        ax.text(0.05, 0.9, "지각+결정+실행 매 프레임(0.17s)", fontsize=9, color=C["blue"], fontweight="bold")
        ax.add_patch(Rectangle((0, 0.15), T, 0.4, color=C["green"], alpha=0.12))
        ax.text(T-0.05, 0.2, "연속 반응형 — 오차 발생 즉시 보정 가능", fontsize=8.5, color=C["green"],
                ha="right", fontweight="bold")
    elif mode == "input_only":
        fine = np.arange(0, T, dt_fast)
        for t in fine:
            ax.plot(t, 0.75, marker="v", color=C["blue"], ms=6)
        # 넓은 기억창 표시(4.6s 폭)
        for t0 in [0.0, 1.7, 3.4]:
            ax.add_patch(Rectangle((t0, 0.55), min(4.6, T-t0), 0.15, color=C["orange"], alpha=0.25))
        ax.text(0.05, 0.9, "결정은 매 프레임(0.17s) — 단, 보는 과거가 4.6초로 흐릿하게 늘어남", fontsize=9,
                color=C["orange"], fontweight="bold")
        ax.add_patch(Rectangle((0, 0.15), T, 0.35, color=C["green"], alpha=0.10))
        ax.text(T-0.05, 0.2, "반응 속도는 그대로(기억만 흐림) → 성능 거의 안 변함", fontsize=8.5,
                color=C["green"], ha="right", fontweight="bold")
    else:  # held
        ticks = np.arange(0, T, dt_slow)
        for i, t in enumerate(ticks):
            ax.plot(t, 0.75, marker="v", color=C["verm"], ms=9)
            t_end = ticks[i+1] if i+1 < len(ticks) else T
            ax.add_patch(Rectangle((t, 0.3), t_end-t, 0.25, color=C["verm"], alpha=0.28-0.05*(i%2)))
            ax.annotate("", xy=(t_end-0.02, 0.42), xytext=(t+0.05, 0.42),
                        arrowprops=dict(arrowstyle="->", color=C["verm"], lw=1.3))
            ax.text((t+t_end)/2, 0.15, "그대로 유지\n(무반응 구간)", fontsize=7, ha="center",
                    color=C["verm"])
        ax.text(0.05, 0.9, "결정은 0.83s에 1번 — 그 사이엔 직전 명령을 맹목적으로 실행(zero-order hold)",
                fontsize=9, color=C["verm"], fontweight="bold")
        ax.add_patch(Rectangle((0, -0.02), T, 0.06, color=C["verm"], alpha=0.5))

for ax, mode, title in zip(axes,
        ["baseline", "input_only", "held"],
        ["① baseline (stride=1) — 연속 반응형 제어, 6Hz",
         "② 입력만 stride=5 — 판단은 여전히 6Hz, 기억(맥락)만 4.6초로 흐림",
         "③ HELD (진짜 실서빙 재현) — 판단 자체가 1.3Hz, 그 사이 개루프(무반응)"]):
    draw_row(ax, mode, title)

axes[-1].set_xlabel("시간 (초)", fontsize=10)
fig.suptitle("64-10 · 세 조건의 제어 루프 구조 비교 — HELD만 '결정 빈도' 자체가 다름",
             fontsize=12.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "fig_64_10_control_loop_diagram.png")
print("saved fig_64_10_control_loop_diagram.png")

# 결과 막대 (baseline/inputonly/HELD 요약, 자리표시자 — 실측값 있으면 스크립트에서 이후 갱신)
fig2, ax = plt.subplots(figsize=(6, 3.8))
labels = ["① baseline\n(6Hz 연속반응)", "② 입력만 흐림\n(맥락만 4.6s)", "③ HELD\n(진짜 실서빙, 1.3Hz)"]
vals = [39.4, 36.4, 24.2]
cols = [C["blue"], C["orange"], C["verm"]]
bars = ax.bar(labels, vals, color=cols)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("Closed-loop Success@0.5m (%)"); ax.set_ylim(0, 50)
ax.set_title("64-10 · '결정 빈도' 자체를 낮추면(HELD) 급락", fontweight="bold", fontsize=11)
fig2.tight_layout()
fig2.savefig(OUT / "fig_64_10_held_impact.png")
print("saved fig_64_10_held_impact.png")
