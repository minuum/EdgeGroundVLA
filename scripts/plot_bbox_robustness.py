"""C 상황(박스 훼손) 결과 그래프. 출력 docs/v5/figures/ch79/bbox_robustness.png"""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
plt.rcParams["font.family"] = "NanumGothic" if any("Nanum" in f.name for f in fm.fontManager.ttflist) else "DejaVu Sans"
ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / "docs/v5/egvla2_fm/bbox_robustness.json"))
VS = [("V0_img_only", "V0 현재 A3 방식", "#94a3b8"), ("V0R_bbox_aug", "V0R 훼손 증강(보완안)", "#34d399"), ("V4_big_vocab", "V4 텍스트 조건화", "#fbbf24")]
CONDS = [("clean", "정상 박스"), ("drop50", "50% 제거"), ("drop100", "전부 제거"), ("jitter", "흔들림"), ("wrong", "반대편 오검출")]
fig, axs = plt.subplots(1, 4, figsize=(20, 4.6))
for ax, key, ttl, yl in [(axs[0], "servo_ang", "서보 라벨(바구니 방향) 대비 각도 (°, 낮을수록 좋음)", "°"), (axs[1], "d_clean", "박스 의존도: 훼손 시 명령 변화 (0 = 박스를 안 씀)", ""), (axs[2], "ly_sign", "좌우 부호가 바구니 쪽과 일치 (%, 높을수록 좋음)", "%"), (axs[3], "stop", "정상 장면에서 정지 (%, 낮을수록 좋음)", "%")]:
    w = 0.26
    for j, (v, lab, col) in enumerate(VS):
        m, s = [], []
        for c, _ in CONDS:
            x = [r[c][key] for k, r in d.items() if k.split("|")[1] == v]
            f = 100 if key in ("stop", "ly_sign") else 1
            m.append(np.mean(x) * f if x else np.nan); s.append(np.std(x) * f if x else 0)
        ax.bar(np.arange(5) + (j - 1) * w, m, w, yerr=s, color=col, capsize=2, label=lab, edgecolor="#1e293b")
        for i, a in enumerate(m):
            if not np.isnan(a): ax.text(i + (j - 1) * w, a + max(s) * 0.0 + (0.01 if key == "d_clean" else 0.8), (f"{a:.3f}" if key == "d_clean" else f"{a:.1f}"), ha="center", fontsize=6)
    ax.set_xticks(range(5)); ax.set_xticklabels([n for _, n in CONDS], fontsize=9); ax.set_title(ttl); ax.set_ylabel(yl); ax.grid(axis="y", alpha=.3)
axs[0].legend(fontsize=7, loc="upper left")
fig.suptitle("C 상황 — 박스를 훼손했을 때 (3분할 × 시드 3, 평균 ± sd, 오프라인·합성 훼손)", fontsize=12)
fig.tight_layout(); fig.savefig(ROOT / "docs/v5/figures/ch79/bbox_robustness.png", dpi=110); print("ok")
