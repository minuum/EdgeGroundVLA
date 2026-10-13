#!/usr/bin/env python3
"""CH76용 실제 데이터 그림 생성: 파일럿 v2/v1 h5 프레임 + 스틱 입력 + F2 bbox 오버레이.
출력: docs/v5/egvla2_fm/figures/*.jpg
실행: .venv_lerobot/bin/python3 scripts/gen_ch76_figures.py   (HF/GPU 불필요, FM 예측 그림만 GPU 사용)
"""
import glob
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
D = ROOT / "ROS_action/mobile_vla_dataset_egvla2"
OUT = ROOT / "docs/v5/egvla2_fm/figures"
CACHE = np.load(ROOT / "docs/v5/egvla2_fm/v3_florence2_cache.npz")
fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf")
plt.rcParams.update({"font.family": "NanumSquareRound", "axes.unicode_minus": False, "figure.facecolor": "#0a1628", "axes.facecolor": "#0f1d33",
                     "text.color": "#e2e8f0", "axes.labelcolor": "#cbd5e1", "xtick.color": "#94a3b8", "ytick.color": "#94a3b8", "axes.edgecolor": "#334155"})
RED, GRN, AMB, BLU, PUR = "#f87171", "#4ade80", "#fbbf24", "#38bdf8", "#a78bfa"
LEVEL_NAME = {0.4: "느림", 0.7: "보통", 1.0: "빠름"}
PILOT = [x if x.endswith(".h5") else x + ".h5" for x in json.load(open(ROOT / "docs/v5/egvla2_fm/meta/manual_dataset_sets.json"))["sets"]["pilot_v2_gauge_20261013"]]


def load(name):
    with h5py.File(D / name) as f:
        return dict(img=np.array(f["images"])[:, :, :, ::-1], raw=np.array(f["actions_raw"]), exec_=np.array(f["actions_exec"]),
                    scaled=np.array(f["actions_scaled"]), t=np.array(f["frame_timestamps"]), name=name,
                    bbox=CACHE[f"{name}|bbox"] if f"{name}|bbox" in CACHE else None)


def hud(ax, lx, ly, az, col=AMB, size=0.9):
    """왼쪽 아래 미니 스틱 HUD: 위=전진(lx), 오른쪽=우(ly<0)."""
    c = plt.Circle((0, 0), 1, fill=False, ec="#94a3b8", lw=1); ax.add_patch(c)
    ax.plot([-1, 1], [0, 0], color="#334155", lw=0.6); ax.plot([0, 0], [-1, 1], color="#334155", lw=0.6)
    m = np.hypot(lx, ly)
    if m > 1e-3:
        ax.annotate("", xy=(-ly, lx), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); ax.axis("off")


def fig_sticks():
    v1, v2 = [], []
    for f in sorted(glob.glob(str(D / "*.h5"))):
        with h5py.File(f) as h:
            sp = h.attrs.get("action_space"); n = Path(f).name
            if sp == "egvla2_continuous_v1": v1.append(np.array(h["actions"])[:, :2])
            elif n in PILOT: v2.append(np.array(h["actions_raw"])[:, :2])
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, L, title, col in ((axs[0], v1, "v1 수집 40건 — 스틱이 모서리·끝에 몰림", RED), (axs[1], v2, "파일럿 v2 수집 15건 — 입력이 고르게 분포", GRN)):
        a = np.concatenate(L); a = a[np.hypot(a[:, 0], a[:, 1]) > 1e-3]
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 1.0, 1.0, color=AMB, alpha=0.18, lw=0))
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ec="#94a3b8", ls="--", lw=1))
        jit = np.random.default_rng(0).normal(0, 0.012, a.shape)
        ax.scatter(-(a[:, 1] + jit[:, 1]), a[:, 0] + jit[:, 0], s=9, c=col, alpha=0.35, lw=0)
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal"); ax.grid(color="#1e293b")
        ax.set_title(f"{title}\n(비영 프레임 n={len(a)})", fontsize=11, color=col)
        ax.set_xlabel("좌우 (왼쪽 ← → 오른쪽)"); ax.set_ylabel("전후 (후진 ↓ ↑ 전진)")
        ax.text(0, 0, "v1 게이트\n(축 모두<0.5)\n발행 안 됨", ha="center", va="center", fontsize=8, color=AMB)
    fig.suptitle("스틱 평면 (lx, ly) 분포 — 실제 수집 데이터", fontsize=13, color="#fcd34d")
    fig.tight_layout(); fig.savefig(OUT / "ch76_stick_plane.jpg", dpi=110, pil_kwargs={"quality": 85}); plt.close(fig)


def pick_episode():
    best, bs = None, -1
    for n in PILOT:
        with h5py.File(D / n) as f:
            r = np.array(f["actions_raw"]); e = np.array(f["actions_exec"]); m = np.hypot(r[:, 0], r[:, 1])
            lv = len({round(float(x), 2) for x in np.hypot(e[:, 0], e[:, 1]) if x > 1e-3})
            sc = m.std() + 0.2 * lv + 0.002 * len(r)
            if sc > bs: best, bs = n, sc
    return best


def fig_episode(name):
    d = load(name); n = len(d["raw"]); idx = np.linspace(0, n - 1, 8).astype(int)
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(3, 8, height_ratios=[1.35, 1.2, 2.3], hspace=0.28, wspace=0.05)
    for k, i in enumerate(idx):
        ax = fig.add_subplot(gs[0, k]); ax.imshow(d["img"][i]); ax.axis("off")
        if d["bbox"] is not None and d["bbox"][i, 3] > 0:
            cx, cy = d["bbox"][i, 0] * d["img"].shape[2], d["bbox"][i, 1] * d["img"].shape[1]
            ax.plot(cx, cy, "+", color=GRN, ms=16, mew=2.4)
        ax.set_title(f"t={i} ({d['t'][i] - d['t'][0]:.1f}s)", fontsize=8.5, color="#cbd5e1")
        ax2 = fig.add_subplot(gs[1, k]); hud(ax2, d["raw"][i, 0], d["raw"][i, 1], d["raw"][i, 2])
        mag = float(np.hypot(d["exec_"][i, 0], d["exec_"][i, 1]))
        lvl = LEVEL_NAME.get(round(mag, 1), "정지" if mag < 1e-3 else f"{mag:.2f}")
        ax2.text(0, -1.45, f"raw ({d['raw'][i,0]:+.2f},{d['raw'][i,1]:+.2f})\n속력 단계: {lvl}", ha="center", va="top", fontsize=7.5, color=AMB)
    ax = fig.add_subplot(gs[2, :]); ts = d["t"] - d["t"][0]
    ax.plot(ts, d["raw"][:, 0], color=BLU, lw=1.8, label="raw lx (전후)"); ax.plot(ts, d["raw"][:, 1], color=PUR, lw=1.8, label="raw ly (좌우)")
    ax.step(ts, np.hypot(d["exec_"][:, 0], d["exec_"][:, 1]), color=AMB, lw=1.6, where="post", label="exec 속력 단계(0.4/0.7/1.0)")
    ax.axhspan(-0.5, 0.5, color=AMB, alpha=0.10)
    for i in idx: ax.axvline(ts[i], color="#475569", lw=0.8, ls=":")
    ax.set_xlabel("시간 (s)"); ax.set_ylabel("값"); ax.grid(color="#1e293b"); ax.legend(loc="lower left", fontsize=8, ncol=3)
    fig.suptitle(f"파일럿 v2 실제 에피소드 {name[:-3]} — 프레임·스틱(HUD)·속력 단계 (초록 + = F2가 찾은 바구니 중심, 음영 = v1 게이트가 지우던 |값|<0.5 구간)", fontsize=10.5, color="#fcd34d")
    fig.savefig(OUT / "ch76_pilot_episode.jpg", dpi=105, pil_kwargs={"quality": 85}, bbox_inches="tight"); plt.close(fig)


def fig_weak(names):
    picks = []
    for n in names:
        d = load(n); m = np.hypot(d["raw"][:, 0], d["raw"][:, 1])
        c = np.where((m > 0.15) & (m < 0.45))[0]
        if len(c): picks.append((d, int(c[len(c) // 2])))
        if len(picks) == 4: break
    fig, axs = plt.subplots(2, 4, figsize=(14, 4.6), gridspec_kw=dict(height_ratios=[2.6, 1.2], hspace=0.02, wspace=0.04))
    for k, (d, i) in enumerate(picks):
        axs[0, k].imshow(d["img"][i]); axs[0, k].axis("off")
        axs[0, k].set_title(f"{d['name'][8:-3]} · t={i}", fontsize=9, color="#cbd5e1")
        hud(axs[1, k], d["raw"][i, 0], d["raw"][i, 1], d["raw"][i, 2], col=GRN)
        axs[1, k].text(0, -1.45, f"raw ({d['raw'][i,0]:+.2f}, {d['raw'][i,1]:+.2f})  |v|={np.hypot(d['raw'][i,0], d['raw'][i,1]):.2f}", ha="center", va="top", fontsize=8.5, color=GRN)
    fig.suptitle("약한 스틱 입력이 실제로 기록된 프레임 — v1 게이트(축 모두 <0.5)였다면 발행·기록되지 않았을 입력", fontsize=11.5, color="#fcd34d")
    fig.savefig(OUT / "ch76_weak_stick_frames.jpg", dpi=105, pil_kwargs={"quality": 85}, bbox_inches="tight"); plt.close(fig)


def fig_convert(name):
    d = load(name); m = np.hypot(d["raw"][:, 0], d["raw"][:, 1])
    i = int(np.argmin(np.abs(np.where((m > 0.4) & (np.abs(d["raw"][:, 1]) > 0.15), m, 9) - 0.65)))
    r, s, e = d["raw"][i], d["scaled"][i], d["exec_"][i]
    ang = np.degrees(np.arctan2(r[1], r[0])); mag = np.hypot(r[0], r[1])
    lvl_mag = round(float(np.hypot(e[0], e[1])), 1)
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    axs[0].imshow(d["img"][i]); axs[0].axis("off"); axs[0].set_title(f"{name[:-3]} · 프레임 {i}", fontsize=10, color="#cbd5e1")
    ax = axs[1]; ax.axis("off")
    rows = [("① 스틱 raw (actions_raw)", f"lx={r[0]:+.2f}  ly={r[1]:+.2f}  az={r[2]:+.2f}", BLU),
            ("② 변환 A (scaled, 원형게이트)", f"lx={s[0]:+.2f}  ly={s[1]:+.2f}  (|v|={np.hypot(s[0], s[1]):.2f} ≤ 1)", GRN),
            ("③ 변환 B (각도 + GO)", f"각도 {ang:+.0f}°  크기 {mag:.2f} → GO", PUR),
            ("④ 변환 C (각도 + 속력 단계)", f"각도 {ang:+.0f}°  속력 단계 {LEVEL_NAME.get(lvl_mag, lvl_mag)} ({lvl_mag})", AMB),
            ("⑤ 실제 구동 (actions_exec)", f"lx={e[0]:+.2f}  ly={e[1]:+.2f}  az={e[2]:+.2f}", RED)]
    y = 0.94
    for t, v, c in rows:
        ax.text(0, y, t, fontsize=10.5, color=c, fontweight="bold", va="top"); ax.text(0.03, y - 0.075, v, fontsize=10, color="#e2e8f0", va="top"); y -= 0.19
    fig.suptitle("한 프레임에서 raw → A/B/C 라벨 → 실제 구동이 어떻게 만들어지는가 (실제 데이터)", fontsize=12, color="#fcd34d")
    fig.savefig(OUT / "ch76_conversion_example.jpg", dpi=105, pil_kwargs={"quality": 85}, bbox_inches="tight"); plt.close(fig)


def fig_prediction():
    import torch
    import train_egvla2_fm_head as T
    T.EPOCHS = 60
    eps = T.load_episodes()
    te_i = 3; tr_i = [i for i in range(len(eps)) if i != te_i]
    tr, te = T.to_t(eps, tr_i), T.to_t(eps, [te_i])
    out = {}
    for name in ("A_mlp_fm", "C_pi0_adaln"):
        m, _ = T.train_one(T.ARCH[name], tr, te, 0, epochs=60)
        S = T.sample(m, te[0], te[1], k=16).cpu().numpy()      # (N,K,H,3)
        out[name] = S
    gt = te[2].cpu().numpy()[:, 0]                              # 첫 스텝 정답 (exec)
    fig, axs = plt.subplots(2, 2, figsize=(13, 6.4), sharex=True)
    for r_, (name, col) in enumerate((("A_mlp_fm", GRN), ("C_pi0_adaln", BLU))):
        for c_, (ax_i, lab) in enumerate(((0, "lx 전후"), (1, "ly 좌우"))):
            ax = axs[r_, c_]; S = out[name][:, :, 0, ax_i]
            ax.plot(gt[:, ax_i], color=AMB, lw=2.2, label="정답 (exec)")
            ax.fill_between(range(len(S)), S.min(1), S.max(1), color=col, alpha=0.18, label="샘플 16개 범위")
            ax.plot(S.mean(1), color=col, lw=1.8, label="샘플 평균")
            ax.set_title(f"{'A MLP-FM' if r_ == 0 else 'C π0형'} · {lab}", fontsize=10.5, color=col); ax.grid(color="#1e293b")
            if r_ == 1: ax.set_xlabel("프레임 (hold-out 에피소드, 학습에 쓰지 않음)")
            if c_ == 0: ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"hold-out 에피소드 {eps[te_i]['name'][8:-3]} — FM이 생성한 첫 스텝 액션 vs 정답 (60 epoch, seed 0)", fontsize=11.5, color="#fcd34d")
    fig.savefig(OUT / "ch76_fm_prediction.jpg", dpi=105, pil_kwargs={"quality": 85}, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["sticks", "episode", "weak", "convert", "pred"]
    ep = pick_episode(); print("대표 에피소드:", ep)
    if "sticks" in which: fig_sticks()
    if "episode" in which: fig_episode(ep)
    if "weak" in which: fig_weak(PILOT)
    if "convert" in which: fig_convert(ep)
    if "pred" in which: fig_prediction()
    print("완료", sorted(p.name for p in OUT.glob("*.jpg")))
