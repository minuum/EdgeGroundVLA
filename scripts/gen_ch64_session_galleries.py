#!/usr/bin/env python3
"""CH64 카드용 실제 세션 이미지 갤러리 — 추상 수치를 실물 프레임으로 뒷받침.

각 갤러리는 실제 서빙 세션 H5의 원본 프레임(720x1280)에 그라운딩 결과를 오버레이한다.
오버레이 규칙(허구 없음): bbox는 [cx, cy, area, has_bbox]만 저장되므로 박스 대신
 - 초록 수직선 = 검출된 cx (헤드가 실제로 쓰는 조향 신호)
 - 초록 점 = (cx, cy), 점 크기 ∝ area
 - 빨강 "미검출" = has_bbox=False (fallback cx=0.5 사용됨)
출력: docs/v5/ch64_figs/gal_*.png
"""
import glob
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_kf = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
fm.fontManager.addfont(_kf)
plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                     "savefig.facecolor": "white", "figure.facecolor": "white"})

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/v5/ch64_figs"; OUT.mkdir(parents=True, exist_ok=True)
RECV = Path("/home/minum/MoNaVLA/inference_sessions_recv")
POS = {"trackA_strong_left": "강좌", "trackA_weak_left": "약좌",
       "trackF_center": "중앙", "trackA_weak_right": "약우",
       "trackA_strong_right": "강우"}
IDX = {Path(f).stem.replace("session_", ""): f
       for f in glob.glob(str(RECV / "2026*" / "h5" / "*.h5"))}


def load_log():
    df = pd.read_csv(RECV / "20260731" / "episode_log.csv")
    df.columns = [c.strip() for c in df.columns]
    d = df[df["경로"].isin(POS)].copy()
    d["pos"] = d["경로"].map(POS); d["ok"] = d["결과"] == "성공"
    return d


def sess(sid):
    """세션 열어서 (images, bbox, runtime_config) 반환."""
    with h5py.File(IDX[str(sid)], "r") as h:
        return (np.array(h["observations/images"]), np.array(h["grounding/bbox"]),
                json.loads(h.attrs["runtime_config"]))


def draw(ax, img, bb, caption, sub=""):
    ax.imshow(img)
    H, W = img.shape[:2]
    cx, cy, area, has = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    if has:
        ax.axvline(cx * W, color="#00e070", lw=2.0, alpha=0.95)
        ax.plot([cx * W], [cy * H], "o", ms=4 + 26 * np.sqrt(max(area, 0)),
                mfc="none", mec="#00e070", mew=2.2)
        ax.text(0.02, 0.03, f"cx={cx:.2f} area={area:.3f}", transform=ax.transAxes,
                fontsize=7, color="#00e070", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#04140a", ec="none", alpha=0.8))
    else:
        ax.axvline(0.5 * W, color="#ff4d4d", lw=2.0, ls="--", alpha=0.9)
        ax.text(0.02, 0.03, "미검출 → fallback cx=0.50", transform=ax.transAxes,
                fontsize=7, color="#ff6b6b", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#1a0505", ec="none", alpha=0.85))
    ax.set_title(caption, fontsize=8.2, pad=3,
                 color="#0a7d4a" if sub == "성공" else ("#b3261e" if sub == "실패" else "#333"))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#0a7d4a" if sub == "성공" else ("#b3261e" if sub == "실패" else "#bbb"))
        s.set_linewidth(2.2 if sub in ("성공", "실패") else 0.8)


def grid(items, path, suptitle, ncol=5):
    """items: [(img, bb, caption, sub)]"""
    n = len(items); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 2.15 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, it in zip(axes, items):
        draw(ax, *it)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12, y=1.0 + 0.015 * nrow, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / path); plt.close(fig)
    print("saved", path)


d = load_log()
K = "exp73_owl_trackF_v6_mlp_holdaware_seed0.pt"
d100 = d[(d["체크포인트"] == K) & (pd.to_datetime(d["날짜"]) >= pd.Timestamp("2026-07-30 19:25"))
         & ~d["#"].between(200, 209) & (d["#"] != 230)]

# ---------- G1 (64-18): 100세트 위치별 성공 vs 실패 실물 ----------
items = []
for p in ["강좌", "약좌", "중앙", "약우", "강우"]:
    g = d100[d100.pos == p]
    for want in (True, False):
        cand = g[g.ok == want]
        if not len(cand):
            continue
        r = cand.iloc[len(cand) // 2]
        imgs, bb, _ = sess(r["session_id"])
        k = len(imgs) - 1
        items.append((imgs[k], bb[k], f"{p} · {'성공' if want else '실패'} "
                                      f"(gnd {r['gnd%']:.0f}%, {int(r['steps'])}스텝)",
                      "성공" if want else "실패"))
grid(items, "gal_64_18_positions.png",
     "64-18 갤러리 — 100회 스크리닝 위치별 실물 최종 프레임 (초록=검출 cx, 빨강점선=미검출)")

# ---------- G2 (64-15/64-20): 그라운딩 실패 실물 ----------
gap = json.load(open(ROOT / "docs/v5/grounding_analysis/jetson_local_gap.json"))
mi = gap["frames"]["miss"]
T = 0.20


def pick(lo, hi, k, tag):
    """구간별 대표 표본 — 극단만 뽑아 오해를 만들지 않도록 분포 비중대로 고른다."""
    sel = sorted([r for r in mi if lo <= r["local_score"] < hi],
                 key=lambda r: r["local_score"])
    if not sel:
        return []
    step = max(1, len(sel) // k)
    out = []
    for r in sel[::step][:k]:
        imgs, bb, _ = sess(r["sid"])
        out.append((imgs[r["fi"]], bb[r["fi"]],
                    f"{r['pos']} · {tag}\n로컬 score {r['local_score']:.3f}", ""))
    return out


items = (pick(0.10, 0.20, 5, "경계밴드(지배구간 54.8%)")
         + pick(0.20, 1.01, 3, "젯슨만 놓침(gap 21.3%)")
         + pick(0.00, 0.05, 2, "타겟 부재 추정(2.5%)"))
grid(items, "gal_64_20_misses.png",
     "64-20 갤러리 — 미검출 프레임을 로컬 score 구간별 대표 표본으로 (지배구간은 0.10~0.20 경계밴드)")

# ---------- G3 (64-19): arm별 실물 — A는 H5 없음을 시각적으로 명시 ----------
B = d[(d["체크포인트"].isna()) & (d["날짜"].astype(str) >= "2026-07-23 18")
      & (d["날짜"].astype(str) < "2026-07-24")]
items = []
for r in list(B.itertuples())[:5]:
    sid = str(r.session_id)
    if sid not in IDX:
        continue
    imgs, bb, rc = sess(sid)
    k = len(imgs) - 1
    items.append((imgs[k], bb[k], f"B arm · {r.pos} · {'성공' if r.ok else '실패'}\n"
                                  f"thr={rc['owlv2_thresh']} head={rc['head']}",
                  "성공" if r.ok else "실패"))
for r in list(d100.itertuples())[:5]:
    imgs, bb, rc = sess(str(r.session_id))
    k = len(imgs) - 1
    items.append((imgs[k], bb[k], f"C arm · {r.pos} · {'성공' if r.ok else '실패'}\n"
                                  f"thr={rc['owlv2_thresh']} freg={rc.get('force_reground_on_miss')}",
                  "성공" if r.ok else "실패"))
grid(items, "gal_64_19_arms.png",
     "64-19 갤러리 — B arm(thr 0.25) vs C arm(thr 0.20+가드) 동일 체크포인트 실물 비교")

# ---------- G4 (64-9/이력): 배포 세대별 실물 ----------
items, seen = [], set()
for f in sorted(glob.glob(str(RECV / "2026*" / "h5" / "*.h5"))):
    try:
        with h5py.File(f, "r") as h:
            rc = h.attrs.get("runtime_config")
            if rc is None:
                continue
            c = json.loads(rc)
            key = (Path(c.get("checkpoint_path", "?")).name, c.get("head"))
            if key in seen:
                continue
            seen.add(key)
            imgs = np.array(h["observations/images"]); bb = np.array(h["grounding/bbox"])
    except Exception:
        continue
    k = len(imgs) - 1
    items.append((imgs[k], bb[k],
                  f"{Path(f).stem.replace('session_','')}\nhead={c.get('head')}\n"
                  f"{Path(c.get('checkpoint_path','?')).name[:34]}", ""))
grid(items[:10], "gal_64_19_deploy_history.png",
     "배포 세대별 실물 — head=transformer(exp71) → head=exp73_mlp 전환 이력", ncol=5)
