#!/usr/bin/env python3
"""Florence-2 vs OWL-v2 그라운딩 성공/실패 케이스 갤러리 (2026-08-19).

florence2_grounding_0807_raw.json(같은 100세션 1087프레임)의 raw 검출을 이용해
Florence-2 최선 조합(OD∪DENSE, beam5, DENSE 우선)이 OWL-v2 정답 대비:
  1. HIT  — 일치(|Δcx|<=0.05)
  2. WRONG— 뭔가 골랐지만 위치가 틀림
  3. MISS — 아예 못 골랐음(OWL은 성공)
세 카테고리에서 실제 프레임을 뽑아 OWL bbox(초록)·Florence-2 선택(빨강)을 그려
grid 이미지로 저장한다.

출력: docs/v5/ch64_figs/fig_florence2_case_gallery.png
"""
import glob
import json
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
H5_DIR = "/home/minum/MoNaVLA/inference_sessions_recv/20260807/h5"
RAW = ROOT / "docs/v5/detector/florence2_grounding_0807_raw.json"
OUT = ROOT / "docs/v5/ch64_figs/fig_florence2_case_gallery.png"
HIT_TOL = 0.05
N_PER_CAT = 8

KEYWORDS = ["hamper", "basket", "trash can", "trash bin", "waste container",
            "waste bin", "wastebasket", "bin", "container"]
KW_RE = [re.compile(r"\b" + re.escape(k) + r"\b") for k in KEYWORDS]


def pick(dets):
    hits = [d for d in dets if any(rx.search(d["label"]) for rx in KW_RE)]
    if not hits:
        return None
    return max(hits, key=lambda d: d["area"])


def best_pick(row):
    """DENSE_b5 우선, 없으면 OD_b5 폴백 — variants.py의 union_b5_DENSEfirst와 동일."""
    p = pick(row["DENSE_b5"])
    if p:
        return p
    return pick(row["OD_b5"])


def iter_paths():
    for path in sorted(glob.glob(f"{H5_DIR}/*.h5")):
        with h5py.File(path, "r") as hf:
            n = hf["observations/images"].shape[0]
            for i in range(n):
                yield path, i


def main():
    rows = json.loads(RAW.read_text())
    paths = list(iter_paths())
    assert len(paths) == len(rows), f"{len(paths)} vs {len(rows)}"

    hit, wrong, miss = [], [], []
    for (path, fi), row in zip(paths, rows):
        if not row["owl_success"]:
            continue
        p = best_pick(row)
        if p is None:
            miss.append((path, fi, row, None))
        elif abs(p["cx"] - row["gt_cx"]) <= HIT_TOL:
            hit.append((path, fi, row, p))
        else:
            wrong.append((path, fi, row, p))

    print(f"HIT={len(hit)}  WRONG={len(wrong)}  MISS={len(miss)}  (owl_success 대상 {len(hit)+len(wrong)+len(miss)})")

    rng = np.random.default_rng(0)
    def sample(lst, n):
        if len(lst) <= n:
            return lst
        idx = rng.choice(len(lst), size=n, replace=False)
        return [lst[i] for i in sorted(idx)]

    cats = [("HIT (일치)", sample(hit, N_PER_CAT), "#22c55e"),
            ("WRONG (위치 틀림)", sample(wrong, N_PER_CAT), "#f59e0b"),
            ("MISS (미검출)", sample(miss, N_PER_CAT), "#ef4444")]

    ncols = N_PER_CAT
    nrows = len(cats)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.7))

    h5_cache = {}
    for r, (title, items, color) in enumerate(cats):
        for c in range(ncols):
            ax = axes[r][c]
            ax.axis("off")
            if c >= len(items):
                continue
            path, fi, row, p = items[c]
            if path not in h5_cache:
                h5_cache[path] = h5py.File(path, "r")
            im = h5_cache[path]["observations/images"][fi]
            # 주의(2026-08-20): 0807 실기 세션은 이미 RGB 저장 — 반전하면 색이 뒤집힘(육안 확인).
            img_rgb = im
            H, W = img_rgb.shape[:2]
            ax.imshow(img_rgb)
            gt_cx_px = row["gt_cx"] * W
            ax.axvline(gt_cx_px, color="#22c55e", linewidth=2, alpha=0.9)
            if p is not None:
                p_cx_px = p["cx"] * W
                ax.axvline(p_cx_px, color="#ef4444", linewidth=2, alpha=0.9, linestyle="--")
                lbl = p["label"][:18]
                ax.set_title(f"F2: {lbl}", fontsize=6.5, color="#ef4444")
            else:
                ax.set_title("F2: (미검출)", fontsize=6.5, color="#999999")
        axes[r][0].set_ylabel(title, fontsize=10, color=color, rotation=0,
                              ha="right", va="center", labelpad=40)
        axes[r][0].axis("on")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for spine in axes[r][0].spines.values():
            spine.set_visible(False)

    fig.suptitle("Florence-2(OD∪DENSE beam5) vs OWL-v2 그라운딩 — 실제 프레임 비교\n"
                  "초록 실선=OWL-v2 정답(cx) · 빨강 점선=Florence-2 선택(cx)  (2026-08-07 100세션 배치)",
                  fontsize=11)
    fig.tight_layout(rect=[0.06, 0, 1, 0.93])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, facecolor="white")
    print(f"저장 → {OUT}")

    for f in h5_cache.values():
        f.close()


if __name__ == "__main__":
    main()
