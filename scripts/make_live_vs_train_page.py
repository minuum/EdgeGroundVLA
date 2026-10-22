#!/usr/bin/env python3
"""학습 프레임 vs 실기 프레임 사진 비교 페이지 생성: docs/v5/egvla2_fm/live_vs_train.html (+ live_vs_train_photos/*.jpg).
섹션: (1) 같은 구도 비교(학습 vs 실기), (2) 바구니 확대 비교(색), (3) 실기 이탈 타임라인(세션 210745), (4) 방해물이 보이는 실기 프레임.
실행: .venv_lerobot/bin/python3 scripts/make_live_vs_train_page.py
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import train_egvla2_s1 as S1  # noqa: E402
T = S1.T
OUT = ROOT / "docs/v5/egvla2_fm"; PH = OUT / "live_vs_train_photos"; PH.mkdir(exist_ok=True, parents=True)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", 26)
W, H = 640, 360


def card(arr, label, bbox=None, color=(255, 80, 80)):
    im = Image.fromarray(np.asarray(arr)).resize((W, H)); d = ImageDraw.Draw(im)
    if bbox is not None and bbox[3] > 0.5:
        cx, cy, area = bbox[0] * W, bbox[1] * H, bbox[2]
        r = max(8, int(np.sqrt(area * W * H) / 2))                 # 면적(이미지 대비 비율)에서 환산한 정사각 근사
        d.rectangle([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
        d.line([cx - 10, cy, cx + 10, cy], fill=color, width=2); d.line([cx, cy - 10, cx, cy + 10], fill=color, width=2)
    d.rectangle([0, 0, W, 38], fill=(0, 0, 0)); d.text((8, 4), label, fill=(255, 255, 255), font=FONT)
    return im


def save(im, name):
    p = PH / name; im.convert("RGB").save(p, "JPEG", quality=88); return f"live_vs_train_photos/{name}"


def main():
    eps = S1.load_main()
    pick = {}
    for cell in ("weak_right|straight", "strong_left|left_curve", "center|straight"):
        pick[cell] = next(e for e in eps if e["cell"] == cell)
    sec1 = []
    for cell, e in pick.items():
        n = len(e["bbox"])
        with h5py.File(T.DATA / e["name"], "r") as f:
            for frac in (0.0, 0.5, 0.95):
                t = min(int(frac * (n - 1)), n - 1); b = e["bbox"][t][8:12]
                sec1.append((cell, frac, save(card(np.ascontiguousarray(f["images"][t][:, :, ::-1]), f"학습 {cell} {int(frac*100)}% cx{b[0]:.2f} 면적{b[2]:.2f}", b, (80, 220, 80)), f"train_{cell.replace('|','_')}_{int(frac*100)}.jpg")))
    sid = "20261022_210745"
    rep = json.load(open(ROOT / f"ROS_action/dryrun_closedloop_20261022/reports/session_{sid}.json"))["history"]
    with h5py.File(ROOT / f"ROS_action/dryrun_closedloop_20261022/h5/session_{sid}.h5", "r") as f:
        imgs = f["observations/images"]; nimg = imgs.shape[0]
        steps = [0, 1, 3, 6, 10, 14, 17, 18, 22, 27]
        sec3 = []
        for s in steps:
            if s >= nimg or s >= len(rep):
                continue
            r = rep[s]; b = r["bbox"]; bb = np.array([b["cx"], b["cy"], b["area"], 1.0 if b["has_bbox"] else 0.0]) if b else None
            a = r["action"]; lab = f"실기 스텝{s} {r['predicted_label'].split(':')[-1][:10]}  cx{b['cx']:.2f} 면적{b['area']:.3f}  az{a[2]:+.2f} ly{a[1]:+.2f}"
            sec3.append((s, save(card(imgs[s], lab, bb, (255, 80, 80)), f"live_{sid}_{s}.jpg")))
    # 방해물이 보이는 프레임 + 바구니 크롭
    with h5py.File(ROOT / "ROS_action/dryrun_closedloop_20261022/h5/session_20261022_210947.h5", "r") as f:
        im = f["observations/images"]; gb = f["grounding/bbox"][:]; t = 14 if len(gb) > 14 else 0
        live_full = save(card(im[t], f"실기 210947 스텝{t}", gb[t], (255, 80, 80)), "live_210947_full.jpg")
        def crop(arr, b, size=0.14):
            Hh, Ww, _ = arr.shape; cx, cy = int(b[0] * Ww), int(b[1] * Hh); r = int(size * Ww)
            return np.asarray(arr)[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        lc = Image.fromarray(crop(im[t], gb[t])).resize((360, 360))
    e = pick["weak_right|straight"]; t0 = len(e["bbox"]) // 2
    with h5py.File(T.DATA / e["name"], "r") as f:
        tc = Image.fromarray(np.ascontiguousarray(crop(f["images"][t0], e["bbox"][t0][8:12])[:, :, ::-1])).resize((360, 360))
    both = Image.new("RGB", (740, 400), (20, 20, 20)); both.paste(tc, (10, 30)); both.paste(lc, (380, 30)); d = ImageDraw.Draw(both)
    d.text((10, 0), "학습 바구니(BGR→RGB 변환 후)", fill=(255, 255, 255), font=FONT); d.text((380, 0), "실기 바구니", fill=(255, 255, 255), font=FONT)
    crops = save(both, "basket_crops.jpg")
    st = json.load(open(OUT / "live_vs_train.json"))
    html = ['<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Live vs Train photos</title>',
            '<style>body{margin:0;background:#0b1220;color:#e2e8f0;font-family:-apple-system,"Noto Sans KR",sans-serif;line-height:1.6}.w{max-width:1100px;margin:0 auto;padding:24px 16px 60px}h1{font-size:1.5rem}h2{margin-top:34px;border-bottom:2px solid #334155;padding-bottom:6px}',
            '.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px}.g img{width:100%;border-radius:6px;border:1px solid #334155}.n{color:#94a3b8;font-size:.86rem}a{color:#60a5fa}table{border-collapse:collapse;font-size:.9rem}td,th{border:1px solid #334155;padding:6px 10px}</style></head><body><div class="w">',
            '<h1>학습 프레임 vs 실기 프레임 — 실제 사진 비교 (학습은 H5의 BGR을 RGB로 변환해 표시)</h1><div class="n">2026-10-23 · 분석: docs/plans/analysis_20261023_train_serve_gap.md · 빨간 상자/초록 상자 = Florence-2 bbox(면적에서 환산한 근사 크기)</div>',
            '<h2>1. 학습 프레임(BGR→RGB 변환) — 실기는 아래 3절</h2><div class="g">']
    for cell, frac, p in sec1:
        html.append(f'<div><img src="{p}" loading="lazy"></div>')
    html += ['</div><h2>2. 바구니 확대 비교</h2>', f'<img src="{crops}" style="max-width:740px;width:100%;border-radius:6px"><div class="n">바구니 중심 패치 평균 RGB(학습은 BGR→RGB 변환 후): 학습 (112,117,122) R−B=−10.6 / 실기 (107,110,111) R−B=−4.2 → 색 계열 같음</div>',
             '<h2>3. 실기 이탈 타임라인 — 세션 210745 (타깃 오른쪽 → 회전 → 직진 중 cx가 화면 밖으로)</h2><div class="g">']
    for s, p in sec3:
        html.append(f'<div><img src="{p}" loading="lazy"></div>')
    html += ['</div><h2>4. 방해물이 보이는 실기 프레임(오른쪽 흰 패널·책상)</h2>', f'<img src="{live_full}" style="max-width:720px;width:100%;border-radius:6px">',
             '<h2>5. 수치 요약</h2><table><tr><th></th><th>학습</th><th>실기</th></tr>',
             f'<tr><td>특징 최근접 코사인 평균/p10</td><td>{st["nn_cos_train_heldout_episode"]["mean"]:.4f} / {st["nn_cos_train_heldout_episode"]["p10"]:.4f}</td><td>{st["nn_cos_live"]["mean"]:.4f} / {st["nn_cos_live"]["p10"]:.4f}</td></tr>',
             f'<tr><td>bbox 검출률</td><td>{st["bbox_train"]["has_rate"]*100:.1f}%</td><td>{st["bbox_live"]["has_rate"]*100:.1f}%</td></tr>',
             f'<tr><td>bbox 면적 중앙값</td><td>{st["bbox_train"]["area_median"]:.3f}</td><td>{st["bbox_live"]["area_median"]:.3f}</td></tr></table></div></body></html>']
    (OUT / "live_vs_train.html").write_text("\n".join(html))
    print("ok", len(sec1), len(sec3))


if __name__ == "__main__":
    main()
