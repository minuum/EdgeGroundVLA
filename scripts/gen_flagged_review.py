#!/usr/bin/env python3
"""검증 경고가 난 에피소드를 프레임 단위로 볼 수 있는 HTML 생성. 출력: docs/v5/egvla2_fm/review/flagged.html (+ jpg). 실행: .venv/bin/python3 -I scripts/gen_flagged_review.py"""
import html
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/v5/egvla2_fm/review"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "ROS_action/mobile_vla_dataset_egvla2"
CN = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
V = json.load(open(ROOT / "docs/v5/egvla2_fm/validation_batch.json"))["files"]
flag = [r["file"] for r in V if r["warnings"] or r["problems"]]
flag.sort(key=lambda n: n != "episode_20261014_170756_weak_right_straight.h5")  # 문제의 18프레임 건을 맨 앞에
secs = []
for fn in flag:
    stem = fn[:-3]
    with h5py.File(DATA / fn, "r") as h:
        im = h["images"]; raw = np.array(h["actions_raw"]); ex = np.array(h["actions_exec"]); cl = np.array(h["action_classes"]); t = np.array(h["frame_timestamps"])
        n = len(raw); cards = []; dups = 0
        for i in range(n):
            f = im[i]; Image.fromarray(np.ascontiguousarray(f[:, :, ::-1])).resize((288, 162)).save(OUT / f"{stem}_{i:03d}.jpg", quality=72)
            d = float(np.abs(f[::16, ::16].astype(np.int16) - im[i - 1][::16, ::16].astype(np.int16)).mean()) if i else None
            dup = d is not None and d < 0.4; dups += dup
            dt = (t[i] - t[i - 1]) if i else 0
            col = "#f87171" if dup else "#1e293b"
            cards.append(f'<div class="c" style="border-color:{col}"><img loading="lazy" src="{stem}_{i:03d}.jpg"><div><b>#{i}</b> t={t[i] - t[0]:.2f}s Δ{dt:.2f}s'
                         f'{" <span class=r>직전과 동일</span>" if dup else ""}<br>픽셀차 {"-" if d is None else f"{d:.2f}"}<br>'
                         f'raw ({raw[i][0]:.2f},{raw[i][1]:.2f},{raw[i][2]:.2f})<br>exec ({ex[i][0]:.2f},{ex[i][1]:.2f},{ex[i][2]:.2f}) {CN[int(cl[i])]}</div></div>')
        a = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h.attrs.items() if k in ("cx_position", "cx_path", "drive_mode", "total_duration")}
    w = next(r for r in V if r["file"] == fn)["warnings"]
    secs.append(f'<h2>{html.escape(stem)}</h2><p>{n}프레임 · 동일 프레임 {dups}/{n - 1}쌍 ({dups / max(1, n - 1):.0%}) · {html.escape(str(a))}<br>경고: {html.escape(" / ".join(w))}</p><div class="g">{"".join(cards)}</div>')
HTML = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>검증 경고 에피소드 검토</title>'
        '<style>body{background:#0a0f1a;color:#e2e8f0;font:14px/1.5 sans-serif;margin:0;padding:16px}h1{font-size:1.2rem}h2{font-size:1rem;margin:28px 0 4px;color:#fcd34d}'
        '.g{display:flex;flex-wrap:wrap;gap:8px}.c{width:288px;background:#0a1220;border:2px solid;border-radius:8px;padding:4px;font-size:11px}.c img{width:100%;border-radius:4px}.r{color:#f87171;font-weight:800}'
        'p{font-size:12px;color:#94a3b8}</style><h1>검증 경고 에피소드 검토 (빨간 테두리 = 직전 프레임과 거의 동일)</h1>'
        '<p>동일 판정: 직전 프레임과 1/16 다운샘플 평균 절대 픽셀 차 &lt; 0.4 (validate_egvla2_batch.py와 같은 기준). 제외는 첫 번째 에피소드 1건뿐.</p>' + "".join(secs))
(OUT / "flagged.html").write_text(HTML, encoding="utf-8")
print(len(flag), "episodes ->", OUT / "flagged.html")
