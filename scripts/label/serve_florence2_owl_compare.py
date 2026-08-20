#!/usr/bin/env python3
"""Florence-2 vs OWL-v2 인터랙티브 비교 도구 (2026-08-20).

2026-08-07 100세션(1087프레임) 배치에서 프레임을 골라 OWL-v2 정답과
Florence-2 결과를 나란히 비교한다. 두 탭:

  /            브라우저 — HIT/WRONG/MISS로 필터, prev/next로 미리 계산된
               raw 검출(OD/DENSE × beam3/beam5)을 즉시 확인 (모델 재실행 없음)
  /live        실시간 테스트 — 프레임 선택 + 태스크(OD/DENSE/CAPTION_TO_PHRASE_GROUNDING
               with 커스텀 phrase) + beam 지정 → Florence-2를 그 자리에서 돌려 결과 확인.
               아직 안 해본 "명시적 phrase로 <CAPTION_TO_PHRASE_GROUNDING> 테스트"를
               여기서 바로 시도할 수 있다.

주의: 파이프라인에 끼워넣지 않는다. 연구용 로컬 비교 도구.

실행: .venv/bin/python3 scripts/label/serve_florence2_owl_compare.py
접속: http://localhost:7794
"""
import base64
import glob
import io
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from flask import Flask, jsonify, request, Response
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent.parent
H5_DIR = "/home/minum/MoNaVLA/inference_sessions_recv/20260807/h5"
RAW_PATH = ROOT / "docs/v5/detector/florence2_grounding_0807_raw.json"
PORT = 7795
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "microsoft/Florence-2-base"
HIT_TOL = 0.05

KEYWORDS = ["hamper", "basket", "trash can", "trash bin", "waste container",
            "waste bin", "wastebasket", "bin", "container"]
KW_RE = [re.compile(r"\b" + re.escape(k) + r"\b") for k in KEYWORDS]

app = Flask(__name__)
_model = {"model": None, "proc": None}


def get_model():
    if _model["model"] is None:
        print("[Florence-2] 로딩...", flush=True)
        from transformers import AutoModelForCausalLM, AutoProcessor
        _model["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
        _model["proc"] = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("[Florence-2] 로딩 완료", flush=True)
    return _model["model"], _model["proc"]


def pick_kw(dets):
    hits = [d for d in dets if any(rx.search(d["label"]) for rx in KW_RE)]
    if not hits:
        return None
    return max(hits, key=lambda d: d["area"])


def load_index():
    """iter_frames 순서를 재현해 (path, frame_idx) ↔ raw row index 매핑을 만든다."""
    rows = json.loads(RAW_PATH.read_text())
    frames = []
    i = 0
    for path in sorted(glob.glob(f"{H5_DIR}/*.h5")):
        with h5py.File(path, "r") as hf:
            n = hf["observations/images"].shape[0]
        for fi in range(n):
            frames.append((path, fi, rows[i]))
            i += 1
    assert i == len(rows)
    return frames


FRAMES = load_index()


def categorize(row):
    if not row["owl_success"]:
        return "owl_fail"
    p = pick_kw(row["DENSE_b5"]) or pick_kw(row["OD_b5"])
    if p is None:
        return "miss"
    return "hit" if abs(p["cx"] - row["gt_cx"]) <= HIT_TOL else "wrong"


def get_image(path, fi):
    with h5py.File(path, "r") as hf:
        im = hf["observations/images"][fi]
    return Image.fromarray(im[:, :, ::-1].astype(np.uint8)).convert("RGB")


def draw_overlay(img, gt_cx, picks):
    """picks: list of (label, cx, color)"""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    draw.line([(gt_cx * W, 0), (gt_cx * W, H)], fill=(34, 197, 94), width=3)
    for label, cx, color in picks:
        if cx is None:
            continue
        x = cx * W
        draw.line([(x, 0), (x, H)], fill=color, width=3)
    return img


def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<title>Florence-2 vs OWL-v2 비교</title>
<style>
body{background:#0a0f1a;color:#e2e8f0;font-family:'Segoe UI',sans-serif;padding:20px;max-width:1100px;margin:auto}
h1{font-size:1.2rem} a{color:#38bdf8;margin-right:12px}
.card{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px;margin-top:14px}
img{max-width:100%;border-radius:8px}
button,select,input{background:#1f2937;color:#e2e8f0;border:1px solid #374151;border-radius:6px;padding:6px 10px;margin:4px 4px 4px 0}
button:hover{background:#374151;cursor:pointer}
.tag{display:inline-block;padding:2px 8px;border-radius:6px;font-size:0.75rem;margin-right:6px}
.hit{background:#14532d;color:#4ade80} .wrong{background:#78350f;color:#fbbf24}
.miss{background:#7f1d1d;color:#f87171} .owl_fail{background:#374151;color:#9ca3af}
.legend span{margin-right:14px;font-size:0.82rem}
pre{background:#0a0f1a;padding:8px;border-radius:6px;overflow-x:auto;font-size:0.78rem}
</style></head><body>
<h1>Florence-2 vs OWL-v2 그라운딩 비교 (2026-08-07 100세션 배치)</h1>
<div><a href="/">브라우저(사전계산)</a><a href="/live">실시간 테스트</a></div>
{body}
</body></html>
"""


@app.route("/")
def browse():
    idx = int(request.args.get("idx", 0))
    cat_filter = request.args.get("cat", "all")
    idx = max(0, min(idx, len(FRAMES) - 1))

    order = list(range(len(FRAMES)))
    if cat_filter != "all":
        order = [i for i in order if categorize(FRAMES[i][2]) == cat_filter]
        if not order:
            order = list(range(len(FRAMES)))
    if idx not in order:
        idx = order[0]
    pos = order.index(idx)
    prev_idx = order[pos - 1] if pos > 0 else order[pos]
    next_idx = order[pos + 1] if pos < len(order) - 1 else order[pos]

    path, fi, row = FRAMES[idx]
    cat = categorize(row)
    img = get_image(path, fi)

    dense5 = pick_kw(row["DENSE_b5"])
    od5 = pick_kw(row["OD_b5"])
    picks = []
    if dense5:
        picks.append((dense5["label"], dense5["cx"], (239, 68, 68)))
    if od5 and (not dense5 or abs(od5["cx"] - dense5["cx"]) > 0.01):
        picks.append((od5["label"], od5["cx"], (250, 204, 21)))
    overlay = draw_overlay(img, row["gt_cx"], picks)

    def det_table(name, dets):
        if not dets:
            return f"<b>{name}</b>: (검출 없음)"
        rows = "".join(f"<tr><td>{d['label']}</td><td>{d['cx']:.3f}</td><td>{d['area']:.4f}</td></tr>" for d in dets)
        return f"<b>{name}</b><table><tr><th>label</th><th>cx</th><th>area</th></tr>{rows}</table>"

    body = f"""
    <div class="legend">
      <span><a href="/?idx=0&cat=all">전체</a></span>
      <span><a href="/?idx=0&cat=hit">HIT만</a></span>
      <span><a href="/?idx=0&cat=wrong">WRONG만</a></span>
      <span><a href="/?idx=0&cat=miss">MISS만</a></span>
      <span><a href="/?idx=0&cat=owl_fail">OWL 실패만</a></span>
    </div>
    <div class="card">
      <div>프레임 {pos+1}/{len(order)} (전체 인덱스 {idx}) &nbsp; <span class="tag {cat}">{cat.upper()}</span>
      &nbsp; {Path(path).stem} #frame{fi}</div>
      <form style="display:inline">
        <input type="hidden" name="cat" value="{cat_filter}">
        <button formaction="/?idx={prev_idx}&cat={cat_filter}">◀ 이전</button>
        <button formaction="/?idx={next_idx}&cat={cat_filter}">다음 ▶</button>
      </form>
      <div style="margin-top:10px"><img src="{img_to_b64(overlay)}"></div>
      <div style="margin-top:8px;font-size:0.82rem;color:#94a3b8">
        초록=OWL 정답(cx={row['gt_cx']:.3f}, success={bool(row['owl_success'])}) ·
        빨강=DENSE(beam5) 선택 · 노랑=OD(beam5) 선택(DENSE와 다를 때만 표시)
      </div>
      <div style="display:flex;gap:16px;margin-top:12px;font-size:0.78rem">
        <div style="flex:1">{det_table("OD(beam3)", row["OD_b3"])}</div>
        <div style="flex:1">{det_table("DENSE(beam3)", row["DENSE_b3"])}</div>
      </div>
      <div style="display:flex;gap:16px;margin-top:8px;font-size:0.78rem">
        <div style="flex:1">{det_table("OD(beam5)", row["OD_b5"])}</div>
        <div style="flex:1">{det_table("DENSE(beam5)", row["DENSE_b5"])}</div>
      </div>
    </div>
    <style>table{{width:100%;border-collapse:collapse}} td,th{{padding:3px 6px;border-bottom:1px solid #1f2937;text-align:left}}</style>
    """
    return PAGE.replace("{body}", body)


LIVE_PAGE = """
<div class="card">
  <form id="f">
    <label>프레임 인덱스 (0~{max_idx}): <input type="number" id="idx" value="0" min="0" max="{max_idx}"></label>
    <button type="button" onclick="rnd()">랜덤</button><br>
    <label>태스크:
      <select id="task">
        <option value="&lt;OD&gt;">&lt;OD&gt;</option>
        <option value="&lt;DENSE_REGION_CAPTION&gt;">&lt;DENSE_REGION_CAPTION&gt;</option>
        <option value="&lt;CAPTION_TO_PHRASE_GROUNDING&gt;">&lt;CAPTION_TO_PHRASE_GROUNDING&gt; (phrase 지정, 아직 안 해본 방식)</option>
        <option value="&lt;OPEN_VOCABULARY_DETECTION&gt;">&lt;OPEN_VOCABULARY_DETECTION&gt;</option>
      </select>
    </label>
    <label>phrase(CAPTION_TO_PHRASE_GROUNDING/OPEN_VOCAB 전용): <input type="text" id="phrase" value="gray basket" size="20"></label><br>
    <label>beam: <input type="number" id="beam" value="3" min="1" max="8"></label>
    <button type="button" onclick="run()">실행</button>
    <span id="status" style="color:#94a3b8"></span>
  </form>
  <div id="result" style="margin-top:12px"></div>
</div>
<script>
function rnd(){ document.getElementById('idx').value = Math.floor(Math.random()*__MAX_IDX__); }
async function run(){
  const idx = document.getElementById('idx').value;
  const task = document.getElementById('task').value;
  const phrase = document.getElementById('phrase').value;
  const beam = document.getElementById('beam').value;
  document.getElementById('status').innerText = '실행 중... (첫 호출은 모델 로딩으로 느릴 수 있음)';
  const res = await fetch(`/api/run?idx=${idx}&task=${encodeURIComponent(task)}&phrase=${encodeURIComponent(phrase)}&beam=${beam}`);
  const data = await res.json();
  document.getElementById('status').innerText = data.elapsed_s ? `완료 (${data.elapsed_s.toFixed(2)}s)` : '';
  let html = `<img src="${data.image}"><div style="font-size:0.82rem;color:#94a3b8;margin-top:6px">
    초록=OWL 정답(cx=${data.gt_cx.toFixed(3)}) · 빨강=이번 실행 결과</div>`;
  html += '<table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:0.8rem">';
  html += '<tr><th>label</th><th>cx</th><th>area</th></tr>';
  for (const d of data.dets){
    html += `<tr><td>${d.label}</td><td>${d.cx.toFixed(3)}</td><td>${d.area.toFixed(4)}</td></tr>`;
  }
  html += '</table>';
  document.getElementById('result').innerHTML = html;
}
</script>
"""


@app.route("/live")
def live():
    max_idx = str(len(FRAMES) - 1)
    live_body = LIVE_PAGE.replace("{max_idx}", max_idx).replace("__MAX_IDX__", max_idx)
    return PAGE.replace("{body}", live_body)


@app.route("/api/run")
def api_run():
    import time
    idx = int(request.args.get("idx", 0))
    task = request.args.get("task", "<OD>")
    phrase = request.args.get("phrase", "gray basket")
    beam = int(request.args.get("beam", 3))
    idx = max(0, min(idx, len(FRAMES) - 1))
    path, fi, row = FRAMES[idx]
    img = get_image(path, fi)

    model, proc = get_model()
    t0 = time.time()
    text = f"{task}{phrase}" if task == "<CAPTION_TO_PHRASE_GROUNDING>" or task == "<OPEN_VOCABULARY_DETECTION>" else task
    W, H = img.width, img.height
    inp = proc(text=text, images=img, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids=inp["input_ids"].to(DEV),
            pixel_values=inp["pixel_values"].to(DEV, torch.float16),
            max_new_tokens=256, num_beams=beam, do_sample=False)
    txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(txt, task=task, image_size=(W, H))[task]
    boxes = parsed.get("bboxes", []) or []
    labels = (parsed.get("labels") or parsed.get("bboxes_labels") or [])
    dets = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        lb = str(labels[i]).lower().strip() if i < len(labels) else phrase
        dets.append(dict(label=lb, cx=(x1 + x2) / 2 / W, area=(x2 - x1) * (y2 - y1) / (W * H)))
    elapsed = time.time() - t0

    picks = [(d["label"], d["cx"], (239, 68, 68)) for d in dets]
    overlay = draw_overlay(img, row["gt_cx"], picks)
    return jsonify(image=img_to_b64(overlay), gt_cx=row["gt_cx"], dets=dets, elapsed_s=elapsed)


if __name__ == "__main__":
    print(f"프레임 {len(FRAMES)}개 로드 완료. 브라우저 → http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
