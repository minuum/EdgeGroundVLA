#!/usr/bin/env python3
"""
PG2 fallback 프레임 — 한 페이지 그리드 + 인라인 라벨링 버튼 서버.
gen_fallback_multimodel_gallery.py가 만든 meta.json/thumb_*.jpg를 그대로 읽어서
카드마다 target_visible(O/X/부분) + coarse_position(L/C/R) 버튼을 붙여 서빙.
183/185 셀프라벨링과 동일한 라벨 스키마, but 한 번에 다 보이는 그리드 형태.

실행: python3 scripts/label/fallback_grid_labeler.py
접속: http://localhost:7791
"""
import json
from pathlib import Path
from flask import Flask, request, jsonify, Response

ROOT = Path("/home/minum/26CS/MoNaVLA")
GALLERY_DIR = ROOT / "docs" / "v5" / "fallback_multimodel_20260703"
META_PATH = GALLERY_DIR / "meta.json"
LABELS_PATH = GALLERY_DIR / "human_labels.json"
PORT = 7791

app = Flask(__name__)


def load_meta():
    return json.loads(META_PATH.read_text())


def load_labels():
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}


def save_labels(labels):
    LABELS_PATH.write_text(json.dumps(labels, indent=2, ensure_ascii=False))


@app.route("/")
def index():
    meta = load_meta()
    labels = load_labels()
    cards = []
    for m in meta:
        key = m["key"]
        lab = labels.get(key, {})
        vis = lab.get("target_visible")
        pos = lab.get("coarse_position")
        cards.append(f"""
  <div class="card" data-key="{key}" id="card-{key}">
    <img src="/thumb/{key}" loading="lazy">
    <div class="card-header">
      {key}<br>서버: cx={m['server_cx']:.2f} area={m['server_area']:.3f}
    </div>
    <div class="label-row">
      <div class="btn-group" data-field="target_visible">
        <button class="lbl-btn {'active' if vis=='visible' else ''}" data-val="visible">O 보임</button>
        <button class="lbl-btn {'active' if vis=='partial' else ''}" data-val="partial">△ 일부</button>
        <button class="lbl-btn {'active' if vis=='hidden' else ''}" data-val="hidden">X 안보임</button>
      </div>
      <div class="btn-group" data-field="coarse_position">
        <button class="lbl-btn pos {'active' if pos=='L' else ''}" data-val="L">L</button>
        <button class="lbl-btn pos {'active' if pos=='C' else ''}" data-val="C">C</button>
        <button class="lbl-btn pos {'active' if pos=='R' else ''}" data-val="R">R</button>
        <button class="lbl-btn pos {'active' if pos=='N/A' else ''}" data-val="N/A">N/A</button>
      </div>
    </div>
  </div>""")

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>Fallback 그리드 라벨링</title>
<style>
* {{ box-sizing: border-box; margin:0; padding:0; }}
body {{ background:#0a0f1a; color:#e2e8f0; font-family:'Segoe UI',sans-serif; padding:24px; }}
h1 {{ font-size:1.2rem; margin-bottom:4px; }}
.subtitle {{ color:#64748b; font-size:0.82rem; margin-bottom:16px; }}
#progress {{ position:sticky; top:0; background:#0a0f1a; padding:8px 0; z-index:10; font-size:0.85rem; color:#38bdf8; margin-bottom:12px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:10px; }}
.card {{ background:#0d1117; border:1px solid #1e293b; border-radius:8px; overflow:hidden; }}
.card.labeled {{ border-color:#22c55e; }}
.card img {{ width:100%; display:block; }}
.card-header {{ padding:4px 8px; font-size:0.62rem; color:#94a3b8; background:#111827; line-height:1.4; }}
.label-row {{ display:flex; justify-content:space-between; padding:5px; gap:4px; }}
.btn-group {{ display:flex; gap:3px; }}
.lbl-btn {{ font-size:0.68rem; padding:3px 6px; border-radius:4px; border:1px solid #334155; background:#1e293b; color:#94a3b8; cursor:pointer; }}
.lbl-btn.active {{ background:#2563eb; color:#fff; border-color:#2563eb; }}
.lbl-btn.pos.active {{ background:#f59e0b; border-color:#f59e0b; }}
</style></head>
<body>
<h1>PG2 Fallback 프레임 그리드 라벨링</h1>
<p class="subtitle">각 카드에 O(보임)/△(일부)/X(안보임) + L/C/R 버튼 클릭 → 즉시 자동저장. 새로고침해도 유지됩니다.</p>
<div id="progress">라벨링: <span id="done-count">0</span> / {len(meta)}</div>
<div class="grid">{''.join(cards)}</div>
<script>
function updateProgress() {{
  const done = document.querySelectorAll('.card.labeled').length;
  document.getElementById('done-count').textContent = done;
}}
document.querySelectorAll('.card').forEach(card => {{
  const key = card.dataset.key;
  card.querySelectorAll('.lbl-btn').forEach(btn => {{
    btn.addEventListener('click', async () => {{
      const field = btn.parentElement.dataset.field;
      const val = btn.dataset.val;
      card.querySelectorAll(`[data-field="${{field}}"] .lbl-btn`).forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      await fetch('/api/label/' + key, {{
        method: 'POST', headers: {{'Content-Type':'application/json'}},
        body: JSON.stringify({{field, val}})
      }});
      const vis = card.querySelector('[data-field="target_visible"] .active');
      const pos = card.querySelector('[data-field="coarse_position"] .active');
      if (vis && pos) card.classList.add('labeled');
      updateProgress();
    }});
  }});
  const vis = card.querySelector('[data-field="target_visible"] .active');
  const pos = card.querySelector('[data-field="coarse_position"] .active');
  if (vis && pos) card.classList.add('labeled');
}});
updateProgress();
</script>
</body></html>"""
    return Response(html, mimetype="text/html")


@app.route("/thumb/<key>")
def thumb(key):
    from flask import send_file
    p = GALLERY_DIR / f"thumb_{key}.jpg"
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(p, mimetype="image/jpeg")


@app.route("/api/label/<key>", methods=["POST"])
def label(key):
    payload = request.json
    labels = load_labels()
    entry = labels.setdefault(key, {})
    if payload["field"] == "target_visible":
        entry["target_visible"] = payload["val"]
    elif payload["field"] == "coarse_position":
        entry["coarse_position"] = payload["val"]
    save_labels(labels)
    return jsonify({"ok": True})


@app.route("/api/export")
def export():
    return jsonify(load_labels())


if __name__ == "__main__":
    print(f"브라우저 → http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
