#!/usr/bin/env python3
"""
exp73 챔피언 val 33ep 셀프검증 라벨러 — serve_hsv_owlv2_labeler.py 패턴 재사용.

각 에피소드마다 궤적 그림(정답 vs 예측) + path_type 라벨 + FPE + 자동판정(success)을
보여주고, 사람이 두 가지를 O/X로 확정:
  1) path_type 라벨이 실제 궤적/도착위치와 맞는가
  2) success/failure 자동판정(FPE<0.5m)이 실제로 맞는가

실행: .venv/bin/python3 scripts/label/serve_exp73_val_review.py
접속: http://localhost:7794
"""
import json
from pathlib import Path

from flask import Flask, request, jsonify, send_file

ROOT = Path("/home/minum/26CS/MoNaVLA/docs/v5/exp73_val_review")
META_PATH = ROOT / "meta.json"
LABELS_PATH = ROOT / "human_labels.json"
PORT = 7794

app = Flask(__name__)


def load_meta():
    return json.loads(META_PATH.read_text())


def load_labels():
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}


def save_labels(labels):
    LABELS_PATH.write_text(json.dumps(labels, indent=2, ensure_ascii=False))


BASE_CSS = """
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#0a0f1a; color:#e2e8f0; font-family:'Segoe UI',sans-serif; padding:20px; }
h1 { font-size:1.3rem; margin-bottom:6px; }
.sub { color:#94a3b8; font-size:0.85rem; margin-bottom:18px; }
.grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; }
.card { background:#111827; border:1px solid #223; border-radius:10px; padding:10px; }
.shots { display:grid; grid-template-columns:repeat(4, 1fr); gap:5px; margin-bottom:6px; }
.shots img { width:100%; height:90px; object-fit:cover; border-radius:5px; background:#fff; }
.shots .lbl { font-size:0.62rem; color:#64748b; text-align:center; margin-top:2px; }
.card img { width:100%; border-radius:6px; background:#fff; }
@media (max-width:1100px) { .grid { grid-template-columns:repeat(2,1fr); } }
@media (max-width:700px) { .grid { grid-template-columns:1fr; } .shots { grid-template-columns:repeat(2,1fr); } }
.meta { font-size:0.78rem; color:#94a3b8; margin:6px 0; line-height:1.6; }
.meta b { color:#e2e8f0; }
.row { display:flex; gap:6px; margin-top:6px; }
.btn { flex:1; padding:6px; border-radius:6px; border:1px solid #334155; background:#1e293b;
       color:#e2e8f0; cursor:pointer; font-size:0.78rem; text-align:center; }
.btn.active-o { background:#065f46; border-color:#10b981; }
.btn.active-x { background:#7f1d1d; border-color:#ef4444; }
.progress { position:sticky; top:0; background:#0a0f1a; padding:10px 0; z-index:10; }
"""

JS = """
function label(idx, kind, val) {
  fetch('/label', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({idx: idx, kind: kind, val: val})})
    .then(r => r.json()).then(d => {
      document.querySelectorAll(`[data-idx="${idx}"][data-kind="${kind}"]`).forEach(b => {
        b.classList.remove('active-o','active-x');
      });
      const btn = document.querySelector(`[data-idx="${idx}"][data-kind="${kind}"][data-val="${val}"]`);
      if (btn) btn.classList.add(val === 'o' ? 'active-o' : 'active-x');
      document.getElementById('progress').innerText = d.done + ' / ' + d.total + ' 항목 확정';
    });
}
"""


@app.route("/")
def index():
    meta = load_meta()
    labels = load_labels()
    cards = []
    for e in meta:
        idx = e["idx"]
        lb = labels.get(str(idx), {})
        path_o = "active-o" if lb.get("path") == "o" else ""
        path_x = "active-x" if lb.get("path") == "x" else ""
        succ_o = "active-o" if lb.get("success") == "o" else ""
        succ_x = "active-x" if lb.get("success") == "x" else ""
        succ_txt = "성공" if e["auto_success"] else "실패"
        arrive = f"{e['arrive_cx']:.3f}" if e["arrive_cx"] is not None else "N/A"
        cards.append(f"""
        <div class="card">
          <div class="shots">
            <div><img src="/frame/ep{idx:02d}_f0.jpg"><div class="lbl">초반</div></div>
            <div><img src="/frame/ep{idx:02d}_f1.jpg"><div class="lbl">중반</div></div>
            <div><img src="/frame/ep{idx:02d}_f2.jpg"><div class="lbl">종반</div></div>
            <div><img src="/img/{e['traj_img']}"><div class="lbl">궤적그래프</div></div>
          </div>
          <div class="meta">
            <b>{e['path_type']}</b> (#{idx})<br>
            FPE={e['fpe']:.3f}m · 자동판정=<b>{succ_txt}</b> · 도착cx={arrive}
          </div>
          <div class="row">
            <span style="font-size:0.72rem;color:#64748b;align-self:center">라벨:</span>
            <button class="btn {path_o}" data-idx="{idx}" data-kind="path" data-val="o" onclick="label({idx},'path','o')">✓ 맞음</button>
            <button class="btn {path_x}" data-idx="{idx}" data-kind="path" data-val="x" onclick="label({idx},'path','x')">✗ 틀림</button>
          </div>
          <div class="row">
            <span style="font-size:0.72rem;color:#64748b;align-self:center">판정:</span>
            <button class="btn {succ_o}" data-idx="{idx}" data-kind="success" data-val="o" onclick="label({idx},'success','o')">✓ 맞음</button>
            <button class="btn {succ_x}" data-idx="{idx}" data-kind="success" data-val="x" onclick="label({idx},'success','x')">✗ 틀림</button>
          </div>
        </div>""")
    done = sum(1 for v in labels.values() if v.get("path") and v.get("success"))
    return f"""<!doctype html><html><head><meta charset="utf-8">
    <title>exp73 val 33ep 셀프검증</title><style>{BASE_CSS}</style></head>
    <body>
      <div class="progress"><h1>exp73 챔피언 val 33ep 셀프검증</h1>
      <div class="sub">각 카드: 궤적(검정=정답, 주황=예측) · path_type 라벨이 맞는지, FPE 자동판정(성공/실패)이 맞는지 확정</div>
      <div id="progress">{done} / {len(meta)} 항목 확정</div></div>
      <div class="grid">{''.join(cards)}</div>
      <script>{JS}</script>
    </body></html>"""


@app.route("/img/<path:fname>")
def img(fname):
    return send_file(ROOT / "traj" / fname)


@app.route("/frame/<path:fname>")
def frame(fname):
    return send_file(ROOT / "frames" / fname)


@app.route("/label", methods=["POST"])
def label_post():
    data = request.get_json()
    idx, kind, val = str(data["idx"]), data["kind"], data["val"]
    labels = load_labels()
    labels.setdefault(idx, {})[kind] = val
    save_labels(labels)
    meta = load_meta()
    done = sum(1 for v in labels.values() if v.get("path") and v.get("success"))
    return jsonify({"status": "ok", "done": done, "total": len(meta)})


if __name__ == "__main__":
    print(f"exp73 val review labeler → http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
