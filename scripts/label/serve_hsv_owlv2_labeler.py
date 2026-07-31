#!/usr/bin/env python3
"""
CH54 프리뷰 후보(H1/H2/OWL-v2/PG2) 셀프검증 — O/X 라벨링 + 분석 탭 서버.
gen_hsv_owlv2_preview_gallery.py가 만든 meta.json/thumb_*.jpg를 읽어서
카드마다 H1/H2/Ow/PG 각각 "basket을 제대로 잡았는가" O/X 버튼을 붙여 서빙.
PG(PaliGemma2)도 항상 맞는다는 보장이 없으므로 동일하게 라벨 대상에 포함.

탭 구성 (plan_20260704_labeler_analysis_tabs.md):
  /          라벨링 그리드 (?session=<stem> 필터 지원)
  /report    모델별 정확도·검출률 리포트
  /disagree  불일치/미라벨 필터 뷰 (여기서도 바로 라벨 가능)
  /sessions  세션별 요약 테이블 + cx 스파크라인

실행: .venv/bin/python3 scripts/label/serve_hsv_owlv2_labeler.py  (또는 alias: labeler)
접속: http://localhost:7793
"""
import json
from collections import defaultdict
from pathlib import Path

from flask import Flask, request, jsonify, Response, send_file

GALLERY_DIR = Path("/home/minum/26CS/MoNaVLA/docs/v5/hsv_owlv2_preview_20260704")
META_PATH = GALLERY_DIR / "meta.json"
LABELS_PATH = GALLERY_DIR / "human_labels.json"
PORT = 7793

MODELS = ["h1", "h2", "ow", "pg", "kr"]
MODEL_LABEL = {"h1": "H1 (HSV 원본)", "h2": "H2 (HSV 재튜닝)", "ow": "Ow (OWL-v2)",
               "pg": "PG (PaliGemma2)", "kr": "Kr (Kosmos-2 refexp)"}
# 썸네일 bbox 색과 동일 (gen_hsv_owlv2_preview_gallery.py MODEL_COLORS)
MODEL_CSS_COLOR = {"h1": "rgb(60,220,60)", "h2": "rgb(0,200,255)", "ow": "rgb(240,200,0)",
                   "pg": "rgb(255,90,90)", "kr": "rgb(200,120,255)"}

app = Flask(__name__)


def load_meta():
    return json.loads(META_PATH.read_text())


def load_labels():
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}


def save_labels(labels):
    LABELS_PATH.write_text(json.dumps(labels, indent=2, ensure_ascii=False))


def fmt(r):
    return "미검출" if r is None else f"cx={r['cx']:.2f}"


# ── 공용 HTML 조각 ────────────────────────────────────────────────────────────

BASE_CSS = """
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#0a0f1a; color:#e2e8f0; font-family:'Segoe UI',sans-serif; padding:24px; }
h1 { font-size:1.2rem; margin-bottom:4px; }
h2 { font-size:1.0rem; margin:18px 0 8px; color:#cbd5e1; }
.subtitle { color:#64748b; font-size:0.82rem; margin-bottom:14px; line-height:1.6; }
.tabbar { display:flex; gap:6px; margin-bottom:18px; border-bottom:1px solid #1e293b; padding-bottom:0; }
.tabbar a { padding:8px 18px; font-size:0.85rem; color:#94a3b8; text-decoration:none;
            border:1px solid transparent; border-bottom:none; border-radius:8px 8px 0 0; }
.tabbar a.active { color:#38bdf8; background:#0d1117; border-color:#1e293b; font-weight:bold; }
.tabbar a:hover { color:#e2e8f0; }
#progress { position:sticky; top:0; background:#0a0f1acc; backdrop-filter:blur(4px); padding:10px 0;
             z-index:10; font-size:0.85rem; color:#38bdf8; margin-bottom:12px; }
#progress .stat { color:#94a3b8; margin-left:14px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:10px; }
.card { background:#0d1117; border:1px solid #1e293b; border-radius:8px; overflow:hidden; }
.card.labeled { border-color:#22c55e; }
.card img { width:100%; display:block; }
.card-header { padding:4px 8px; font-size:0.62rem; color:#94a3b8; background:#111827; line-height:1.4; }
.label-block { padding:5px 8px; display:flex; flex-direction:column; gap:4px; }
.model-row { display:flex; align-items:center; gap:6px; font-size:0.68rem; }
.model-name { flex:1; }
.model-val { font-size:0.62rem; }
.btn-group { display:flex; gap:3px; }
.lbl-btn { font-size:0.7rem; padding:2px 8px; border-radius:4px; border:1px solid #334155;
            background:#1e293b; color:#94a3b8; cursor:pointer; font-weight:bold; }
.lbl-btn.ok.active { background:#22c55e; color:#fff; border-color:#22c55e; }
.lbl-btn.ng.active { background:#ef4444; color:#fff; border-color:#ef4444; }
.lbl-btn.nt.active { background:#f97316; color:#fff; border-color:#f97316; }
.no-target-row { border-bottom:1px solid #1e293b; padding-bottom:4px; margin-bottom:2px; }
.card.no-target { border-color:#f97316; }
.card.no-target .model-row:not(.no-target-row) { opacity:0.35; }
.card.selected {
  outline:3px solid #fff !important; outline-offset:2px;
  box-shadow:0 0 24px 4px #38bdf8aa;
  transform:scale(1.02); z-index:5; position:relative;
  transition:transform 0.1s;
}
.kbd-help { font-size:0.72rem; color:#64748b; margin-left:16px; }
.kbd-help b { color:#94a3b8; border:1px solid #334155; border-radius:3px; padding:0 4px; font-size:0.68rem; }
.key-badge { font-size:0.6rem; color:#64748b; border:1px solid #334155; border-radius:3px;
             padding:0 4px; min-width:14px; text-align:center; }
.legend { display:flex; gap:16px; margin-bottom:16px; font-size:0.8rem; flex-wrap:wrap; }
.legend span { display:inline-flex; align-items:center; gap:6px; }
.dot { width:12px; height:12px; border-radius:3px; display:inline-block; }
table.rep { border-collapse:collapse; font-size:0.8rem; margin-bottom:16px; }
table.rep th, table.rep td { border:1px solid #1e293b; padding:6px 12px; text-align:right; }
table.rep th { background:#111827; color:#cbd5e1; text-align:center; }
table.rep td:first-child, table.rep th:first-child { text-align:left; }
.best { color:#22c55e; font-weight:bold; }
.conclusion { background:#0d1117; border:1px solid #1e293b; border-left:3px solid #38bdf8;
              border-radius:6px; padding:12px 16px; font-size:0.88rem; margin:14px 0; }
.filterbar { display:flex; gap:8px; margin-bottom:14px; flex-wrap:wrap; }
.filterbar a { font-size:0.78rem; padding:5px 12px; border-radius:14px; border:1px solid #334155;
               color:#94a3b8; text-decoration:none; }
.filterbar a.active { background:#2563eb; color:#fff; border-color:#2563eb; }
a.sess-link { color:#38bdf8; text-decoration:none; }
a.sess-link:hover { text-decoration:underline; }
.muted { color:#64748b; font-size:0.78rem; }
"""

LEGEND_HTML = """
<div class="legend">
  <span><span class="dot" style="background:rgb(60,220,60)"></span>H1 (HSV 원본)</span>
  <span><span class="dot" style="background:rgb(0,200,255)"></span>H2 (HSV 재튜닝)</span>
  <span><span class="dot" style="background:rgb(240,200,0)"></span>Ow (OWL-v2)</span>
  <span><span class="dot" style="background:rgb(255,90,90)"></span>PG (PaliGemma2)</span>
  <span><span class="dot" style="background:rgb(200,120,255)"></span>Kr (Kosmos-2 refexp)</span>
</div>"""

LABEL_JS = """
<script>
const MODELS = ["h1","h2","ow","pg","kr"];
function computeStats() {
  const stats = {};
  MODELS.forEach(m => stats[m] = {ok:0, ng:0});
  document.querySelectorAll('.card').forEach(card => {
    MODELS.forEach(m => {
      const active = card.querySelector(`[data-field="${m}"] .lbl-btn.active`);
      if (active) stats[m][active.dataset.val === 'ok' ? 'ok' : 'ng']++;
    });
  });
  return stats;
}
function updateProgress() {
  const el = document.getElementById('done-count');
  if (!el) return;
  el.textContent = document.querySelectorAll('.card.labeled').length;
  const s = computeStats();
  MODELS.forEach(m => {
    const t = s[m].ok + s[m].ng;
    const pct = t ? Math.round(100*s[m].ok/t) : 0;
    const st = document.getElementById('stat-' + m);
    if (st) st.textContent = `${m.toUpperCase()}: ${s[m].ok}/${t} (${pct}%)`;
  });
}
function isNoTarget(card) {
  return !!card.querySelector('[data-field="no_target"] .lbl-btn.active');
}
function checkLabeled(card) {
  const nt = isNoTarget(card);
  card.classList.toggle('no-target', nt);
  const allSet = MODELS.every(m => card.querySelector(`[data-field="${m}"] .lbl-btn.active`));
  card.classList.toggle('labeled', nt || allSet);
}
document.querySelectorAll('.card').forEach(card => {
  const key = card.dataset.key;
  card.querySelectorAll('.lbl-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const field = btn.parentElement.parentElement.dataset.field;
      let val = btn.dataset.val;
      if (field === 'no_target') {
        // 토글: 이미 켜져있으면 해제
        const wasActive = btn.classList.contains('active');
        btn.classList.toggle('active', !wasActive);
        val = wasActive ? 'no' : 'yes';
      } else {
        card.querySelectorAll(`[data-field="${field}"] .lbl-btn`).forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      }
      await fetch('/api/label/' + key, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({field, val})
      });
      checkLabeled(card);
      updateProgress();
    });
  });
  checkLabeled(card);
});
updateProgress();

// ── 키보드 라벨링 ──────────────────────────────────────────────
// ←→↑↓: 프레임 이동 / 1~5: 각 모델(H1,H2,Ow,PG,Kr) O↔X 사이클
// (첫 누름 O, 다시 누르면 X, 또 누르면 O ...) / 0: 객체 없음 토글
// 5개 모델 전부 라벨되면 자동으로 다음 프레임 선택.
const cards = Array.from(document.querySelectorAll('.card'));
let selIdx = -1;

function clearCursor() {
  document.querySelectorAll('.card.selected').forEach(c => c.classList.remove('selected'));
}
function paintCursor() {
  if (selIdx < 0 || selIdx >= cards.length) return;
  const card = cards[selIdx];
  card.classList.add('selected');
  card.scrollIntoView({block:'center'});
}
function selectCard(i) {
  clearCursor();
  selIdx = Math.max(0, Math.min(cards.length - 1, i));
  paintCursor();
}
function colsInGrid() {
  const grid = document.querySelector('.grid');
  if (!grid || !cards.length) return 1;
  const style = getComputedStyle(grid);
  return style.gridTemplateColumns.split(' ').length || 1;
}
function pressModelKey(mIdx) {
  if (selIdx < 0) { selectCard(0); return; }
  const card = cards[selIdx];
  const mid = MODELS[mIdx];
  if (!mid) return;
  const row = card.querySelector(`[data-field="${mid}"]`);
  if (!row) return;
  // 사이클: 미라벨→O, O→X, X→O
  const okBtn = row.querySelector('.lbl-btn.ok');
  const ngBtn = row.querySelector('.lbl-btn.ng');
  (okBtn.classList.contains('active') ? ngBtn : okBtn).click();
  // 5개 모델 전부 라벨 완료 시 자동으로 다음 프레임
  const allSet = MODELS.every(m => card.querySelector(`[data-field="${m}"] .lbl-btn.active`));
  if (allSet && selIdx < cards.length - 1) selectCard(selIdx + 1);
}
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const cols = colsInGrid();
  if (e.key >= '1' && e.key <= String(MODELS.length)) {
    e.preventDefault();
    pressModelKey(parseInt(e.key, 10) - 1);
    return;
  }
  switch (e.key) {
    case 'ArrowRight': e.preventDefault(); selectCard(selIdx < 0 ? 0 : selIdx + 1); break;
    case 'ArrowLeft':  e.preventDefault(); selectCard(selIdx < 0 ? 0 : selIdx - 1); break;
    case 'ArrowDown':  e.preventDefault(); selectCard(selIdx < 0 ? 0 : selIdx + cols); break;
    case 'ArrowUp':    e.preventDefault(); selectCard(selIdx < 0 ? 0 : selIdx - cols); break;
    case '0': {
      e.preventDefault();
      if (selIdx < 0) { selectCard(0); break; }
      const ntBtn = cards[selIdx].querySelector('[data-field="no_target"] .lbl-btn');
      if (ntBtn) ntBtn.click();
      // 객체 없음 = 이 프레임 완료 → 다음 프레임으로
      selectCard(selIdx + 1);
      break;
    }
  }
});
// 카드 클릭으로도 선택 가능
cards.forEach((card, i) => {
  card.addEventListener('click', (e) => {
    if (e.target.classList.contains('lbl-btn')) return;
    selectCard(i);
  });
});
// 시작하자마자 첫 미라벨 카드 자동 선택 — 방향키가 바로 반응하도록
if (cards.length) {
  const firstUnlabeled = cards.findIndex(c => !c.classList.contains('labeled'));
  selectCard(firstUnlabeled >= 0 ? firstUnlabeled : 0);
}
</script>"""


def tabbar(active: str) -> str:
    tabs = [("/", "라벨링"), ("/disagree", "불일치"), ("/report", "리포트"),
            ("/sessions", "세션별"), ("/threshold", "Threshold")]
    links = "".join(
        f'<a href="{url}" class="{"active" if url == active else ""}">{name}</a>'
        for url, name in tabs
    )
    return f'<div class="tabbar">{links}</div>'


def page(title: str, active_tab: str, body: str, with_label_js: bool = False) -> Response:
    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>{title}</title>
<style>{BASE_CSS}</style></head>
<body>
{tabbar(active_tab)}
{body}
{LABEL_JS if with_label_js else ""}
</body></html>"""
    return Response(html, mimetype="text/html")


def render_card(m: dict, lab: dict) -> str:
    key = m["key"]
    nt = lab.get("no_target") == "yes"
    rows = [f"""
      <div class="model-row no-target-row" data-field="no_target">
        <span class="model-name" style="color:#f97316">🚫 원하는 객체가 이미지에 없음</span>
        <div class="btn-group">
          <button class="lbl-btn nt {'active' if nt else ''}" data-val="yes">없음</button>
        </div>
      </div>"""]
    for i, mid in enumerate(MODELS):
        val = lab.get(mid)
        rows.append(f"""
      <div class="model-row" data-field="{mid}">
        <span class="key-badge">{i + 1}</span>
        <span class="model-name" style="color:{MODEL_CSS_COLOR[mid]}">{MODEL_LABEL[mid]}</span>
        <span class="model-val" style="color:{MODEL_CSS_COLOR[mid]}; opacity:0.75">{fmt(m.get(mid))}</span>
        <div class="btn-group">
          <button class="lbl-btn ok {'active' if val == 'ok' else ''}" data-val="ok">O</button>
          <button class="lbl-btn ng {'active' if val == 'ng' else ''}" data-val="ng">X</button>
        </div>
      </div>""")
    return f"""
  <div class="card" data-key="{key}" id="card-{key}">
    <img src="/thumb/{key}" loading="lazy">
    <div class="card-header">{key}</div>
    <div class="label-block">{''.join(rows)}</div>
  </div>"""


def card_grid(frames: list, labels: dict) -> str:
    cards = "".join(render_card(m, labels.get(m["key"], {})) for m in frames)
    stats_spans = "".join(f'<span class="stat" id="stat-{m}"></span>' for m in MODELS)
    return f"""
<div id="progress">라벨링: <span id="done-count">0</span> / {len(frames)} 프레임 완료 {stats_spans}
  <span class="kbd-help"><b>←→↑↓</b> 프레임 이동 · <b>1</b>H1 <b>2</b>H2 <b>3</b>Ow <b>4</b>PG <b>5</b>Kr (첫 누름 O, 다시 누르면 X) · <b>0</b> 객체 없음 · 5개 다 되면 자동 다음 프레임</span></div>
<div class="grid">{cards}</div>"""


# ── 집계 ──────────────────────────────────────────────────────────────────────

def cx_spread(m: dict) -> float | None:
    cxs = [m[k]["cx"] for k in MODELS if m.get(k)]
    if len(cxs) < 2:
        return None
    return max(cxs) - min(cxs)


def compute_stats():
    meta = load_meta()
    labels = load_labels()
    total = len(meta)
    no_target_keys = {k for k, v in labels.items() if v.get("no_target") == "yes"}
    per_model = {}
    for mid in MODELS:
        ok = ng = 0
        detected = sum(1 for m in meta if m.get(mid) is not None)
        missed_but_present = 0  # 해당 모델 null인데 다른 모델이 O 받은 프레임
        false_on_absent = 0     # 객체 없음 프레임인데 뭔가 검출한 경우 (오탐)
        for m in meta:
            lab = labels.get(m["key"], {})
            if m["key"] in no_target_keys:
                if m.get(mid) is not None:
                    false_on_absent += 1
                continue  # 객체 없음 프레임은 O/X 정확도 집계에서 제외
            v = lab.get(mid)
            if v == "ok":
                ok += 1
            elif v == "ng":
                ng += 1
            if m.get(mid) is None and any(lab.get(o) == "ok" for o in MODELS if o != mid):
                missed_but_present += 1
        per_model[mid] = {
            "ok": ok, "ng": ng, "labeled": ok + ng,
            "acc": (ok / (ok + ng)) if (ok + ng) else None,
            "detect_rate": detected / total if total else 0.0,
            "missed_but_present": missed_but_present,
            "false_on_absent": false_on_absent,
        }
    # 세션별
    sessions = defaultdict(lambda: {"frames": 0, "labeled": 0, "no_target": 0,
                                     **{f"{m}_ok": 0 for m in MODELS},
                                     **{f"{m}_lab": 0 for m in MODELS}})
    for m in meta:
        s = sessions[m["episode"]]
        s["frames"] += 1
        lab = labels.get(m["key"], {})
        if m["key"] in no_target_keys:
            s["no_target"] += 1
            s["labeled"] += 1  # 객체 없음도 라벨 완료로 취급
            continue
        if all(lab.get(mid) in ("ok", "ng") for mid in MODELS):
            s["labeled"] += 1
        for mid in MODELS:
            if lab.get(mid) in ("ok", "ng"):
                s[f"{mid}_lab"] += 1
                if lab[mid] == "ok":
                    s[f"{mid}_ok"] += 1
    return {"total": total, "no_target_count": len(no_target_keys),
            "per_model": per_model, "sessions": dict(sessions)}


# ── 라우트 ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    meta = load_meta()
    labels = load_labels()
    session_filter = request.args.get("session")
    if session_filter:
        meta = [m for m in meta if m["episode"] == session_filter]
    head = f"""
<h1>CH54 프리뷰 후보 O/X 라벨링 — H1/H2/OWL-v2/PG2 각각 실제 basket을 잡았는지 확인</h1>
<p class="subtitle">사진 속 박스 색으로 어느 모델인지 구분하세요. PG(빨강)도 항상 맞다는 보장 없음 —
4개 모델 전부 박스 위치가 실제 basket과 맞으면 O, 아니면 X. 클릭 즉시 자동저장.
{f'<b>세션 필터: {session_filter}</b> (<a class="sess-link" href="/">전체 보기</a>)' if session_filter else ''}</p>
{LEGEND_HTML}"""
    return page("CH54 라벨링", "/", head + card_grid(meta, labels), with_label_js=True)


@app.route("/disagree")
def disagree():
    meta = load_meta()
    labels = load_labels()
    mode = request.args.get("f", "unlabeled")

    def is_unlabeled(m):
        lab = labels.get(m["key"], {})
        if lab.get("no_target") == "yes":
            return False  # 객체 없음 = 라벨 완료
        return not all(lab.get(mid) in ("ok", "ng") for mid in MODELS)

    def is_no_target(m):
        return labels.get(m["key"], {}).get("no_target") == "yes"

    def is_split(m):
        lab = labels.get(m["key"], {})
        vals = {lab.get(mid) for mid in MODELS if lab.get(mid) in ("ok", "ng")}
        return len(vals) == 2

    def is_pg_ng(m):
        return labels.get(m["key"], {}).get("pg") == "ng"

    def is_all_ng(m):
        lab = labels.get(m["key"], {})
        return all(lab.get(mid) == "ng" for mid in MODELS)

    def is_spread(m):
        sp = cx_spread(m)
        return sp is not None and sp > 0.15

    filters = {
        "unlabeled": ("미라벨만", is_unlabeled),
        "split": ("모델간 O/X 갈림", is_split),
        "pg_ng": ("PG가 X", is_pg_ng),
        "all_ng": ("전원 X (진짜 안보임 후보)", is_all_ng),
        "no_target": ("객체 없음 라벨", is_no_target),
        "spread": ("cx 스프레드 > 0.15", is_spread),
    }
    if mode not in filters:
        mode = "unlabeled"
    frames = [m for m in meta if filters[mode][1](m)]

    fbar = "".join(
        f'<a href="/disagree?f={k}" class="{"active" if k == mode else ""}">{name}</a>'
        for k, (name, _) in filters.items()
    )
    head = f"""
<h1>불일치/미라벨 필터 — 사람이 봐야 할 프레임부터</h1>
<p class="subtitle">필터: <b>{filters[mode][0]}</b> — {len(frames)}개 / 전체 {len(meta)}개.
cx 스프레드 필터는 라벨 없이도 동작(모델 예측끼리 어긋난 프레임). 여기서도 바로 O/X 라벨 가능.</p>
<div class="filterbar">{fbar}</div>
{LEGEND_HTML}"""
    return page("불일치 필터", "/disagree", head + card_grid(frames, labels), with_label_js=True)


@app.route("/report")
def report():
    st = compute_stats()
    pm = st["per_model"]
    any_labeled = any(pm[m]["labeled"] for m in MODELS)

    def pct(v):
        return "—" if v is None else f"{100*v:.1f}%"

    best = None
    if any_labeled:
        cands = [(mid, pm[mid]["acc"]) for mid in MODELS if pm[mid]["acc"] is not None]
        if cands:
            best = max(cands, key=lambda t: t[1])

    rows = []
    for mid in MODELS:
        p = pm[mid]
        cls = "best" if best and best[0] == mid else ""
        rows.append(f"""<tr>
  <td style="color:{MODEL_CSS_COLOR[mid]}">{MODEL_LABEL[mid]}</td>
  <td class="{cls}">{pct(p['acc'])}</td>
  <td>{p['ok']} / {p['labeled']}</td>
  <td>{pct(p['detect_rate'])}</td>
  <td>{p['missed_but_present']}</td>
  <td>{p['false_on_absent']}</td>
</tr>""")

    sess_rows = []
    for name, s in sorted(st["sessions"].items()):
        cells = []
        for mid in MODELS:
            lab, ok = s[f"{mid}_lab"], s[f"{mid}_ok"]
            cells.append(f"<td>{f'{100*ok/lab:.0f}%' if lab else '—'}</td>")
        sess_rows.append(
            f'<tr><td><a class="sess-link" href="/?session={name}">{name}</a></td>'
            f'<td>{s["labeled"]}/{s["frames"]}</td>{"".join(cells)}</tr>'
        )

    conclusion = (
        f'현재 라벨 기준 프리뷰 후보 1위: <b style="color:{MODEL_CSS_COLOR[best[0]]}">{MODEL_LABEL[best[0]]}</b> '
        f'(정확도 {100*best[1]:.1f}%, {pm[best[0]]["labeled"]}프레임 라벨 기준)'
        if best else "아직 라벨이 없습니다 — 라벨링 탭에서 O/X를 눌러주세요."
    )

    body = f"""
<h1>리포트 — 모델별 성적표</h1>
<p class="subtitle">human_labels.json 실시간 집계. 새로고침하면 최신 반영. 전체 {st['total']}프레임,
그중 <b style="color:#f97316">객체 없음 라벨 {st['no_target_count']}개</b> (정확도 집계에서 제외, 오탐 집계에 사용).</p>
<div class="conclusion">{conclusion}</div>
<table class="rep">
<tr><th>모델</th><th>정확도 (O비율)</th><th>O / 라벨수</th><th>검출률 (전체 {st['total']})</th><th>미검출인데 실제 있었음*</th><th>객체없음인데 검출 (오탐)**</th></tr>
{''.join(rows)}
</table>
<p class="muted">* 해당 모델이 미검출(null)인데 같은 프레임에서 다른 모델이 O를 받은 경우 — "놓친" 프레임 수.<br>
** 사람이 "객체 없음"으로 라벨한 프레임에서 해당 모델이 뭔가를 검출한 경우 — 허위 양성. 낮을수록 좋음.</p>
<h2>세션별 정확도 (라벨완료수 / 모델별 O비율)</h2>
<table class="rep">
<tr><th>세션</th><th>라벨</th>{''.join(f'<th>{m.upper()}</th>' for m in MODELS)}</tr>
{''.join(sess_rows)}
</table>"""
    return page("리포트", "/report", body)


@app.route("/sessions")
def sessions_view():
    meta = load_meta()
    st = compute_stats()

    by_sess = defaultdict(list)
    for m in meta:
        by_sess[m["episode"]].append(m)

    def sparkline(frames, mid):
        pts = [(i, f[mid]["cx"]) for i, f in enumerate(frames) if f.get(mid)]
        if len(pts) < 2:
            return '<span class="muted">—</span>'
        w, h = 120, 24
        n = len(frames) - 1 or 1
        path = " ".join(f"{'M' if j == 0 else 'L'}{x/n*w:.1f},{h - cx*h:.1f}" for j, (x, cx) in enumerate(pts))
        return (f'<svg width="{w}" height="{h}" style="background:#111827;border-radius:3px">'
                f'<line x1="0" y1="{h/2}" x2="{w}" y2="{h/2}" stroke="#334155" stroke-width="0.5"/>'
                f'<path d="{path}" fill="none" stroke="{MODEL_CSS_COLOR[mid]}" stroke-width="1.2"/></svg>')

    rows = []
    for name in sorted(by_sess):
        frames = sorted(by_sess[name], key=lambda m: m["frame_idx"])
        s = st["sessions"][name]
        accs = []
        for mid in MODELS:
            lab, ok = s[f"{mid}_lab"], s[f"{mid}_ok"]
            accs.append(f"<td>{f'{100*ok/lab:.0f}%' if lab else '—'}</td>")
        sparks = "".join(f"<td>{sparkline(frames, mid)}</td>" for mid in MODELS)
        rows.append(f"""<tr>
  <td><a class="sess-link" href="/?session={name}">{name}</a></td>
  <td>{s['frames']}</td><td>{s['labeled']}</td>
  {''.join(accs)}
  {sparks}
</tr>""")

    body = f"""
<h1>세션별 요약</h1>
<p class="subtitle">세션명 클릭 → 해당 세션만 필터된 라벨링 뷰. 스파크라인 = 프레임 진행에 따른 각 모델 cx 궤적
(위=오른쪽 cx=1.0, 아래=왼쪽 cx=0.0, 가운데 선=cx 0.5). 궤적이 요동치면 해당 모델 예측이 불안정.</p>
{LEGEND_HTML}
<table class="rep">
<tr><th>세션</th><th>프레임</th><th>라벨</th>
    {''.join(f'<th>{m.upper()}</th>' for m in MODELS)}
    {''.join(f'<th>{m.upper()} cx</th>' for m in MODELS)}</tr>
{''.join(rows)}
</table>"""
    return page("세션별", "/sessions", body)


@app.route("/threshold")
def threshold_view():
    """OWL-v2 confidence threshold 슬라이더 — 프레임별 TP/FN/FP/TN 실시간 분류."""
    meta = load_meta()
    labels = load_labels()
    scores_path = GALLERY_DIR / "owlv2_scores.json"
    if not scores_path.exists():
        return page("Threshold", "/threshold",
                    "<h1>owlv2_scores.json 없음</h1><p class='subtitle'>"
                    "scripts/eval/owlv2_threshold_roc.py를 먼저 실행하세요.</p>")
    scores = json.loads(scores_path.read_text())

    items = []
    for m in meta:
        key = m["key"]
        lab = labels.get(key, {})
        absent = lab.get("no_target") == "yes"
        items.append({"key": key, "score": scores.get(key, 0.0),
                      "absent": absent, "ow_ok": lab.get("ow") == "ok"})
    items.sort(key=lambda x: x["score"])

    cards = "".join(f"""
  <div class="card th-card" data-score="{it['score']:.4f}" data-absent="{1 if it['absent'] else 0}">
    <img src="/thumb/{it['key']}" loading="lazy">
    <div class="card-header">{it['key']}<br>
      score=<b>{it['score']:.3f}</b> · 사람라벨: {'🚫 객체없음' if it['absent'] else ('OWL 정답(O)' if it['ow_ok'] else '객체있음')}
      <span class="verdict"></span>
    </div>
  </div>""" for it in items)

    body = f"""
<h1>OWL-v2 Threshold 탐색 — 프레임별 판정 확인</h1>
<p class="subtitle">슬라이더를 움직이면 각 프레임이 해당 threshold에서 어떻게 판정되는지 실시간 표시.
score 오름차순 정렬 — 위쪽이 낮은(거부되는) 프레임. ROC 권장값 0.25.</p>
<div id="progress">
  threshold = <b id="th-val">0.25</b>
  <input type="range" id="th-slider" min="0" max="0.7" step="0.01" value="0.25"
         style="width:280px; vertical-align:middle; margin:0 12px;">
  <span class="stat" id="th-summary"></span>
  <span class="kbd-help">필터:
    <a href="#" class="th-filter active" data-f="all">전체</a> ·
    <a href="#" class="th-filter" data-f="fp">오탐(FP)</a> ·
    <a href="#" class="th-filter" data-f="fn">놓침(FN)</a>
  </span>
</div>
<div class="grid">{cards}</div>
<style>
.th-card.tp {{ border-color:#22c55e; }}
.th-card.tn {{ border-color:#334155; opacity:0.55; }}
.th-card.fp {{ border-color:#ef4444; box-shadow:0 0 8px #ef444455; }}
.th-card.fn {{ border-color:#f59e0b; box-shadow:0 0 8px #f59e0b55; }}
.verdict {{ font-weight:bold; }}
.th-filter {{ color:#94a3b8; text-decoration:none; padding:2px 8px; border-radius:10px; }}
.th-filter.active {{ background:#2563eb; color:#fff; }}
</style>
<script>
const cardsTh = Array.from(document.querySelectorAll('.th-card'));
let filterMode = 'all';
function applyThreshold() {{
  const t = parseFloat(document.getElementById('th-slider').value);
  document.getElementById('th-val').textContent = t.toFixed(2);
  let tp=0, tn=0, fp=0, fn=0;
  cardsTh.forEach(c => {{
    const s = parseFloat(c.dataset.score);
    const absent = c.dataset.absent === '1';
    const detected = s >= t;
    let cls, label;
    if (absent && detected)      {{ cls='fp'; label='❌ 오탐 (없는데 검출)'; fp++; }}
    else if (absent)             {{ cls='tn'; label='✅ 정거부 (없음→거부)'; tn++; }}
    else if (detected)           {{ cls='tp'; label='✅ 정탐'; tp++; }}
    else                         {{ cls='fn'; label='⚠️ 놓침 (있는데 거부)'; fn++; }}
    c.classList.remove('tp','tn','fp','fn');
    c.classList.add(cls);
    c.querySelector('.verdict').textContent = ' → ' + label;
    c.style.display = (filterMode==='all' || filterMode===cls) ? '' : 'none';
  }});
  const nAbs = fp+tn, nPre = tp+fn;
  document.getElementById('th-summary').textContent =
    `정탐 ${{tp}}/${{nPre}} (${{(100*tp/nPre).toFixed(1)}}%) · 오탐 ${{fp}}/${{nAbs}} (${{(100*fp/nAbs).toFixed(1)}}%) · 놓침 ${{fn}} · 정거부 ${{tn}}`;
}}
document.getElementById('th-slider').addEventListener('input', applyThreshold);
document.querySelectorAll('.th-filter').forEach(a => {{
  a.addEventListener('click', (e) => {{
    e.preventDefault();
    document.querySelectorAll('.th-filter').forEach(x => x.classList.remove('active'));
    a.classList.add('active');
    filterMode = a.dataset.f;
    applyThreshold();
  }});
}});
applyThreshold();
</script>"""
    return page("Threshold 탐색", "/threshold", body)


@app.route("/thumb/<key>")
def thumb(key):
    p = GALLERY_DIR / f"thumb_{key}.jpg"
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(p, mimetype="image/jpeg")


@app.route("/api/label/<key>", methods=["POST"])
def label(key):
    payload = request.json
    labels = load_labels()
    entry = labels.setdefault(key, {})
    entry[payload["field"]] = payload["val"]
    save_labels(labels)
    return jsonify({"ok": True})


@app.route("/api/stats")
def api_stats():
    return jsonify(compute_stats())


@app.route("/api/export")
def export():
    return jsonify(load_labels())


if __name__ == "__main__":
    print(f"브라우저 → http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
