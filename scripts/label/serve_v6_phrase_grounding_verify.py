#!/usr/bin/env python3
"""V6 학습셋(16599프레임) 대상 Florence-2 phrase-grounding 사람 검증 (2026-08-20).

이전 버전(serve_florence2_owl_compare.py)은 0807 실기 세션(1087프레임) 기준이었는데,
실제로 검증해야 할 건 **학습에 쓰이는 V6 데이터셋**이라는 지적에 따라 새로 만든다.

형식: serve_hsv_owlv2_labeler.py(CH54 그라운딩 라벨링)의 카드 그리드 + O/X 버튼
스타일을 따른다.

표본 설계: 225개 에피소드 × 프레임 내 위치(초/중/후) × OWL 검출여부, 6칸에
최대한 서로 다른 에피소드로 채운다(한 에피소드에 몰리지 않게).

색상 채널 주의(2026-08-20 확인): V6 원본 h5(`ROS_action/mobile_vla_dataset_v5/*.h5`)는
BGR로 저장되어 있다 — `im[:, :, ::-1]` 반전 필수(육안 확인: 반전 안 하면 파란 색조가
낌 — 예: 나무 책상이 파랗게 보임). 0807 실기 세션(inference_sessions_recv)과는
반대 규칙이니 섞어 쓰지 말 것.

Florence-2 태스크: <CAPTION_TO_PHRASE_GROUNDING> + phrase="gray basket"
(0807 배치에서 재현율 84.96% 나온 것과 동일 방식) — OWL bbox(V6 주석)를 참고선으로,
Florence-2 실시간 예측을 겹쳐서 보여주고 사람이 최종 판정한다.

주의: 파이프라인에 끼워넣지 않는다. 연구용 로컬 검증 도구.

실행: .venv/bin/python3 scripts/label/serve_v6_phrase_grounding_verify.py
접속: http://localhost:7796
"""
import base64
import io
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
ANN_OWL = ROOT / "docs/v5/bbox_nav_owl/bbox_dataset_v6_owl.json"
LABELS_PATH = ROOT / "docs/v5/detector/v6_phrase_grounding_human_labels.json"
PORT = 7796
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "microsoft/Florence-2-base"
TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
PHRASE = "gray basket"

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


# 목표 5개(direction) × 경로 접근방식 3개(path_type 접미사) — V6 설계 그대로
DIRECTION_COLOR = {
    "center": "#94a3b8", "weak_left": "#60a5fa", "weak_right": "#fbbf24",
    "strong_left": "#a78bfa", "strong_right": "#fb923c",
}
DIRECTION_LABEL = {
    "center": "중앙", "weak_left": "약좌", "weak_right": "약우",
    "strong_left": "강좌", "strong_right": "강우",
}
APPROACH_COLOR = {"straight": "#4ade80", "left_curve": "#38bdf8", "right_curve": "#f472b6"}
APPROACH_LABEL = {"straight": "직진", "left_curve": "좌커브", "right_curve": "우커브"}


def parse_approach(path_type, direction):
    suffix = path_type[len(direction) + 1:] if path_type.startswith(direction + "_") else path_type
    if suffix.endswith("left_curve"):
        return "left_curve"
    if suffix.endswith("right_curve"):
        return "right_curve"
    return "straight"


def load_index():
    """(h5_path, frame_idx, gt_cx, owl_has_bbox, bin, episode_stem, direction, approach) 리스트."""
    ann = json.loads(ANN_OWL.read_text())
    frames = []
    for ep in ann:
        h5_path = ep["episode"]
        direction = ep.get("direction", "center")
        approach = parse_approach(ep.get("path_type", ""), direction)
        fs = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        n = len(fs)
        if n == 0:
            continue
        for pos, fr in enumerate(fs):
            bin_ = "early" if pos < n / 3 else ("mid" if pos < 2 * n / 3 else "late")
            frames.append(dict(
                path=h5_path, frame_idx=fr["frame_idx"], gt_cx=fr.get("cx_det", 0.5),
                owl_ok=bool(fr.get("has_bbox", False)), bin=bin_, stem=Path(h5_path).stem,
                direction=direction, approach=approach,
            ))
    return frames


FRAMES = load_index()
print(f"V6 프레임 {len(FRAMES)}개 인덱싱 완료 (기대값 16599)", flush=True)

# 층(bin × owl_ok)별 모집단 크기 — 표본이 균등하지 않으니(succ는 소량, fail은 전수) 모집단 비율로
# 가중해야 전체 정확도 추정치가 왜곡되지 않는다(예: OWL 실패 프레임은 전체의 일부인데
# 표본에선 절반).
CELL_POP = {}
for fr in FRAMES:
    key = (fr["bin"], "succ" if fr["owl_ok"] else "fail")
    CELL_POP[key] = CELL_POP.get(key, 0) + 1


# succ은 OWL 자체가 이미 어느 정도 신뢰도가 있다고 알려진 영역(과거 ROC 분석
# 정탐 94.9%)이라 소량 샌티티체크만, fail은 완전 미검증 영역이라 있는 대로 전부
# — 사용자 요청(2026-08-20)에 따른 재배분.
N_PER_CELL_SUCC = 10
FAIL_TAKE_ALL = True


def build_sample():
    """6칸(bin × owl_ok) — succ은 칸당 N_PER_CELL_SUCC개 스팟체크, fail은 가능한 전부.
    칸 안에서는 (목표 5 × 접근 3) 15조합을 라운드로빈으로 순회해 최대한 골고루 뽑는다."""
    rng = np.random.default_rng(42)
    cells = {}
    for i, fr in enumerate(FRAMES):
        key = (fr["bin"], "succ" if fr["owl_ok"] else "fail")
        cells.setdefault(key, []).append(i)

    sample = []
    for key, idxs in cells.items():
        by_combo = {}
        for i in idxs:
            fr = FRAMES[i]
            combo = (fr["direction"], fr["approach"], fr["stem"])
            by_combo.setdefault(combo, []).append(i)
        combos = list(by_combo.keys())
        rng.shuffle(combos)
        by_da = {}
        for combo in combos:
            da = (combo[0], combo[1])
            by_da.setdefault(da, []).append(combo)
        da_order = list(by_da.keys())
        rng.shuffle(da_order)

        target_n = len(combos) if (key[1] == "fail" and FAIL_TAKE_ALL) else N_PER_CELL_SUCC
        picked = []
        while len(picked) < target_n and any(by_da.values()):
            for da in list(da_order):
                if not by_da[da]:
                    continue
                combo = by_da[da].pop()
                cand = by_combo[combo]
                picked.append(cand[rng.integers(0, len(cand))])
                if len(picked) >= target_n:
                    break
        sample.extend([(i, key[0], key[1]) for i in picked])
    rng.shuffle(sample)
    return sample


SAMPLE = build_sample()
_n_succ = sum(1 for _, _, ok in SAMPLE if ok == "succ")
_n_fail = sum(1 for _, _, ok in SAMPLE if ok == "fail")
print(f"검증 표본 {len(SAMPLE)}개 구성 완료 (succ {_n_succ}개 스팟체크 + fail {_n_fail}개 전수)", flush=True)

_h5_cache = {}


def get_image(path, fi):
    if path not in _h5_cache:
        _h5_cache[path] = h5py.File(path, "r")
    hf = _h5_cache[path]
    imgs = hf["images"] if "images" in hf else hf["observations"]["images"]
    im = imgs[fi]
    # V6는 BGR 저장 — 반전 필수(0807과 반대, 육안 확인 완료)
    return Image.fromarray(im[:, :, ::-1].astype(np.uint8)).convert("RGB")


def run_phrase_grounding(img):
    model, proc = get_model()
    W, H = img.width, img.height
    text = TASK + PHRASE
    inp = proc(text=text, images=img, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids=inp["input_ids"].to(DEV),
            pixel_values=inp["pixel_values"].to(DEV, torch.float16),
            max_new_tokens=128, num_beams=3, do_sample=False)
    txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(txt, task=TASK, image_size=(W, H))[TASK]
    boxes = parsed.get("bboxes", []) or []
    return (boxes[0][0] + boxes[0][2]) / 2 / W if boxes else None


def img_to_b64(img, quality=78):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


_render_cache = {}


def render_card(sample_i):
    if sample_i in _render_cache:
        return _render_cache[sample_i]
    fidx, bin_, owl_ok = SAMPLE[sample_i]
    fr = FRAMES[fidx]
    img = get_image(fr["path"], fr["frame_idx"])
    pred_cx = run_phrase_grounding(img)
    # 화질 손실 방지: 선/글씨를 이미지에 굽지 않는다. 원본 해상도(1280×720)를
    # 그대로 고품질(quality=92)로 인코딩하고, cx는 값만 넘겨 프론트엔드에서
    # CSS 오버레이(위치 %)로 그린다 — 카드가 어떤 크기로 표시되든 선명하게 유지.
    data = dict(image=img_to_b64(img, quality=92), bin=bin_, owl_ok=owl_ok,
                stem=fr["stem"], frame_idx=fr["frame_idx"], gt_cx=fr["gt_cx"], pred_cx=pred_cx,
                direction=fr["direction"], approach=fr["approach"])
    _render_cache[sample_i] = data
    return data


def load_labels():
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}


def save_label(key, data):
    labels = load_labels()
    labels[key] = data
    LABELS_PATH.write_text(json.dumps(labels, indent=2, ensure_ascii=False))


PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<title>V6 Florence-2 phrase-grounding 검증</title>
<style>
body{background:#0a0f1a;color:#e2e8f0;font-family:'Segoe UI',sans-serif;padding:20px}
h1{font-size:1.15rem}
#progress{position:sticky;top:0;background:#0a0f1acc;backdrop-filter:blur(4px);padding:10px 0;
  z-index:10;font-size:0.85rem;color:#38bdf8;margin-bottom:10px}
#progress .stat{color:#94a3b8;margin-left:14px}
#groupstats{display:flex;gap:24px;flex-wrap:wrap;font-size:0.75rem;color:#94a3b8;margin-bottom:12px}
#groupstats table{border-collapse:collapse}
#groupstats td{padding:1px 8px 1px 0}
#legend{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:12px;font-size:0.78rem;color:#94a3b8}
#legend .lg-group{display:flex;gap:8px;align-items:center}
.swatch{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:4px;vertical-align:-1px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}
.card{background:#0d1117;border:1px solid #1e293b;border-left-width:5px;border-radius:8px;overflow:hidden}
.card.labeled{outline:2px solid #22c55e}
.imgwrap{position:relative;width:100%;line-height:0}
.card img{width:100%;display:block}
.vline{position:absolute;top:0;bottom:0;width:2px;pointer-events:none}
.vline .tag-label{position:absolute;top:2px;left:4px;font-size:0.68rem;font-weight:bold;
  white-space:nowrap;text-shadow:-1px -1px 0 #000,1px -1px 0 #000,-1px 1px 0 #000,1px 1px 0 #000}
.vline.pred .tag-label{top:16px}
.card-header{padding:4px 8px;font-size:0.65rem;color:#94a3b8;background:#111827;line-height:1.5}
.tag{display:inline-block;padding:0 6px;border-radius:4px;font-size:0.68rem;font-weight:bold;color:#0a0f1a;margin-right:4px}
.label-block{padding:6px 8px;display:flex;flex-direction:column;gap:5px}
.btn-group{display:flex;gap:3px}
.lbl-btn{font-size:0.72rem;padding:3px 9px;border-radius:4px;border:1px solid #334155;
  background:#1e293b;color:#94a3b8;cursor:pointer;font-weight:bold}
.lbl-btn.ok.active{background:#22c55e;color:#fff;border-color:#22c55e}
.lbl-btn.ng.active{background:#ef4444;color:#fff;border-color:#ef4444}
.lbl-btn.nt.active{background:#f97316;color:#fff;border-color:#f97316}
</style></head><body>
<h1>V6 학습셋(16599프레임) — Florence-2 phrase-grounding("gray basket") 사람 검증</h1>
<div id="legend">
  <span>초록선=OWL bbox(참고) · 빨강선=Florence-2 예측</span>
</div>
<div id="progress">진행 <span id="prog">0/0</span>
  <span class="stat" id="stats"></span>
</div>
<div id="groupstats"></div>
<div class="grid" id="grid"></div>
<script>
let cards = [];
let LEGEND = {directions:[], approaches:[]};
async function loadAll(){
  const lg = await (await fetch('/api/legend')).json();
  LEGEND = lg;
  const legendDiv = document.getElementById('legend');
  let html = '<span>초록선=OWL bbox(참고) · 빨강선=Florence-2 예측</span>';
  html += '<span class="lg-group"><b>목표:</b>' + lg.directions.map(d =>
    `<span class="swatch" style="background:${d.color}"></span>${d.label}`).join(' ') + '</span>';
  html += '<span class="lg-group"><b>접근:</b>' + lg.approaches.map(a =>
    `<span class="swatch" style="background:${a.color}"></span>${a.label}`).join(' ') + '</span>';
  legendDiv.innerHTML = html;
  const res = await fetch('/api/frames');
  cards = await res.json();
  render();
}
function render(){
  // 최초 1회만 그리드 DOM을 만든다 — 이후 setLabel은 해당 카드만 patch한다
  // (전체 innerHTML 재생성을 하면 <img>가 새로 생성돼 썸네일을 매번 다시 불러오는
  // 문제가 있었음. 클릭 반응성 문제의 원인이었음).
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  cards.forEach((c, i) => {
    const div = document.createElement('div');
    div.id = `card-${i}`;
    div.className = 'card';
    div.style.borderLeftColor = c.direction_color;
    div.innerHTML = `
      <div class="imgwrap">
        <img id="img-${i}" src="" loading="lazy">
        <div class="vline gt" id="gtline-${i}" style="display:none;border-left:2px solid #22c55e">
          <span class="tag-label" style="color:#22c55e">OWLv2</span></div>
        <div class="vline pred" id="predline-${i}" style="display:none;border-left:2px solid #ef4444">
          <span class="tag-label" style="color:#ef4444">Flo2</span></div>
      </div>
      <div class="card-header">#${i} ${c.stem} f${c.frame_idx} · bin=${c.bin} · owl=${c.owl_ok}<br>
        <span class="tag" style="background:${c.direction_color}">${c.direction_label}</span>
        <span class="tag" style="background:${c.approach_color}">${c.approach_label}</span>
      </div>
      <div class="label-block">
        <div class="btn-group" id="btns-${i}">
          <button class="lbl-btn ok" onclick="setLabel(${i},'ok')">정확</button>
          <button class="lbl-btn ng" onclick="setLabel(${i},'ng')">오탐(위치틀림)</button>
          <button class="lbl-btn nt" onclick="setLabel(${i},'nt')">타겟없음</button>
        </div>
      </div>`;
    grid.appendChild(div);
    patchCard(i);  // 초기 라벨 상태 반영(새로고침 시 기존 라벨 표시)
  });
  updateProg();
  loadThumbs();
  updateStats();
}
function patchCard(i){
  const c = cards[i];
  document.getElementById(`card-${i}`).classList.toggle('labeled', !!c.label);
  const btns = document.getElementById(`btns-${i}`).children;
  const map = {ok:0, ng:1, nt:2};
  for (const [k, idx] of Object.entries(map)){
    btns[idx].classList.toggle('active', c.label === k);
  }
}
function updateProg(){
  const labeled = cards.filter(c => c.label).length;
  document.getElementById('prog').innerText = `${labeled}/${cards.length}`;
}
async function loadThumbs(){
  // 순차 로딩 유지(모델이 순차 추론이라 동시 요청해도 이득 없음), 이미 로드된 건 건너뜀
  for (let i = 0; i < cards.length; i++){
    const im = document.getElementById(`img-${i}`);
    if (!im || im.src.startsWith('data:')) continue;
    const res = await fetch(`/api/card?i=${i}`);
    const d = await res.json();
    if (im) im.src = d.image;
    const gt = document.getElementById(`gtline-${i}`);
    const pr = document.getElementById(`predline-${i}`);
    if (gt){ gt.style.left = `${d.gt_cx*100}%`; gt.style.display = 'block'; }
    if (pr && d.pred_cx !== null){ pr.style.left = `${d.pred_cx*100}%`; pr.style.display = 'block'; }
  }
}
async function setLabel(i, lbl){
  cards[i].label = lbl;
  patchCard(i);       // 이 카드만 즉시 갱신 — 전체 재로딩 없음
  updateProg();
  await fetch(`/api/label?i=${i}&lbl=${lbl}`);
  updateStats();       // 통계만 갱신, 그리드/썸네일은 안 건드림
}
function groupTable(title, obj, colorMap, labelKey){
  let rows = Object.entries(obj).map(([k,v]) => {
    const item = (colorMap||[]).find(x => x.label === k);
    const sw = item ? `<span class="swatch" style="background:${item.color}"></span>` : '';
    return `<tr><td>${sw}${k}</td><td>${v}</td></tr>`;
  }).join('');
  return `<div><b>${title}</b><table>${rows || '<tr><td>-</td></tr>'}</table></div>`;
}
async function updateStats(){
  const res = await fetch('/api/stats');
  const s = await res.json();
  document.getElementById('stats').innerText =
    `정확 ${s.ok} · 오탐(위치틀림) ${s.ng} · 타겟없음 ${s.nt} · ` +
    (s.n_judged ? `정밀도(타겟있음 기준) ${s.precision}` : '');
  document.getElementById('groupstats').innerHTML =
    groupTable('목표별 정확도(정확/판정)', s.by_direction, LEGEND.directions) +
    groupTable('접근방식별 정확도', s.by_approach, LEGEND.approaches);
}
loadAll();
</script>
</body></html>
"""


@app.route("/")
def index():
    return PAGE


@app.route("/api/frames")
def api_frames():
    labels = load_labels()
    out = []
    for i, (fidx, bin_, owl_ok) in enumerate(SAMPLE):
        fr = FRAMES[fidx]
        key = f"{fr['stem']}|{fr['frame_idx']}"
        out.append(dict(stem=fr["stem"], frame_idx=fr["frame_idx"], bin=bin_, owl_ok=owl_ok,
                        direction=fr["direction"], approach=fr["approach"],
                        direction_label=DIRECTION_LABEL[fr["direction"]],
                        direction_color=DIRECTION_COLOR[fr["direction"]],
                        approach_label=APPROACH_LABEL[fr["approach"]],
                        approach_color=APPROACH_COLOR[fr["approach"]],
                        label=labels.get(key, {}).get("label")))
    return jsonify(out)


@app.route("/api/legend")
def api_legend():
    return jsonify(
        directions=[dict(key=k, label=DIRECTION_LABEL[k], color=v) for k, v in DIRECTION_COLOR.items()],
        approaches=[dict(key=k, label=APPROACH_LABEL[k], color=v) for k, v in APPROACH_COLOR.items()],
    )


@app.route("/api/card")
def api_card():
    i = int(request.args.get("i", 0))
    i = max(0, min(i, len(SAMPLE) - 1))
    data = render_card(i)
    return jsonify(image=data["image"], gt_cx=data["gt_cx"], pred_cx=data["pred_cx"])


@app.route("/api/label")
def api_label():
    i = int(request.args.get("i", 0))
    lbl = request.args.get("lbl")
    fidx, bin_, owl_ok = SAMPLE[i]
    fr = FRAMES[fidx]
    data = render_card(i)
    key = f"{fr['stem']}|{fr['frame_idx']}"
    save_label(key, dict(label=lbl, bin=bin_, owl_ok=owl_ok, gt_cx=fr["gt_cx"], pred_cx=data["pred_cx"],
                         direction=fr["direction"], approach=fr["approach"]))
    return jsonify(ok=True)


@app.route("/api/stats")
def api_stats():
    labels = load_labels()
    ok = sum(1 for v in labels.values() if v["label"] == "ok")
    ng = sum(1 for v in labels.values() if v["label"] == "ng")
    nt = sum(1 for v in labels.values() if v["label"] == "nt")
    n_judged = ok + ng

    def group_precision(key):
        out = {}
        for v in labels.values():
            g = v.get(key, "?")
            out.setdefault(g, {"ok": 0, "ng": 0})
            if v["label"] == "ok":
                out[g]["ok"] += 1
            elif v["label"] == "ng":
                out[g]["ng"] += 1
        return {g: f"{c['ok']}/{c['ok']+c['ng']}" + (f" ({c['ok']/(c['ok']+c['ng'])*100:.0f}%)" if c['ok']+c['ng'] else "")
                for g, c in out.items() if c["ok"] + c["ng"] > 0}

    # 층화 가중 추정 — 표본은 칸(bin×owl_ok)당 균등(20개)이지만 모집단 비율은 다르다.
    # 각 칸의 표본 정확도를 그 칸의 모집단 비중으로 가중해 전체 추정치를 왜곡 없이 낸다.
    cell_stat = {}
    for v in labels.values():
        key = (v.get("bin"), v.get("owl_ok"))
        cell_stat.setdefault(key, {"ok": 0, "ng": 0, "nt": 0})
        if v["label"] in ("ok", "ng", "nt"):
            cell_stat[key][v["label"]] += 1

    total_pop = sum(CELL_POP.values())
    weighted_target_present_rate = 0.0  # 타겟이 실제 존재하는 비율(가중)
    weighted_precision_num, weighted_precision_den = 0.0, 0.0
    cells_ready = []
    for key, pop in CELL_POP.items():
        w = pop / total_pop
        cs = cell_stat.get(key, {"ok": 0, "ng": 0, "nt": 0})
        n_cell = cs["ok"] + cs["ng"] + cs["nt"]
        if n_cell == 0:
            continue
        present_rate = (cs["ok"] + cs["ng"]) / n_cell
        weighted_target_present_rate += w * present_rate
        weighted_precision_num += w * cs["ok"]
        weighted_precision_den += w * (cs["ok"] + cs["ng"])
        cells_ready.append(key)

    weighted_precision = (weighted_precision_num / weighted_precision_den
                           if weighted_precision_den else None)
    coverage_note = f"{len(cells_ready)}/{len(CELL_POP)}칸 라벨 있음"

    return jsonify(ok=ok, ng=ng, nt=nt, n_judged=n_judged,
                   precision=f"{ok/n_judged*100:.1f}%" if n_judged else "N/A",
                   by_direction={DIRECTION_LABEL.get(k, k): v for k, v in group_precision("direction").items()},
                   by_approach={APPROACH_LABEL.get(k, k): v for k, v in group_precision("approach").items()},
                   weighted_precision=(f"{weighted_precision*100:.1f}%" if weighted_precision is not None else "N/A"),
                   weighted_target_present_rate=f"{weighted_target_present_rate*100:.1f}%",
                   weighted_coverage=coverage_note)


if __name__ == "__main__":
    print(f"브라우저 → http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
