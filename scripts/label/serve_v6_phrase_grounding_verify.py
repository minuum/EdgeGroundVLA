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
# 정탐 94.9%)이라 소량 샌티티체크, fail은 완전 미검증 영역이라 있는 대로 최대한
# — 사용자 요청(2026-08-20 재배분, 2026-08-24 표본 확대)에 따름.
# fail 쪽은 이전엔 에피소드당 1프레임만 뽑았는데(67개 = 에피소드 수만큼),
# 같은 에피소드 안에서도 여러 프레임을 더 뽑아 표본을 키운다(최대 FRAMES_PER_EP_FAIL개/에피소드).
N_PER_CELL_SUCC = 30
FRAMES_PER_EP_FAIL = 3
FAIL_TAKE_ALL = True


def build_sample():
    """6칸(bin × owl_ok) — succ은 칸당 N_PER_CELL_SUCC개 스팟체크, fail은 에피소드당
    최대 FRAMES_PER_EP_FAIL개까지. 칸 안에서는 (목표 5 × 접근 3) 15조합을 라운드로빈으로
    순회해 최대한 골고루 뽑는다(같은 조합/에피소드에 몰리지 않게)."""
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
        for combo in combos:
            rng.shuffle(by_combo[combo])  # 각 에피소드 안에서도 무작위 순서로 소비
        by_da = {}
        for combo in combos:
            da = (combo[0], combo[1])
            by_da.setdefault(da, []).append(combo)
        da_order = list(by_da.keys())
        rng.shuffle(da_order)

        per_combo_cap = FRAMES_PER_EP_FAIL if key[1] == "fail" else 1
        per_combo_used = {c: 0 for c in combos}
        total_available = sum(min(len(by_combo[c]), per_combo_cap) for c in combos)
        target_n = total_available if (key[1] == "fail" and FAIL_TAKE_ALL) else min(N_PER_CELL_SUCC, total_available)

        picked = []
        while len(picked) < target_n:
            progressed = False
            for da in da_order:
                if len(picked) >= target_n:
                    break
                for combo in by_da[da]:
                    cand = by_combo[combo]
                    used = per_combo_used[combo]
                    if used >= per_combo_cap or used >= len(cand):
                        continue
                    picked.append(cand[used])
                    per_combo_used[combo] += 1
                    progressed = True
                    break
            if not progressed:
                break
        sample.extend([(i, key[0], key[1]) for i in picked])
    rng.shuffle(sample)
    return sample


SAMPLE = build_sample()
_n_succ = sum(1 for _, _, ok in SAMPLE if ok == "succ")
_n_fail = sum(1 for _, _, ok in SAMPLE if ok == "fail")
print(f"검증 표본 {len(SAMPLE)}개 구성 완료 (succ {_n_succ}개 스팟체크 + fail {_n_fail}개 전수)", flush=True)

# build_v6_verification_dataset.py가 미리 계산해둔 합의추정/불일치 분류 — "불일치만
# 보기" 필터에 쓴다. 없으면(스크립트 실행 전) 전부 needs_human=True로 취급(필터 무의미해짐).
VERIF_STATUS = {}
_VERIF_PATH = ROOT / "docs/v5/detector/v6_verification_dataset.json"
if _VERIF_PATH.exists():
    for row in json.loads(_VERIF_PATH.read_text()):
        VERIF_STATUS[row["key"]] = row["status"]
    print(f"검증셋 상태 {len(VERIF_STATUS)}개 로드 완료 "
          f"(needs_human={sum(1 for v in VERIF_STATUS.values() if v=='needs_human')}개)", flush=True)

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


def save_label(key, patch, base_meta):
    labels = load_labels()
    cur = labels.get(key, dict(owlv2=None, florence2=None, no_target=False))
    cur.update(patch)
    cur.update(base_meta)
    labels[key] = cur
    LABELS_PATH.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    return cur


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
.card.selected{outline:3px solid #fff !important;outline-offset:2px;box-shadow:0 0 24px 4px #38bdf8aa;
  transform:scale(1.02);z-index:5;position:relative;transition:transform 0.1s}
.kbd-help{font-size:0.72rem;color:#64748b;margin:6px 0 12px}
.kbd-help b{color:#94a3b8;border:1px solid #334155;border-radius:3px;padding:0 4px;font-size:0.68rem}
.imgwrap{position:relative;width:100%;line-height:0;cursor:crosshair}
.card img{width:100%;display:block}
.vline{position:absolute;top:0;bottom:0;width:2px;pointer-events:none}
.vline .tag-label{position:absolute;top:2px;left:4px;font-size:0.68rem;font-weight:bold;
  white-space:nowrap;text-shadow:-1px -1px 0 #000,1px -1px 0 #000,-1px 1px 0 #000,1px 1px 0 #000}
.vline.pred .tag-label{top:16px}
.vline.truth{border-left:3px solid #38bdf8 !important}
.vline.truth .tag-label{top:30px}
.card-header{padding:4px 8px;font-size:0.65rem;color:#94a3b8;background:#111827;line-height:1.5}
.tag{display:inline-block;padding:0 6px;border-radius:4px;font-size:0.68rem;font-weight:bold;color:#0a0f1a;margin-right:4px}
.label-block{padding:6px 8px;display:flex;flex-direction:column;gap:5px}
.row{display:flex;align-items:center;gap:8px;font-size:0.72rem}
.row .mname{flex:1;color:#94a3b8}
.btn-group{display:flex;gap:3px}
.lbl-btn{font-size:0.72rem;padding:3px 9px;border-radius:4px;border:1px solid #334155;
  background:#1e293b;color:#94a3b8;cursor:pointer;font-weight:bold}
.lbl-btn.ok.active{background:#22c55e;color:#fff;border-color:#22c55e}
.lbl-btn.ng.active{background:#ef4444;color:#fff;border-color:#ef4444}
.lbl-btn.nt.active{background:#f97316;color:#fff;border-color:#f97316}
</style></head><body>
<h1>V6 학습셋(16599프레임) — OWLv2 vs Florence-2 그라운딩 사람 검증</h1>
<div id="legend">
  <span>초록선=OWLv2 · 빨강선=Florence-2("gray basket")</span>
</div>
<div class="kbd-help">
  <b>←→↑↓</b> 카드 이동 · <b>1</b> OWLv2 정오 순환(미판정→O→X→O…) · <b>2</b> Florence-2 정오 순환 ·
  <b>0</b> 타겟없음(토글, 이 프레임 완료 처리) · OWL 실패 칸은 OWLv2가 자동 X 처리되어 있어 2번만 누르면 됨 ·
  1(succ 칸)/2 모두 정해지거나 0을 누르면 자동으로 다음 카드로 이동 ·
  <b style="color:#38bdf8">사진에서 바구니 위치를 직접 클릭</b>하면 파란선(진짜 정답)이 찍힘 — O/X 버튼과 무관한 독립 검증 기준
</div>
<div style="margin-bottom:10px">
  <button id="filter-btn" class="lbl-btn" style="font-size:0.8rem;padding:6px 14px" onclick="toggleFilter()">불일치만 보기 OFF</button>
  <span style="font-size:0.75rem;color:#64748b;margin-left:8px">OWL·Florence가 서로 근접(합의추정)한 카드는 자동으로 숨겨서, 진짜 판단이 필요한
    카드만 빠르게 훑을 수 있습니다 (build_v6_verification_dataset.py 결과 기반)</span>
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
      <div class="imgwrap" id="imgwrap-${i}">
        <img id="img-${i}" src="" loading="lazy">
        <div class="vline gt" id="gtline-${i}" style="display:none;border-left:2px solid #22c55e">
          <span class="tag-label" style="color:#22c55e">OWLv2</span></div>
        <div class="vline pred" id="predline-${i}" style="display:none;border-left:2px solid #ef4444">
          <span class="tag-label" style="color:#ef4444">Flo2</span></div>
        <div class="vline truth" id="truthline-${i}" style="display:none">
          <span class="tag-label" style="color:#38bdf8">정답(클릭)</span></div>
      </div>
      <div class="card-header">#${i} ${c.stem} f${c.frame_idx} · bin=${c.bin} · owl=${c.owl_ok}<br>
        <span class="tag" style="background:${c.direction_color}">${c.direction_label}</span>
        <span class="tag" style="background:${c.approach_color}">${c.approach_label}</span>
        <span class="tag" style="background:${c.verif_status==='needs_human'?'#f87171':'#4ade80'}">
          ${c.verif_status==='needs_human'?'판단필요':'합의추정'}</span>
      </div>
      <div class="label-block">
        <div class="row"><span class="mname">1:OWLv2</span>
          <div class="btn-group" id="owlv2-${i}">
            <button class="lbl-btn ok" onclick="cycleModel(${i},'owlv2')">O</button>
            <button class="lbl-btn ng" onclick="cycleModel(${i},'owlv2')">X</button>
          </div></div>
        <div class="row"><span class="mname">2:Flo2</span>
          <div class="btn-group" id="florence2-${i}">
            <button class="lbl-btn ok" onclick="cycleModel(${i},'florence2')">O</button>
            <button class="lbl-btn ng" onclick="cycleModel(${i},'florence2')">X</button>
          </div></div>
        <div class="row"><span class="mname">0:타겟없음</span>
          <button class="lbl-btn nt" id="nt-${i}" onclick="toggleNoTarget(${i})">토글</button>
        </div>
        <div class="row"><span class="mname" style="color:#38bdf8">사진 클릭=진짜 위치 찍기</span></div>
      </div>`;
    div.addEventListener('click', (e) => {
      if (e.target.classList.contains('lbl-btn')) return;
      selectCard(i);
    });
    const imgwrap = div.querySelector(`#imgwrap-${i}`);
    imgwrap.addEventListener('click', (e) => {
      e.stopPropagation();
      selectCard(i);
      const rect = imgwrap.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      markTruth(i, Math.max(0, Math.min(1, x)));
    });
    grid.appendChild(div);
    patchCard(i);  // 초기 라벨 상태 반영(새로고침 시 기존 라벨 표시)
  });
  computeVisible();
  updateProg();
  loadThumbs();
  updateStats();
  const firstUnlabeled = cards.findIndex(c => !isComplete(c));
  selectCard(firstUnlabeled >= 0 ? firstUnlabeled : (visibleIdx[0] ?? 0));
}
function isComplete(c){
  return c.no_target || (!!c.owlv2 && !!c.florence2);
}
function patchCard(i){
  const c = cards[i];
  document.getElementById(`card-${i}`).classList.toggle('labeled', isComplete(c));
  const owlBtns = document.getElementById(`owlv2-${i}`).children;
  owlBtns[0].classList.toggle('active', c.owlv2 === 'ok');
  owlBtns[1].classList.toggle('active', c.owlv2 === 'ng');
  const floBtns = document.getElementById(`florence2-${i}`).children;
  floBtns[0].classList.toggle('active', c.florence2 === 'ok');
  floBtns[1].classList.toggle('active', c.florence2 === 'ng');
  document.getElementById(`nt-${i}`).classList.toggle('active', !!c.no_target);
  const tl = document.getElementById(`truthline-${i}`);
  if (tl){
    if (c.true_cx !== null && c.true_cx !== undefined){
      tl.style.left = `${c.true_cx*100}%`; tl.style.display = 'block';
    } else {
      tl.style.display = 'none';
    }
  }
}
async function markTruth(i, x){
  cards[i].true_cx = x;
  patchCard(i);
  await fetch(`/api/label?i=${i}&field=true_cx&val=${x}`);
}
function updateProg(){
  const scope = filterOnlyDisagree ? visibleIdx.map(i => cards[i]) : cards;
  const labeled = scope.filter(isComplete).length;
  document.getElementById('prog').innerText =
    `${labeled}/${scope.length}` + (filterOnlyDisagree ? ` (불일치만, 전체 ${cards.length})` : '');
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
function cycleValue(cur){
  // 미판정 → O → X → O → ... (원본 라벨러와 동일한 순환)
  return cur === 'ok' ? 'ng' : 'ok';
}
async function cycleModel(i, field){
  const c = cards[i];
  c[field] = cycleValue(c[field]);
  patchCard(i);
  updateProg();
  await fetch(`/api/label?i=${i}&field=${field}&val=${c[field]}`);
  updateStats();
  if (isComplete(c) && selIdx === i && i < cards.length - 1) selectCard(i + 1);
}
async function toggleNoTarget(i){
  const c = cards[i];
  c.no_target = !c.no_target;
  patchCard(i);
  updateProg();
  await fetch(`/api/label?i=${i}&field=no_target&val=${c.no_target}`);
  updateStats();
  if (c.no_target && selIdx === i && i < cards.length - 1) selectCard(i + 1);
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
    `OWLv2 정확도 ${s.owlv2_precision}(${s.owlv2_n}건) · Florence-2 정확도 ${s.florence2_precision}(${s.florence2_n}건) · ` +
    `타겟없음 확인 ${s.no_target}건 · 가중추정(전체 V6 기준) OWLv2 ${s.weighted_owlv2} / Florence-2 ${s.weighted_florence2} · ` +
    `[독립검증: 사진 클릭한 ${s.n_truthed}건 기준] OWLv2 ${s.owl_vs_truth} / Florence-2 ${s.florence2_vs_truth}`;
  document.getElementById('groupstats').innerHTML =
    groupTable('목표별 Florence-2 정확도', s.by_direction, LEGEND.directions) +
    groupTable('접근방식별 Florence-2 정확도', s.by_approach, LEGEND.approaches);
  updateProg();
}

// ── 키보드 라벨링 (serve_hsv_owlv2_labeler.py 알고리즘 이식) ──────────────
// ←→↑↓: 카드 이동(격자 열수 자동 계산) / 1: OWLv2 순환(O↔X) / 2: Florence-2 순환 /
// 0: 타겟없음 토글. 두 모델 다 판정되거나 0을 누르면 자동으로 다음 카드.
// 카드 클릭으로도 커서 이동 가능.
let selIdx = -1;          // 실제 카드 인덱스(cycleModel 등에 그대로 씀)
let filterOnlyDisagree = false;
let visibleIdx = [];      // 현재 필터 통과한 카드 인덱스 목록(순서대로) — 방향키는 이 안에서만 이동

function computeVisible(){
  visibleIdx = cards.map((c, i) => i).filter(i =>
    !filterOnlyDisagree || cards[i].verif_status === 'needs_human');
}
function applyFilter(){
  computeVisible();
  cards.forEach((c, i) => {
    const el = document.getElementById(`card-${i}`);
    if (el) el.style.display = visibleIdx.includes(i) ? '' : 'none';
  });
  const btn = document.getElementById('filter-btn');
  if (btn){
    btn.textContent = filterOnlyDisagree
      ? `불일치만 보기 ON (${visibleIdx.length}개)` : `불일치만 보기 OFF (전체 ${cards.length}개)`;
    btn.classList.toggle('active', filterOnlyDisagree);
  }
  if (!visibleIdx.includes(selIdx)) selectCard(visibleIdx[0] ?? 0);
  updateProg();
}
function toggleFilter(){
  filterOnlyDisagree = !filterOnlyDisagree;
  applyFilter();
}
function clearCursor(){
  const grid = document.getElementById('grid');
  grid.querySelectorAll('.card.selected').forEach(c => c.classList.remove('selected'));
}
function paintCursor(){
  if (selIdx < 0 || selIdx >= cards.length) return;
  const card = document.getElementById(`card-${selIdx}`);
  if (!card) return;
  card.classList.add('selected');
  card.scrollIntoView({block: 'center'});
}
function selectCard(i){
  clearCursor();
  if (!visibleIdx.length) { selIdx = -1; return; }
  // i가 visibleIdx 안에 없으면 가장 가까운 걸로 스냅
  if (!visibleIdx.includes(i)){
    i = visibleIdx.reduce((a, b) => Math.abs(b - i) < Math.abs(a - i) ? b : a);
  }
  selIdx = i;
  paintCursor();
}
function moveByPos(delta){
  if (!visibleIdx.length) return;
  const pos = visibleIdx.indexOf(selIdx);
  const nextPos = Math.max(0, Math.min(visibleIdx.length - 1, (pos < 0 ? 0 : pos) + delta));
  selectCard(visibleIdx[nextPos]);
}
function colsInGrid(){
  const grid = document.getElementById('grid');
  if (!grid || !cards.length) return 1;
  const style = getComputedStyle(grid);
  return style.gridTemplateColumns.split(' ').length || 1;
}
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (!cards.length) return;
  const cols = colsInGrid();
  if (e.key === '1' || e.key === '2') {
    e.preventDefault();
    if (selIdx < 0) { selectCard(visibleIdx[0] ?? 0); return; }
    cycleModel(selIdx, e.key === '1' ? 'owlv2' : 'florence2');
    return;
  }
  if (e.key === '0') {
    e.preventDefault();
    if (selIdx < 0) { selectCard(visibleIdx[0] ?? 0); return; }
    toggleNoTarget(selIdx);
    return;
  }
  switch (e.key) {
    case 'ArrowRight': e.preventDefault(); moveByPos(1); break;
    case 'ArrowLeft':  e.preventDefault(); moveByPos(-1); break;
    case 'ArrowDown':  e.preventDefault(); moveByPos(cols); break;
    case 'ArrowUp':    e.preventDefault(); moveByPos(-cols); break;
  }
});
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
        saved = labels.get(key, {})
        # OWL이 이미 실패한 걸로 알려진 칸은 재판정할 필요 없이 자동 X — 사용자가
        # 명시적으로 다시 판정하면(저장값 존재) 그 값을 우선한다.
        default_owlv2 = "ng" if owl_ok == "fail" else None
        out.append(dict(stem=fr["stem"], frame_idx=fr["frame_idx"], bin=bin_, owl_ok=owl_ok,
                        direction=fr["direction"], approach=fr["approach"],
                        direction_label=DIRECTION_LABEL[fr["direction"]],
                        direction_color=DIRECTION_COLOR[fr["direction"]],
                        approach_label=APPROACH_LABEL[fr["approach"]],
                        approach_color=APPROACH_COLOR[fr["approach"]],
                        owlv2=saved.get("owlv2", default_owlv2),
                        florence2=saved.get("florence2"),
                        no_target=saved.get("no_target", False),
                        true_cx=saved.get("true_cx"),
                        verif_status=VERIF_STATUS.get(key, "needs_human")))
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
    field = request.args.get("field")  # owlv2 | florence2 | no_target | true_cx
    val = request.args.get("val")
    if field == "no_target":
        val = val == "true"
    elif field == "true_cx":
        val = float(val)
    fidx, bin_, owl_ok = SAMPLE[i]
    fr = FRAMES[fidx]
    data = render_card(i)
    key = f"{fr['stem']}|{fr['frame_idx']}"
    save_label(key, {field: val},
               dict(bin=bin_, owl_ok=owl_ok, gt_cx=fr["gt_cx"], pred_cx=data["pred_cx"],
                    direction=fr["direction"], approach=fr["approach"]))
    return jsonify(ok=True)


def _effective_owlv2(v):
    """fail 칸은 명시적으로 다시 판정 안 했으면 자동 X로 간주(디폴트)."""
    ov = v.get("owlv2")
    if ov is None and v.get("owl_ok") == "fail":
        return "ng"
    return ov


def _fmt_frac(ok, n):
    return f"{ok}/{n}" + (f" ({ok/n*100:.0f}%)" if n else "")


@app.route("/api/stats")
def api_stats():
    labels = load_labels()
    no_target = sum(1 for v in labels.values() if v.get("no_target"))

    judged = [v for v in labels.values() if not v.get("no_target")]
    owlv2_vals = [_effective_owlv2(v) for v in judged]
    owlv2_ok = sum(1 for x in owlv2_vals if x == "ok")
    owlv2_n = sum(1 for x in owlv2_vals if x in ("ok", "ng"))
    florence2_ok = sum(1 for v in judged if v.get("florence2") == "ok")
    florence2_n = sum(1 for v in judged if v.get("florence2") in ("ok", "ng"))

    def group_precision(key):
        out = {}
        for v in judged:
            if v.get("florence2") not in ("ok", "ng"):
                continue
            g = v.get(key, "?")
            out.setdefault(g, {"ok": 0, "n": 0})
            out[g]["n"] += 1
            if v["florence2"] == "ok":
                out[g]["ok"] += 1
        return {g: _fmt_frac(c["ok"], c["n"]) for g, c in out.items() if c["n"] > 0}

    # 층화 가중 추정 — 표본이 칸(bin×owl_ok)마다 크기가 다르다(succ는 소량 스팟체크,
    # fail은 전수). 각 칸의 표본 정확도를 그 칸의 실제 V6 모집단 비중으로 가중해야
    # 전체 추정치가 succ/fail 어느 한쪽으로 쏠리지 않는다.
    cell_stat = {}
    for v in judged:
        key = (v.get("bin"), v.get("owl_ok"))
        cell_stat.setdefault(key, {"owlv2_ok": 0, "owlv2_n": 0, "flo_ok": 0, "flo_n": 0})
        ov = _effective_owlv2(v)
        if ov in ("ok", "ng"):
            cell_stat[key]["owlv2_n"] += 1
            if ov == "ok":
                cell_stat[key]["owlv2_ok"] += 1
        if v.get("florence2") in ("ok", "ng"):
            cell_stat[key]["flo_n"] += 1
            if v["florence2"] == "ok":
                cell_stat[key]["flo_ok"] += 1

    total_pop = sum(CELL_POP.values())
    w_owl_num = w_owl_den = w_flo_num = w_flo_den = 0.0
    for key, pop in CELL_POP.items():
        w = pop / total_pop
        cs = cell_stat.get(key)
        if not cs:
            continue
        w_owl_num += w * cs["owlv2_ok"]; w_owl_den += w * cs["owlv2_n"]
        w_flo_num += w * cs["flo_ok"]; w_flo_den += w * cs["flo_n"]

    weighted_owlv2 = f"{w_owl_num/w_owl_den*100:.1f}%" if w_owl_den else "N/A"
    weighted_florence2 = f"{w_flo_num/w_flo_den*100:.1f}%" if w_flo_den else "N/A"

    # 독립 검증 기준 — 사람이 클릭으로 직접 찍은 true_cx를 기준으로, OWL의 gt_cx와
    # Florence의 pred_cx가 맞는지 "O/X 버튼 판정과 무관하게" 재계산한다. gt_cx/pred_cx
    # 자체를 기준으로 삼는 순환논리 우려를 완전히 해소하는 유일한 지표.
    HIT_TOL = 0.05
    truthed = [v for v in labels.values() if v.get("true_cx") is not None]
    owl_vs_truth_ok = sum(1 for v in truthed if abs(v["gt_cx"] - v["true_cx"]) <= HIT_TOL)
    flo_vs_truth_ok = sum(1 for v in truthed
                          if v.get("pred_cx") is not None and abs(v["pred_cx"] - v["true_cx"]) <= HIT_TOL)

    return jsonify(
        no_target=no_target,
        owlv2_precision=_fmt_frac(owlv2_ok, owlv2_n), owlv2_n=owlv2_n,
        florence2_precision=_fmt_frac(florence2_ok, florence2_n), florence2_n=florence2_n,
        by_direction={DIRECTION_LABEL.get(k, k): v for k, v in group_precision("direction").items()},
        by_approach={APPROACH_LABEL.get(k, k): v for k, v in group_precision("approach").items()},
        weighted_owlv2=weighted_owlv2,
        weighted_florence2=weighted_florence2,
        n_truthed=len(truthed),
        owl_vs_truth=_fmt_frac(owl_vs_truth_ok, len(truthed)),
        florence2_vs_truth=_fmt_frac(flo_vs_truth_ok, len(truthed)),
    )


if __name__ == "__main__":
    print(f"브라우저 → http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
