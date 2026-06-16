#!/usr/bin/env python3
"""
Backbone Recognition Eval — 8-combo 백본 × 태스크 비교

모든 백본에서 3가지를 측정:
  1. Grounding  : bbox IoU (해당 백본이 지원하는 경우)
  2. VQA        : "Is there a basket?" → yes/no
  3. Caption    : 자유 형식 설명 → basket 언급 여부

모든 프레임의 raw 텍스트 출력을 details.json에 저장.

Usage:
  .venv/bin/python3 scripts/eval_backbone_recognition.py --smoke     # 백본별 1프레임 smoke test
  .venv/bin/python3 scripts/eval_backbone_recognition.py --combos C1 # 특정 콤보만
  .venv/bin/python3 scripts/eval_backbone_recognition.py             # 전체 실행
  .venv/bin/python3 scripts/eval_backbone_recognition.py --frames-per-ep 2
"""
import sys, json, time, argparse, gc, warnings, re, traceback
import numpy as np
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import h5py
from PIL import Image

# ─── 경로 ─────────────────────────────────────────────────────────────────────
KOSMOS_PATH     = ROOT / ".vlms" / "kosmos-2-patch14-224"
KOSMOS_LORA_DIR = ROOT / "runs" / "v5_nav" / "mlp" / "exp55" / "stage1_lora" / "lora_adapter"
GOOGLE_ROBOT_PT = ROOT / ".vlms" / "google_robot_pretrain" / "kosmos_ph_google-robot-post-train.pt"
PALI_PT_PATH    = Path.home() / ".cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
PALI_LORA_DIR   = ROOT / "runs" / "v5_nav" / "grounding" / "exp57"
MOONDREAM_PATH  = Path.home() / ".cache/huggingface/hub/models--vikhyatk--moondream2/snapshots/6b714b26eea5cbd9f31e4edb2541c170afa935ba"

def _pali2_path():
    base = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots"
    snaps = sorted(base.iterdir()) if base.exists() else []
    return snaps[-1] if snaps else None

PALI2_PATH = _pali2_path()

DATA_EXP46   = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
DATA_EXP55   = ROOT / "docs" / "v5" / "bbox_nav_exp55" / "bbox_dataset_free.json"
OUT_DIR      = ROOT / "docs" / "v5" / "backbone_recognition"
PROGRESS_JSON = OUT_DIR / "progress.json"
RESULTS_JSON  = OUT_DIR / "results.json"
DETAILS_DIR   = OUT_DIR / "details"

IOU_THR = 0.3
BASKET_WORDS = {"basket", "bin", "container", "box", "trash", "laundry", "tub", "bucket", "hamper"}
LOC_RE = re.compile(r"<loc(\d{4})>")

# ─── 콤보 정의 ────────────────────────────────────────────────────────────────
# tasks: grounding / vqa / caption (각 백본이 지원하는 것)
COMBOS = [
    {"id": "C1", "backbone": "kosmos-base",    "label": "Kosmos-2 base",
     "tasks": ["grounding", "vqa", "caption"]},
    {"id": "C2", "backbone": "kosmos-lora",    "label": "Kosmos-2+exp55LoRA",
     "tasks": ["grounding", "vqa", "caption"]},
    {"id": "C3", "backbone": "kosmos-google",  "label": "Google-robot (neg.)",
     "tasks": ["grounding", "vqa", "caption"]},  # expected: all fail
    {"id": "C4", "backbone": "paligemma-base", "label": "PaliGemma-3B-pt",
     "tasks": ["grounding"]},                    # pt: detect prompt만 지원
    {"id": "C5", "backbone": "paligemma-lora", "label": "PaliGemma+exp57LoRA",
     "tasks": ["grounding"]},
    {"id": "C6", "backbone": "paligemma2-mix", "label": "PaliGemma2-mix",
     "tasks": ["vqa", "caption"]},               # SFT: VQA+캡션
    {"id": "C7", "backbone": "moondream",      "label": "MoonDream2",
     "tasks": ["grounding", "vqa", "caption"]},  # 모든 태스크 지원
]

# ─── 데이터 ──────────────────────────────────────────────────────────────────
def load_frames(json_path, frames_per_ep, seed=42):
    data = json.loads(Path(json_path).read_text())
    rng  = np.random.RandomState(seed)
    samples = []
    for ep in data:
        frames = ep.get("frames", [])
        if not frames: continue
        det = [f for f in frames if f.get("has_bbox", True)]
        pool = det if det else frames
        n = min(frames_per_ep, len(pool))
        for idx in rng.choice(len(pool), n, replace=False):
            fr = pool[idx]
            samples.append({
                "episode":   ep["episode"],
                "path_type": ep["path_type"],
                "frame_idx": fr["frame_idx"],
                "cx":    fr.get("cx",  fr.get("cx_det",  0.5)),
                "cy":    fr.get("cy",  fr.get("cy_det",  0.5)),
                "area":  fr.get("area",fr.get("area_det",0.05)),
                "has_bbox": fr.get("has_bbox", True),
            })
    return samples


def load_image(ep_path, frame_idx):
    try:
        with h5py.File(ep_path, "r") as f:
            return Image.fromarray(f["observations"]["images"][frame_idx].astype("uint8")).convert("RGB")
    except Exception:
        return None


def compute_iou(box, cx, cy, area):
    """box: [x1,y1,x2,y2] normalized 0-1"""
    s = area ** 0.5
    gx1,gy1,gx2,gy2 = cx-s/2, cy-s/2, cx+s/2, cy+s/2
    x1,y1,x2,y2 = box
    ix1,iy1 = max(x1,gx1), max(y1,gy1)
    ix2,iy2 = min(x2,gx2), min(y2,gy2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    union = (x2-x1)*(y2-y1)+(gx2-gx1)*(gy2-gy1)-inter
    return inter/union if union>1e-6 else 0.0


def mentions_basket(text):
    t = text.lower()
    return any(w in t for w in BASKET_WORDS)


# ─── 모델 로더 ────────────────────────────────────────────────────────────────
def unload_all(model_objects):
    for v in model_objects.values():
        del v
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_kosmos(device, lora_dir=None, google_pt=None):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    proc  = AutoProcessor.from_pretrained(str(KOSMOS_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(KOSMOS_PATH), torch_dtype=torch.float16).to(device).eval()

    if google_pt and Path(google_pt).exists():
        ckpt = torch.load(google_pt, map_location="cpu", weights_only=False)
        sd   = ckpt.get("state_dict", ckpt)
        pfix = "model.backbone."
        new_sd = {(k[len(pfix):] if k.startswith(pfix) else k): v for k,v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
        print(f"  [LOAD] Google-robot weights applied")
    elif lora_dir and Path(lora_dir).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(lora_dir)).eval()
        print(f"  [LOAD] exp55 LoRA applied")
    return proc, model


def load_paligemma(device, lora_dir=None):
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    proc  = PaliGemmaProcessor.from_pretrained(str(PALI_PT_PATH))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PALI_PT_PATH), torch_dtype=torch.bfloat16).to(device).eval()
    if lora_dir and Path(lora_dir).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(lora_dir)).eval()
        print(f"  [LOAD] exp57 LoRA applied")
    return proc, model


def load_paligemma2(device):
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    if PALI2_PATH is None:
        raise RuntimeError("PaliGemma2 mix snapshot not found")
    proc  = PaliGemmaProcessor.from_pretrained(str(PALI2_PATH))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PALI2_PATH), torch_dtype=torch.bfloat16).to(device).eval()
    return proc, model


def load_moondream(device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(MOONDREAM_PATH), trust_remote_code=True,
        torch_dtype=torch.float16).to(device).eval()
    return model


# ─── 태스크별 단일 프레임 추론 ────────────────────────────────────────────────
def infer_kosmos(proc, model, img, s, device):
    """Kosmos-2: grounding + VQA + caption 한 번에"""
    out = {}

    # 1) Grounding
    try:
        prompt  = "<grounding> An image of gray basket."
        inputs  = proc(text=prompt, images=img, return_tensors="pt")
        inputs  = {k: v.to(device) if hasattr(v,"to") else v for k,v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        raw = proc.decode(gen[0], skip_special_tokens=False)
        _, entities = proc.post_process_generation(raw, cleanup_and_extract=True)
        boxes = []
        for ep_phrase, _, eb in entities:
            if "basket" in ep_phrase.lower() or "gray" in ep_phrase.lower():
                boxes.extend(eb)
        if not boxes:
            for _, _, eb in entities: boxes.extend(eb)
        has_box = len(boxes) > 0
        iou = 0.0
        if has_box and s["has_bbox"]:
            iou = max(compute_iou(list(b), s["cx"], s["cy"], s["area"]) for b in boxes)
        out["grounding"] = {
            "raw":     raw[:200],
            "has_bbox": has_box,
            "iou":     round(iou, 4),
            "hit":     iou >= IOU_THR,
            "n_boxes": len(boxes),
        }
    except Exception as e:
        out["grounding"] = {"raw": f"ERROR:{e}", "has_bbox": False, "iou": 0.0, "hit": False}

    # 2) VQA
    try:
        vqa_prompt = "<image> Is there a gray basket or bin in this image? Answer yes or no:"
        inputs = proc(text=vqa_prompt, images=img, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(device) if hasattr(v,"to") else v for k,v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        answer = proc.tokenizer.decode(gen[0][input_len:], skip_special_tokens=True).strip()
        out["vqa"] = {
            "prompt":  vqa_prompt,
            "answer":  answer,
            "yes":     "yes" in answer.lower()[:20],
            "mention": mentions_basket(answer),
        }
    except Exception as e:
        out["vqa"] = {"answer": f"ERROR:{e}", "yes": False, "mention": False}

    # 3) Caption
    try:
        cap_prompt = "<image> Describe what objects you see in this image:"
        inputs = proc(text=cap_prompt, images=img, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(device) if hasattr(v,"to") else v for k,v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        caption = proc.tokenizer.decode(gen[0][input_len:], skip_special_tokens=True).strip()
        out["caption"] = {
            "text":    caption,
            "mention": mentions_basket(caption),
        }
    except Exception as e:
        out["caption"] = {"text": f"ERROR:{e}", "mention": False}

    return out


def infer_paligemma_grounding(proc, model, img, s, device):
    """PaliGemma pt: grounding only (detect 프롬프트)"""
    out = {}
    try:
        inputs = proc(text="<image> detect gray basket", images=img,
                      return_tensors="pt", padding="longest").to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        raw  = proc.decode(gen[0], skip_special_tokens=True)
        vals = [int(v)/1024.0 for v in LOC_RE.findall(raw)]
        has_box = len(vals) >= 4
        iou = 0.0
        if has_box and s["has_bbox"]:
            y1,x1,y2,x2 = vals[:4]
            iou = compute_iou([x1,y1,x2,y2], s["cx"], s["cy"], s["area"])
        out["grounding"] = {
            "raw":      raw[:200],
            "has_bbox": has_box,
            "iou":      round(iou, 4),
            "hit":      iou >= IOU_THR,
            "loc_vals": vals[:4] if has_box else [],
        }
    except Exception as e:
        out["grounding"] = {"raw": f"ERROR:{e}", "has_bbox": False, "iou": 0.0, "hit": False}
    return out


def infer_paligemma2_vqa_caption(proc, model, img, s, device):
    """PaliGemma2 mix: VQA + caption"""
    out = {}

    # VQA
    try:
        vqa_prompt = "<image> Is there a basket or bin in this image?"
        inputs = proc(text=vqa_prompt, images=img,
                      return_tensors="pt", padding="longest").to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        answer = proc.decode(gen[0], skip_special_tokens=True).strip()
        # PaliGemma2 mix repeats prompt in output — strip it
        if answer.lower().startswith(vqa_prompt.replace("<image> ", "").lower()):
            answer = answer[len(vqa_prompt):].strip()
        out["vqa"] = {
            "prompt":  vqa_prompt,
            "answer":  answer,
            "yes":     "yes" in answer.lower()[:20],
            "mention": mentions_basket(answer),
        }
    except Exception as e:
        out["vqa"] = {"answer": f"ERROR:{e}", "yes": False, "mention": False}

    # Caption
    try:
        cap_prompt = "<image> caption en"
        inputs = proc(text=cap_prompt, images=img,
                      return_tensors="pt", padding="longest").to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        caption = proc.decode(gen[0], skip_special_tokens=True).strip()
        out["caption"] = {
            "text":    caption,
            "mention": mentions_basket(caption),
        }
    except Exception as e:
        out["caption"] = {"text": f"ERROR:{e}", "mention": False}

    return out


def infer_moondream(model, img, s, device):
    """MoonDream: grounding + VQA + caption"""
    out = {}

    # Grounding
    try:
        det = model.detect(img, "gray basket")
        objs = det.get("objects", [])
        has_box = len(objs) > 0
        iou = 0.0
        if has_box and s["has_bbox"]:
            for o in objs:
                b = [o["x_min"], o["y_min"], o["x_max"], o["y_max"]]
                iou = max(iou, compute_iou(b, s["cx"], s["cy"], s["area"]))
        out["grounding"] = {
            "raw":      str(objs[:2]),
            "has_bbox": has_box,
            "iou":      round(iou, 4),
            "hit":      iou >= IOU_THR,
            "n_boxes":  len(objs),
        }
    except Exception as e:
        out["grounding"] = {"raw": f"ERROR:{e}", "has_bbox": False, "iou": 0.0, "hit": False}

    # VQA
    try:
        r = model.query(img, "Is there a gray basket or bin visible in this image?")
        answer = r.get("answer", str(r)) if isinstance(r, dict) else str(r)
        out["vqa"] = {
            "answer":  answer,
            "yes":     "yes" in answer.lower()[:20],
            "mention": mentions_basket(answer),
        }
    except Exception as e:
        out["vqa"] = {"answer": f"ERROR:{e}", "yes": False, "mention": False}

    # Caption
    try:
        r = model.caption(img)
        caption = r.get("caption", str(r)) if isinstance(r, dict) else str(r)
        out["caption"] = {
            "text":    caption,
            "mention": mentions_basket(caption),
        }
    except Exception as e:
        out["caption"] = {"text": f"ERROR:{e}", "mention": False}

    return out


# ─── 집계 ─────────────────────────────────────────────────────────────────────
def aggregate(frame_results):
    """frame_results: list of per-frame task dicts"""
    agg = {}
    tasks_seen = set()
    for fr in frame_results:
        tasks_seen.update(fr.keys())

    for task in tasks_seen:
        vals = [fr[task] for fr in frame_results if task in fr]
        n    = len(vals)
        if not n: continue

        if task == "grounding":
            agg["grounding"] = {
                "hit_rate":      round(sum(v["hit"]     for v in vals) / n, 3),
                "mean_iou":      round(np.mean([v["iou"]  for v in vals]), 3),
                "any_bbox_rate": round(sum(v["has_bbox"] for v in vals) / n, 3),
                "n": n,
            }
        elif task in ("vqa", "vqa_caption"):
            agg["vqa"] = {
                "yes_rate":     round(sum(v["yes"]     for v in vals) / n, 3),
                "mention_rate": round(sum(v["mention"] for v in vals) / n, 3),
                "n": n,
                "sample_answers": [v["answer"] for v in vals[:5]],
            }
        elif task == "caption":
            agg["caption"] = {
                "mention_rate": round(sum(v["mention"] for v in vals) / n, 3),
                "n": n,
                "sample_captions": [v["text"][:80] for v in vals[:5]],
            }

    return agg


# ─── HTML 생성 ────────────────────────────────────────────────────────────────
def generate_html(state):
    results  = state.get("results", {})
    cur      = state.get("current_combo", "")
    status   = state.get("status", "pending")
    elapsed  = int(state.get("elapsed_s", 0))
    eta      = int(state.get("estimated_remaining_s", 0))
    done_ids = state.get("completed_combos", [])
    frame_cur= state.get("current_frame", 0)
    frame_tot= state.get("total_frames", 1)
    pct      = int(frame_cur / max(frame_tot, 1) * 100)

    status_color = {"running":"#f59e0b","done":"#22c55e","error":"#ef4444","smoke_done":"#06b6d4"}.get(status,"#94a3b8")
    cur_label = next((c["label"] for c in COMBOS if c["id"]==cur), cur)

    def metric_cell(cid, ds, task, metric):
        r = results.get(cid, {}).get(ds, {}).get(task, {})
        v = r.get(metric)
        if v is None: return "<td class='na'>—</td>"
        pct_v = v * 100
        if metric in ("hit_rate","yes_rate","mention_rate","any_bbox_rate"):
            col = "#22c55e" if pct_v>=50 else "#f59e0b" if pct_v>=20 else "#ef4444"
            return f"<td style='color:{col};font-weight:700'>{pct_v:.0f}%</td>"
        return f"<td class='num'>{v:.3f}</td>"

    METRICS = [
        ("Grounding hit% (IoU≥0.3)", "grounding","hit_rate"),
        ("Grounding mean IoU",         "grounding","mean_iou"),
        ("Any bbox predicted",         "grounding","any_bbox_rate"),
        ("VQA yes%",                   "vqa",      "yes_rate"),
        ("VQA basket mention%",        "vqa",      "mention_rate"),
        ("Caption mention%",           "caption",  "mention_rate"),
    ]
    DATASETS = [("exp46","in-dist"),("exp55","OOD")]

    header = "".join(
        f'<th class="{"active" if c["id"]==cur else ""}">'
        f'{c["id"]}<br><small>{c["label"]}</small></th>'
        for c in COMBOS
    )

    rows = ""
    for (m_label, task, metric) in METRICS:
        for (ds, ds_label) in DATASETS:
            cells = "".join(metric_cell(c["id"], ds, task, metric) for c in COMBOS)
            rows += f'<tr><td class="rowlabel">{m_label}<br><small class="ds">{ds_label}</small></td>{cells}</tr>\n'

    # 샘플 텍스트 출력 섹션 (최근 완료 백본)
    sample_html = ""
    for cid in done_ids[-2:]:  # 최근 2개
        r = results.get(cid, {})
        c_info = next((c for c in COMBOS if c["id"]==cid), {})
        label  = c_info.get("label","?")
        for ds in ["exp46","exp55"]:
            ds_r = r.get(ds,{})
            for task in ["grounding","vqa","caption"]:
                t_r = ds_r.get(task,{})
                samples = t_r.get("sample_answers") or t_r.get("sample_captions") or []
                if t_r.get("sample_raw"): samples = t_r["sample_raw"]
                if samples:
                    items = "".join(f'<li>{s[:120]}</li>' for s in samples[:3])
                    sample_html += f'<div class="sample-block"><b>{cid} {label} / {task} / {ds}</b><ul>{items}</ul></div>'

    badges = "".join(
        f'<span class="badge {"done" if c["id"] in done_ids else "running" if c["id"]==cur else "pending"}">'
        f'{c["id"]}</span>'
        for c in COMBOS
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="6">
<title>Backbone Recognition Eval</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0f1a;color:#e2e8f0;padding:20px}}
h1{{font-size:1.3rem;font-weight:800;margin-bottom:4px}}
.sub{{color:#64748b;font-size:0.8rem;margin-bottom:16px}}
.bar{{background:#1e293b;border-radius:10px;padding:14px 18px;margin-bottom:16px;display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap}}
.dot{{width:10px;height:10px;border-radius:50%;background:{status_color};box-shadow:0 0 6px {status_color};margin-top:4px;flex-shrink:0}}
.st{{font-weight:700;color:{status_color}}}
.badges{{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}}
.badge{{padding:2px 9px;border-radius:20px;font-size:0.72rem;font-weight:700}}
.badge.done{{background:#166534;color:#86efac}}.badge.running{{background:#78350f;color:#fde68a}}.badge.pending{{background:#1e293b;color:#64748b}}
.prog-wrap{{background:#0f172a;border-radius:6px;height:7px;width:180px;margin-top:8px}}
.prog-fill{{height:100%;background:{status_color};width:{pct}%;border-radius:6px}}
table{{width:100%;border-collapse:collapse;background:#111827;border-radius:10px;overflow:hidden;font-size:0.82rem;margin-bottom:20px}}
th,td{{padding:9px 10px;text-align:center;border-bottom:1px solid #1e293b}}
th{{background:#0f172a;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:#94a3b8}}
th.active{{background:#1e3a5f;color:#60a5fa}}
td.rowlabel{{text-align:left;min-width:170px;color:#94a3b8;font-size:0.75rem}}
td.na{{color:#334155}}td.num{{color:#94a3b8}}
small.ds{{color:#475569;font-size:0.68rem}}
.sample-block{{background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:10px;font-size:0.78rem}}
.sample-block b{{color:#94a3b8;display:block;margin-bottom:6px}}
.sample-block ul{{padding-left:16px;color:#e2e8f0;line-height:1.7}}
.time{{color:#64748b;font-size:0.75rem;text-align:right;margin-top:8px}}
</style>
</head>
<body>
<h1>Backbone Recognition Eval</h1>
<div class="sub">basket 인식 — {len(COMBOS)} backbones × grounding/VQA/caption | IoU≥{IOU_THR} | 6초 자동새로고침</div>

<div class="bar">
  <div class="dot"></div>
  <div>
    <div class="st">{status.upper()}{f" — {cur_label} ({frame_cur}/{frame_tot})" if status=="running" else ""}</div>
    <div class="badges">{badges}</div>
  </div>
  <div style="margin-left:auto">
    <div class="prog-wrap"><div class="prog-fill"></div></div>
    <div class="time">경과 {elapsed//60}m{elapsed%60:02d}s{"  ETA "+str(eta//60)+"m"+str(eta%60)+"s" if eta>0 else ""}</div>
  </div>
</div>

<table>
<thead><tr><th style="text-align:left">Metric</th>{header}</tr></thead>
<tbody>{rows}</tbody>
</table>

{"<h3 style='margin-bottom:10px;font-size:0.9rem;color:#94a3b8'>샘플 텍스트 출력</h3>" + sample_html if sample_html else ""}

<div class="time">{time.strftime('%Y-%m-%d %H:%M:%S')} | details/ 에 프레임별 raw 저장</div>
</body></html>"""

    (OUT_DIR / "index.html").write_text(html)


# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke",         action="store_true", help="1프레임으로 각 백본 smoke test")
    ap.add_argument("--max-frames",    type=int,  default=None, help="smoke 시 사용할 최대 총 프레임 수 (기본: smoke=10, full=전체)")
    ap.add_argument("--combos",        nargs="*", default=None)
    ap.add_argument("--frames-per-ep", type=int,  default=3)
    ap.add_argument("--iou-thr",       type=float,default=0.3)
    args = ap.parse_args()

    global IOU_THR
    IOU_THR = args.iou_thr

    frames_per_ep = 1 if args.smoke else args.frames_per_ep
    target = [c for c in COMBOS if args.combos is None or c["id"] in (args.combos or [])]

    OUT_DIR.mkdir(exist_ok=True)
    DETAILS_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory//1024**3}GB")

    print(f"\n[DATA] frames_per_ep={frames_per_ep} {'(SMOKE TEST)' if args.smoke else ''}")
    frames_exp46 = load_frames(DATA_EXP46, frames_per_ep)
    frames_exp55 = load_frames(DATA_EXP55, frames_per_ep)

    # smoke 모드: 최대 프레임 수 제한
    max_f = args.max_frames if args.max_frames else (10 if args.smoke else None)
    if max_f:
        # exp46에서 최대 max_f*2//3, exp55에서 나머지
        n46 = max(1, max_f * 2 // 3)
        n55 = max(1, max_f - n46)
        frames_exp46 = frames_exp46[:n46]
        frames_exp55 = frames_exp55[:n55]
    print(f"  exp46: {len(frames_exp46)} frames | exp55: {len(frames_exp55)} frames")

    state = {
        "status": "running",
        "current_combo": "",
        "current_frame": 0,
        "total_frames":  len(frames_exp46) + len(frames_exp55),
        "completed_combos": [],
        "results":  {},
        "started_at": time.time(),
        "elapsed_s": 0,
        "estimated_remaining_s": 0,
        "smoke": args.smoke,
    }
    PROGRESS_JSON.write_text(json.dumps(state, indent=2))
    generate_html(state)

    prev_bb = None
    model_objs = {}
    combo_times = []

    for combo in target:
        cid  = combo["id"]
        bb   = combo["backbone"]
        tasks= combo["tasks"]

        state.update({"current_combo": cid, "current_frame": 0,
                      "elapsed_s": int(time.time()-state["started_at"])})
        PROGRESS_JSON.write_text(json.dumps(state, indent=2))
        generate_html(state)

        print(f"\n{'='*60}")
        print(f"[{cid}] {combo['label']}  tasks={tasks}")
        print(f"{'='*60}")
        t0 = time.time()

        # ── 모델 로드 ──────────────────────────────────────────────────────
        if bb != prev_bb:
            if model_objs:
                print(f"  [UNLOAD] {prev_bb}")
                unload_all(model_objs)
                model_objs = {}
            try:
                print(f"  [LOAD] {bb}...")
                if   bb == "kosmos-base":    mo = {"proc": None, "model": None}; mo["proc"], mo["model"] = load_kosmos(device)
                elif bb == "kosmos-lora":    mo = {"proc": None, "model": None}; mo["proc"], mo["model"] = load_kosmos(device, lora_dir=KOSMOS_LORA_DIR)
                elif bb == "kosmos-google":  mo = {"proc": None, "model": None}; mo["proc"], mo["model"] = load_kosmos(device, google_pt=GOOGLE_ROBOT_PT)
                elif bb == "paligemma-base": mo = {"proc": None, "model": None}; mo["proc"], mo["model"] = load_paligemma(device)
                elif bb == "paligemma-lora": mo = {"proc": None, "model": None}; mo["proc"], mo["model"] = load_paligemma(device, lora_dir=PALI_LORA_DIR)
                elif bb == "paligemma2-mix": mo = {"proc": None, "model": None}; mo["proc"], mo["model"] = load_paligemma2(device)
                elif bb == "moondream":      mo = {"model": load_moondream(device)}
                model_objs = mo
                prev_bb = bb
            except Exception as e:
                print(f"  [ERROR] 로드 실패: {e}"); traceback.print_exc()
                state["results"][cid] = {"error": str(e)}
                state["completed_combos"].append(cid)
                PROGRESS_JSON.write_text(json.dumps(state, indent=2))
                generate_html(state)
                continue

        # ── 추론 루프 ──────────────────────────────────────────────────────
        details = {"exp46": [], "exp55": []}
        all_frame_results = {"exp46": [], "exp55": []}

        for ds_key, frames in [("exp46", frames_exp46), ("exp55", frames_exp55)]:
            for fi, s in enumerate(frames):
                global_fi = (len(frames_exp46) if ds_key=="exp55" else 0) + fi
                state["current_frame"] = global_fi + 1
                state["elapsed_s"]     = int(time.time()-state["started_at"])
                if fi % 5 == 0:
                    PROGRESS_JSON.write_text(json.dumps(state, indent=2))
                    generate_html(state)

                img = load_image(s["episode"], s["frame_idx"])
                if img is None:
                    print(f"  [SKIP] {Path(s['episode']).name}:{s['frame_idx']}")
                    continue

                # 추론
                try:
                    if bb.startswith("kosmos"):
                        frame_out = infer_kosmos(model_objs["proc"], model_objs["model"], img, s, device)
                    elif bb.startswith("paligemma-"):
                        frame_out = infer_paligemma_grounding(model_objs["proc"], model_objs["model"], img, s, device)
                    elif bb == "paligemma2-mix":
                        frame_out = infer_paligemma2_vqa_caption(model_objs["proc"], model_objs["model"], img, s, device)
                    elif bb == "moondream":
                        frame_out = infer_moondream(model_objs["model"], img, s, device)
                    else:
                        frame_out = {}
                except Exception as e:
                    frame_out = {"error": str(e)}

                # 로깅
                g = frame_out.get("grounding",{})
                v = frame_out.get("vqa",{})
                c = frame_out.get("caption",{})
                parts = []
                if g: parts.append(f"grd={'✅' if g.get('hit') else '❌'} IoU={g.get('iou',0):.2f}")
                if v: parts.append(f"vqa={'✅' if v.get('yes') else '❌'} '{v.get('answer','')[:30]}'")
                if c: parts.append(f"cap={'✅' if c.get('mention') else '❌'}")
                print(f"  [{cid}] {ds_key} {fi+1:3d}/{len(frames)} {' | '.join(parts)}")

                details[ds_key].append({
                    "frame_idx":  s["frame_idx"],
                    "episode":    Path(s["episode"]).name,
                    "path_type":  s["path_type"],
                    "has_bbox_gt": s["has_bbox"],
                    **frame_out,
                })
                all_frame_results[ds_key].append(frame_out)

        # ── 집계 및 저장 ────────────────────────────────────────────────────
        elapsed_c = time.time() - t0
        combo_times.append(elapsed_c)

        metrics = {}
        for ds_key in ["exp46","exp55"]:
            metrics[ds_key] = aggregate(all_frame_results[ds_key])
        metrics["elapsed_s"] = round(elapsed_c, 1)
        metrics["backbone"]  = bb
        metrics["tasks"]     = tasks

        # sample_raw 추가 (처음 5개 raw 출력 저장)
        for ds_key in ["exp46","exp55"]:
            for task in ["grounding","vqa","caption"]:
                t_m = metrics[ds_key].get(task, {})
                raws = [d.get(task,{}).get("raw") or d.get(task,{}).get("answer") or d.get(task,{}).get("text","")
                        for d in details[ds_key][:5]]
                t_m["sample_raw"] = [r for r in raws if r]

        # 집계 요약 출력
        print(f"\n  [{cid}] 완료 ({elapsed_c:.0f}s)")
        for ds_key in ["exp46","exp55"]:
            m = metrics[ds_key]
            g = m.get("grounding",{})
            v = m.get("vqa",{})
            cp= m.get("caption",{})
            print(f"  {ds_key}: grd_hit={g.get('hit_rate','—')} IoU={g.get('mean_iou','—')} "
                  f"vqa_yes={v.get('yes_rate','—')} cap_mention={cp.get('mention_rate','—')}")

        state["results"][cid] = metrics
        state["completed_combos"].append(cid)

        # ETA 추정
        remaining = len(target) - len(state["completed_combos"])
        if combo_times:
            state["estimated_remaining_s"] = int(np.mean(combo_times) * remaining)

        PROGRESS_JSON.write_text(json.dumps(state, indent=2))
        RESULTS_JSON.write_text(json.dumps(state["results"], indent=2))
        (DETAILS_DIR / f"{cid}.json").write_text(json.dumps(details, indent=2))
        generate_html(state)

    # 완료
    state.update({"status": "smoke_done" if args.smoke else "done",
                  "current_combo": "",
                  "elapsed_s": int(time.time()-state["started_at"]),
                  "estimated_remaining_s": 0})
    PROGRESS_JSON.write_text(json.dumps(state, indent=2))
    RESULTS_JSON.write_text(json.dumps(state["results"], indent=2))
    generate_html(state)

    total_s = state["elapsed_s"]
    print(f"\n{'='*60}")
    print(f"  완료: {total_s//60}m{total_s%60:02d}s")
    print(f"  결과: {RESULTS_JSON}")
    print(f"  details/: {DETAILS_DIR}")
    print(f"{'='*60}\n")

    # 최종 요약
    print(f"{'ID':<4} {'Backbone':<22} {'exp46 grd_hit':>13} {'exp46 vqa_yes':>13} {'exp55 grd_hit':>13} {'exp55 vqa_yes':>13}")
    print("-"*80)
    for c in target:
        r = state["results"].get(c["id"],{})
        if "error" in r: print(f"{c['id']:<4} {c['label']:<22} ERROR"); continue
        g46 = r.get("exp46",{}).get("grounding",{}).get("hit_rate","—")
        v46 = r.get("exp46",{}).get("vqa",{}).get("yes_rate","—")
        g55 = r.get("exp55",{}).get("grounding",{}).get("hit_rate","—")
        v55 = r.get("exp55",{}).get("vqa",{}).get("yes_rate","—")
        fmt = lambda x: f"{x*100:.0f}%" if isinstance(x,float) else str(x)
        print(f"{c['id']:<4} {c['label']:<22} {fmt(g46):>13} {fmt(v46):>13} {fmt(g55):>13} {fmt(v55):>13}")


if __name__ == "__main__":
    main()
