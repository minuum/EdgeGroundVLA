#!/usr/bin/env python3
"""
Counterfactual Visual Checker — 교수님 요구사항 시각화 플랫폼

탭 1: Condition A~D  (path_type 필터 + cx 표시)
탭 2: Object Substitution  (basket 유무 비교)
탭 3: Batch Sweep  (cx편향 프레임 전수 자동 실행 + flip 통계)
탭 4: VLM Text Output  (Kosmos-2가 프레임을 보고 뭐라고 하는지)

Usage:
  python3 scripts/viz_counterfactual_checker.py [--port 7861]
"""

import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import gradio as gr
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH    = ROOT / ".vlms" / "kosmos-2-patch14-224"
RECOG_CSV   = ROOT / "docs" / "v5" / "vlm_recognition" / "vlm_text_recognition.csv"
RECOG_JSON  = ROOT / "docs" / "v5" / "vlm_recognition" / "vlm_text_recognition.json"
DATA_PATH = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
STAGE1_V2 = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
STAGE2_V2 = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage2_v2" / "stage2_v2_mlp.pt"

CLASS_NAMES  = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
ACTION_EMOJI = {"STOP":"⬛","FORWARD":"⬆️","LEFT":"⬅️","RIGHT":"➡️",
                "FWD+L":"↖️","FWD+R":"↗️","ROT_L":"↺","ROT_R":"↻"}
NUM_CLASSES = 8
WINDOW = 8
VIS_DIM, PROJ_DIM = 1024, 256
D_IN = WINDOW * 4 + PROJ_DIM   # 288

ALL_PATH_TYPES = [
    "center_straight","center_left","center_right",
    "left_straight","left_left","left_right",
    "right_straight","right_left","right_right",
]


# ─── 모델 ────────────────────────────────────────────────

class FrozenCLIPV2(nn.Module):
    def __init__(self, vlm_path, ckpt_path, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        self.processor = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vision_model = base.vision_model.to(device)
        self.image_proj   = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.image_proj.load_state_dict(ckpt["image_proj"])
        for p in list(self.vision_model.parameters()) + list(self.image_proj.parameters()):
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, pil_images, device, batch=32):
        all_feats = []
        for i in range(0, len(pil_images), batch):
            imgs = pil_images[i:i+batch]
            inputs = self.processor(images=imgs, return_tensors="pt")
            pv = inputs["pixel_values"].to(device, dtype=torch.float16)
            out = self.vision_model(pixel_values=pv)
            feat = out.last_hidden_state.mean(dim=1).float()
            all_feats.append(F.normalize(self.image_proj(feat), dim=-1))
        return torch.cat(all_feats, dim=0)


class ActionMLP(nn.Module):
    def __init__(self, d_in=D_IN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES),
        )
    def forward(self, x): return self.net(x)


# ─── 전역 상태 ───────────────────────────────────────────

G = {"device":None,"enc":None,"mlp":None,"data":None,"vis_cache":{}}
VLM_G = {"proc": None, "model": None}   # lazy-loaded on Tab 4 first use
SUBST_PAIRS = []

# Kosmos-2 프롬프트 템플릿
VLM_PROMPTS = {
    "🔍 Grounding — gray basket":    ("<grounding><phrase>gray basket</phrase>", True),
    "🔍 Grounding — basket":          ("<grounding><phrase>basket</phrase>", True),
    "🔍 Grounding — gray object":     ("<grounding><phrase>gray object</phrase>", True),
    "📝 Free description":            ("An image of", False),
    "📍 Where is the basket?":        ("Question: Where is the gray basket? Answer:", False),
    "🎯 Describe the scene":          ("Describe what you see in this image:", False),
    "✏️ Custom prompt":               ("", False),
}


def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}", flush=True)
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_V2, device); enc.eval()
    ckpt = torch.load(str(STAGE2_V2), map_location=device, weights_only=False)
    mlp  = ActionMLP().to(device); mlp.load_state_dict(ckpt["mlp"]); mlp.eval()
    data = json.loads(DATA_PATH.read_text())
    G.update({"device":device,"enc":enc,"mlp":mlp,"data":data})
    print(f"[DATA] {len(data)} episodes  val_acc={ckpt.get('val_acc','?'):.4f}", flush=True)
    _cache_ep(data[:8])


def load_vlm_lazy():
    """Tab 4 첫 사용 시에만 Kosmos-2 full model 로드 (3~4GB VRAM)."""
    if VLM_G["model"] is not None:
        return True
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        device = G["device"]
        print("[VLM] Kosmos-2 full model 로딩...", flush=True)
        proc  = AutoProcessor.from_pretrained(str(VLM_PATH))
        model = AutoModelForVision2Seq.from_pretrained(
            str(VLM_PATH), torch_dtype=torch.float16
        ).to(device)
        model.eval()
        VLM_G["proc"]  = proc
        VLM_G["model"] = model
        print("[VLM] 로드 완료", flush=True)
        return True
    except Exception as e:
        print(f"[VLM] 로드 실패: {e}", flush=True)
        return False


def vlm_generate(h5_path, frame_idx, prompt_text, is_grounding, max_tokens=128):
    """
    Returns:
        decoded_raw: str  (special token 포함)
        decoded_clean: str  (cleaned)
        entities: list of (phrase, span, boxes)
        bbox_info: str  (human-readable)
    """
    proc  = VLM_G["proc"]
    model = VLM_G["model"]
    device = G["device"]

    with h5py.File(h5_path, "r") as f:
        arr = f["observations"]["images"][frame_idx]
    pil_img = Image.fromarray(arr)

    inputs = proc(text=prompt_text, images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

    with torch.no_grad():
        gen = model.generate(**inputs, use_cache=True, max_new_tokens=max_tokens)

    decoded_raw   = proc.batch_decode(gen, skip_special_tokens=False)[0]
    decoded_clean = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

    entities = []
    bbox_info = "(grounding 모드 아님)"
    if is_grounding:
        try:
            _, entities = proc.post_process_generation(decoded_raw, cleanup_and_extract=True)
            if entities:
                lines = []
                for phrase, span, boxes in entities:
                    for b in boxes:
                        lines.append(f"  phrase='{phrase}'  box=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f})")
                bbox_info = "\n".join(lines) if lines else "bbox 없음 (grounding 실패)"
            else:
                bbox_info = "entity 없음 (grounding 실패)"
        except Exception as e:
            bbox_info = f"post_process 오류: {e}"

    return decoded_raw, decoded_clean, entities, bbox_info


def draw_img_with_vlm_bbox(h5_path, frame_idx, entities):
    """VLM grounding 결과 bbox를 이미지 위에 그림."""
    with h5py.File(h5_path, "r") as f:
        arr = f["observations"]["images"][frame_idx]
    img  = Image.fromarray(arr).resize((400, 300))
    draw = ImageDraw.Draw(img)
    W, H = img.size
    colors = [(0, 220, 100), (0, 120, 255), (255, 120, 0), (220, 0, 200)]
    for ci, (phrase, _, boxes) in enumerate(entities):
        color = colors[ci % len(colors)]
        for b in boxes:
            x1, y1, x2, y2 = int(b[0]*W), int(b[1]*H), int(b[2]*W), int(b[3]*H)
            draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
            draw.rectangle([x1,y1,x1+len(phrase)*7+6,y1+18], fill=color)
            draw.text((x1+3, y1+1), phrase, fill="white")
    return img


def _cache_ep(eps):
    enc, device = G["enc"], G["device"]
    for ep in eps:
        key = ep["episode"]
        if key in G["vis_cache"]: continue
        try:
            with h5py.File(key,"r") as f:
                imgs = [Image.fromarray(f["observations"]["images"][i])
                        for i in range(len(ep["frames"]))]
            G["vis_cache"][key] = enc.encode_batch(imgs, device).cpu()
        except Exception as e:
            G["vis_cache"][key] = None


def _ensure_cached(ep):
    if ep["episode"] not in G["vis_cache"]:
        _cache_ep([ep])
    return G["vis_cache"].get(ep["episode"])


# ─── 이미지 그리기 ────────────────────────────────────────

def draw_img(h5_path, frame_idx, fr, label, bbox_mode="normal", color=(0,200,0)):
    with h5py.File(h5_path,"r") as f:
        arr = f["observations"]["images"][frame_idx]
    img  = Image.fromarray(arr).resize((320,240))
    draw = ImageDraw.Draw(img)
    W, H = img.size

    if fr.get("has_bbox") and bbox_mode not in ("none","zeros"):
        cx = (1.0 - fr["cx"]) if bbox_mode == "flip" else fr["cx"]
        side = max(fr.get("area",0.05),0.005)**0.5 * 1.5
        x1 = int((cx - side/2)*W); y1 = int((fr["cy"]-side/2)*H)
        x2 = int((cx + side/2)*W); y2 = int((fr["cy"]+side/2)*H)
        draw.rectangle([x1,y1,x2,y2], outline=color, width=3)

    draw.rectangle([0,0,len(label)*8+6,22], fill=(30,30,30))
    draw.text((4,3), label, fill="white")
    return img


def draw_absent(h5_path, frame_idx, label="🔴 NO BASKET"):
    with h5py.File(h5_path,"r") as f:
        arr = f["observations"]["images"][frame_idx]
    img  = Image.fromarray(arr).resize((320,240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([2,2,317,237], outline=(220,50,50), width=4)
    draw.rectangle([0,0,len(label)*8+6,22], fill=(160,20,20))
    draw.text((4,3), label, fill="white")
    return img


# ─── 추론 유틸 ───────────────────────────────────────────

@torch.no_grad()
def infer(vis_feat, frames, t, bbox_mode="normal"):
    mlp, device = G["mlp"], G["device"]
    arr = []
    for k in range(WINDOW):
        f2 = frames[max(0, t-(WINDOW-1-k))]
        if bbox_mode == "zeros":
            arr.extend([0.0,0.0,0.0,0.0])
        elif bbox_mode == "flip":
            arr.extend([1.0-f2["cx"], f2["cy"], f2["area"], float(f2["has_bbox"])])
        else:
            arr.extend([f2["cx"], f2["cy"], f2["area"], float(f2["has_bbox"])])
    bf = torch.tensor(arr, dtype=torch.float32)
    x  = torch.cat([bf, vis_feat]).unsqueeze(0).to(device)
    logits = mlp(x)[0]
    pred   = logits.argmax().item()
    probs  = torch.softmax(logits, dim=0).cpu().numpy()
    return pred, probs


def prob_bar(probs, highlight):
    lines = []
    for i,(n,p) in enumerate(zip(CLASS_NAMES, probs)):
        bar = "█"*int(p*20) + "░"*(20-int(p*20))
        mk  = " ◀" if i==highlight else ""
        lines.append(f"{n:8s} {bar} {p*100:5.1f}%{mk}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════
# 탭 1: Condition A~D  (path_type 필터 + cx 표시)
# ════════════════════════════════════════════════════════

def get_path_types():
    pts = sorted(set(ep["path_type"] for ep in G["data"]))
    return ["전체"] + pts


def ep_choices_filtered(pt_filter="전체"):
    out = []
    for i, ep in enumerate(G["data"]):
        if pt_filter != "전체" and ep["path_type"] != pt_filter:
            continue
        out.append(f"[{i:03d}] {ep['path_type']} — {Path(ep['episode']).stem[-22:]}")
    return out


def frame_choices(ep_str):
    ei = int(ep_str.split("]")[0].strip("["))
    ep = G["data"][ei]
    rows = []
    for j, fr in enumerate(ep["frames"]):
        bbox_tag = "🟢" if fr["has_bbox"] else "🔴"
        # cx 편향도 표시 (0=좌, 0.5=중앙, 1=우)
        if fr["has_bbox"]:
            cx = fr["cx"]
            if cx < 0.35:
                bias = f"cx={cx:.2f}◀LEFT"
            elif cx > 0.65:
                bias = f"cx={cx:.2f}▶RIGHT"
            else:
                bias = f"cx={cx:.2f}·CENTER"
        else:
            bias = "bbox없음"
        gt = CLASS_NAMES[fr["gt_class"]]
        rows.append(f"Frame {j:03d} {bbox_tag} {bias} | gt={gt}")
    return rows


def run_abcd(ep_str, frame_str):
    if G["data"] is None or not ep_str or not frame_str:
        return [None]*4, "", ""

    ei = int(ep_str.split("]")[0].strip("["))
    ti = int(frame_str.split(" ")[1])
    ep = G["data"][ei]; frames = ep["frames"]; fr = frames[ti]
    gt = fr["gt_class"]; key = ep["episode"]

    vis = _ensure_cached(ep)
    if vis is None:
        return [None]*4, "캐시 오류", ""

    # Condition D: 반대방향 에피소드
    pt = ep.get("path_type","")
    swap_vis, swap_ep = None, None
    if "left" in pt and "right" not in pt:
        targets = [e for e in G["data"] if "right" in e["path_type"] and "left" not in e["path_type"]]
    elif "right" in pt and "left" not in pt:
        targets = [e for e in G["data"] if "left" in e["path_type"] and "right" not in e["path_type"]]
    else:
        targets = []

    for e in targets:
        sv = _ensure_cached(e)
        if sv is not None and len(sv) > ti:
            swap_vis, swap_ep = sv, e; break

    results = {}
    for cond, bm, vf in [
        ("A","normal",  vis[ti]),
        ("B","zeros",   vis[ti]),
        ("C","flip",    vis[ti]),
        ("D","normal",  swap_vis[ti] if swap_vis is not None else vis[ti]),
    ]:
        results[cond] = infer(vf, frames, ti, bm)

    # cx 경고
    cx_val = fr.get("cx", 0.5)
    cx_warn = ""
    if fr.get("has_bbox") and 0.35 <= cx_val <= 0.65:
        cx_warn = "\n\n> ⚠️ **basket이 중앙 근처 (cx≈0.5) — Condition C 효과 미미.** left/right 에피소드로 바꿔서 테스트하세요."

    # D swap 정보
    d_note = f"(swap ep: `{swap_ep['path_type']}`)" if swap_ep else "(반대방향 에피소드 없음 — 동일 이미지 사용)"

    # 이미지 생성
    img_A = draw_img(key, fr.get("frame_idx",ti), fr, "A 정상",    "normal", (0,200,0))
    img_B = draw_img(key, fr.get("frame_idx",ti), fr, "B bbox=0",  "zeros",  (200,50,50))
    img_C = draw_img(key, fr.get("frame_idx",ti), fr, f"C cx반전({cx_val:.2f}→{1-cx_val:.2f})", "flip", (255,140,0))
    if swap_ep:
        d_ti  = min(ti, len(swap_ep["frames"])-1)
        d_fr  = swap_ep["frames"][d_ti]
        img_D = draw_img(swap_ep["episode"], d_fr.get("frame_idx",d_ti),
                         d_fr, f"D {swap_ep['path_type']}", "normal", (150,50,220))
    else:
        img_D = draw_img(key, fr.get("frame_idx",ti), fr, "D (동일)", "normal", (150,50,220))

    # 요약 마크다운
    rows = [f"**정답: {ACTION_EMOJI.get(CLASS_NAMES[gt],'')} {CLASS_NAMES[gt]}**  |  "
            f"basket cx={cx_val:.2f}  |  has_bbox={fr.get('has_bbox')}\n"]
    rows += ["| 조건 | 예측 | 판정 | 의미 |",
             "|------|------|------|------|"]
    meanings = {"A":"기준선","B":"basket 제거","C":"위치 속임","D":f"다른방향 visual {d_note}"}
    for cond,(p,_) in results.items():
        em = ACTION_EMOJI.get(CLASS_NAMES[p],'')
        vd = "✅" if p==gt else "❌"
        rows.append(f"| **{cond}** | {em} {CLASS_NAMES[p]} | {vd} | {meanings[cond]} |")

    rows.append(cx_warn)
    summary = "\n".join(rows)

    bars = []
    for cond,(p,probs) in results.items():
        bars.append(f"─── COND {cond} ({'✅' if p==gt else '❌'}) ───")
        bars.append(prob_bar(probs, p)); bars.append("")
    bar_txt = "\n".join(bars)

    return img_A, img_B, img_C, img_D, summary, bar_txt


# ════════════════════════════════════════════════════════
# 탭 2: Object Substitution (has_bbox=False 프레임)
# ════════════════════════════════════════════════════════

def build_subst_pairs():
    pairs = []
    for ei, ep in enumerate(G["data"]):
        present = [(j,fr) for j,fr in enumerate(ep["frames"]) if fr["has_bbox"]]
        absent  = [(j,fr) for j,fr in enumerate(ep["frames"]) if not fr["has_bbox"]]
        if not present or not absent: continue
        for aj, afr in absent[:4]:
            closest = min(present, key=lambda x: abs(x[0]-aj))
            pj, pfr = closest
            pairs.append(dict(ep_idx=ei, ep_key=ep["episode"],
                              path_type=ep["path_type"],
                              present_t=pj, present_fr=pfr,
                              absent_t=aj,  absent_fr=afr))
    return pairs


def subst_choices():
    return [f"[{i:03d}] {p['path_type']} | 🟢t={p['present_t']} cx={p['present_fr']['cx']:.2f} → 🔴t={p['absent_t']}"
            for i,p in enumerate(SUBST_PAIRS)]


def run_substitution(pair_str):
    if not SUBST_PAIRS or not pair_str: return None, None, "", ""
    pi   = int(pair_str.split("]")[0].strip("["))
    pair = SUBST_PAIRS[pi]
    ep   = G["data"][pair["ep_idx"]]
    key  = pair["ep_key"]; frames = ep["frames"]

    vis = _ensure_cached(ep)
    if vis is None: return None, None, "캐시 오류", ""

    pt = pair["present_t"]; at = pair["absent_t"]
    pfr = pair["present_fr"]; afr = pair["absent_fr"]

    pred_p, probs_p = infer(vis[pt], frames, pt, "normal")
    pred_a, probs_a = infer(vis[at], frames, at, "normal")

    img_pres = draw_img(key, pfr.get("frame_idx",pt), pfr,
                        f"🟢 basket 있음 t={pt} cx={pfr['cx']:.2f}", "normal", (0,200,80))
    img_abs  = draw_absent(key, afr.get("frame_idx",at),
                           f"🔴 basket 없음 t={at}")

    gt_p, gt_a = pfr["gt_class"], afr["gt_class"]
    changed = pred_p != pred_a
    change_icon = "🔄 **예측 바뀜!**" if changed else "➡️ 예측 동일 (바뀌지 않음)"

    em_p = ACTION_EMOJI.get(CLASS_NAMES[pred_p],'')
    em_a = ACTION_EMOJI.get(CLASS_NAMES[pred_a],'')

    summary = "\n".join([
        f"### `{pair['path_type']}` 에피소드\n",
        "| | 🟢 Basket 있음 | 🔴 Basket 없음 |",
        "|---|---|---|",
        f"| **정답** | {ACTION_EMOJI.get(CLASS_NAMES[gt_p],'')} {CLASS_NAMES[gt_p]} | {ACTION_EMOJI.get(CLASS_NAMES[gt_a],'')} {CLASS_NAMES[gt_a]} |",
        f"| **예측** | {em_p} {CLASS_NAMES[pred_p]} | {em_a} {CLASS_NAMES[pred_a]} |",
        f"| **판정** | {'✅' if pred_p==gt_p else '❌'} | {'✅' if pred_a==gt_a else '❌'} |",
        f"\n{change_icon}",
        "",
        "**해석:**",
        ("- basket 사라지자 예측 변화 → ✅ 모델이 basket 유무를 감지함"
         if changed else
         "- basket 유무 무관하게 동일 예측 → ⚠️ trajectory 암기 가능성"),
    ])

    bar = ("─── 🟢 Basket 있음 ───\n" + prob_bar(probs_p, pred_p) +
           "\n\n─── 🔴 Basket 없음 ───\n" + prob_bar(probs_a, pred_a))
    return img_pres, img_abs, summary, bar


# ════════════════════════════════════════════════════════
# 탭 3: Batch Sweep — cx 편향 프레임 전수 자동 실행
# ════════════════════════════════════════════════════════

def run_batch_sweep(cx_thresh, path_filter, progress=gr.Progress()):
    """
    cx < cx_thresh (basket 좌편향) 또는 cx > 1-cx_thresh (우편향) 프레임을 전수 실행.
    각 조건(A/B/C/D)에서 A 대비 예측이 바뀌는 비율(flip rate) 계산.
    """
    data   = G["data"]
    thresh = float(cx_thresh)

    # 대상 에피소드 필터
    eps = [ep for ep in data
           if path_filter == "전체" or ep["path_type"] == path_filter]

    results = []   # {cond, flipped, ep_idx, frame_t, gt, pred_A, pred_X, cx}
    flip_counts = defaultdict(int)
    total = 0

    ep_bar = [ep for ep in eps]
    for ep_i, ep in enumerate(ep_bar):
        progress((ep_i+1)/len(ep_bar), desc=f"에피소드 {ep_i+1}/{len(ep_bar)}")

        vis = _ensure_cached(ep)
        if vis is None: continue

        frames = ep["frames"]

        # 반대방향 swap 에피소드 찾기
        pt = ep.get("path_type","")
        swap_vis = None
        if "left" in pt and "right" not in pt:
            for e in data:
                if "right" in e["path_type"] and "left" not in e["path_type"]:
                    sv = _ensure_cached(e)
                    if sv is not None:
                        swap_vis = sv; break
        elif "right" in pt and "left" not in pt:
            for e in data:
                if "left" in e["path_type"] and "right" not in e["path_type"]:
                    sv = _ensure_cached(e)
                    if sv is not None:
                        swap_vis = sv; break

        for t, fr in enumerate(frames):
            if not fr.get("has_bbox"): continue
            cx = fr["cx"]
            # cx 편향 필터 (좌편향 or 우편향)
            if not (cx < thresh or cx > 1.0 - thresh):
                continue

            gt = fr["gt_class"]
            vf = vis[t]
            pred_A, _ = infer(vf, frames, t, "normal")

            row = {"ep": ep["path_type"], "t": t, "gt": gt,
                   "cx": cx, "pred_A": pred_A}

            for cond, bm, vf2 in [
                ("B","zeros",   vf),
                ("C","flip",    vf),
                ("D","normal",  swap_vis[min(t, len(swap_vis)-1)] if swap_vis is not None else vf),
            ]:
                pred_X, _ = infer(vf2, frames, t, bm)
                flipped = (pred_X != pred_A)
                row[f"pred_{cond}"] = pred_X
                row[f"flip_{cond}"] = flipped
                if flipped: flip_counts[cond] += 1

            results.append(row)
            total += 1

    if total == 0:
        return "해당 조건에 맞는 프레임 없음", "", None

    # 통계 텍스트
    lines = [
        f"## Batch Sweep 결과",
        f"- cx 임계값: **{thresh}** (cx < {thresh} 또는 cx > {1-thresh:.2f})",
        f"- path_type 필터: **{path_filter}**",
        f"- 대상 프레임: **{total}개**\n",
        "| 조건 | Flip 수 | Flip Rate | 의미 |",
        "|------|---------|-----------|------|",
        f"| B (bbox 제거) | {flip_counts['B']} | **{flip_counts['B']/total*100:.1f}%** | bbox가 방향에 기여하는가 |",
        f"| C (cx 반전)   | {flip_counts['C']} | **{flip_counts['C']/total*100:.1f}%** | bbox cx가 방향 결정하는가 |",
        f"| D (visual 교체)| {flip_counts['D']} | **{flip_counts['D']/total*100:.1f}%** | visual이 방향 결정하는가 |",
        "",
        "**해석 기준:**",
        "- flip(C) ≥ 50% → bbox cx가 방향 결정  (basket 위치 추적 증거)",
        "- flip(C) < 20% → visual이 방향 결정  (이미지에서 직접 본다)",
        "- flip(B) ≥ 50% → bbox 존재 자체가 필수적",
    ]

    # path_type별 세부 통계
    pt_stats = defaultdict(lambda: defaultdict(int))
    pt_total = defaultdict(int)
    for row in results:
        pt = row["ep"]
        pt_total[pt] += 1
        for cond in ("B","C","D"):
            if row.get(f"flip_{cond}"): pt_stats[pt][cond] += 1

    lines += ["", "### Path Type별 flip(C) 상세"]
    lines += ["| path_type | 프레임수 | flip_B | flip_C | flip_D |",
              "|-----------|---------|--------|--------|--------|"]
    for pt in sorted(pt_total.keys()):
        n = pt_total[pt]
        fB = pt_stats[pt]["B"]; fC = pt_stats[pt]["C"]; fD = pt_stats[pt]["D"]
        lines.append(f"| {pt} | {n} | {fB/n*100:.0f}% | {fC/n*100:.0f}% | {fD/n*100:.0f}% |")

    # 샘플 프레임 상세 (flip_C 발생한 것 최대 10개)
    flipped_C = [r for r in results if r.get("flip_C")]
    sample_lines = ["\n### flip_C 발생 샘플 (최대 10개)"]
    sample_lines += ["| ep | t | cx | gt | pred_A | pred_C |",
                     "|----|----|-----|-----|--------|--------|"]
    for r in flipped_C[:10]:
        sample_lines.append(
            f"| {r['ep']} | {r['t']} | {r['cx']:.2f} | {CLASS_NAMES[r['gt']]} "
            f"| {CLASS_NAMES[r['pred_A']]} | {CLASS_NAMES[r['pred_C']]} |"
        )
    if not flipped_C:
        sample_lines.append("| (없음) | — | — | — | — | — |")

    full = "\n".join(lines + sample_lines)

    # 결론 한 줄
    fr_C = flip_counts["C"] / total
    if fr_C >= 0.5:
        conclusion = f"✅ **flip(C)={fr_C*100:.1f}% — bbox cx가 방향 결정. 교수님께 basket 위치 추적 증거 제시 가능**"
    elif fr_C >= 0.2:
        conclusion = f"⚠️ **flip(C)={fr_C*100:.1f}% — 부분적 bbox cx 의존. visual도 함께 기여**"
    else:
        conclusion = f"❌ **flip(C)={fr_C*100:.1f}% — bbox cx 반전 효과 없음. visual이 방향 결정 (교수님 질문 재검토 필요)**"

    return full, conclusion, results


# ════════════════════════════════════════════════════════
# Gradio UI
# ════════════════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="MoNaVLA Checker") as app:

        gr.Markdown(
            "# 🔍 MoNaVLA — 객체 인식 시각 검증\n"
            "**교수님 핵심 질문**: basket 없으면 이상한 행동 하는가? bbox cx가 방향 결정하는가?"
        )

        with gr.Tabs():

            # ── 탭 1 ─────────────────────────────────────
            with gr.TabItem("🔬 탭1 — Condition A~D"):
                gr.Markdown(
                    "**cx < 0.35 또는 cx > 0.65** 인 프레임을 골라야 Condition C 효과가 나타납니다.\n"
                    "프레임 드롭다운에 `cx=0.xx◀LEFT / ▶RIGHT` 표시를 보고 선택하세요."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        pt_filter = gr.Dropdown(label="path_type 필터",
                                                choices=["전체"]+ALL_PATH_TYPES,
                                                value="전체")
                        ep_dd     = gr.Dropdown(label="에피소드", choices=ep_choices_filtered())
                        frame_dd  = gr.Dropdown(label="프레임  (cx편향 / gt 표시)")
                        run1_btn  = gr.Button("🚀 추론 실행", variant="primary")
                    with gr.Column(scale=2):
                        summary1 = gr.Markdown("← 에피소드/프레임 선택 후 실행")

                with gr.Row():
                    img1A = gr.Image(label="A — 정상",      type="pil", height=220)
                    img1B = gr.Image(label="B — bbox 제거", type="pil", height=220)
                    img1C = gr.Image(label="C — cx 반전",   type="pil", height=220)
                    img1D = gr.Image(label="D — visual 교체",type="pil",height=220)
                bars1 = gr.Textbox(label="확률 분포", lines=14)

                def _filter_ep(pt):
                    choices = ep_choices_filtered(pt)
                    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

                def _update_frames(ep_str):
                    if not ep_str: return gr.Dropdown(choices=[])
                    choices = frame_choices(ep_str)
                    return gr.Dropdown(choices=choices, value=choices[0])

                pt_filter.change(_filter_ep, pt_filter, ep_dd)
                ep_dd.change(_update_frames, ep_dd, frame_dd)
                app.load(_update_frames, ep_dd, frame_dd)
                run1_btn.click(run_abcd,
                               inputs=[ep_dd, frame_dd],
                               outputs=[img1A,img1B,img1C,img1D,summary1,bars1])

            # ── 탭 2 ─────────────────────────────────────
            with gr.TabItem("🧪 탭2 — Object Substitution (basket 유무)"):
                gr.Markdown(
                    "### 교수님 질문: basket 없을 때 행동이 달라지는가?\n"
                    "같은 에피소드에서 **basket 있는 프레임** vs **basket 없는 실제 프레임** 직접 비교\n\n"
                    "드롭다운에 `cx=0.xx` 값이 표시됩니다 — cx가 0.5에서 멀수록 좋은 테스트케이스"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        pair_dd  = gr.Dropdown(label="비교 쌍 선택",
                                               choices=subst_choices() if SUBST_PAIRS else ["(없음)"],
                                               value=subst_choices()[0] if SUBST_PAIRS else "(없음)")
                        run2_btn = gr.Button("🔍 비교 실행", variant="primary")
                    with gr.Column(scale=2):
                        summary2 = gr.Markdown("← 쌍 선택 후 실행")

                with gr.Row():
                    img2P = gr.Image(label="🟢 basket 있음", type="pil", height=260)
                    img2A = gr.Image(label="🔴 basket 없음 (다른물체 proxy)", type="pil", height=260)
                bars2 = gr.Textbox(label="확률 분포", lines=20)
                run2_btn.click(run_substitution, inputs=[pair_dd],
                               outputs=[img2P, img2A, summary2, bars2])

            # ── 탭 3 ─────────────────────────────────────
            with gr.TabItem("📊 탭3 — Batch Sweep (자동 전수 분석)"):
                gr.Markdown(
                    "### cx 편향 프레임 전부 자동 실행 → 조건별 flip rate 통계\n"
                    "cx 임계값: 0.35면 basket이 왼쪽 35% 또는 오른쪽 35% 에 있는 프레임만 선택"
                )
                with gr.Row():
                    cx_slider   = gr.Slider(0.1, 0.5, value=0.35, step=0.05,
                                            label="cx 편향 임계값 (이 값보다 작거나 1-이 값보다 크면 선택)")
                    pt_filter3  = gr.Dropdown(label="path_type 필터",
                                              choices=["전체"]+ALL_PATH_TYPES, value="전체")
                    run3_btn    = gr.Button("⚡ 전수 분석 시작", variant="primary")

                conclusion3 = gr.Markdown("← 위 설정 후 전수 분석 시작")
                report3     = gr.Markdown()

                def _run_batch(cx_thresh, path_filter, progress=gr.Progress()):
                    full, conclusion, _ = run_batch_sweep(cx_thresh, path_filter, progress)
                    return conclusion, full

                run3_btn.click(_run_batch,
                               inputs=[cx_slider, pt_filter3],
                               outputs=[conclusion3, report3])

            # ── 탭 4 ─────────────────────────────────────
            with gr.TabItem("🧠 탭4 — VLM Text Output (Kosmos-2)"):
                gr.Markdown(
                    "### Kosmos-2 VLM이 우리 복도 이미지를 보고 실제로 뭐라고 하는가?\n"
                    "우리 액션 모델(Stage 2 MLP)과 **별개 레이어** — VLM 자체의 언어 출력을 봄\n\n"
                    "> ⚠️ 첫 실행 시 Kosmos-2 full model 로드 (3~4GB VRAM, 약 30초 소요)"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        pt_filter4 = gr.Dropdown(label="path_type 필터",
                                                 choices=["전체"]+ALL_PATH_TYPES,
                                                 value="전체")
                        _ep4_choices = ep_choices_filtered()
                        ep_dd4  = gr.Dropdown(label="에피소드",
                                              choices=_ep4_choices,
                                              value=_ep4_choices[0] if _ep4_choices else None)
                        _fr4_init = frame_choices(_ep4_choices[0]) if _ep4_choices else []
                        frame_dd4 = gr.Dropdown(label="프레임",
                                                choices=_fr4_init,
                                                value=_fr4_init[0] if _fr4_init else None)
                        prompt_dd = gr.Dropdown(
                            label="프롬프트 타입",
                            choices=list(VLM_PROMPTS.keys()),
                            value="🔍 Grounding — gray basket"
                        )
                        custom_box = gr.Textbox(
                            label="Custom prompt (✏️ 선택 시 활성화)",
                            placeholder="예: An image of a corridor with",
                            visible=False
                        )
                        max_tok = gr.Slider(32, 256, value=128, step=16,
                                            label="max_new_tokens")
                        run4_btn = gr.Button("🧠 VLM Generate", variant="primary")

                    with gr.Column(scale=2):
                        vlm_img     = gr.Image(label="프레임 + VLM bbox", type="pil", height=300)
                        vlm_clean   = gr.Textbox(label="📝 VLM 출력 (clean)", lines=3)
                        vlm_bbox    = gr.Textbox(label="📦 Grounding 결과 (bbox 좌표)", lines=5)
                        vlm_raw     = gr.Textbox(label="🔤 Raw decode (special tokens 포함)", lines=4)

                def _show_custom(prompt_key):
                    return gr.Textbox(visible=(prompt_key == "✏️ Custom prompt"))

                def _filter_ep4(pt):
                    choices = ep_choices_filtered(pt)
                    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

                def _update_frames4(ep_str):
                    if not ep_str: return gr.Dropdown(choices=[])
                    choices = frame_choices(ep_str)
                    return gr.Dropdown(choices=choices, value=choices[0])

                def run_vlm(ep_str, frame_str, prompt_key, custom_text, max_tokens):
                    if not ep_str or not frame_str:
                        return None, "에피소드/프레임을 선택하세요", "", ""

                    # VLM lazy load
                    if not load_vlm_lazy():
                        return None, "VLM 로드 실패 (로그 확인)", "", ""

                    ei = int(ep_str.split("]")[0].strip("["))
                    ti = int(frame_str.split(" ")[1])
                    ep = G["data"][ei]
                    fr = ep["frames"][ti]
                    h5_path  = ep["episode"]
                    frame_idx = fr.get("frame_idx", ti)

                    prompt_text, is_grounding = VLM_PROMPTS[prompt_key]
                    if prompt_key == "✏️ Custom prompt":
                        prompt_text  = custom_text.strip() or "An image of"
                        is_grounding = False

                    try:
                        raw, clean, entities, bbox_info = vlm_generate(
                            h5_path, frame_idx, prompt_text, is_grounding, int(max_tokens)
                        )
                    except Exception as e:
                        return None, f"오류: {e}", "", ""

                    # bbox 시각화
                    if entities:
                        img_out = draw_img_with_vlm_bbox(h5_path, frame_idx, entities)
                    else:
                        with h5py.File(h5_path, "r") as f:
                            arr = f["observations"]["images"][frame_idx]
                        img_out = Image.fromarray(arr).resize((400,300))

                    # 헤더 정보 추가
                    gt_name = CLASS_NAMES[fr["gt_class"]]
                    has_bbox = fr.get("has_bbox", False)
                    cx_str = f"cx={fr['cx']:.2f}" if has_bbox else "bbox없음"
                    header = (f"[프롬프트] {prompt_text}\n"
                              f"[gt_class={gt_name}  {cx_str}  path={ep['path_type']}]\n\n"
                              + clean)
                    return img_out, header, bbox_info, raw

                prompt_dd.change(_show_custom, prompt_dd, custom_box)
                pt_filter4.change(_filter_ep4, pt_filter4, ep_dd4)
                ep_dd4.change(_update_frames4, ep_dd4, frame_dd4)
                run4_btn.click(run_vlm,
                               inputs=[ep_dd4, frame_dd4, prompt_dd, custom_box, max_tok],
                               outputs=[vlm_img, vlm_clean, vlm_bbox, vlm_raw])

            # ── 탭 5 ─────────────────────────────────────
            with gr.TabItem("📈 탭5 — VLM Recognition 분석"):
                gr.Markdown(
                    "### `run_vlm_text_recognition.py` 배치 결과 분석\n"
                    f"결과 파일: `{RECOG_CSV.relative_to(ROOT)}`\n\n"
                    "**Refresh** 버튼으로 진행 중인 배치의 중간 결과도 볼 수 있음"
                )
                refresh5_btn = gr.Button("🔄 결과 불러오기 / Refresh", variant="primary")
                status5      = gr.Markdown("← Refresh 버튼 클릭")

                with gr.Row():
                    summary5_md  = gr.Markdown()
                    phrase5_md   = gr.Markdown()

                with gr.Row():
                    bypath5_md   = gr.Markdown()
                    iou5_md      = gr.Markdown()

                samples5_md = gr.Markdown()

                def load_recog_analysis():
                    if not RECOG_CSV.exists():
                        return ("⚠️ 결과 파일 없음 — `run_vlm_text_recognition.py` 먼저 실행하세요",
                                "","","","","")

                    import csv as _csv
                    from collections import Counter, defaultdict

                    rows = []
                    with open(RECOG_CSV, newline="", encoding="utf-8") as f:
                        for r in _csv.DictReader(f):
                            r["has_bbox"]                = r["has_bbox"] == "True"
                            r["grounding_any_success"]   = r.get("grounding_any_success","") == "True"
                            r["basket_grounding_success"]= r.get("basket_grounding_success","") == "True"
                            r["basket_iou"]              = float(r.get("basket_iou") or 0)
                            r["grounding_n_boxes"]       = int(r.get("grounding_n_boxes") or 0)
                            rows.append(r)

                    if not rows:
                        return "데이터 없음","","","","",""

                    pres = [r for r in rows if r["has_bbox"]]
                    abse = [r for r in rows if not r["has_bbox"]]

                    # ── 전체 요약 ──
                    n_ep       = len(set(r["ep_stem"] for r in rows))
                    # basket-specific (신뢰 가능)
                    bsk_ok     = sum(1 for r in pres if r["basket_grounding_success"])
                    bsk_fp     = sum(1 for r in abse if r["basket_grounding_success"])
                    avg_bsk_iou= (sum(r["basket_iou"] for r in pres) / len(pres)) if pres else 0
                    # any-phrase (부풀려진 수치)
                    any_ok     = sum(1 for r in pres if r["grounding_any_success"])
                    any_fp     = sum(1 for r in abse if r["grounding_any_success"])

                    iou_hi = sum(1 for r in pres if r["basket_iou"] >= 0.5)
                    iou_md = sum(1 for r in pres if 0.2 <= r["basket_iou"] < 0.5)
                    iou_lo = sum(1 for r in pres if 0 < r["basket_iou"] < 0.2)
                    iou_z  = sum(1 for r in pres if r["basket_iou"] == 0)

                    summary = "\n".join([
                        f"## 전체 요약  ({n_ep}/150 에피소드)",
                        f"- 총 프레임: **{len(rows)}** (basket 있음 {len(pres)}, 없음 {len(abse)})\n",
                        "| 지표 | 값 | 비고 |",
                        "|------|-----|------|",
                        (f"| **basket phrase grounding** | **{bsk_ok}/{len(pres)} = {bsk_ok/len(pres)*100:.1f}%** | 신뢰 가능 ✅ |" if pres else ""),
                        (f"| any phrase grounding | {any_ok}/{len(pres)} = {any_ok/len(pres)*100:.1f}% | 부풀려짐 ⚠️ |" if pres else ""),
                        (f"| Avg basket IoU | **{avg_bsk_iou:.3f}** | basket bbox 기준만 |"),
                        (f"| IoU ≥ 0.5 (좋음) | {iou_hi} ({iou_hi/len(pres)*100:.1f}%) | |" if pres else ""),
                        (f"| IoU 0.2~0.5 (보통) | {iou_md} ({iou_md/len(pres)*100:.1f}%) | |" if pres else ""),
                        (f"| IoU 0~0.2 (낮음) | {iou_lo} ({iou_lo/len(pres)*100:.1f}%) | |" if pres else ""),
                        (f"| basket bbox 없음 | {iou_z} ({iou_z/len(pres)*100:.1f}%) | basket phrase grounding 실패 |" if pres else ""),
                        "",
                        "| FP 지표 | 값 | 비고 |",
                        "|---------|-----|------|",
                        (f"| basket FP (없는데 감지) | {bsk_fp}/{len(abse)} = {bsk_fp/len(abse)*100:.1f}% | basket phrase 기준 |" if abse else ""),
                        (f"| any FP | {any_fp}/{len(abse)} = {any_fp/len(abse)*100:.1f}% | any phrase 기준 |" if abse else ""),
                    ])

                    # ── Side phrase 분포 (basket 아닌 것들 — hallucination) ──
                    side_ctr  = Counter()
                    multi_box = sum(1 for r in pres if r["grounding_n_boxes"] > 1)
                    for r in rows:
                        for ph in r.get("side_phrases","").split("|"):
                            ph = ph.strip().lower()
                            if ph:
                                side_ctr[ph] += 1

                    phrase_lines = [
                        "## Side Phrase 분포 (basket 아닌 것들 — hallucination)\n",
                        f"- 복수 bbox 프레임: {multi_box}/{len(pres)} ({multi_box/len(pres)*100:.1f}%)\n" if pres else "",
                        "| Side Phrase | 빈도 | 해석 |",
                        "|-------------|------|------|",
                    ]
                    total_side = sum(side_ctr.values()) or 1
                    for ph, cnt in side_ctr.most_common(20):
                        bar = "█" * min(15, int(cnt/total_side*100))
                        phrase_lines.append(f"| `{ph}` | {cnt} | {bar} |")
                    if not side_ctr:
                        phrase_lines.append("| (없음) — side phrase 없이 basket만 감지 | 0 | ✅ 깔끔 |")

                    # ── path_type별 ──
                    by_pt = defaultdict(lambda: {"n":0,"bsk":0,"iou_sum":0.0})
                    for r in pres:
                        pt = r["path_type"]
                        by_pt[pt]["n"]      += 1
                        by_pt[pt]["iou_sum"]+= r["basket_iou"]
                        if r["basket_grounding_success"]:
                            by_pt[pt]["bsk"] += 1

                    path_lines = [
                        "## path_type별 basket Grounding 성공률\n",
                        "| path_type | 프레임 | basket hit% | avg basket IoU |",
                        "|-----------|--------|-------------|----------------|",
                    ]
                    for pt in sorted(by_pt):
                        d = by_pt[pt]
                        hit = d["bsk"]/d["n"]*100 if d["n"] else 0
                        avg = d["iou_sum"]/d["n"] if d["n"] else 0
                        bar = "█"*int(hit/10)
                        path_lines.append(f"| {pt} | {d['n']} | **{hit:.0f}%** {bar} | {avg:.3f} |")

                    # ── IoU 분포 히스토그램 (basket bbox 기준) ──
                    bins = [0.001, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01]
                    bin_labels = ["0.001~0.1","0.1~0.2","0.2~0.3","0.3~0.5","0.5~0.7","0.7~1.0"]
                    iou_vals = [r["basket_iou"] for r in pres if r["basket_grounding_success"]]
                    iou_hist = [0]*len(bin_labels)
                    for v in iou_vals:
                        for i in range(len(bins)-1):
                            if bins[i] <= v < bins[i+1]:
                                iou_hist[i] += 1; break

                    iou_lines = [
                        "## basket Grounding IoU 분포\n",
                        f"(basket phrase grounding 성공 {len(iou_vals)}개 기준)\n",
                        "| IoU 구간 | 수 | 분포 |",
                        "|---------|-----|------|",
                    ]
                    max_h = max(iou_hist) if iou_hist else 1
                    for lbl, cnt in zip(bin_labels, iou_hist):
                        bar = "█"*int(cnt/max_h*20) if max_h else ""
                        iou_lines.append(f"| {lbl} | {cnt} | {bar} |")

                    # ── basket grounding 실패 케이스 샘플 ──
                    fail_cases = [r for r in pres if not r["basket_grounding_success"]]
                    sample_lines = [
                        "## basket grounding 실패 케이스 (basket 있는데 못 찾은 것)\n",
                        f"총 {len(fail_cases)}개 / {len(pres)}개 ({len(fail_cases)/len(pres)*100:.1f}%)\n" if pres else "",
                        "| path_type | t | cx | side_phrases | gt |",
                        "|-----------|---|----|--------------|----|",
                    ]
                    for r in fail_cases[:15]:
                        sample_lines.append(
                            f"| {r['path_type']} | {r['frame_t']} | {r['cx']} "
                            f"| `{r.get('side_phrases','')}` | {r['gt_class']} |"
                        )
                    if not fail_cases:
                        sample_lines.append("| ✅ 모든 basket 프레임에서 grounding 성공 | | | | |")

                    status = f"✅ {len(rows)}개 프레임 로드됨 ({n_ep}/150 에피소드)"
                    return (status,
                            summary,
                            "\n".join(phrase_lines),
                            "\n".join(path_lines),
                            "\n".join(iou_lines),
                            "\n".join(sample_lines))

                refresh5_btn.click(
                    load_recog_analysis,
                    inputs=[],
                    outputs=[status5, summary5_md, phrase5_md, bypath5_md, iou5_md, samples5_md]
                )

    return app


# ─── 메인 ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("[INIT] 모델 로딩...", flush=True)
    load_models()

    global SUBST_PAIRS
    SUBST_PAIRS = build_subst_pairs()
    print(f"[INIT] Substitution 쌍: {len(SUBST_PAIRS)}개", flush=True)

    app = build_ui()
    app.launch(server_port=args.port, share=args.share,
               server_name="0.0.0.0", show_error=True)


if __name__ == "__main__":
    main()
