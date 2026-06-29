"""
CH55: 각 perception model이 실제 세션 frame 0에서 basket을 어떻게 탐지하는지 시각화.

출력: docs/v5/ch55_viz/
  - frame_HHMMSS_pg2.jpg    : PG2 oracle bbox
  - frame_HHMMSS_kosmos.jpg : Kosmos-2 bbox
  - frame_HHMMSS_owlv2.jpg  : OWL-v2 bbox
  - frame_HHMMSS_clip.jpg   : CLIP direction 화살표
  - frame_HHMMSS_compare.jpg: 4개 나란히 비교
"""

import io, json, time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

SESSIONS_DIR = Path("docs/inference_sessions")
OUT_DIR      = Path("docs/v5/ch55_viz")
OUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda"
PHRASE = "gray basket"

# ── 대표 세션 선택 (다양한 basket 위치: left/center/right) ──────────────────
TARGET_SESSIONS = []
for sp in sorted(SESSIONS_DIR.glob("session_20260626*.h5")):
    with h5py.File(sp) as f:
        if "grounding/bbox" not in f: continue
        bbox = f["grounding/bbox"][:]
        if len(bbox) < 2: continue
        if bbox[1,3] < 0.5: continue
        cx = float(bbox[1,0])
        if "observations" not in f: continue
        raw = f["observations"]["images"][0]
        TARGET_SESSIONS.append((sp, cx, raw))
        if len(TARGET_SESSIONS) >= 8:
            break

print(f"대표 프레임 {len(TARGET_SESSIONS)}개 선택")


def load_img(raw):
    if raw.ndim == 1:
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    return Image.fromarray(raw.astype(np.uint8)).convert("RGB")


def draw_bbox(img, cx_norm, cy_norm=0.5, area_norm=0.05, color=(0,255,0), label="", linewidth=3):
    """normalized cx,cy,area → 실제 픽셀 bbox 그리기."""
    W, H = img.size
    w = int(np.sqrt(area_norm) * W * 1.5)
    h = int(np.sqrt(area_norm) * H * 2.0)
    x1 = int(cx_norm * W - w//2); x2 = int(cx_norm * W + w//2)
    y1 = int(cy_norm * H - h//2); y2 = int(cy_norm * H + h//2)
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(W,x2), min(H,y2)
    draw = ImageDraw.Draw(img)
    for i in range(linewidth):
        draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=color)
    if label:
        draw.rectangle([x1, y1-18, x1+len(label)*8+4, y1], fill=color)
        draw.text((x1+2, y1-17), label, fill=(0,0,0))
    return img


def draw_xyxy(img, x1, y1, x2, y2, color=(0,255,0), label="", linewidth=3):
    draw = ImageDraw.Draw(img)
    for i in range(linewidth):
        draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=color)
    if label:
        draw.rectangle([x1, y1-18, x1+len(label)*8+4, y1], fill=color)
        draw.text((x1+2, y1-17), label, fill=(0,0,0))
    return img


def draw_direction(img, direction, color=(255,200,0), label="CLIP"):
    """방향 화살표 오버레이."""
    W, H = img.size
    draw = ImageDraw.Draw(img)
    cx, cy = W//2, H//2
    arrow = {"L": (-80,0), "C": (0,0), "R": (80,0)}
    dx, dy = arrow.get(direction, (0,0))
    if direction != "C":
        draw.line([(cx, cy),(cx+dx, cy+dy)], fill=color, width=6)
        # 화살촉
        sign = 1 if dx>0 else -1
        draw.polygon([(cx+dx+sign*15,cy), (cx+dx-sign*8,cy-10), (cx+dx-sign*8,cy+10)], fill=color)
    # 라벨
    draw.rectangle([4, 4, 90, 22], fill=(0,0,0,180))
    draw.text((6,5), f"{label}: {direction}", fill=color)
    return img


def add_header(img, text, color=(200,200,200)):
    W, H = img.size
    new = Image.new("RGB", (W, H+24), (30,30,30))
    new.paste(img, (0,24))
    draw = ImageDraw.Draw(new)
    draw.rectangle([0,0,W,23], fill=(40,40,60))
    draw.text((W//2 - len(text)*4, 4), text, fill=color)
    return new


# ── 모델 로드 ───────────────────────────────────────────────────────────────

print("PG2는 h5 oracle 사용 (별도 로드 불필요)")

print("CLIP 로딩...")
from transformers import CLIPProcessor, CLIPModel as HF_CLIP
clip_model = HF_CLIP.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
CLIP_TEXTS = [
    "a gray basket on the left side",
    "a gray basket in the center",
    "a gray basket on the right side",
    "no basket visible",
]

print("Kosmos-2 로딩...")
from transformers import AutoProcessor, AutoModelForVision2Seq
kosmos_model = AutoModelForVision2Seq.from_pretrained(".vlms/kosmos-2-patch14-224").to(DEVICE)
kosmos_proc  = AutoProcessor.from_pretrained(".vlms/kosmos-2-patch14-224")

print("OWL-v2 로딩...")
from transformers import Owlv2Processor, Owlv2ForObjectDetection
owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEVICE)
owl_proc  = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")


def infer_clip(img):
    inputs = clip_proc(text=CLIP_TEXTS, images=img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        logits = clip_model(**inputs).logits_per_image[0]
        probs  = logits.softmax(dim=0).cpu().numpy()
    best = int(np.argmax(probs))
    if best == 3 or probs[best] < 0.3:
        return None, probs
    return ["L","C","R"][best], probs


def infer_kosmos(img):
    prompt = "<grounding><phrase>gray basket</phrase>"
    inputs = kosmos_proc(text=prompt, images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = kosmos_model.generate(**inputs, max_new_tokens=64)
    out    = kosmos_proc.decode(ids[0], skip_special_tokens=False)
    result = kosmos_proc.post_process_generation(out, cleanup_and_extract=True)
    entities = result[1] if isinstance(result, tuple) and len(result)>1 else []
    for ent in entities:
        bboxes = ent[2] if len(ent)>=3 else []
        if bboxes:
            return bboxes[0]  # (x1,y1,x2,y2) normalized
    return None


def infer_owlv2(img):
    W, H = img.width, img.height
    inputs = owl_proc(text=[[PHRASE]], images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = owl_model(**inputs)
    res = owl_proc.post_process_object_detection(out, threshold=0.1,
          target_sizes=[(H, W)])[0]
    boxes  = res["boxes"]
    scores = res["scores"].cpu().numpy()
    if len(boxes)==0: return None
    best = int(np.argmax(scores))
    x1,y1,x2,y2 = boxes[best].cpu().tolist()
    return x1/W, y1/H, x2/W, y2/H  # normalized


# ── 시각화 생성 ─────────────────────────────────────────────────────────────

compare_strips = []

for sp, pg2_cx, raw in TARGET_SESSIONS:
    tag = sp.stem[-6:]
    img_orig = load_img(raw)
    W, H = img_orig.size

    # PG2 oracle (frame 1 결과를 frame 0에 그림)
    with h5py.File(sp) as f:
        b = f["grounding/bbox"][1]
        pg2_cx_v, pg2_cy_v, pg2_area_v = float(b[0]), float(b[1]), float(b[2])
    img_pg2 = img_orig.copy()
    draw_bbox(img_pg2, pg2_cx_v, pg2_cy_v, pg2_area_v, color=(0,200,255), label=f"PG2 cx={pg2_cx_v:.2f}")
    img_pg2 = add_header(img_pg2, f"PG2 (oracle)  cx={pg2_cx_v:.2f}", (0,200,255))

    # CLIP
    t0=time.time()
    clip_dir, clip_probs = infer_clip(img_orig)
    clip_lat = int((time.time()-t0)*1000)
    img_clip = img_orig.copy()
    if clip_dir:
        draw_direction(img_clip, clip_dir, label="CLIP")
    else:
        ImageDraw.Draw(img_clip).text((6,5), "CLIP: N/D", fill=(255,100,100))
    img_clip = add_header(img_clip, f"CLIP  {clip_dir or 'N/D'}  {clip_lat}ms", (255,220,100))

    # Kosmos-2
    t0=time.time()
    k_bbox = infer_kosmos(img_orig)
    k_lat = int((time.time()-t0)*1000)
    img_kosmos = img_orig.copy()
    if k_bbox:
        x1,y1,x2,y2 = k_bbox
        draw_xyxy(img_kosmos, int(x1*W), int(y1*H), int(x2*W), int(y2*H),
                  color=(150,255,100), label=f"K2 cx={(x1+x2)/2:.2f}")
    else:
        ImageDraw.Draw(img_kosmos).text((6,5), "Kosmos: N/D", fill=(255,100,100))
    k_cx = (k_bbox[0]+k_bbox[2])/2 if k_bbox else None
    k_cx_str = f"{k_cx:.2f}" if k_cx is not None else "N/D"
    img_kosmos = add_header(img_kosmos, f"Kosmos-2  cx={k_cx_str}  {k_lat}ms", (150,255,100))

    # OWL-v2
    t0=time.time()
    o_bbox = infer_owlv2(img_orig)
    o_lat = int((time.time()-t0)*1000)
    img_owl = img_orig.copy()
    if o_bbox:
        x1,y1,x2,y2 = o_bbox
        draw_xyxy(img_owl, int(x1*W), int(y1*H), int(x2*W), int(y2*H),
                  color=(255,130,200), label=f"OWL cx={(x1+x2)/2:.2f}")
    else:
        ImageDraw.Draw(img_owl).text((6,5), "OWL-v2: N/D", fill=(255,100,100))
    o_cx = (o_bbox[0]+o_bbox[2])/2 if o_bbox else None
    o_cx_str = f"{o_cx:.2f}" if o_cx is not None else "N/D"
    img_owl = add_header(img_owl, f"OWL-v2  cx={o_cx_str}  {o_lat}ms", (255,130,200))

    # 4-panel compare
    panels = [img_pg2, img_clip, img_kosmos, img_owl]
    panel_w = 320
    panel_h = int(panels[0].height * panel_w / panels[0].width)
    panels_r = [p.resize((panel_w, panel_h)) for p in panels]
    compare = Image.new("RGB", (panel_w*4+6, panel_h), (20,20,20))
    for i, p in enumerate(panels_r):
        compare.paste(p, (i*(panel_w+2), 0))
    compare.save(OUT_DIR / f"compare_{tag}.jpg", quality=88)
    compare_strips.append(f"compare_{tag}.jpg")

    # 개별 저장
    for name, img_v in [("pg2",img_pg2),("clip",img_clip),("kosmos",img_kosmos),("owl",img_owl)]:
        img_v.save(OUT_DIR / f"{tag}_{name}.jpg", quality=88)

    print(f"  {tag}  PG2cx={pg2_cx_v:.2f}  CLIP={clip_dir}  K2={k_cx_str}  OWL={o_cx_str}")

# 결과 JSON
out = {"compare_strips": compare_strips}
(OUT_DIR / "manifest.json").write_text(json.dumps(out, indent=2))
print(f"\n완료: {OUT_DIR}")
