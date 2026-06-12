# -*- coding: utf-8 -*-
"""
의자(Chair) 인식 검증 — base PaliGemma2(LoRA 없이)가 다양한 외부 의자 이미지를
`detect chair`로 안정적으로 grounding하는지 실측.

probe_pg2_objects.py와의 차이:
  - probe_pg2_objects: 우리 basket 프레임에 chair 던져 "오탐" 측정
  - 이 스크립트       : 실제 의자 이미지 다종에서 "정탐률 + phrase 변형 영향" 측정

목적(6/10 plan_20260610_chair_object_pipeline):
  1. PG2가 chair 개념을 아는가 (필요조건 sanity)
  2. phrase 변형(chair / white chair / stool / office chair)이 hit율에 미치는 영향
     → 색 수식어 제거("그냥 의자") 결정의 근거/반증

⚠️ 한계: 스톡 이미지는 스튜디오 정면 샷 — 로봇 low-angle(30cm)·224px·원거리 POV와 다름.
   이 검증은 필요조건이지 충분조건 아님. 최종 검증은 로봇 프레임으로 별도 수행.

산출:
  docs/v5/chair_probe/images/*.jpg          (다운로드 캐시)
  docs/v5/chair_probe/chair_recognition.json (수치)
  docs/v5/chair_probe/chair_recognition_grid.png (bbox 오버레이 그리드)

Usage:
  .venv/bin/python3 scripts/probe_chair_recognition.py
"""
import sys, re, json, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image, ImageDraw, ImageFont

PG2 = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
LOC = re.compile(r"<loc(\d{4})>")
CDIR = ROOT / "docs/v5/chair_probe"
IMGDIR = CDIR / "images"

# Openverse/Wikimedia에서 큐레이션한 실제 의자 이미지 (종류·색·각도·배경 다양)
CHAIR_URLS = [
    ("leather_ergonomic",  "https://live.staticflickr.com/2473/3603638141_9e99fd630f_m.jpg"),
    ("eames_zenith",       "https://live.staticflickr.com/7149/6462248701_c164fc8bfb.jpg"),
    ("steelcase_office",   "https://live.staticflickr.com/3254/2730613340_13097c67ea_b.jpg"),
    ("white_office_rows",  "https://live.staticflickr.com/2712/4272145651_6b8cd01f7d_b.jpg"),
    ("arper_aston_office", "https://live.staticflickr.com/3071/2629764552_0520445520_b.jpg"),
    ("office_at_work",     "https://live.staticflickr.com/6038/6269216038_b2ea82bdb2_b.jpg"),
    ("wooden_dining",      "https://upload.wikimedia.org/wikipedia/commons/e/e3/Wooden_Dining_Chair.jpg"),
    ("bar_stools_four",    "https://live.staticflickr.com/3743/12275789166_9d47fecd51_b.jpg"),
    ("bar_stools_two",     "https://live.staticflickr.com/3190/2897079476_8d97d6cd1f_b.jpg"),
    ("white_chair_room",   "https://live.staticflickr.com/7163/6637673595_e5663fbcef_b.jpg"),
    ("vanity_stool",       "https://live.staticflickr.com/2937/13827499454_5a959ce151.jpg"),
]

PHRASES = ["chair", "white chair", "stool", "office chair"]


def download():
    IMGDIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name, url in CHAIR_URLS:
        dst = IMGDIR / f"{name}.jpg"
        if not dst.exists():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "MoNaVLA-research/1.0 (academic)"})
                dst.write_bytes(urllib.request.urlopen(req, timeout=30).read())
                print(f"  ↓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                continue
        try:
            img = Image.open(dst).convert("RGB")
            out.append((name, img))
        except Exception as e:
            print(f"  ✗ open {name}: {e}")
    return out


def main():
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    CDIR.mkdir(parents=True, exist_ok=True)
    print("[DL] 의자 이미지 다운로드...")
    imgs = download()
    print(f"[DL] {len(imgs)}장 확보\n")

    # exp64 학습 중 GPU VRAM 보호 — 여유 < 10GB면 CPU로 안전하게
    use_cpu = True
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        use_cpu = free < 10 * 1e9
    dev = torch.device("cpu" if use_cpu else "cuda")
    dtype = torch.float32 if use_cpu else torch.bfloat16
    print(f"[DEV] {'CPU (GPU 여유 부족 — exp64 보호)' if use_cpu else 'CUDA'}")
    proc = PaliGemmaProcessor.from_pretrained(str(PG2))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG2), torch_dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
    print("[LOAD] base PaliGemma2 (NO LoRA)\n")

    def detect(img, phrase):
        inp = proc(text=f"<image>detect {phrase}", images=img, return_tensors="pt").to(dev)
        inp["pixel_values"] = inp["pixel_values"].to(dtype)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=48, do_sample=False)
        raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        locs = [int(v) / 1023 for v in LOC.findall(raw)]
        if len(locs) >= 4:
            y1, x1, y2, x2 = locs[:4]
            return {"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                    "area": (x2 - x1) * (y2 - y1), "box": [x1, y1, x2, y2]}
        return None

    # phrase × image detect
    results = {ph: {} for ph in PHRASES}
    per_image = {}  # name -> chair phrase box (그리드 오버레이용)
    print(f"{'phrase':<14}{'hit':>8}{'area_mean':>11}{'area_std':>10}{'cx_std':>9}")
    print("-" * 52)
    for ph in PHRASES:
        cxs, areas, hits, boxes = [], [], 0, {}
        for name, img in imgs:
            det = detect(img, ph)
            if det is not None:
                hits += 1
                cxs.append(det["cx"]); areas.append(det["area"]); boxes[name] = det["box"]
        n = len(imgs)
        results[ph] = {
            "hit_rate": hits / max(n, 1),
            "area_mean": float(np.mean(areas)) if areas else None,
            "area_std": float(np.std(areas)) if areas else None,
            "cx_std": float(np.std(cxs)) if cxs else None,
            "hits": hits, "n": n,
        }
        if ph == "chair":
            per_image = boxes
        r = results[ph]
        am = f"{r['area_mean']:.3f}" if r['area_mean'] is not None else "  -  "
        ast = f"{r['area_std']:.3f}" if r['area_std'] is not None else "  -  "
        cst = f"{r['cx_std']:.3f}" if r['cx_std'] is not None else "  -  "
        print(f"{ph:<14}{hits}/{n:<6}{am:>11}{ast:>10}{cst:>9}")

    # 시각 그리드 (detect chair bbox 오버레이)
    cols = 4
    rows = (len(imgs) + cols - 1) // cols
    cell = 280
    grid = Image.new("RGB", (cols * cell, rows * cell), (15, 22, 36))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for i, (name, img) in enumerate(imgs):
        r, c = divmod(i, cols)
        th = img.copy(); th.thumbnail((cell - 16, cell - 40))
        ox, oy = c * cell + 8, r * cell + 8
        grid.paste(th, (ox, oy))
        box = per_image.get(name)
        if box:
            x1, y1, x2, y2 = box
            bx1, by1 = ox + x1 * th.width, oy + y1 * th.height
            bx2, by2 = ox + x2 * th.width, oy + y2 * th.height
            draw.rectangle([bx1, by1, bx2, by2], outline=(34, 197, 94), width=3)
            tag, col = "✓ chair", (134, 239, 172)
        else:
            tag, col = "✗ miss", (252, 165, 165)
        draw.text((ox, oy + th.height + 4), f"{name}", fill=(148, 163, 184), font=font)
        draw.text((ox, oy + th.height + 22), tag, fill=col, font=font)
    grid_path = CDIR / "chair_recognition_grid.png"
    grid.save(grid_path)

    (CDIR / "chair_recognition.json").write_text(json.dumps(
        {"n_images": len(imgs), "phrases": PHRASES, "results": results,
         "note": "스톡 이미지 — 로봇 POV 아님. 필요조건 검증용."}, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] {CDIR}/chair_recognition.json")
    print(f"[SAVE] {grid_path}")


if __name__ == "__main__":
    main()
