# -*- coding: utf-8 -*-
"""
멀티 객체 zero-shot grounding 사전검증 — plan_20260621_instruction_grounding.md §3.

목적: base PG2(LoRA 없음, 현재 production과 동일 zero-shot)가 "gray basket" 외
다른 객체에도 비슷한 품질로 반응하는지 확인. 여기서 실패하면(바스켓 특화)
instruction을 grounding 프롬프트에 연동하는 코드 변경 자체가 무의미하므로,
서버 코드를 건드리기 전에 먼저 검증한다.

대상: docs/object_test_images/(기존에 이미 있던 5장 — apple/mug/coke/chair/cone).
기준: §B/§H에서 측정한 바스켓 grounding 품질(hit 98%, cx_MAE 0.126)과 비슷한지.

Usage:
  .venv/bin/python3 scripts/eval/eval_multi_object_grounding.py
"""
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent.parent
PG2 = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
IMG_DIR = ROOT / "docs/object_test_images"
OUT_DIR = ROOT / "docs/v5/grounding_hub"
LOC_RE = re.compile(r"<loc(\d{4})>")

# (파일 패턴, detect phrase)
TARGETS = [
    ("test_apple_floor", "green apple"),
    ("test_blue_mug_floor", "blue mug"),
    ("test_coke_can_floor", "red coke can"),
    ("test_chair_obstacle", "chair"),
    ("test_cone_obstacle", "orange cone"),
]


def detect(model, proc, img, phrase):
    inp = proc(text=f"detect {phrase}", images=img, return_tensors="pt").to(model.device)
    inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=48, min_new_tokens=1, do_sample=False)
    raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
    locs = [int(v) / 1023.0 for v in LOC_RE.findall(raw)]
    if len(locs) >= 4:
        y1, x1, y2, x2 = locs[:4]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return {"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "area": (x2 - x1) * (y2 - y1),
                "box": [x1, y1, x2, y2], "raw": raw}
    return {"box": None, "raw": raw}


def grid(samples, path):
    cols = min(3, len(samples)); rows = (len(samples) + cols - 1) // cols
    cell, hdr = 280, 24
    cv = Image.new("RGB", (cols * cell, rows * (cell + hdr)), (15, 22, 36))
    dr = ImageDraw.Draw(cv)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    for i, s in enumerate(samples):
        r, c = divmod(i, cols)
        th = s["img"].copy(); th.thumbnail((cell - 8, cell - 8))
        ox, oy = c * cell + 4, r * (cell + hdr) + hdr
        cv.paste(th, (ox, oy))
        if s["box"]:
            x1, y1, x2, y2 = s["box"]
            full = (x2 - x1) * (y2 - y1) > 0.9
            col = (239, 68, 68) if full else (34, 197, 94)
            dr.rectangle([ox + x1 * th.width, oy + y1 * th.height,
                         ox + x2 * th.width, oy + y2 * th.height], outline=col, width=3)
        label = f"{s['phrase']} ({'HIT' if s['box'] else 'MISS'} a={s['area']:.3f})" if s["box"] else f"{s['phrase']} (MISS)"
        dr.text((ox, r * (cell + hdr) + 4), label[:36], fill=(148, 163, 184), font=font)
    cv.save(path)


def main():
    print(f"[로드] {PG2}")
    proc = PaliGemmaProcessor.from_pretrained(str(PG2))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG2), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": "cuda"},
    ).eval()

    results = []
    samples_for_grid = []
    for stem_prefix, phrase in TARGETS:
        matches = list(IMG_DIR.glob(f"{stem_prefix}*"))
        if not matches:
            print(f"  [스킵] {stem_prefix}* 없음")
            continue
        fpath = matches[0]
        img = Image.open(fpath).convert("RGB")
        d = detect(model, proc, img, phrase)
        hit = d["box"] is not None
        area = d.get("area")
        cx = d.get("cx")
        print(f"  [{phrase:14s}] {fpath.name:40s} hit={hit} area={area and round(area,3)} "
              f"cx={cx and round(cx,3)} raw='{d['raw'][:60]}'")
        results.append({"phrase": phrase, "file": fpath.name, "hit": hit, "area": area, "cx": cx,
                        "cy": d.get("cy"), "raw": d["raw"]})
        samples_for_grid.append({"img": img, "phrase": phrase, "box": d["box"], "area": area or 0.0})

    out_path = OUT_DIR / "multi_object_grounding.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    grid(samples_for_grid, OUT_DIR / "grid_multi_object.png")

    n = len(results)
    hits = sum(1 for r in results if r["hit"])
    full = sum(1 for r in results if r["hit"] and r["area"] > 0.9)
    print(f"\n[요약] hit={hits}/{n} ({hits/max(n,1)*100:.0f}%)  full-frame={full}/{n}")
    print(f"[저장] {out_path}")
    print("\n[판정 기준] base PG2 바스켓 grounding(§B): hit 98%, full-frame 0%")


if __name__ == "__main__":
    main()
