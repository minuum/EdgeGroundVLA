# -*- coding: utf-8 -*-
"""
Grounding Evaluation Hub — exp56~64 grounding 모델 + base PG2 일괄 비교.

대상 (조사 완료):
  base PG2  : PaliGemma2-mix, LoRA 없음 (기준선)
  exp57     : PaliGemma-1 (pt-224) + LM LoRA
  exp58     : PaliGemma-2 (mix)   + LM LoRA
  exp59     : PaliGemma-2 (mix)   + LM LoRA (hard-neg, r=16)
  exp64     : PaliGemma-2 (mix)   + vision(SigLIP) LoRA  ← CH31/32 full-frame collapse
  exp56     : Kosmos-2 + LM LoRA  ← generate 시도, 가비지면 제외

공통 세트: viewpoint 49프레임(basket) + 의자 11장(OOD). 각 모델 × [hit/cx_MAE/cx_std/full-frame/OOD FP].

산출: docs/v5/grounding_hub/hub_results.json + 모델별 bbox 그리드.
Usage:
  .venv/bin/python3 scripts/eval_grounding_hub.py
  .venv/bin/python3 scripts/eval_grounding_hub.py --per-vp 6 --no-kosmos
"""
import sys, re, json, argparse, gc
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image, ImageDraw, ImageFont

from eval_exp64_grounding import build_basket_sample, CHAIRDIR  # 재사용

HUB = Path.home() / ".cache/huggingface/hub"
PG2 = HUB / "models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
PG1 = HUB / "models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
KOSMOS = ROOT / ".vlms/kosmos-2-patch14-224"
GR = ROOT / "runs/v5_nav/grounding"
OUT = ROOT / "docs/v5/grounding_hub"
LOC = re.compile(r"<loc(\d{4})>")

# (key, family, base_path, adapter_path)
MODELS = [
    ("base_pg2",  "pg", PG2, None),
    ("pure_kosmos", "kosmos", KOSMOS, None),   # Pure HF Kosmos-2, LoRA 없음 (grounding 기준선)
    ("exp57_pg1", "pg", PG1, GR / "exp57"),
    ("exp58_pg2", "pg", PG2, GR / "exp58"),
    ("exp59_pg2", "pg", PG2, GR / "exp59"),
    ("exp64_pg2", "pg", PG2, GR / "exp64_visionenc"),
    ("exp56_kosmos", "kosmos", KOSMOS, GR / "exp56"),
]


def load_pg(base, adapter):
    from transformers import PaliGemmaForConditionalGeneration
    m = PaliGemmaForConditionalGeneration.from_pretrained(
        str(base), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(adapter))
    return m.eval()


def make_pg_detect(model, proc):
    @torch.no_grad()
    def detect(img, phrase="gray basket"):
        inp = proc(text=f"<image>detect {phrase}", images=img, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        gen = model.generate(**inp, max_new_tokens=48, do_sample=False)
        raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        locs = [int(v) / 1023 for v in LOC.findall(raw)]
        if len(locs) >= 4:
            y1, x1, y2, x2 = locs[:4]
            return {"cx": (x1+x2)/2, "area": (x2-x1)*(y2-y1), "box": [x1, y1, x2, y2]}
        return None
    return detect


def make_kosmos_detect(model, proc):
    """Kosmos-2 grounding — post_process_generation으로 bbox 추출. 가비지면 None."""
    @torch.no_grad()
    def detect(img, phrase="gray basket"):
        prompt = f"<grounding><phrase> {phrase}</phrase>"
        inp = proc(text=prompt, images=img, return_tensors="pt").to("cuda")
        gen = model.generate(pixel_values=inp["pixel_values"],
                             input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                             image_embeds=None, image_embeds_position_mask=inp["image_embeds_position_mask"],
                             max_new_tokens=64, do_sample=False)
        txt = proc.batch_decode(gen, skip_special_tokens=True)[0]
        _, entities = proc.post_process_generation(txt)
        if entities:
            for _, _, bbs in entities:
                if bbs:
                    x1, y1, x2, y2 = bbs[0]
                    return {"cx": (x1+x2)/2, "area": (x2-x1)*(y2-y1), "box": [x1, y1, x2, y2]}
        return None
    return detect


def eval_model(detect, basket, chairs):
    cxs, errs, areas, hit, full, boxes = [], [], [], 0, 0, []
    for s in basket:
        try: d = detect(s["img"], "gray basket")
        except Exception: d = None
        if d is None: boxes.append(None); continue
        hit += 1; cxs.append(d["cx"]); areas.append(d["area"]); errs.append(abs(d["cx"] - s["cx_ref"]))
        if d["area"] > 0.9: full += 1
        boxes.append(d["box"])
    n = len(basket)
    fp, ood_boxes = 0, []
    for name, img in chairs:
        try: d = detect(img, "gray basket")
        except Exception: d = None
        if d is not None: fp += 1; ood_boxes.append(d["box"])
        else: ood_boxes.append(None)
    return {
        "basket_hit": hit / max(n, 1), "basket_n": n,
        "cx_mae": float(np.mean(errs)) if errs else None,
        "cx_std": float(np.std(cxs)) if cxs else None,
        "area_mean": float(np.mean(areas)) if areas else None,
        "fullframe_rate": full / max(n, 1),
        "ood_fp_rate": fp / max(len(chairs), 1), "ood_fp": fp, "ood_n": len(chairs),
    }, boxes, ood_boxes


def grid(samples, boxes, path, title_each):
    cols = min(4, len(samples)); rows = (len(samples) + cols - 1) // cols
    cell, hdr = 220, 20
    cv = Image.new("RGB", (cols * cell, rows * (cell + hdr)), (15, 22, 36))
    dr = ImageDraw.Draw(cv)
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception: font = ImageFont.load_default()
    for i, s in enumerate(samples):
        r, c = divmod(i, cols)
        th = s["img"].copy(); th.thumbnail((cell - 8, cell - 8))
        ox, oy = c * cell + 4, r * (cell + hdr) + hdr
        cv.paste(th, (ox, oy))
        if boxes[i]:
            x1, y1, x2, y2 = boxes[i]
            full = (x2-x1)*(y2-y1) > 0.9
            col = (239, 68, 68) if full else (34, 197, 94)
            dr.rectangle([ox+x1*th.width, oy+y1*th.height, ox+x2*th.width, oy+y2*th.height], outline=col, width=2)
        dr.text((ox, r*(cell+hdr)+4), title_each(s, i)[:30], fill=(148, 163, 184), font=font)
    cv.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-vp", type=int, default=6)
    ap.add_argument("--no-kosmos", action="store_true")
    ap.add_argument("--only", default="", help="쉼표구분 key만 평가하고 기존 JSON에 병합")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None

    basket = build_basket_sample(args.per_vp)
    chairs = [(p.stem, Image.open(p).convert("RGB")) for p in sorted(CHAIRDIR.glob("*.jpg"))]
    print(f"[SAMPLE] basket={len(basket)} chairs={len(chairs)}\n")

    results = {}
    if only and (OUT / "hub_results.json").exists():
        results = json.loads((OUT / "hub_results.json").read_text())  # 기존 병합
    for key, fam, base, adapter in MODELS:
        if only and key not in only:
            continue
        if fam == "kosmos" and args.no_kosmos:
            results[key] = {"skipped": "no-kosmos"}; continue
        print(f"[{key}] 로드 ({fam}, adapter={adapter and adapter.name})...")
        try:
            if fam == "pg":
                from transformers import PaliGemmaProcessor
                proc = PaliGemmaProcessor.from_pretrained(str(base))
                model = load_pg(base, adapter)
                detect = make_pg_detect(model, proc)
            else:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                proc = AutoProcessor.from_pretrained(str(base))
                model = AutoModelForVision2Seq.from_pretrained(
                    str(base), torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda")
                if adapter:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, str(adapter))
                model.eval()
                detect = make_kosmos_detect(model, proc)
        except Exception as e:
            print(f"  ✗ 로드 실패: {e}"); results[key] = {"error": str(e)[:200]}; continue

        stat, boxes, ood_boxes = eval_model(detect, basket, chairs)
        results[key] = stat
        # Kosmos 가비지 판정: hit 0%면 generate 붕괴로 간주
        if fam == "kosmos" and stat["basket_hit"] == 0:
            results[key]["note"] = "generate 붕괴 추정 (hit 0%) — Kosmos grounding 불가"
        grid(basket[:8], boxes[:8], OUT / f"grid_{key}.png", lambda s, i: f"{s['vp']} {s['tag']}")
        r = stat
        print(f"  basket hit={r['basket_hit']*100:.0f}% cxMAE={r['cx_mae'] and round(r['cx_mae'],3)} "
              f"full-frame={r['fullframe_rate']*100:.0f}% OOD_FP={r['ood_fp_rate']*100:.0f}%")
        del model, detect; gc.collect(); torch.cuda.empty_cache()
        print()

    (OUT / "hub_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("=" * 70)
    print(f"{'model':<16}{'hit':>7}{'cxMAE':>9}{'cx_std':>9}{'full-frame':>12}{'OOD_FP':>9}")
    print("-" * 70)
    for key, *_ in MODELS:
        r = results.get(key, {})
        if "basket_hit" not in r:
            print(f"{key:<16}  {r.get('skipped') or r.get('error','?')[:40]}"); continue
        mae = f"{r['cx_mae']:.3f}" if r['cx_mae'] is not None else "  -"
        std = f"{r['cx_std']:.3f}" if r['cx_std'] is not None else "  -"
        print(f"{key:<16}{r['basket_hit']*100:>6.0f}%{mae:>9}{std:>9}"
              f"{r['fullframe_rate']*100:>11.0f}%{r['ood_fp_rate']*100:>8.0f}%")
    print(f"\n[SAVE] {OUT}/hub_results.json + grids")


if __name__ == "__main__":
    main()
