# -*- coding: utf-8 -*-
"""
exp64 Grounding LoRA 종합 평가 — base PaliGemma2 대비.

"grounding LoRA가 실제로 작동하는가 vs 학습셋 암기인가"를 3개 세트로 판정:
  ① In-dist basket  : V5 basket 프레임 — hit / cx_MAE / cx_std / full-frame (좌표 정밀도)
  ② OOD 미학습 객체  : 의자 이미지에 "detect gray basket" — 오탐(FP) 여부
                       (3-negative 지름길 "NOT{pot,ball,person}→basket" 과적합 검증)
  ③ 시각 그리드      : basket 프레임 base vs exp64 나란히 + 의자 OOD 오버레이

base PG2(LoRA 없음)와 동일 샘플로 대조. exp64 어댑터는 PEFT로 주입.

산출:
  docs/v5/exp64_eval/eval_results.json
  docs/v5/exp64_eval/basket_compare_grid.png
  docs/v5/exp64_eval/ood_chair_grid.png

Usage:
  .venv/bin/python3 scripts/eval_exp64_grounding.py
  .venv/bin/python3 scripts/eval_exp64_grounding.py --per-vp 6
"""
import sys, re, json, argparse, gc
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, h5py
from PIL import Image, ImageDraw, ImageFont

PG2 = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
ADAPTER = ROOT / "runs/v5_nav/grounding/exp64_visionenc"
GDIR = ROOT / "docs/v5/grounding_ablation"
CHAIRDIR = ROOT / "docs/v5/chair_probe/images"
OUT = ROOT / "docs/v5/exp64_eval"
LOC = re.compile(r"<loc(\d{4})>")


def pos_b(cx): return "L" if cx < 0.4 else ("R" if cx > 0.6 else "C")
def dist_b(a): return "far" if a < 0.05 else ("near" if a > 0.3 else "mid")


def build_basket_sample(per_vp):
    hsv = json.loads((GDIR / "grounding_hsv.json").read_text())
    buckets = defaultdict(list)
    for ep in hsv:
        for fr in ep["frames"]:
            if fr["cx"] is None: continue
            vp = f"{pos_b(fr['cx'])}-{dist_b(fr.get('area') or 0)}"
            buckets[vp].append((ep["episode"], fr["frame_idx"], fr["cx"], vp))
    rng = np.random.default_rng(42)
    sample = []
    for vp, items in buckets.items():
        idx = rng.permutation(len(items))[:per_vp]
        sample += [items[i] for i in idx]
    out, cache = [], {}
    for epp, fi, cx_ref, vp in sample:
        if epp not in cache:
            try:
                with h5py.File(epp, "r") as f: cache[epp] = f["observations"]["images"][:]
            except Exception: cache[epp] = None
        imgs = cache[epp]
        if imgs is None or fi >= len(imgs): continue
        out.append({"img": Image.fromarray(imgs[fi].astype("uint8")).convert("RGB"),
                    "cx_ref": cx_ref, "vp": vp, "tag": f"{Path(epp).stem[:14]}#{fi}"})
    return out


def load_model(with_adapter):
    from transformers import PaliGemmaForConditionalGeneration
    dev = torch.device("cuda")
    m = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG2), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(dev)
    if with_adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(ADAPTER))
    return m.eval()


def make_detect(model, proc):
    dev = torch.device("cuda")
    @torch.no_grad()
    def detect(img, phrase="gray basket"):
        inp = proc(text=f"<image>detect {phrase}", images=img, return_tensors="pt").to(dev)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        gen = model.generate(**inp, max_new_tokens=48, do_sample=False)
        raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        locs = [int(v) / 1023 for v in LOC.findall(raw)]
        if len(locs) >= 4:
            y1, x1, y2, x2 = locs[:4]
            return {"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                    "area": (x2 - x1) * (y2 - y1), "box": [x1, y1, x2, y2]}
        return None
    return detect


def eval_basket(detect, sample):
    cxs, errs, areas, hit, full = [], [], [], 0, 0
    boxes, misses = [], []
    pv = defaultdict(lambda: {"n": 0, "hit": 0, "full": 0, "err": []})
    for s in sample:
        b = pv[s["vp"]]; b["n"] += 1
        d = detect(s["img"], "gray basket")
        if d is None:
            boxes.append(None); misses.append({"tag": s["tag"], "vp": s["vp"]}); continue
        hit += 1; b["hit"] += 1
        cxs.append(d["cx"]); areas.append(d["area"]); errs.append(abs(d["cx"] - s["cx_ref"]))
        b["err"].append(abs(d["cx"] - s["cx_ref"]))
        if d["area"] > 0.9: full += 1; b["full"] += 1
        boxes.append(d["box"])
    n = len(sample)
    per_vp = {vp: {"n": b["n"], "hit_rate": b["hit"]/max(b["n"],1),
                   "fullframe_rate": b["full"]/max(b["n"],1),
                   "cx_mae": float(np.mean(b["err"])) if b["err"] else None}
              for vp, b in sorted(pv.items())}
    return {
        "hit_rate": hit / max(n, 1), "n": n,
        "cx_mae": float(np.mean(errs)) if errs else None,
        "cx_std": float(np.std(cxs)) if cxs else None,
        "area_mean": float(np.mean(areas)) if areas else None,
        "fullframe_rate": full / max(n, 1),
        "per_vp": per_vp, "misses": misses,
    }, boxes


def eval_ood(detect, chair_imgs):
    """의자에 'detect gray basket' → 박스 나오면 FP(오탐)."""
    fp, boxes, fp_names = 0, [], []
    for name, img in chair_imgs:
        d = detect(img, "gray basket")
        if d is not None:
            fp += 1; boxes.append(d["box"]); fp_names.append({"name": name, "area": round(d["area"], 3)})
        else:
            boxes.append(None)
    n = len(chair_imgs)
    return {"fp_rate": fp / max(n, 1), "fp": fp, "n": n, "fp_names": fp_names}, boxes


def grid(samples, boxes_base, boxes_lora, path, title_each):
    cols = min(4, len(samples)); rows = (len(samples) + cols - 1) // cols
    cell, hdr = 240, 22
    canvas = Image.new("RGB", (cols * cell, rows * (cell + hdr)), (15, 22, 36))
    dr = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception: font = ImageFont.load_default()
    for i, s in enumerate(samples):
        r, c = divmod(i, cols)
        th = s["img"].copy(); th.thumbnail((cell - 12, cell - 12))
        ox, oy = c * cell + 6, r * (cell + hdr) + hdr
        canvas.paste(th, (ox, oy))
        for box, col in ((boxes_base[i], (96, 165, 250)), (boxes_lora[i], (34, 197, 94))):
            if box:
                x1, y1, x2, y2 = box
                dr.rectangle([ox + x1 * th.width, oy + y1 * th.height,
                              ox + x2 * th.width, oy + y2 * th.height], outline=col, width=2)
        dr.text((ox, r * (cell + hdr) + 4), title_each(s, i)[:34], fill=(148, 163, 184), font=font)
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-vp", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import PaliGemmaProcessor
    proc = PaliGemmaProcessor.from_pretrained(str(PG2))

    print("[SAMPLE] basket 프레임 구성...")
    basket = build_basket_sample(args.per_vp)
    chairs = [(p.stem, Image.open(p).convert("RGB")) for p in sorted(CHAIRDIR.glob("*.jpg"))]
    print(f"  basket={len(basket)} frames, chairs={len(chairs)} imgs\n")

    results = {}
    store = {}
    for label, with_ad in (("base", False), ("exp64", True)):
        print(f"[{label}] 모델 로드 ({'PG2+adapter' if with_ad else 'PG2 base'})...")
        model = load_model(with_ad)
        detect = make_detect(model, proc)
        b_stat, b_boxes = eval_basket(detect, basket)
        o_stat, o_boxes = eval_ood(detect, chairs)
        results[label] = {"basket": b_stat, "ood_chair": o_stat}
        store[label] = {"basket_boxes": b_boxes, "ood_boxes": o_boxes}
        print(f"  basket: hit={b_stat['hit_rate']*100:.0f}% cxMAE={b_stat['cx_mae'] and round(b_stat['cx_mae'],3)} "
              f"cx_std={b_stat['cx_std'] and round(b_stat['cx_std'],3)} full-frame={b_stat['fullframe_rate']*100:.0f}%")
        print(f"  OOD chair FP(basket 오탐)={o_stat['fp_rate']*100:.0f}% ({o_stat['fp']}/{o_stat['n']})\n")
        del model, detect; gc.collect(); torch.cuda.empty_cache()

    # 시각 그리드 (basket: 최대 8장 / chair: 전체)
    bk = basket[:8]
    grid(bk, store["base"]["basket_boxes"][:8], store["exp64"]["basket_boxes"][:8],
         OUT / "basket_compare_grid.png", lambda s, i: f"{s['vp']} {s['tag']}")
    chair_samples = [{"img": img, "name": name} for name, img in chairs]
    grid(chair_samples, store["base"]["ood_boxes"], store["exp64"]["ood_boxes"],
         OUT / "ood_chair_grid.png", lambda s, i: s["name"])

    (OUT / "eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("=" * 56)
    print(f"{'metric':<22}{'base':>14}{'exp64':>14}")
    print("-" * 56)
    b, e = results["base"]["basket"], results["exp64"]["basket"]
    print(f"{'basket hit':<22}{b['hit_rate']*100:>13.0f}%{e['hit_rate']*100:>13.0f}%")
    print(f"{'basket cx_MAE':<22}{(b['cx_mae'] or 0):>14.3f}{(e['cx_mae'] or 0):>14.3f}")
    print(f"{'basket cx_std':<22}{(b['cx_std'] or 0):>14.3f}{(e['cx_std'] or 0):>14.3f}")
    print(f"{'basket full-frame':<22}{b['fullframe_rate']*100:>13.0f}%{e['fullframe_rate']*100:>13.0f}%")
    bo, eo = results["base"]["ood_chair"], results["exp64"]["ood_chair"]
    print(f"{'OOD chair FP':<22}{bo['fp_rate']*100:>13.0f}%{eo['fp_rate']*100:>13.0f}%")
    print(f"\n[SAVE] {OUT}/eval_results.json + 2 grids")


if __name__ == "__main__":
    main()
