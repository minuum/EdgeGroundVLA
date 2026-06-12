# -*- coding: utf-8 -*-
"""
증강(Augmentation) 기반 grounding 강건성 테스트 — base PaliGemma2.

실제 에피소드가 한정적이므로, 기존 basket 프레임에 변형을 가해
"어떤 조건에서 grounding이 흔들리는가"를 체계적으로 점검.
→ 약한 변형 = 데이터셋이 보강해야 할 다양성 축.

변형 종류 (실주행 관련):
  original / bright_up / bright_down / low_contrast / blur / noise /
  downscale(저해상도) / rotate_cw / rotate_ccw / hflip(좌우반전) / occlude(부분가림)

각 (프레임 × 변형) → base PG2 detect.
지표: hit율 / cx drift(원본 대비 중심 이동, hflip은 1-cx 기준) / full-frame율.

산출: docs/v5/grounding_hub/aug_robustness.json + aug_grid.png
Usage:
  .venv/bin/python3 scripts/grounding_augmentation_robustness.py --n 12
"""
import sys, re, json, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from eval_exp64_grounding import build_basket_sample
from eval_grounding_hub import MODELS, load_pg, make_pg_detect, make_kosmos_detect

LOC = re.compile(r"<loc(\d{4})>")
OUT = ROOT / "docs/v5/grounding_hub"


def aug_funcs():
    def noise(im):
        a = np.asarray(im).astype(np.float32)
        a += np.random.default_rng(0).normal(0, 22, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
    def downscale(im):
        w, h = im.size
        return im.resize((w // 3, h // 3)).resize((w, h))
    def occlude(im):
        im = im.copy(); d = ImageDraw.Draw(im); w, h = im.size
        d.rectangle([0, int(h*0.55), int(w*0.4), h], fill=(90, 90, 90))  # 좌하단 가림
        return im
    return [
        ("original",    lambda im: im),
        ("bright_up",   lambda im: ImageEnhance.Brightness(im).enhance(1.6)),
        ("bright_down", lambda im: ImageEnhance.Brightness(im).enhance(0.45)),
        ("low_contrast",lambda im: ImageEnhance.Contrast(im).enhance(0.5)),
        ("blur",        lambda im: im.filter(ImageFilter.GaussianBlur(2.2))),
        ("noise",       noise),
        ("downscale",   downscale),
        ("rotate_cw",   lambda im: im.rotate(-12, expand=False, fillcolor=(60,60,60))),
        ("rotate_ccw",  lambda im: im.rotate(12, expand=False, fillcolor=(60,60,60))),
        ("hflip",       lambda im: im.transpose(Image.FLIP_LEFT_RIGHT)),
        ("occlude",     occlude),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--model", default="base_pg2", help="eval_grounding_hub.MODELS의 key")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mkey = args.model
    spec = {m[0]: m for m in MODELS}[mkey]
    _, fam, base, adapter = spec
    print(f"[LOAD] {mkey} ({fam})\n")
    if fam == "pg":
        from transformers import PaliGemmaProcessor
        proc = PaliGemmaProcessor.from_pretrained(str(base))
        model = load_pg(base, adapter)
        _det = make_pg_detect(model, proc)
    else:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        proc = AutoProcessor.from_pretrained(str(base))
        model = AutoModelForVision2Seq.from_pretrained(
            str(base), torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda")
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()
        _det = make_kosmos_detect(model, proc)

    def detect(img):
        try: return _det(img, "gray basket")
        except Exception: return None

    # 원본에서 검출 성공한 프레임만 기준으로
    sample = build_basket_sample(2)
    base_ok = []
    for s in sample:
        d = detect(s["img"])
        if d is not None and d["area"] <= 0.9:
            base_ok.append({"img": s["img"], "cx0": d["cx"], "tag": s["tag"], "vp": s["vp"]})
        if len(base_ok) >= args.n: break
    print(f"[REF] 원본 검출 성공 {len(base_ok)} 프레임 기준\n")

    augs = aug_funcs()
    results = {}
    for name, fn in augs:
        drifts, hits, full, n = [], 0, 0, 0
        for s in base_ok:
            n += 1
            d = detect(fn(s["img"]))
            if d is None: continue
            hits += 1
            if d["area"] > 0.9: full += 1
            ref = (1 - s["cx0"]) if name == "hflip" else s["cx0"]
            drifts.append(abs(d["cx"] - ref))
        results[name] = {
            "hit_rate": hits / max(n, 1), "n": n,
            "cx_drift_mean": float(np.mean(drifts)) if drifts else None,
            "fullframe_rate": full / max(n, 1),
        }
        r = results[name]
        dr = f"{r['cx_drift_mean']:.3f}" if r['cx_drift_mean'] is not None else "  -"
        print(f"  {name:<13} hit={r['hit_rate']*100:>3.0f}%  cx_drift={dr}  full-frame={r['fullframe_rate']*100:.0f}%")

    # 시각 그리드: 대표 3프레임 × 변형들
    reps = base_ok[:3]
    cols = len(augs); rows = len(reps)
    cell, hdr = 150, 18
    cv = Image.new("RGB", (cols*cell, rows*cell + hdr), (15, 22, 36))
    dr = ImageDraw.Draw(cv)
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception: font = ImageFont.load_default()
    for ci, (name, fn) in enumerate(augs):
        dr.text((ci*cell+4, 4), name[:14], fill=(125, 211, 252), font=font)
        for ri, s in enumerate(reps):
            im = fn(s["img"]); d = detect(im)
            th = im.copy(); th.thumbnail((cell-6, cell-6))
            ox, oy = ci*cell+3, ri*cell+hdr+3
            cv.paste(th, (ox, oy))
            if d:
                x1, y1, x2, y2 = d["box"]; fl = d["area"] > 0.9
                col = (239,68,68) if fl else (34,197,94)
                dr.rectangle([ox+x1*th.width, oy+y1*th.height, ox+x2*th.width, oy+y2*th.height], outline=col, width=2)
            else:
                dr.text((ox+4, oy+4), "MISS", fill=(252,165,165), font=font)
    sfx = "" if mkey == "base_pg2" else f"_{mkey}"
    cv.save(OUT / f"aug_grid{sfx}.png")
    (OUT / f"aug_robustness{sfx}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] {OUT}/aug_robustness{sfx}.json + aug_grid{sfx}.png")


if __name__ == "__main__":
    main()
