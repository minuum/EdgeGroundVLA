# -*- coding: utf-8 -*-
"""
측면 경로(left_left/right_right/left_right/right_left) 에피소드 갤러리 — 모델별.

grounding_sideangle_episodes.py(base PG2 전용)를 7개 모델로 확장.
각 모델 × 각 path type → 에피소드별 시간순 프레임 그리드(bbox + cx 라벨).
모델 1회 로드 후 4경로 일괄 처리 (효율).

대상 모델: eval_grounding_hub.MODELS 재사용 (base PG2 / pure Kosmos / exp57~64).

산출: docs/v5/grounding_hub/sa_<modelkey>_<pathtype>.png + sa_per_model_summary.json
Usage:
  .venv/bin/python3 scripts/sideangle_per_model.py --n-ep 3 --n-fr 6
  .venv/bin/python3 scripts/sideangle_per_model.py --only base_pg2,exp64_pg2
"""
import sys, json, glob, argparse, gc
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from PIL import Image, ImageDraw, ImageFont

from eval_grounding_hub import (MODELS, load_pg, make_pg_detect, make_kosmos_detect, OUT)

DSET = ROOT / "ROS_action/mobile_vla_dataset_v5"
PATH_TYPES = ["left_left", "right_right", "left_right", "right_left"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ep", type=int, default=3)
    ap.add_argument("--n-fr", type=int, default=6)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        fsm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = fsm = ImageFont.load_default()

    # 경로별 에피소드 + 프레임 미리 로드 (모델 간 재사용)
    ep_imgs = {}  # pt -> [(epname, [PIL frames at sampled idxs], [idxs])]
    import h5py
    for pt in PATH_TYPES:
        eps = sorted(glob.glob(str(DSET / f"*{pt}_path__core__fixed_center.h5")))[:args.n_ep]
        lst = []
        for ep in eps:
            with h5py.File(ep, "r") as f:
                imgs = f["observations"]["images"][:]
            idxs = np.linspace(0, len(imgs) - 1, args.n_fr).astype(int)
            frames = [Image.fromarray(imgs[i].astype("uint8")).convert("RGB") for i in idxs]
            lst.append((Path(ep).stem, frames, list(idxs)))
        ep_imgs[pt] = lst

    summary = {}
    for key, fam, base, adapter in MODELS:
        if only and key not in only:
            continue
        print(f"\n[{key}] 로드 ({fam})...")
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
            print(f"  ✗ 로드 실패: {e}"); continue

        summary[key] = {}
        for pt in PATH_TYPES:
            lst = ep_imgs[pt]
            cell, hdr, lab = 220, 24, 16
            cv = Image.new("RGB", (args.n_fr * cell, len(lst) * (cell + hdr)), (15, 22, 36))
            dr = ImageDraw.Draw(cv)
            pt_rec = []
            for ei, (epname, frames, idxs) in enumerate(lst):
                dr.text((6, ei*(cell+hdr)+4), f"[{key}] {epname[-34:]}", fill=(125, 211, 252), font=font)
                rec = {"episode": epname, "frames": []}
                for fi, (img, idx) in enumerate(zip(frames, idxs)):
                    try: d = detect(img, "gray basket")
                    except Exception: d = None
                    th = img.copy(); th.thumbnail((cell-8, cell-lab-8))
                    ox = fi*cell+4; oy = ei*(cell+hdr)+hdr
                    cv.paste(th, (ox, oy))
                    if d:
                        x1, y1, x2, y2 = d["box"]; full = d["area"] > 0.9
                        col = (239, 68, 68) if full else (96, 165, 250)
                        dr.rectangle([ox+x1*th.width, oy+y1*th.height, ox+x2*th.width, oy+y2*th.height], outline=col, width=2)
                        txt = f"f{idx} cx={d['cx']:.2f}" + (" FULL" if full else ""); tc = (252,165,165) if full else (134,239,172)
                    else:
                        txt = f"f{idx} MISS"; tc = (252, 165, 165)
                    dr.text((ox, oy+th.height+2), txt, fill=tc, font=fsm)
                    rec["frames"].append({"idx": int(idx), "cx": d and round(d["cx"],3), "area": d and round(d["area"],3), "hit": d is not None})
                pt_rec.append(rec)
            cv.save(OUT / f"sa_{key}_{pt}.png")
            allf = [f for r in pt_rec for f in r["frames"]]
            hit = sum(1 for f in allf if f["hit"]); full = sum(1 for f in allf if f["hit"] and f["area"] and f["area"]>0.9)
            summary[key][pt] = {"hit_rate": round(hit/max(len(allf),1),3), "fullframe_rate": round(full/max(len(allf),1),3), "n": len(allf)}
            print(f"  {pt:<12} hit={summary[key][pt]['hit_rate']*100:.0f}% full-frame={summary[key][pt]['fullframe_rate']*100:.0f}%")
        del model, detect; gc.collect(); torch.cuda.empty_cache()

    (OUT / "sa_per_model_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] {OUT}/sa_per_model_summary.json")


if __name__ == "__main__":
    main()
