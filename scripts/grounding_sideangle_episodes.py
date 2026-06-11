# -*- coding: utf-8 -*-
"""
측면 꺾임 경로(left_left / right_right) 에피소드별 base PG2 grounding 시연.

목적: 각도가 크게 꺾인 좌좌/우우 진입에서 base PaliGemma2(LoRA 없음)가
      에피소드 전 구간에 걸쳐 바구니를 안정적으로 잡는지 시간순으로 확인.
      (CH31/32 결론 "grounding은 base PG2" 을 가장 어려운 시점에서 검증)

각 에피소드 = 1행, 시간순 프레임 6장. 각 프레임에 detect bbox + cx 라벨.

산출: docs/v5/exp64_eval/sideangle_<pt>.png  (+ json)
Usage:
  .venv/bin/python3 scripts/grounding_sideangle_episodes.py
  .venv/bin/python3 scripts/grounding_sideangle_episodes.py --n-ep 3 --n-fr 6
"""
import sys, re, json, glob, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, h5py
from PIL import Image, ImageDraw, ImageFont

PG2 = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
LOC = re.compile(r"<loc(\d{4})>")
OUT = ROOT / "docs/v5/exp64_eval"
DSET = ROOT / "ROS_action/mobile_vla_dataset_v5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ep", type=int, default=4)
    ap.add_argument("--n-fr", type=int, default=6)
    ap.add_argument("--path-types", default="left_left,right_right,left_right,right_left")
    ap.add_argument("--mode", default="path", choices=["path", "free"],
                    help="path=경로타입별 / free=free_* 극단시나리오별")
    args = ap.parse_args()
    PATH_TYPES = args.path_types.split(",")
    # free 모드: 시나리오 키워드별 그룹 (출발위치 무관하게 묶음)
    FREE_GROUPS = {
        "basket_extreme": "*free*basket_*extreme*.h5",
        "robot_distance": "*free*robot_*.h5",
        "diagonal": "*free*diagonal_*.h5",
        "lighting_diff": "*free*lighting_diff*.h5",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    dev = torch.device("cuda")
    proc = PaliGemmaProcessor.from_pretrained(str(PG2))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG2), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(dev).eval()
    print("[LOAD] base PaliGemma2 (NO LoRA)\n")

    @torch.no_grad()
    def detect(img):
        inp = proc(text="<image>detect gray basket", images=img, return_tensors="pt").to(dev)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        gen = model.generate(**inp, max_new_tokens=48, do_sample=False)
        raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        locs = [int(v) / 1023 for v in LOC.findall(raw)]
        if len(locs) >= 4:
            y1, x1, y2, x2 = locs[:4]
            return {"cx": (x1+x2)/2, "area": (x2-x1)*(y2-y1), "box": [x1, y1, x2, y2]}
        return None

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        fsm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = fsm = ImageFont.load_default()

    if args.mode == "free":
        groups = [(k, pat, 99) for k, pat in FREE_GROUPS.items()]
        prefix = "free"
    else:
        groups = [(pt, f"*{pt}_path__core__fixed_center.h5", args.n_ep) for pt in PATH_TYPES]
        prefix = "sideangle"

    summary = {}
    for pt, pattern, nep in groups:
        eps = sorted(glob.glob(str(DSET / pattern)))[:nep]
        cell, hdr, lab = 230, 26, 18
        canvas = Image.new("RGB", (args.n_fr * cell, len(eps) * (cell + hdr)), (15, 22, 36))
        dr = ImageDraw.Draw(canvas)
        summary[pt] = []
        for ei, ep in enumerate(eps):
            with h5py.File(ep, "r") as f:
                imgs = f["observations"]["images"][:]
            idxs = np.linspace(0, len(imgs) - 1, args.n_fr).astype(int)
            epname = Path(ep).stem
            dr.text((6, ei * (cell + hdr) + 4), f"[{pt}] {epname[:46]}", fill=(125, 211, 252), font=font)
            ep_rec = {"episode": epname, "frames": []}
            for fi, idx in enumerate(idxs):
                img = Image.fromarray(imgs[idx].astype("uint8")).convert("RGB")
                d = detect(img)
                th = img.copy(); th.thumbnail((cell - 8, cell - lab - 8))
                ox = fi * cell + 4; oy = ei * (cell + hdr) + hdr
                canvas.paste(th, (ox, oy))
                if d:
                    x1, y1, x2, y2 = d["box"]
                    full = d["area"] > 0.9
                    col = (239, 68, 68) if full else (96, 165, 250)
                    dr.rectangle([ox + x1*th.width, oy + y1*th.height,
                                  ox + x2*th.width, oy + y2*th.height], outline=col, width=2)
                    txt = f"f{idx} cx={d['cx']:.2f}" + (" FULL" if full else "")
                    tc = (252, 165, 165) if full else (134, 239, 172)
                else:
                    txt = f"f{idx} MISS"; tc = (252, 165, 165)
                dr.text((ox, oy + th.height + 2), txt, fill=tc, font=fsm)
                ep_rec["frames"].append({"idx": int(idx),
                                         "cx": d and round(d["cx"], 3),
                                         "area": d and round(d["area"], 3),
                                         "hit": d is not None})
            summary[pt].append(ep_rec)
            print(f"[{pt}] {epname[:40]} done")
        path = OUT / f"{prefix}_{pt}.png"
        canvas.save(path)
        print(f"  [SAVE] {path}\n")

    # 집계 통계 (path type별)
    agg = {}
    for pt, eps in summary.items():
        allf = [f for ep in eps for f in ep["frames"]]
        hit = sum(1 for f in allf if f["hit"])
        full = sum(1 for f in allf if f["hit"] and f["area"] and f["area"] > 0.9)
        agg[pt] = {"n_ep": len(eps), "n_frames": len(allf),
                   "hit_rate": round(hit / max(len(allf), 1), 3),
                   "fullframe_rate": round(full / max(len(allf), 1), 3),
                   "miss": len(allf) - hit}
    (OUT / f"{prefix}_summary.json").write_text(json.dumps(
        {"per_episode": summary, "aggregate": agg}, indent=2, ensure_ascii=False))
    print(f"\n[AGG] {json.dumps(agg, ensure_ascii=False)}")
    print(f"[SAVE] {OUT}/{prefix}_summary.json")


if __name__ == "__main__":
    main()
