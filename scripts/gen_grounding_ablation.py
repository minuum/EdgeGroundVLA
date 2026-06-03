# -*- coding: utf-8 -*-
"""
Grounding ablation — 여러 grounding 모델을 동일 val 셋에 실행해 비교 (Table 1)
+ 프레임별 cx/cy/area/hit 캐시 저장 (Table 2 CL 파이프라인 공유).

대상 (PaliGemma 계열):
  base   : PaliGemma2-3b-mix (LoRA 없음, zero-shot)
  exp57  : PaliGemma1-3b-pt  + LoRA
  exp58  : PaliGemma2-3b-mix + LoRA (2-class)
  exp59  : PaliGemma2-3b-mix + LoRA (hardneg, 현재 CL용)

Table 1 지표: hit율 / cx MAE(vs HSV) / cx_std(에피소드내) / 캔박스율 / full-frame율 / 선택성 gap

산출:
  docs/v5/grounding_ablation/grounding_{tag}.json   (프레임별 결과, Table2용)
  docs/v5/grounding_ablation/table1.json            (모델별 종합)

Usage:
  .venv/bin/python3 scripts/gen_grounding_ablation.py
  .venv/bin/python3 scripts/gen_grounding_ablation.py --models base,exp59 --neg-every 3
"""
import sys, re, json, argparse, warnings, gc
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, h5py
from PIL import Image

ANN  = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"
OUTD = ROOT / "docs/v5/grounding_ablation"
OUTD.mkdir(parents=True, exist_ok=True)
LOC  = re.compile(r"<loc(\d{4})>")

PG2 = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
PG1 = Path.home() / ".cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"

SPECS = {
    "base":  {"backbone": PG2, "adapter": None},
    "exp57": {"backbone": PG1, "adapter": ROOT / "runs/v5_nav/grounding/exp57"},
    "exp58": {"backbone": PG2, "adapter": ROOT / "runs/v5_nav/grounding/exp58"},
    "exp59": {"backbone": PG2, "adapter": ROOT / "runs/v5_nav/grounding/exp59"},
    "hsv":   None,
}


def load_model(spec, device):
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    proc = PaliGemmaProcessor.from_pretrained(str(spec["backbone"]))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(spec["backbone"]), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    if spec["adapter"] is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(spec["adapter"]))
    return proc, model.eval()


@torch.no_grad()
def detect(proc, model, img, phrase, device):
    inp = proc(text=f"<image>detect {phrase}", images=img, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
    gen = model.generate(**inp, max_new_tokens=48, do_sample=False)
    raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
    locs = [int(v) / 1023 for v in LOC.findall(raw)]
    if len(locs) >= 4:
        y1, x1, y2, x2 = locs[:4]
        return {"cx": (x1+x2)/2, "cy": (y1+y2)/2, "area": (x2-x1)*(y2-y1), "hit": True}
    return {"cx": None, "cy": None, "area": None, "hit": False}


def val_episodes():
    import random
    ann = json.loads(ANN.read_text())
    ann = [ep for ep in ann if ep.get("path_type", "") not in ("", "free", "unknown")]
    random.seed(42); np.random.seed(42); random.shuffle(ann)
    return ann[:max(1, int(len(ann) * 0.15))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="base,exp57,exp58,exp59")
    ap.add_argument("--neg-phrase", default="red ball")
    ap.add_argument("--neg-every", type=int, default=2, help="N프레임마다 네거티브 phrase 평가")
    args = ap.parse_args()
    device = torch.device("cuda")
    tags = args.models.split(",")

    val = val_episodes()
    print(f"[DATA] val {len(val)} ep")

    # 프레임 미리 로드 (이미지 + HSV ref)
    ep_frames = []
    for ep in val:
        h5 = Path(ep["episode"])
        if not h5.exists(): continue
        with h5py.File(str(h5), "r") as f:
            imgs = f["observations"]["images"][:]
        frs = []
        for fr in ep["frames"]:
            fi = fr["frame_idx"]
            if fi < len(imgs):
                frs.append({
                    "frame_idx": fi, 
                    "img": Image.fromarray(imgs[fi].astype("uint8")).convert("RGB"),
                    "cx_hsv": fr.get("cx_det_hsv", fr.get("cx_det")),
                    "cy_hsv": fr.get("cy_det_hsv", fr.get("cy_det")),
                    "area_hsv": fr.get("area_det_hsv", fr.get("area_det")),
                    "gt_class": fr.get("gt_class")
                })
        ep_frames.append({"episode": ep["episode"], "path_type": ep["path_type"], "frames": frs})

    table1 = {}
    for tag in tags:
        if tag not in SPECS:
            print(f"  skip unknown {tag}"); continue
        
        if tag == "hsv":
            print(f"\n[HSV] hsv 데이터를 수집하여 지표 산출...", flush=True)
            proc, model = None, None
        else:
            print(f"\n[MODEL] {tag} 로드...", flush=True)
            proc, model = load_model(SPECS[tag], device)
            
        cache = []
        cx_err, cx_by_ep, areas, hits, n = [], [], [], 0, 0
        neg_hits = neg_n = 0
        for ep in ep_frames:
            ep_cx = []
            ep_out = {"episode": ep["episode"], "path_type": ep["path_type"], "frames": []}
            for j, fr in enumerate(ep["frames"]):
                if tag == "hsv":
                    cx_val = fr.get("cx_hsv")
                    cy_val = fr.get("cy_hsv")
                    area_val = fr.get("area_hsv")
                    hit_val = (cx_val is not None)
                    d = {"cx": cx_val, "cy": cy_val, "area": area_val, "hit": hit_val}
                else:
                    d = detect(proc, model, fr["img"], "gray basket", device)
                    
                rec = {"frame_idx": fr["frame_idx"], "gt_class": fr["gt_class"],
                       "cx": d["cx"], "cy": d["cy"], "area": d["area"], "hit": d["hit"]}
                ep_out["frames"].append(rec)
                n += 1
                if d["hit"]:
                    hits += 1; areas.append(d["area"]); ep_cx.append(d["cx"])
                    if fr["cx_hsv"] is not None:
                        cx_err.append(abs(d["cx"] - fr["cx_hsv"]))
                if j % args.neg_every == 0:
                    if tag == "hsv":
                        nd = {"hit": False}
                    else:
                        nd = detect(proc, model, fr["img"], args.neg_phrase, device)
                    neg_n += 1; neg_hits += int(nd["hit"])
            if len(ep_cx) >= 2:
                cx_by_ep.append(float(np.std(ep_cx)))
            cache.append(ep_out)
        (OUTD / f"grounding_{tag}.json").write_text(json.dumps(cache, ensure_ascii=False))

        ar = np.array(areas) if areas else np.array([])
        table1[tag] = {
            "hit_rate": hits / max(n, 1),
            "cx_mae_vs_hsv": float(np.mean(cx_err)) if cx_err else None,
            "cx_std_in_ep": float(np.mean(cx_by_ep)) if cx_by_ep else None,
            "canned_rate": float(np.mean((ar > 0.04) & (ar < 0.06))) if len(ar) else None,
            "fullframe_rate": float(np.mean(ar > 0.9)) if len(ar) else None,
            "neg_hit_rate": neg_hits / max(neg_n, 1),
            "selectivity_gap": hits / max(n, 1) - neg_hits / max(neg_n, 1),
            "n_frames": n,
        }
        r = table1[tag]
        print(f"  hit={r['hit_rate']*100:.0f}%  cxMAE={r['cx_mae_vs_hsv']}  cx_std={r['cx_std_in_ep']}  "
              f"canned={r['canned_rate']}  full={r['fullframe_rate']}  selGap={r['selectivity_gap']:.2f}", flush=True)
        if tag != "hsv":
            del model, proc; gc.collect(); torch.cuda.empty_cache()

    (OUTD / "table1.json").write_text(json.dumps(table1, indent=2, ensure_ascii=False))
    print("\n===== Table 1 (Grounding 품질) =====")
    print(f"{'model':<8}{'hit':>6}{'cxMAE':>8}{'cx_std':>8}{'canned':>8}{'full':>7}{'selGap':>8}")
    for tag in tags:
        if tag not in table1: continue
        r = table1[tag]
        def f(x, p=3): return f"{x:.{p}f}" if x is not None else "  -  "
        print(f"{tag:<8}{r['hit_rate']*100:>5.0f}%{f(r['cx_mae_vs_hsv']):>8}{f(r['cx_std_in_ep']):>8}"
              f"{f(r['canned_rate'],2):>8}{f(r['fullframe_rate'],2):>7}{r['selectivity_gap']:>8.2f}")
    print(f"\n[SAVE] {OUTD}/table1.json + grounding_*.json")


if __name__ == "__main__":
    main()
