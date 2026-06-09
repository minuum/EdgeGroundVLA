# -*- coding: utf-8 -*-
"""
E2E LoRA-depth ablation 모델의 grounding 박스 점검.

E2E PaliGemma VLA(top2~8 × proj)는 action 예측용이라 bbox를 직접 안 냄.
→ Lightning ckpt에서 vision-tower LoRA만 추출해 base PG1에 주입 후 `detect gray basket`.
   vision-LoRA(action 학습)가 grounding을 망가뜨리는지(벽/의자 트래킹) 시점별로 점검.

base(LoRA 없음)·grounding 모델(CH27)과 동일 지표: full-frame율 / cx오차 / miss / 시점별.

산출: docs/v5/grounding_ablation/e2e_grounding_probe.json
Usage:
  .venv/bin/python3 scripts/probe_e2e_grounding.py
  .venv/bin/python3 scripts/probe_e2e_grounding.py --models top2_proj_frozen --per-vp 2
"""
import sys, re, json, glob, argparse, warnings, gc
from pathlib import Path
from collections import defaultdict
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, h5py
from PIL import Image

GDIR = ROOT / "docs/v5/grounding_ablation"
PG1 = Path.home() / ".cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
LOC = re.compile(r"<loc(\d{4})>")
EXPS = [f"top{n}_proj_{p}" for n in (2, 4, 6, 8) for p in ("frozen", "tuned")]


def pos_b(cx): return "L" if cx < 0.4 else ("R" if cx > 0.6 else "C")
def dist_b(a): return "far" if a < 0.05 else ("near" if a > 0.3 else "mid")


def build_viewpoint_sample(per_vp):
    """grounding_hsv.json에서 시점 버킷별 per_vp개 프레임 샘플 (이미지+hsv cx)."""
    hsv = json.loads((GDIR / "grounding_hsv.json").read_text())
    buckets = defaultdict(list)
    for ep in hsv:
        for fr in ep["frames"]:
            if fr["cx"] is None: continue
            vp = f"{pos_b(fr['cx'])}-{dist_b(fr.get('area') or 0)}"
            buckets[vp].append((ep["episode"], fr["frame_idx"], fr["cx"], vp))
    sample = []
    rng = np.random.default_rng(42)
    for vp, items in buckets.items():
        idx = rng.permutation(len(items))[:per_vp]
        sample += [items[i] for i in idx]
    # 이미지 로드 (에피소드별 캐시)
    out = []
    imgcache = {}
    for epp, fi, cx_ref, vp in sample:
        if epp not in imgcache:
            try:
                with h5py.File(epp, "r") as f: imgcache[epp] = f["observations"]["images"][:]
            except Exception: imgcache[epp] = None
        imgs = imgcache[epp]
        if imgs is None or fi >= len(imgs): continue
        out.append({"img": Image.fromarray(imgs[fi].astype("uint8")).convert("RGB"),
                    "cx_ref": cx_ref, "vp": vp})
    return out


def extract_lora(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = sd.get("state_dict", sd)
    lora = {}
    for k, v in sd.items():
        if "lora" in k.lower() and "vision_tower" in k:
            nk = k.replace("model.backbone.", "")   # → base_model.model.vision_tower...
            lora[nk] = v.clone()
    return lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(EXPS))
    ap.add_argument("--per-vp", type=int, default=4)
    args = ap.parse_args()
    dev = torch.device("cuda")
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    sample = build_viewpoint_sample(args.per_vp)
    print(f"[SAMPLE] {len(sample)} frames across viewpoints", flush=True)
    proc = PaliGemmaProcessor.from_pretrained(str(PG1))

    @torch.no_grad()
    def detect(model, img):
        inp = proc(text="<image>detect gray basket", images=img, return_tensors="pt").to(dev)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        gen = model.generate(**inp, max_new_tokens=48, do_sample=False)
        raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        locs = [int(v)/1023 for v in LOC.findall(raw)]
        if len(locs) >= 4:
            y1, x1, y2, x2 = locs[:4]
            return (x1+x2)/2, (y1+y2)/2, (x2-x1)*(y2-y1), True
        return None, None, None, False

    results = {}
    for exp in args.models.split(","):
        ckpts = glob.glob(str(ROOT / f"runs/mobile_vla_paligemma/v5_ablation_{exp}/**/last.ckpt"), recursive=True)
        cfg = ROOT / f"configs/v5_ablation/v5_ablation_{exp}.json"
        if not ckpts or not cfg.exists():
            print(f"  skip {exp} (no ckpt/config)"); continue
        targets = json.loads(cfg.read_text())["train_setup"]["lora_target_modules"]
        print(f"\n[{exp}] LoRA 추출 + base PG1 주입 ({len(targets)} targets)...", flush=True)
        lora = extract_lora(ckpts[0])
        base = PaliGemmaForConditionalGeneration.from_pretrained(
            str(PG1), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(dev)
        lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", target_modules=targets)
        model = get_peft_model(base, lcfg)
        missing, unexpected = model.load_state_dict(
            {**{k: v.to(dev, torch.bfloat16) for k, v in lora.items()}}, strict=False)
        loaded = len(lora) - len([m for m in missing if m in lora])  # 근사
        model.eval()
        print(f"  lora keys 주입: {len(lora)} (unexpected={len(unexpected)})", flush=True)

        per_vp = defaultdict(lambda: {"err": [], "full": 0, "miss": 0, "n": 0})
        full = miss = n = 0; errs = []
        for s in sample:
            cx, cy, ar, hit = detect(model, s["img"])
            vp = s["vp"]; pv = per_vp[vp]; pv["n"] += 1; n += 1
            if not hit:
                miss += 1; pv["miss"] += 1; continue
            errs.append(abs(cx - s["cx_ref"])); pv["err"].append(abs(cx - s["cx_ref"]))
            if ar > 0.9: full += 1; pv["full"] += 1
        results[exp] = {
            "hit_rate": (n-miss)/max(n,1), "cx_mae": float(np.mean(errs)) if errs else None,
            "fullframe_rate": full/max(n,1), "miss_rate": miss/max(n,1), "n": n,
            "per_vp": {vp: {"cx_mae": (float(np.mean(p["err"])) if p["err"] else None),
                            "fullframe_rate": p["full"]/max(p["n"],1), "miss_rate": p["miss"]/max(p["n"],1),
                            "n": p["n"]} for vp, p in per_vp.items()},
        }
        r = results[exp]
        print(f"  → hit={r['hit_rate']*100:.0f}% cxMAE={r['cx_mae'] and round(r['cx_mae'],3)} "
              f"full-frame={r['fullframe_rate']*100:.0f}% miss={r['miss_rate']*100:.0f}%", flush=True)
        del model, base; gc.collect(); torch.cuda.empty_cache()

    (GDIR / "e2e_grounding_probe.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n===== E2E LoRA-depth grounding 점검 =====")
    print(f"{'model':<18}{'hit':>6}{'cxMAE':>8}{'full-frame':>12}{'miss':>7}")
    for exp in args.models.split(","):
        if exp not in results: continue
        r = results[exp]
        mae = f"{r['cx_mae']:.3f}" if r['cx_mae'] is not None else "  -  "
        print(f"{exp:<18}{r['hit_rate']*100:>5.0f}%{mae:>8}{r['fullframe_rate']*100:>11.0f}%{r['miss_rate']*100:>6.0f}%")
    print(f"\n[SAVE] {GDIR}/e2e_grounding_probe.json")


if __name__ == "__main__":
    main()
