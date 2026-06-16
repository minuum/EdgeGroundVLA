#!/usr/bin/env python3
"""
Exp59 V5-only 교차 테스트
같은 V5 이미지에 4가지 쿼리 → gray basket만 bbox, 나머지는 <eos>

Usage:
  .venv/bin/python3 scripts/eval_exp59_v5_cross.py
"""
import json, re, random
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image

ROOT    = Path(__file__).resolve().parent.parent
ANN     = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_frame_level.json"
ADAPTER = ROOT / "runs/v5_nav/grounding/exp59"
PG2     = Path.home() / ".cache/huggingface/hub" \
          / "models--google--paligemma2-3b-mix-224" \
          / "snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
LOC_RE  = re.compile(r"<loc(\d{4})>")

PHRASES = {
    "gray basket": "TP (맞아야)",
    "brown pot":   "TN (없어야)",
    "red ball":    "TN (없어야)",
    "person":      "TN (없어야)",
}


def load_model(device):
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    from peft import PeftModel
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print("[LOAD] PaliGemma2 + Exp59 LoRA...")
    proc  = PaliGemmaProcessor.from_pretrained(str(PG2))
    base  = PaliGemmaForConditionalGeneration.from_pretrained(
                str(PG2), torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model = PeftModel.from_pretrained(base, str(ADAPTER)).eval()
    print("[LOAD] 완료\n")
    return proc, model, dtype


@torch.no_grad()
def detect(model, proc, img_np, phrase, device, dtype):
    pil = Image.fromarray(img_np).convert("RGB")
    inp = proc(text=f"detect {phrase}", images=pil, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    gen = model.generate(**inp, max_new_tokens=48, do_sample=False)
    raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                             skip_special_tokens=False)[0]
    locs = [int(v)/1023.0 for v in LOC_RE.findall(raw)]
    hit  = len(locs) >= 4
    bbox = [locs[1], locs[0], locs[3], locs[2]] if hit else []  # x1,y1,x2,y2
    return hit, bbox, raw.strip()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc, model, dtype = load_model(device)

    with open(ANN) as f:
        ann = json.load(f)

    # 다양한 path_type에서 20장 샘플링
    random.seed(42)
    samples = []
    seen_types = set()
    for ep in random.sample(ann, len(ann)):
        pt = ep["path_type"]
        frs = [fr for fr in ep["frames"]
               if fr.get("detected") and 0.03 < fr.get("area_det", 0) < 0.25]
        if not frs: continue
        fr = random.choice(frs)
        try:
            with h5py.File(ep["episode"], "r") as f:
                img = f["observations"]["images"][fr["frame_idx"]].astype(np.uint8)
        except: continue
        samples.append({
            "path_type": pt, "episode": ep["episode"],
            "frame_idx": fr["frame_idx"],
            "cx_gt": fr["cx_det"], "img": img,
        })
        if len(samples) >= 20: break

    print(f"테스트: {len(samples)}장 이미지 × {len(PHRASES)}쿼리 = {len(samples)*len(PHRASES)}회 추론\n")

    # 결과 집계
    stats = {p: {"hit": 0, "total": 0} for p in PHRASES}
    rows  = []

    for s in samples:
        for phrase, expect in PHRASES.items():
            hit, bbox, raw = detect(model, proc, s["img"], phrase, device, dtype)
            stats[phrase]["hit"]   += int(hit)
            stats[phrase]["total"] += 1
            is_tp = (phrase == "gray basket")
            ok = hit if is_tp else not hit
            mark = "✅" if ok else "❌"
            print(f"  {mark} {s['path_type'][:12]:12s} | '{phrase:<12}' | "
                  f"{'HIT' if hit else 'miss':4s} | {raw[:50]}")
            rows.append({"path_type": s["path_type"], "phrase": phrase,
                          "hit": hit, "ok": ok, "raw": raw})
        print()

    # 최종 결과
    print("=" * 60)
    print("결과 요약")
    print("=" * 60)
    for phrase, expect in PHRASES.items():
        s = stats[phrase]
        rate = s["hit"] / s["total"] * 100
        ok_str = "✅" if (phrase == "gray basket" and rate >= 90) or \
                        (phrase != "gray basket" and rate <= 10) else "❌"
        print(f"  {ok_str} '{phrase}': {s['hit']}/{s['total']} = {rate:.1f}%  ({expect})")

    gb = stats["gray basket"]
    fp_avg = sum(stats[p]["hit"] for p in PHRASES if p != "gray basket") / \
             sum(stats[p]["total"] for p in PHRASES if p != "gray basket"  ) * 100
    tp = gb["hit"] / gb["total"] * 100
    gap = tp - fp_avg
    print(f"\n  분리도: TP={tp:.0f}%  FP평균={fp_avg:.0f}%  gap={gap:+.0f}%p")

    if gap >= 80:
        print("\n✅ 목표 달성 — 텍스트로 gray basket 특정 가능")
        print("   = Goal-Conditioned Grounding 증명")
    else:
        print(f"\n⚠️  gap={gap:.0f}%p < 80%p — 추가 학습 필요")

    # JSON 저장
    result = {
        "stats": {p: {"hit": v["hit"], "total": v["total"],
                       "rate": v["hit"]/v["total"]*100} for p,v in stats.items()},
        "tp_rate": tp, "fp_avg": fp_avg, "gap": gap,
    }
    out = ROOT / "docs/v5/exp59_cross_object"
    out.mkdir(exist_ok=True)
    (out / "v5_cross_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  JSON → {out}/v5_cross_results.json")


if __name__ == "__main__":
    main()
