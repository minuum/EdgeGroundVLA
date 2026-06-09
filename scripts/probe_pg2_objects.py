# -*- coding: utf-8 -*-
"""
Zero-shot 객체 probe — 베이스 PaliGemma2(LoRA 없이)가 우리 환경에서
어떤 객체명을 신뢰성 있게 grounding하는지 실측.

목적: "PG2 사전학습에 우리 객체가 있나 / 무엇을 아는가"의 실용적 답 +
      Exp59 LoRA 붕괴(CH23)와 베이스 모델 안정성 비교.

각 (프레임 × phrase)에 detect → hit율 / cx / area / box 크기 안정성(std) 측정.

Usage:
  .venv/bin/python3 scripts/probe_pg2_objects.py
"""
import sys, re, glob, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, h5py
from PIL import Image

PG2 = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
LOC = re.compile(r"<loc(\d{4})>")
OUT = ROOT / "docs/v5/grounding_collapse/pg2_object_probe.json"

PHRASES = ["gray basket", "laundry basket", "basket", "hamper", "box",
           "container", "trash can", "chair", "bottle", "red ball"]

EPISODES = [
    "episode_260408_174654_target_center_left",
    "episode_260408_175129_target_center_left",
]
FRAMES = [0, 5, 8, 10, 13, 16]


def main():
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    dev = torch.device("cuda")
    proc = PaliGemmaProcessor.from_pretrained(str(PG2))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG2), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(dev).eval()
    print("[LOAD] base PaliGemma2 (NO LoRA) 완료\n")

    def detect(img, phrase):
        inp = proc(text=f"<image>detect {phrase}", images=img, return_tensors="pt").to(dev)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=48, do_sample=False)
        raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        locs = [int(v) / 1023 for v in LOC.findall(raw)]
        if len(locs) >= 4:
            y1, x1, y2, x2 = locs[:4]
            return (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * (y2 - y1)
        return None

    # 프레임 로드
    frames_data = []
    for epname in EPISODES:
        cands = glob.glob(f"ROS_action/**/{epname}*.h5", recursive=True)
        if not cands:
            continue
        with h5py.File(cands[0], "r") as f:
            imgs = f["observations"]["images"][:]
        for fi in FRAMES:
            if fi < len(imgs):
                frames_data.append(Image.fromarray(imgs[fi].astype("uint8")).convert("RGB"))

    results = {}
    print(f"{'phrase':<16}{'hit':>6}{'cx_mean':>9}{'cx_std':>8}{'area_mean':>11}{'area_std':>10}")
    print("-" * 60)
    for ph in PHRASES:
        cxs, areas, hits = [], [], 0
        for img in frames_data:
            det = detect(img, ph)
            if det is not None:
                hits += 1
                cxs.append(det[0]); areas.append(det[2])
        n = len(frames_data)
        rec = {
            "hit_rate": hits / max(n, 1),
            "cx_mean": float(np.mean(cxs)) if cxs else None,
            "cx_std": float(np.std(cxs)) if cxs else None,
            "area_mean": float(np.mean(areas)) if areas else None,
            "area_std": float(np.std(areas)) if areas else None,
            "n": n,
        }
        results[ph] = rec
        cxm = f"{rec['cx_mean']:.3f}" if rec['cx_mean'] is not None else "  -  "
        cxs_ = f"{rec['cx_std']:.3f}" if rec['cx_std'] is not None else "  -  "
        am = f"{rec['area_mean']:.3f}" if rec['area_mean'] is not None else "  -  "
        as_ = f"{rec['area_std']:.3f}" if rec['area_std'] is not None else "  -  "
        print(f"{ph:<16}{hits}/{n:<4}{cxm:>9}{cxs_:>8}{am:>11}{as_:>10}")

    OUT.write_text(json.dumps({"episodes": EPISODES, "frames": FRAMES,
                               "n_images": len(frames_data), "results": results},
                              indent=2, ensure_ascii=False))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
