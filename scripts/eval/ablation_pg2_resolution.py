#!/usr/bin/env python3
"""
PG2 해상도 ablation: 동일 테스트 이미지를 224 / 448 / 896 체크포인트로 비교.
로컬 GB10 머신에서 실행 (soda 운영 서버는 건드리지 않음).

대상 프레임: S6 dead-zone(정지장면 풀프레임 환각), S7 근접-미스 bbox,
S7 정상 검출 베이스라인 — 이미 식별된 실패/성공 케이스로 실질적 비교.

사용:
    .venv/bin/python3 scripts/eval/ablation_pg2_resolution.py
"""
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

LOC_RE = re.compile(r"<loc(\d{4})>")
PROMPT = "detect gray basket"

CHECKPOINTS = [
    "google/paligemma2-3b-mix-224",
    "google/paligemma2-3b-mix-448",
    # 896은 mix(downstream 태스크 파인튜닝) 버전이 3B에 없음 — pt-896(순수 사전학습)만 존재.
    # detect 태스크 <loc> 포맷 출력을 보장하지 않아 비교에서 제외 (별도 표기).
]

# (세션, 프레임번호, 라벨, 의미)
FRAMES = [
    ("s6", 1,  "S6 baseline",   "초반 정상 접근 구간"),
    ("s6", 56, "S6 dead-zone",  "정지장면 풀프레임 환각 시작"),
    ("s6", 70, "S6 dead-zone",  "동일 collapse 지속 (14프레임 후)"),
    ("s6", 85, "S6 dead-zone",  "동일 collapse 지속 (29프레임 후)"),
    ("s7", 39, "S7 near-miss",  "근접-미스 클러스터 시작 — 빈 바닥 박싱"),
    ("s7", 41, "S7 near-miss",  "근접-미스 지속"),
    ("s7", 84, "S7 correct",    "정상 검출 (HSV로도 확인된 케이스)"),
]


def parse_bbox(raw: str):
    locs = [int(v) / 1023.0 for v in LOC_RE.findall(raw)]
    if len(locs) < 4:
        return None
    y1, x1, y2, x2 = locs[:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    area = abs((x2 - x1) * (y2 - y1))
    return {"cx": cx, "cy": cy, "area": area, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def main():
    results = {ckpt: [] for ckpt in CHECKPOINTS}

    for ckpt in CHECKPOINTS:
        print(f"\n{'=' * 70}\n[로드] {ckpt}\n{'=' * 70}")
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(ckpt)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map={"": "cuda"},
        ).eval()
        load_s = time.time() - t0
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  로드 완료 {load_s:.1f}s | image size={processor.image_processor.size} | "
              f"GPU alloc={alloc_gb:.2f}GB")

        for session, fn, label, desc in FRAMES:
            fpath = Path(f"docs/v5/grounding_frames/{session}/frame_{fn:04d}.jpg")
            if not fpath.exists():
                print(f"  [스킵] {fpath} 없음")
                continue
            pil = Image.open(fpath).convert("RGB")
            inp = processor(text=PROMPT, images=pil, return_tensors="pt").to("cuda")
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)

            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=48, min_new_tokens=1, do_sample=False)
            torch.cuda.synchronize()
            lat_ms = (time.time() - t0) * 1000

            raw = processor.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
            bbox = parse_bbox(raw)
            peak_gb = torch.cuda.max_memory_allocated() / 1e9

            tag = f"a={bbox['area']:.3f} cx={bbox['cx']:.3f}" if bbox else "NO_BBOX"
            print(f"  [{session}:{fn:03d}] {label:16s} {lat_ms:6.0f}ms | {tag:22s} | peak={peak_gb:.2f}GB")

            results[ckpt].append({
                "session": session, "frame": fn, "label": label, "desc": desc,
                "raw": raw, "bbox": bbox, "latency_ms": lat_ms,
            })

        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    out_path = Path("docs/v5/pg2_resolution_ablation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[완료] {out_path}")

    # 요약 비교
    print(f"\n{'=' * 70}\n요약 비교\n{'=' * 70}")
    for session, fn, label, desc in FRAMES:
        print(f"\n[{session}:{fn:03d}] {label} — {desc}")
        for ckpt in CHECKPOINTS:
            matches = [r for r in results[ckpt] if r["session"] == session and r["frame"] == fn]
            if not matches:
                continue
            r = matches[0]
            res = (f"a={r['bbox']['area']:.3f} cx={r['bbox']['cx']:.3f}" if r["bbox"] else "NO_BBOX")
            short_ckpt = ckpt.split("-")[-1]
            print(f"  {short_ckpt:6s}: {res:22s} ({r['latency_ms']:.0f}ms)")


if __name__ == "__main__":
    main()
