#!/usr/bin/env python3
"""
PG2 해상도 업그레이드(224→448/896) 실측 테스트.
운영 서버(stage2_v2_inference_server.py, port 8001)는 건드리지 않고
별도 프로세스에서 독립적으로 모델을 로드해 latency/메모리를 실측.

사용:
    python3 test_pg2_resolution.py google/paligemma2-3b-mix-448 \
        --frames docs/v5/grounding_frames/s7 --limit 10
"""
import argparse, time, sys
from pathlib import Path

import torch
from PIL import Image

LOC_RE = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", help="HF repo id, e.g. google/paligemma2-3b-mix-448")
    ap.add_argument("--frames", required=True, help="frame_*.jpg 디렉토리")
    ap.add_argument("--limit", type=int, default=10, help="테스트할 프레임 수")
    ap.add_argument("--prompt", default="detect gray basket")
    args = ap.parse_args()

    import re
    global LOC_RE
    LOC_RE = re.compile(r"<loc(\d{4})>")

    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    print(f"[로드 시작] {args.model_id}")
    t0 = time.time()
    try:
        processor = AutoProcessor.from_pretrained(args.model_id)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            device_map={"": "cuda"},
        ).eval()
    except Exception as e:
        print(f"[로드 실패] {type(e).__name__}: {e}")
        sys.exit(1)
    load_s = time.time() - t0
    alloc_gb = torch.cuda.memory_allocated() / 1e9
    reserved_gb = torch.cuda.memory_reserved() / 1e9
    print(f"[로드 완료] {load_s:.1f}s | GPU allocated={alloc_gb:.2f}GB reserved={reserved_gb:.2f}GB")
    print(f"  image_processor size: {processor.image_processor.size}")

    frame_dir = Path(args.frames)
    frames = sorted(frame_dir.glob("frame_*.jpg"))[: args.limit]
    print(f"\n[추론 테스트] {len(frames)}프레임 × prompt='{args.prompt}'")
    print("=" * 70)

    lats = []
    for i, fpath in enumerate(frames):
        pil = Image.open(fpath).convert("RGB")
        inp = processor(text=args.prompt, images=pil, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=48, min_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        lat_ms = (time.time() - t0) * 1000
        lats.append(lat_ms)

        raw = processor.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        locs = [int(v) / 1023.0 for v in LOC_RE.findall(raw)]
        bbox_str = f"locs={len(locs)//4 if len(locs)>=4 else 0}box" if locs else "no-bbox"
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  f{i+1:03d}: {lat_ms:6.0f}ms | {bbox_str:10s} | peak_mem={peak_gb:.2f}GB | raw='{raw[:50]}'")

    print("=" * 70)
    if lats:
        import statistics
        print(f"[결과] n={len(lats)} mean={statistics.mean(lats):.0f}ms median={statistics.median(lats):.0f}ms "
              f"min={min(lats):.0f}ms max={max(lats):.0f}ms")
    print(f"[메모리] load_alloc={alloc_gb:.2f}GB peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")


if __name__ == "__main__":
    main()
