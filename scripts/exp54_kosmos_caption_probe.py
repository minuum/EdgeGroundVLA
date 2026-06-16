#!/usr/bin/env python3
"""
Track 1: Kosmos-2 텍스트 생성 실험

Pure HF Kosmos-2가 basket 프레임을 보고 어떤 텍스트를 생성하는지 확인.
- 프롬프트 A (grounding): "<grounding>An image of a gray basket"
- 프롬프트 B (caption):   "An image of"
- 프롬프트 C (question):  "Question: What objects are visible? Answer:"

방향(left/center/right) × 구간(early/mid/late) × N_SAMPLE 프레임
basket 관련 키워드 hit율로 "CLIP이 basket을 텍스트로 인식하는가" 측정.

Usage:
  .venv/bin/python3 scripts/exp54_kosmos_caption_probe.py
"""

import json, sys, warnings, time
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"
OUT_PATH  = ROOT / "logs" / "exp54_kosmos_caption_probe.json"

N_SAMPLE = 5   # 방향 × 구간당 샘플 수

PROMPTS = {
    "grounding": "<grounding>An image of a gray basket",
    "caption":   "An image of",
    "question":  "Question: What objects are visible in this image? Answer:",
}

KEYWORDS = ("basket", "box", "container", "bin", "crate", "bucket")


def load_model(device):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16
    ).to(device).eval()
    return processor, model


@torch.no_grad()
def run_prompt(model, processor, img_pil, prompt_text, device):
    inputs = processor(text=prompt_text, images=img_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pv = inputs["pixel_values"].to(torch.float16)

    gen = model.generate(
        pixel_values=pv,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs.get("image_embeds_position_mask"),
        use_cache=True,
        max_new_tokens=64,
    )
    new_ids = gen[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(new_ids, skip_special_tokens=False)[0]
    caption, entities = processor.post_process_generation(raw)
    return caption.strip(), entities


def keyword_hit(text):
    return any(k in text.lower() for k in KEYWORDS)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[MODEL] {VLM_PATH.name} 로드 중...")
    processor, model = load_model(device)
    print(f"[MODEL] 로드 완료\n")

    data = json.loads(DATA_PATH.read_text())

    # 방향 × 구간별 샘플 수집
    dir_seg_frames = defaultdict(list)  # (direction, seg) → list of (ep, frame)
    for ep in data:
        d = ep["direction"]
        frames = [f for f in ep["frames"] if f["consistent"] and f["label"]]
        if not frames:
            continue
        n = len(frames)
        e_cut = max(1, n // 3)
        l_cut = n - max(1, n // 3)
        segs = {
            "early": frames[:e_cut],
            "mid":   frames[e_cut:l_cut],
            "late":  frames[l_cut:],
        }
        for seg, seg_frames in segs.items():
            key = (d, seg)
            if len(dir_seg_frames[key]) < N_SAMPLE and seg_frames:
                mid = seg_frames[len(seg_frames) // 2]
                dir_seg_frames[key].append((ep["episode"], mid["frame_idx"], mid.get("cx_det"), mid.get("area_det")))

    results = []
    summary = defaultdict(lambda: defaultdict(lambda: {"hit": 0, "total": 0}))

    for direction in ["left", "center", "right"]:
        for seg in ["early", "mid", "late"]:
            key = (direction, seg)
            samples = dir_seg_frames[key]
            if not samples:
                print(f"  [{direction}/{seg}] 샘플 없음")
                continue

            print(f"  [{direction}/{seg}] {len(samples)}개 프레임 처리 중...")
            for ep_path, fidx, cx, area in samples:
                try:
                    with h5py.File(ep_path, "r") as f:
                        img = Image.fromarray(f["observations"]["images"][fidx]).convert("RGB")
                except Exception as e:
                    print(f"    이미지 로드 실패: {e}")
                    continue

                row = {
                    "direction": direction, "seg": seg,
                    "frame_idx": fidx, "cx_det": cx, "area_det": area,
                    "outputs": {}
                }

                for pname, ptext in PROMPTS.items():
                    t0 = time.time()
                    caption, entities = run_prompt(model, processor, img, ptext, device)
                    ms = (time.time() - t0) * 1000
                    hit = keyword_hit(caption)

                    row["outputs"][pname] = {
                        "caption": caption[:120],
                        "entities": [(e[0], e[2]) for e in entities] if entities else [],
                        "keyword_hit": hit,
                        "latency_ms": round(ms),
                    }
                    summary[pname][(direction, seg)]["hit"]   += int(hit)
                    summary[pname][(direction, seg)]["total"] += 1

                # 즉시 출력 (grounding 결과만)
                g = row["outputs"]["grounding"]
                c = row["outputs"]["caption"]
                q = row["outputs"]["question"]
                cx_str = f"{cx:.2f}" if cx is not None else "N/A"
                print(
                    f"    cx={cx_str}  "
                    f"grounding: {'✅' if g['keyword_hit'] else '❌'} \"{g['caption'][:50]}\"  "
                    f"caption: {'✅' if c['keyword_hit'] else '❌'} \"{c['caption'][:40]}\""
                )
                results.append(row)

        print()

    # 저장
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(str(OUT_PATH), "w"), indent=2)

    # 최종 요약
    print(f"\n{'='*65}")
    print("  Track 1: Kosmos-2 텍스트 생성 결과")
    print(f"{'='*65}")
    print(f"  키워드: {KEYWORDS}")
    print()

    for pname in ["grounding", "caption", "question"]:
        print(f"  [ {pname} 프롬프트 ]")
        print(f"  {'방향':<8} {'early':>8} {'mid':>8} {'late':>8}  {'전체':>8}")
        print("  " + "-" * 48)
        for d in ["left", "center", "right"]:
            row_str = f"  {d:<8}"
            total_hit = total_n = 0
            for seg in ["early", "mid", "late"]:
                s = summary[pname].get((d, seg), {"hit": 0, "total": 0})
                if s["total"] > 0:
                    pct = s["hit"] / s["total"] * 100
                    row_str += f"  {pct:>5.0f}%({s['hit']}/{s['total']})"
                else:
                    row_str += f"  {'N/A':>8}"
                total_hit += s["hit"]
                total_n   += s["total"]
            overall = f"{total_hit/total_n*100:.0f}%" if total_n > 0 else "N/A"
            row_str += f"  {overall:>8}"
            print(row_str)

        # 전체 hit율
        all_hit = sum(s["hit"] for s in summary[pname].values())
        all_n   = sum(s["total"] for s in summary[pname].values())
        overall_pct = all_hit / all_n * 100 if all_n > 0 else 0
        verdict = (
            "CLIP이 basket을 텍스트로 인식 ✅" if overall_pct >= 50 else
            "부분 인식 (30~50%)" if overall_pct >= 30 else
            "텍스트 생성 불안정 ⚠️"
        )
        print(f"  전체 keyword hit: {all_hit}/{all_n} ({overall_pct:.1f}%)  → {verdict}")
        print()

    print(f"  상세 결과: {OUT_PATH}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
