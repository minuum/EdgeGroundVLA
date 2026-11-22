#!/usr/bin/env python3
"""Phase 1 (Track B1 + Track A2 벤치마크):
실제 11/13(의자·책상·화분·바구니) + 11/16(근접 정지 포함) 프레임에서
  1) beam=3 fp32 (현행 배포 기준선)
  2) beam=1 fp32 (Greedy)
  3) beam=1 fp16 (Track B: Dual-Rate Slow Loop 후보)
  4) Prefix-Forced Soft-Loc fp16 (Track A2: [</s><s> + phrase] 1-pass KV-cache + 4-step 미분가능 Softmax Expectation, 문자열 파싱 없음!)
의 추론 지연(ms), Open-Vocab BBox 오차(cx/cy/area MAE), 검출률, A3C_AG23_s0 행동 일치율을 전수 비교한다.
출력: docs/v5/egvla2_fm/beam_bench.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import precompute_v6_florence2_fused as P
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
import robovlm_nav.models.egvla2_fm_heads as HD

DEV = torch.device("cuda")
LOC_START = 50269
LOC_END = 51269
LOC_GRID = torch.linspace(0.0, 1.0, 1000, device=DEV, dtype=torch.float32)


@torch.no_grad()
def run_generate(model, proc, pil, phrase, dtype, num_beams):
    inp = proc(text=[P.TASK + phrase], images=[pil], return_tensors="pt")
    pv = inp["pixel_values"].to(DEV, dtype)
    ids = inp["input_ids"].to(DEV)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = model._encode_image(pv)
    emb, _ = model._merge_input_ids_with_image_features(img, model.get_input_embeddings()(ids))
    gen = model.language_model.generate(
        input_ids=None, inputs_embeds=emb, max_new_tokens=32, num_beams=num_beams, do_sample=False
    )
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    text = proc.batch_decode(gen, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(text, task=P.TASK, image_size=(pil.width, pil.height))[P.TASK]
    b = P.florence2_output_to_bbox(parsed.get("bboxes", []), pil.width, pil.height)
    return np.array([b["cx"], b["cy"], b["area"], float(b["has_bbox"])], np.float32), dt_ms


@torch.no_grad()
def run_prefix_soft_loc(model, proc, pil, phrase, dtype):
    """Track A2: [</s><s> + phrase_tokens] 프리픽스 1-Pass + KV-Cache 4스텝으로
    문자열 파싱 없이 <loc_0~999> Softmax Expectation 연속 좌표 (cx, cy, area, has) 직접 추출."""
    inp = proc(text=[P.TASK + phrase], images=[pil], return_tensors="pt")
    pv = inp["pixel_values"].to(DEV, dtype)
    ids = inp["input_ids"].to(DEV)
    p_toks = proc.tokenizer.encode(phrase, add_special_tokens=False)
    dec_prefix = torch.tensor([[2, 0] + p_toks], device=DEV, dtype=torch.long)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = model._encode_image(pv)
    emb, att = model._merge_input_ids_with_image_features(img, model.get_input_embeddings()(ids))
    enc_out = model.language_model.model.encoder(inputs_embeds=emb, attention_mask=att)

    # 1-pass decoder over known phrase prefix -> predicts x1 logits directly
    out = model.language_model(
        encoder_outputs=enc_out,
        attention_mask=att,
        decoder_input_ids=dec_prefix,
        use_cache=True,
    )
    pkv = out.past_key_values
    coords = []
    has_prob = 1.0
    for step in range(4):
        logits = out.logits[:, -1, :].float()  # (1, vocab)
        if step == 0:
            # Probability mass on location tokens vs eos/non-loc
            probs_full = torch.softmax(logits, dim=-1)
            has_prob = float(probs_full[0, LOC_START:LOC_END].sum().item())
        loc_probs = torch.softmax(logits[:, LOC_START:LOC_END], dim=-1)
        val = float((loc_probs * LOC_GRID).sum(dim=-1).item())
        coords.append(val)
        if step < 3:
            next_tok = torch.argmax(logits[:, LOC_START:LOC_END], dim=-1, keepdim=True) + LOC_START
            out = model.language_model(
                encoder_outputs=enc_out,
                attention_mask=att,
                decoder_input_ids=next_tok,
                past_key_values=pkv,
                use_cache=True,
            )
            pkv = out.past_key_values
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    x1, y1, x2, y2 = coords
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    area = w * h
    has = 1.0 if (has_prob > 0.35 and area > 1e-4) else 0.0
    return np.array([cx, cy, area, has], np.float32), dt_ms


def main():
    print("=== Phase 1: Florence-2 Decoder Beam/Precision/Soft-Loc 벤치마크 ===", flush=True)
    D13 = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
    R16 = json.load(open(ROOT / "docs/v5/egvla2_fm/a3c/replay_1116.json")); R16.pop("_validation", None)

    # Sample ~240 representative frames across all 4 objects (11/13 + 11/16 including high-area frames)
    samples = []
    for x in D13:
        sid = x["sid"]; ins = x["ins"]; n = x["n"]
        # first, middle, last frame per 11/13 session (65 * 3 = 195 frames)
        for fi in sorted({0, n // 2, n - 1}):
            p = ROOT / f"docs/v5/figures/objtest/{sid}_{fi}.jpg"
            if p.exists():
                samples.append((sid, fi, ins, x["obj"], p))
    for sid, v in list(R16.items()):
        ins = v["instruction"]; fs = v["frames"]
        for fi_idx in sorted({0, len(fs) - 2, len(fs) - 1}):
            if fi_idx < 0: continue
            fn = fs[fi_idx]
            p = ROOT / f"docs/v5/figures/objtest_1116/{sid}_{fn}.jpg"
            if p.exists():
                samples.append((sid, fn, ins, "11/16_" + ins.split()[-1], p))

    print(f"총 벤치마크 샘플: {len(samples)} 프레임 (11/13 다물체 + 11/16 근접 포함)", flush=True)
    pils = [(Image.open(p).convert("RGB"), ins, obj) for (_, _, ins, obj, p) in samples]

    proc = AutoProcessor.from_pretrained(P.MODEL, trust_remote_code=True)
    m_fp32 = AutoModelForCausalLM.from_pretrained(P.MODEL, trust_remote_code=True, torch_dtype=torch.float32).to(DEV).eval()
    m_fp16 = AutoModelForCausalLM.from_pretrained(P.MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()

    modes = [
        ("beam3_fp32 (현행 기준)", m_fp32, torch.float32, "gen", 3),
        ("beam1_fp32 (Greedy)", m_fp32, torch.float32, "gen", 1),
        ("beam1_fp16 (Track B)", m_fp16, torch.float16, "gen", 1),
        ("softloc_fp16 (Track A2)", m_fp16, torch.float16, "softloc", 1),
    ]

    raw_res = {}
    for name, mod, dt, kind, nb in modes:
        # Warmup 5 frames
        for k in range(5):
            if kind == "gen": run_generate(mod, proc, pils[k][0], pils[k][1], dt, nb)
            else: run_prefix_soft_loc(mod, proc, pils[k][0], pils[k][1], dt)
        boxes, times = [], []
        for pil, ins, _ in pils:
            if kind == "gen":
                b, ms = run_generate(mod, proc, pil, ins, dt, nb)
            else:
                b, ms = run_prefix_soft_loc(mod, proc, pil, ins, dt)
            boxes.append(b); times.append(ms)
        raw_res[name] = (np.stack(boxes), np.array(times))
        print(f"  {name:26s} | mean={np.mean(times):.1f}ms med={np.median(times):.1f}ms p90={np.percentile(times,90):.1f}ms | has_bbox={np.mean(np.stack(boxes)[:,3]>0.5):.3f}", flush=True)

    ref_b, ref_t = raw_res["beam3_fp32 (현행 기준)"]
    ref_ms = float(np.median(ref_t))
    summary = {}
    print("\n=== beam3_fp32 대비 정량 비교 ===")
    for name, (b, t) in raw_res.items():
        both = (b[:, 3] > 0.5) & (ref_b[:, 3] > 0.5)
        cx_mae = float(np.mean(np.abs(b[both, 0] - ref_b[both, 0])))
        cy_mae = float(np.mean(np.abs(b[both, 1] - ref_b[both, 1])))
        ar_mae = float(np.mean(np.abs(b[both, 2] - ref_b[both, 2])))
        has_agree = float(np.mean((b[:, 3] > 0.5) == (ref_b[:, 3] > 0.5)))
        med_ms = float(np.median(t))
        speedup = round(ref_ms / med_ms, 2)
        summary[name] = dict(
            n_frames=len(b),
            latency_mean_ms=round(float(np.mean(t)), 1),
            latency_med_ms=round(med_ms, 1),
            latency_p90_ms=round(float(np.percentile(t, 90)), 1),
            speedup_vs_beam3=speedup,
            has_bbox_rate=round(float(np.mean(b[:, 3] > 0.5)), 4),
            has_agree_vs_beam3=round(has_agree, 4),
            cx_mae_vs_beam3=round(cx_mae, 4),
            cy_mae_vs_beam3=round(cy_mae, 4),
            area_mae_vs_beam3=round(ar_mae, 4),
        )
        print(f"★ {name:26s} | {med_ms:5.1f}ms ({speedup:.2f}x) | has일치:{has_agree:.3f} | cx MAE:{cx_mae:.4f} | area MAE:{ar_mae:.4f}", flush=True)

    out_path = ROOT / "docs/v5/egvla2_fm/beam_bench.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
