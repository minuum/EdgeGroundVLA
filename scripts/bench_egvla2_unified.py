#!/usr/bin/env python3
"""EGVLA2 규격 벤치 — 학습서버/젯슨에서 **동일 코드·동일 입력·동일 조건**으로 3개 구성 비교.

  A. 현재(프로덕션)  : OwlV2Grounder.run() → Stage1Encoder.encode_image()  순차 (stage2_v2_inference_server 클래스 그대로,
                      OWLv2 fp16 thresh=0.2 — sequential_vs_threaded_* 측정과 동일 설정)
  B. F2 분리 호출    : (processor+generate+박스디코드+어댑터) + (processor+vision_tower.forward_features_unpool)
  C. F2 fused (1회)  : processor 1회 → _encode_image 1회 → LM.generate + 같은 특징 평균풀링 → 박스디코드+어댑터

공통 조건: 입력=실제 V6 프레임 1장(docs/v5/egvla2_florence2/bench_frame.jpg, 1280x720), fp16, warmup 10, iters 50,
매 반복 cuda sync, phrase="gray basket", beam=3. 환경(디바이스/torch/transformers)은 JSON에 함께 기록.

실행: .venv/bin/python3 scripts/bench_egvla2_unified.py --tag server     (젯슨은 --tag jetson)
출력: docs/v5/detector/bench_egvla2_unified_<tag>.json
"""
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("VLA_OWLV2_FP16", "1")
os.environ.setdefault("VLA_OWLV2_THRESH", "0.2")
os.environ.setdefault("VLA_OWLV2_AREA_SCALE", "3.0")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from PIL import Image

FRAME = ROOT / "docs/v5/egvla2_florence2/bench_frame.jpg"
STAGE1 = ROOT / "runs/v5_nav/mlp/stage1_v3_5cls/stage1_v3_5cls_owl_projs.pt"
PHRASE, TASK = "gray basket", "<CAPTION_TO_PHRASE_GROUNDING>"


def stats(ts):
    a = np.array(ts)
    return dict(mean=float(a.mean()), std=float(a.std()), p95=float(np.percentile(a, 95)), min=float(a.min()), n=len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    import transformers
    env = dict(device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", torch=torch.__version__,
               transformers=transformers.__version__, fp16=True, iters=a.iters, warmup=a.warmup, frame=str(FRAME.name))
    print(env, flush=True)
    pil = Image.open(FRAME).convert("RGB"); rgb = np.array(pil)
    W, H = pil.size
    res = dict(env=env)

    def bench(fn):
        for _ in range(a.warmup): fn()
        sync(); ts = []
        for _ in range(a.iters):
            t0 = time.time(); fn(); sync(); ts.append((time.time() - t0) * 1000)
        return ts

    # ---- A. 프로덕션 (현재) ----
    from robovlm_nav.serve.stage2_v2_inference_server import DEFAULT_VLM, OwlV2Grounder, Stage1Encoder
    enc = Stage1Encoder(DEFAULT_VLM, STAGE1, dev); enc.eval()
    gr = OwlV2Grounder(dev); gr._ensure_loaded()
    tg, tv = [], []
    def run_a():
        t0 = time.time(); gr.run(rgb, phrase=PHRASE); sync(); t1 = time.time()
        enc.encode_image(pil); sync(); t2 = time.time()
        tg.append((t1 - t0) * 1000); tv.append((t2 - t1) * 1000)
    ts = bench(run_a)
    res["A_current_production_seq"] = dict(total=stats(ts), grounder_owlv2=stats(tg[-a.iters:]), vision_kosmos2=stats(tv[-a.iters:]))
    print("A", res["A_current_production_seq"]["total"], flush=True)
    del gr, enc; torch.cuda.empty_cache() if dev.type == "cuda" else None

    # ---- B/C. Florence-2 ----
    from transformers import AutoModelForCausalLM, AutoProcessor
    from florence2_bbox_adapter import florence2_output_to_bbox
    m = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float16).to(dev).eval()
    p = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

    def prep():
        inp = p(text=TASK + PHRASE, images=pil, return_tensors="pt")
        return inp["input_ids"].to(dev), inp["pixel_values"].to(dev, torch.float16)

    def decode(out):
        t = p.batch_decode(out, skip_special_tokens=False)[0]
        parsed = p.post_process_generation(t, task=TASK, image_size=(W, H))[TASK]
        return florence2_output_to_bbox(parsed.get("bboxes", []), W, H)

    @torch.no_grad()
    def run_b():
        ids, pv = prep()
        out = m.generate(input_ids=ids, pixel_values=pv, max_new_tokens=128, num_beams=3, do_sample=False)
        decode(out)
        ids, pv = prep()
        m.vision_tower.forward_features_unpool(pv)
    res["B_florence2_split"] = dict(total=stats(bench(run_b)))
    print("B", res["B_florence2_split"]["total"], flush=True)

    @torch.no_grad()
    def run_c():
        ids, pv = prep()
        feats = m._encode_image(pv)
        emb = m.get_input_embeddings()(ids)
        emb, _ = m._merge_input_ids_with_image_features(feats, emb)
        out = m.language_model.generate(input_ids=None, inputs_embeds=emb, max_new_tokens=128, num_beams=3, do_sample=False)
        decode(out)
        feats.float().mean(dim=1)
    res["C_florence2_fused"] = dict(total=stats(bench(run_c)))
    print("C", res["C_florence2_fused"]["total"], flush=True)

    a_ = res["A_current_production_seq"]["total"]["mean"]
    res["speedup_fused_vs_current"] = a_ / res["C_florence2_fused"]["total"]["mean"]
    res["speedup_fused_vs_split"] = res["B_florence2_split"]["total"]["mean"] / res["C_florence2_fused"]["total"]["mean"]
    out = ROOT / f"docs/v5/detector/bench_egvla2_unified_{a.tag}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"저장: {out}\nfused vs 현재 {res['speedup_fused_vs_current']:.2f}x · fused vs 분리 {res['speedup_fused_vs_split']:.2f}x")


if __name__ == "__main__":
    main()
