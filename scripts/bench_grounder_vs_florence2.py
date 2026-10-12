#!/usr/bin/env python3
"""OWLv2(그라운더) vs Florence-2(그라운더) vs Florence-2(비전 인코더만) 레이턴시 벤치.

EGVLA2 Florence-2 단일화 plan(docs/plans/plan_20261012_egvla2_florence2_unification.md)
§2.3/2.4용 — 학습서버(GPU)와 젯슨(로봇) 양쪽에서 각각 돌려서 비교한다.
동일 GPU라면 디바이스만 다를 뿐 코드는 그대로 — torch.cuda 체크로 자동 분기.

학습서버 결과 (2026-10-12, 1회 측정, n_iters=20):
  OWLv2 (forward+postprocess)            : 529.6±52.6 ms
  Florence-2 그라운더 (generate, beam=3)  : 247.6±5.9 ms   ← 서버 기준은 OWLv2보다 빠름
  Florence-2 비전 인코더만 (별도 forward) : 93.7±0.9 ms
  → 그라운딩+비전 따로 부르면 247.6+93.7=341.3ms, generate() 결과의 이미지 인코딩을
    재사용하면 247.6ms로 끝남 (약 27% 절감 가능 — _encode_image가 generate() 내부에
    이미 포함돼 있고 커스텀 오버라이드로 확인됨)

실행:
  .venv/bin/python3 scripts/bench_grounder_vs_florence2.py
  (젯슨에서 돌릴 때도 동일 커맨드 — torch.cuda.is_available()로 디바이스 자동 선택,
   fp16 비교가 필요하면 환경 맞춰 dtype만 조정)

출력 해석: 세 줄의 레이턴시(ms) + 마지막에 "합쳐쓰기 가능성" 확인 로그.
"""
import time
import torch
import numpy as np
from PIL import Image

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WARMUP, N_ITERS = 3, 20


def bench(fn, n=N_ITERS, warmup=N_WARMUP):
    for _ in range(warmup):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.time()
        fn()
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1000)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    img = Image.fromarray((np.random.rand(720, 1280, 3) * 255).astype("uint8"))
    print(f"device={DEV}", flush=True)

    # ---- OWLv2 ----
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    owl_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl_model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(DEV).eval()

    def run_owl():
        inp = owl_proc(text=[["gray basket"]], images=img, return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = owl_model(**inp)
        _ = owl_proc.post_process_grounded_object_detection(
            out, threshold=0.1, target_sizes=torch.tensor([img.size[::-1]]))

    m, s = bench(run_owl)
    print(f"OWLv2 (forward+postprocess): {m:.1f}±{s:.1f} ms", flush=True)
    del owl_model
    if DEV.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Florence-2 그라운더 (generate) ----
    from transformers import AutoModelForCausalLM, AutoProcessor
    f2_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    f2_proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

    TASK = "<CAPTION_TO_PHRASE_GROUNDING>"

    def run_f2_grounder():
        inp = f2_proc(text=TASK + "gray basket", images=img, return_tensors="pt")
        with torch.no_grad():
            ids = f2_model.generate(
                input_ids=inp["input_ids"].to(DEV),
                pixel_values=inp["pixel_values"].to(DEV, torch.float16),
                max_new_tokens=128, num_beams=3, do_sample=False)
        _ = f2_proc.batch_decode(ids, skip_special_tokens=False)[0]

    m, s = bench(run_f2_grounder)
    print(f"Florence-2 그라운더 (generate, beam=3): {m:.1f}±{s:.1f} ms", flush=True)

    # ---- Florence-2 비전 인코더만 (별도 forward, 현재 비전백본 용도) ----
    def run_f2_vision_only():
        inp = f2_proc(text=TASK + "gray basket", images=img, return_tensors="pt")
        with torch.no_grad():
            _ = f2_model.vision_tower.forward_features_unpool(inp["pixel_values"].to(DEV, torch.float16))

    m, s = bench(run_f2_vision_only)
    print(f"Florence-2 비전 인코더만 (forward_features_unpool): {m:.1f}±{s:.1f} ms", flush=True)

    # ---- "합쳐쓰기" 가능성 확인 ----
    print("\n--- 합쳐쓰기(비전 재사용) 가능성 테스트 ---", flush=True)
    has_encode_image = hasattr(f2_model, "_encode_image") or hasattr(f2_model, "encode_image")
    print(f"model에 _encode_image/encode_image 존재: {has_encode_image}")
    import inspect
    gen_src = inspect.getsource(f2_model.generate) if hasattr(f2_model, "generate") else ""
    print("generate가 커스텀 오버라이드인지:", "def generate" in gen_src[:200] if gen_src else "확인불가")


if __name__ == "__main__":
    main()
