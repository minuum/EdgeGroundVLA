#!/usr/bin/env python3
"""
Hybrid backbone 생성 스크립트.

vision_model + image_to_text_projection → Google-robot (강한 로봇 비전)
text_model                              → Pure HF Kosmos-2 (text attn 22.7% 보존)

결과물: .vlms/kosmos-gr-vision-hf-text/ (HF 포맷, AutoModelForVision2Seq 로드 가능)
"""
import sys
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent

HF_PATH   = ROOT / ".vlms" / "kosmos-2-patch14-224"
GR_CKPT   = ROOT / ".vlms" / "checkpoints" / "kosmos_ph_google-robot-post-train.pt"
OUT_PATH  = ROOT / ".vlms" / "kosmos-gr-vision-hf-text"

VISION_PREFIXES = ("vision_model.", "image_to_text_projection.")


def main():
    print("=== Hybrid Backbone 생성 ===")
    print(f"  vision:  Google-robot ({GR_CKPT.name})")
    print(f"  text:    Pure HF Kosmos-2")
    print(f"  output:  {OUT_PATH}")

    # 1. Pure HF 모델 로드 (base)
    print("\n[1] Pure HF 모델 로드...")
    model = AutoModelForVision2Seq.from_pretrained(str(HF_PATH), trust_remote_code=True)
    sd = model.state_dict()
    print(f"    총 {len(sd)} 파라미터")

    # 2. Google-robot state_dict 로드
    print("[2] Google-robot 체크포인트 로드...")
    gr_raw = torch.load(str(GR_CKPT), map_location="cpu")
    gr_sd = {
        k.replace("model.backbone.", ""): v
        for k, v in gr_raw["state_dict"].items()
        if k.startswith("model.backbone.")
    }
    print(f"    총 {len(gr_sd)} 파라미터")

    # 3. vision + bridge 레이어 이식
    print("[3] Vision 레이어 이식...")
    replaced, skipped, shape_err = 0, 0, 0
    for k, v in gr_sd.items():
        if not any(k.startswith(p) for p in VISION_PREFIXES):
            skipped += 1
            continue
        if k not in sd:
            print(f"    WARNING: {k} not in HF model — skip")
            shape_err += 1
            continue
        if sd[k].shape != v.shape:
            print(f"    SHAPE MISMATCH: {k} HF{list(sd[k].shape)} != GR{list(v.shape)} — skip")
            shape_err += 1
            continue
        sd[k] = v
        replaced += 1

    print(f"    이식: {replaced}  유지(text): {skipped}  에러: {shape_err}")
    assert shape_err == 0, "Shape mismatch 발생 — 중단"

    # 4. 검증: text 레이어가 Pure HF 값인지 확인
    print("[4] 검증...")
    orig_sd = AutoModelForVision2Seq.from_pretrained(str(HF_PATH), trust_remote_code=True).state_dict()
    text_ok = all(
        torch.equal(sd[k], orig_sd[k])
        for k in sd if "text_model" in k
    )
    vision_ok = all(
        torch.equal(sd[k], gr_sd[k])
        for k in gr_sd if any(k.startswith(p) for p in VISION_PREFIXES) and k in sd
    )
    print(f"    text == Pure HF: {text_ok}")
    print(f"    vision == Google-robot: {vision_ok}")
    assert text_ok and vision_ok

    # 5. HF 포맷으로 저장
    print(f"[5] 저장 → {OUT_PATH}")
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    # config/tokenizer 파일은 Pure HF에서 복사
    for f in HF_PATH.iterdir():
        if f.suffix in (".json", ".txt", ".model", ".tiktoken") or f.name in ("vocab.json", "merges.txt", "tokenizer.model"):
            shutil.copy2(f, OUT_PATH / f.name)

    # weights 저장
    model.load_state_dict(sd)
    model.save_pretrained(str(OUT_PATH))

    print("\n완료!")
    print(f"사용법: model_path = \"{OUT_PATH}\"")
    print("       pretrained_vlm_path = null")


if __name__ == "__main__":
    main()
