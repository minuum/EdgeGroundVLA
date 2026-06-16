#!/usr/bin/env python3
"""
Catastrophic Forgetting Test — L3 계층 검증

교수님 5/22 미팅 (line 126~132):
  "만약에 V-encoder 하이레벨만 로라고 했다 그러면
   기존에 했던 '개'도 되고 새롭게 봤습니다. 그래서 바스켓도 되고 둘 다 돼야 제대로."

Stage 1 LoRA (exp53)가 basket 인식을 추가했을 때,
기존 pretrain 객체들(RT-1, OXE)도 여전히 grounding 가능한지 확인.

테스트 객체:
  - RoboVLM pretrain (RT-1/OXE): orange, blue bowl, 7-up can, apple, rubiks cube
  - 우리 학습 객체: gray basket (LoRA로 추가)
  - 일반 객체: book, bottle, cup (Kosmos-2 pretrain에 있을 것으로 추정)

이미지: PIL로 직접 다운로드하거나, 색상 패턴 합성 이미지 사용.

Usage:
  .venv/bin/python3 scripts/test_catastrophic_forgetting.py
  .venv/bin/python3 scripts/test_catastrophic_forgetting.py --use-synthetic
  .venv/bin/python3 scripts/test_catastrophic_forgetting.py --image-dir /path/to/test_images
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
LORA_DIR  = ROOT / "runs" / "v5_nav" / "mlp" / "exp53"

MAX_NEW_TOKENS = 128

# 테스트 객체: (이름, grounding phrase, 기대 결과)
# RT-1 데이터셋의 대표 객체들 (교수님 지적)
TEST_OBJECTS = [
    # (display_name, grounding_phrase, category)
    ("orange",      "an orange",           "RT-1 pretrain"),
    ("blue bowl",   "a blue bowl",         "RT-1 pretrain"),
    ("7-up can",    "a 7-up can",          "RT-1 pretrain"),
    ("apple",       "an apple",            "RT-1 pretrain"),
    ("rubiks cube", "a rubiks cube",       "RT-1 pretrain"),
    ("book",        "a book",              "general"),
    ("bottle",      "a bottle",            "general"),
    ("cup",         "a cup",               "general"),
    ("gray basket", "a gray basket",       "our object"),  # LoRA로 학습한 것
]


# ─── 합성 이미지 생성 ─────────────────────────────────────

def make_synthetic_image(object_name):
    """
    단순 색상 블록으로 객체를 표현한 224×224 이미지.
    실제 이미지 없이 grounding 테스트용.
    (Kosmos-2는 실제 이미지에서 학습됐으므로, 합성 이미지에서는 성능 낮음.)
    """
    COLOR_MAP = {
        "orange":       ((255, 165, 0), "orange"),
        "blue bowl":    ((0, 0, 200), "blue"),
        "7-up can":     ((0, 200, 0), "green"),
        "apple":        ((200, 0, 0), "red"),
        "rubiks cube":  ((255, 200, 0), "colorful"),
        "book":         ((180, 140, 100), "brown"),
        "bottle":       ((100, 180, 220), "light blue"),
        "cup":          ((240, 240, 240), "white"),
        "gray basket":  ((128, 128, 128), "gray"),
    }
    color, desc = COLOR_MAP.get(object_name, ((200, 200, 200), "gray"))
    img = Image.new("RGB", (224, 224), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # 중앙에 색상 블록
    cx, cy = 112, 112
    w, h = 80, 60
    draw.rectangle(
        [cx - w//2, cy - h//2, cx + w//2, cy + h//2],
        fill=color, outline=(0, 0, 0), width=2,
    )
    # 텍스트 레이블
    try:
        draw.text((10, 10), object_name, fill=(0, 0, 0))
    except Exception:
        pass
    return img


# ─── 모델 로드 ───────────────────────────────────────────

def load_base_model(device):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    print("[MODEL] Base Kosmos-2 로딩...", flush=True)
    proc  = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16
    ).to(device)
    model.eval()
    return proc, model


def load_lora_model(device):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    try:
        from peft import PeftModel
    except ImportError:
        print("[WARN] peft 없음 → base 모델 사용", flush=True)
        return load_base_model(device)

    adapter_dir = LORA_DIR / "clip_lora_adapter"
    if not adapter_dir.exists():
        print(f"[WARN] LoRA adapter 없음: {adapter_dir}", flush=True)
        return load_base_model(device)

    print(f"[MODEL] Stage1 LoRA (exp53) 로딩: {adapter_dir}", flush=True)
    proc = AutoProcessor.from_pretrained(str(VLM_PATH))
    base = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    model.eval()
    return proc, model


# ─── Grounding 추론 ──────────────────────────────────────

def grounding_inference(proc, model, pil_image, phrase, device):
    """
    Returns: (grounded: bool, boxes: list of (x1,y1,x2,y2), decoded_text: str)
    """
    prompt = f"<grounding><phrase>{phrase}</phrase>"
    inputs = proc(text=prompt, images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

    with torch.no_grad():
        gen = model.generate(**inputs, use_cache=True, max_new_tokens=MAX_NEW_TOKENS)

    decoded = proc.batch_decode(gen, skip_special_tokens=False)[0]
    _, entities = proc.post_process_generation(decoded, cleanup_and_extract=True)

    boxes = []
    for ent_phrase, _, bboxes in entities:
        if bboxes:
            boxes.extend(bboxes)

    grounded = len(boxes) > 0
    return grounded, boxes, decoded


# ─── 평가 ────────────────────────────────────────────────

def evaluate_model(label, proc, model, test_items, device):
    """
    test_items: list of (object_name, phrase, category, pil_image)
    Returns: dict {object_name: {"grounded": bool, "boxes": list}}
    """
    print(f"\n{'='*60}", flush=True)
    print(f"[{label}]", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}
    for obj_name, phrase, category, img in test_items:
        grounded, boxes, decoded = grounding_inference(proc, model, img, phrase, device)
        results[obj_name] = {
            "grounded": grounded,
            "boxes": boxes,
            "category": category,
            "phrase": phrase,
        }
        status = "✅" if grounded else "❌"
        box_str = f" → {boxes[0]}" if boxes else ""
        print(f"  {status} [{category:15s}] '{obj_name}': grounded={grounded}{box_str}",
              flush=True)

    return results


def print_comparison(base_results, lora_results):
    print(f"\n{'='*60}")
    print("COMPARISON: Base Kosmos-2 vs Stage1 LoRA (exp53)")
    print(f"{'='*60}")
    print(f"  {'Object':15s} {'Category':15s} {'Base':6s} {'LoRA':6s} {'변화'}")
    print(f"  {'-'*60}")

    for obj_name in base_results:
        b = base_results[obj_name]
        l = lora_results.get(obj_name, {})
        b_g = "✅" if b.get("grounded") else "❌"
        l_g = "✅" if l.get("grounded") else "❌"
        cat = b.get("category", "?")

        if b.get("grounded") and not l.get("grounded"):
            change = "⚠️  LOST"
        elif not b.get("grounded") and l.get("grounded"):
            change = "🆕 GAINED"
        elif b.get("grounded") and l.get("grounded"):
            change = "✅ KEPT"
        else:
            change = "❌ BOTH FAIL"

        print(f"  {obj_name:15s} {cat:15s} {b_g:6s} {l_g:6s} {change}")

    # 요약
    pretrain_objs = {k: v for k, v in base_results.items()
                     if v["category"] != "our object"}
    lost = sum(1 for k, v in pretrain_objs.items()
               if v["grounded"] and not lora_results.get(k, {}).get("grounded"))
    total_base_ok = sum(1 for v in pretrain_objs.values() if v["grounded"])

    print(f"\n  Pretrain objects (Base 기준 OK {total_base_ok}개 중):")
    print(f"    LoRA 후 유지: {total_base_ok - lost}개")
    print(f"    LoRA 후 실패: {lost}개  ← catastrophic forgetting 여부")

    basket_base = base_results.get("gray basket", {}).get("grounded", False)
    basket_lora = lora_results.get("gray basket", {}).get("grounded", False)
    print(f"\n  gray basket: Base={basket_base}, LoRA={basket_lora}")

    print("\n[결론]")
    if lost == 0:
        print("  ✅ Catastrophic forgetting 없음 — pretrain 객체 유지하며 basket 추가")
    elif lost <= 2:
        print("  ⚠️  일부 pretrain 객체 성능 저하 — 허용 범위 여부 교수님 판단 필요")
    else:
        print("  ❌ Catastrophic forgetting 발생 — LoRA가 기존 표현 파괴")

    if not basket_base and basket_lora:
        print("  ✅ LoRA가 basket 인식 능력을 새로 추가함")
    elif basket_base and basket_lora:
        print("  ℹ️  Base도 basket 인식 → LoRA 기여 불분명")


# ─── 메인 ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-synthetic", action="store_true",
                        help="합성 이미지 사용 (실제 이미지 없을 때)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="실제 테스트 이미지 디렉토리 (object_name.jpg 형식)")
    parser.add_argument("--model", choices=["base", "lora", "both"], default="both")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}", flush=True)

    # 이미지 준비
    test_items = []
    image_dir = Path(args.image_dir) if args.image_dir else None

    for obj_name, phrase, category in TEST_OBJECTS:
        img = None
        if image_dir is not None:
            # 실제 이미지 우선
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                candidate = image_dir / (obj_name.replace(" ", "_") + ext)
                if candidate.exists():
                    img = Image.open(candidate).convert("RGB")
                    print(f"  [IMAGE] {obj_name}: {candidate.name}", flush=True)
                    break

        if img is None:
            if not args.use_synthetic and image_dir is not None:
                print(f"  [SKIP] {obj_name}: 이미지 없음 (--use-synthetic 없이는 스킵)", flush=True)
                continue
            # 합성 이미지
            img = make_synthetic_image(obj_name)
            print(f"  [SYNTH] {obj_name}: 합성 이미지 생성", flush=True)

        test_items.append((obj_name, phrase, category, img))

    if not test_items:
        print("[ERROR] 테스트 이미지가 없습니다.")
        print("  옵션 1: --use-synthetic (합성 이미지)")
        print("  옵션 2: --image-dir /path/to/images (실제 이미지)")
        sys.exit(1)

    print(f"\n[DATA] {len(test_items)} 객체 테스트 준비 완료", flush=True)

    # 모델별 평가
    base_results, lora_results = None, None

    if args.model in ("base", "both"):
        proc, model = load_base_model(device)
        base_results = evaluate_model("Base Kosmos-2", proc, model, test_items, device)
        del proc, model
        torch.cuda.empty_cache()

    if args.model in ("lora", "both"):
        proc, model = load_lora_model(device)
        lora_results = evaluate_model("Stage1 LoRA (exp53)", proc, model, test_items, device)
        del proc, model
        torch.cuda.empty_cache()

    if base_results and lora_results:
        print_comparison(base_results, lora_results)

    print("\n[완료]")
    print("  이 결과를 교수님께 보고:")
    print("  - 'LoRA 적용 후에도 pretrain 객체들이 여전히 grounding 됩니다'")
    print("  - 혹은 'N개 객체에서 성능 저하 → 재학습 방향 조정 필요'")
    print()
    print("  주의: 합성 이미지(--use-synthetic)는 실제 이미지보다 정확도 낮음.")
    print("  RT-1 실제 이미지나 웹 이미지로 테스트하면 더 신뢰성 높음.")
    print("  실제 이미지 경로: --image-dir docs/v5/test_images/")


if __name__ == "__main__":
    main()
