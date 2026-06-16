#!/usr/bin/env python3
"""
Exp53 테스트케이스 — 방향어 있음/없음 비교

교수님 요구 형식 (5/15 미팅):
  방향어 포함 ("basket under left") vs 방향어 없음 ("go to the box")
  각 방향 × N회 → 성공률 테이블

  판별 기준:
  - 방향어 있을 때 + 없을 때 모두 성공 → 박스 시각 인식 근거
  - 방향어 있을 때만 성공 → 텍스트 패턴 암기 의심

Usage:
  # 모델 로드 후 오프라인 테스트 (H5 에피소드 기반)
  python3 scripts/test_clip_lora_testcases.py

  # 방향어 있음/없음 지정
  python3 scripts/test_clip_lora_testcases.py --prompt-type directional
  python3 scripts/test_clip_lora_testcases.py --prompt-type neutral
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP46_DIR  = ROOT / "docs" / "v5" / "bbox_nav_exp46"
EXP53_DIR  = ROOT / "docs" / "v5" / "bbox_nav_exp53"
MLP_DIR    = ROOT / "runs" / "v5_nav" / "mlp" / "exp53"
VLM_PATH   = ROOT / ".vlms" / "kosmos-2-patch14-224"
CKPT_PATH  = MLP_DIR / "exp53_clip_lora.pt"
LORA_DIR   = MLP_DIR / "clip_lora_adapter"

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
GOAL_DIM    = 3
D_IN        = WINDOW * 4 + VIS_DIM + GOAL_DIM

# 방향어 포함 프롬프트 (그라운딩에 사용)
DIRECTIONAL_PROMPTS = {
    "left":   "<grounding>The gray basket is at the left",
    "center": "<grounding>The gray basket is at the center",
    "right":  "<grounding>The gray basket is at the right",
}
# 방향어 없는 프롬프트
NEUTRAL_PROMPT = "<grounding>The gray basket is at"

# path_type → 바스켓 위치 방향 매핑
PATH_TO_DIRECTION = {
    "left_straight":  "left",
    "left_left":      "left",
    "left_right":     "left",
    "center_straight":"center",
    "center_left":    "center",
    "center_right":   "center",
    "right_straight": "right",
    "right_left":     "right",
    "right_right":    "right",
}

LORA_R      = 16
LORA_ALPHA  = 32
LORA_LAYERS = list(range(16, 24))
LORA_TARGET = ["q_proj", "v_proj"]


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
class GoalNavMLP(nn.Module):
    def __init__(self, d_in: int = D_IN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def load_model(device: torch.device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model, PeftModel

    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    base = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    if LORA_DIR.exists():
        vision_model = PeftModel.from_pretrained(base.vision_model, str(LORA_DIR))
        print(f"[MODEL] LoRA 어댑터 로드: {LORA_DIR}")
    else:
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET,
            layers_to_transform=LORA_LAYERS,
            layers_pattern="layers",
            bias="none",
        )
        vision_model = get_peft_model(base.vision_model, lora_cfg)
        print("[MODEL] LoRA 어댑터 없음 — 랜덤 초기화 (학습 후 재실행)")

    vision_model = vision_model.to(device).eval()

    mlp = GoalNavMLP(D_IN).to(device)
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        mlp.load_state_dict(ckpt["mlp"])
        print(f"[MODEL] MLP 로드: {CKPT_PATH}  val_acc={ckpt.get('val_acc', '?'):.4f}")
    else:
        print(f"[WARNING] {CKPT_PATH} 없음 — 학습 먼저 실행하세요.")

    mlp.eval()
    return processor, vision_model, mlp


# ──────────────────────────────────────────────
# Grounding → goal(cx, cy, area)
# ──────────────────────────────────────────────
@torch.no_grad()
def ground_image(processor, vision_model, pil_image: Image.Image,
                 prompt: str, base_model, device: torch.device) -> tuple[float, float, float]:
    """Kosmos-2 grounding → (cx, cy, area). 실패 시 (0.5, 0.5, 0.0)."""
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pv = inputs["pixel_values"].to(
        torch.float16 if device.type == "cuda" else torch.float32
    )
    generated = base_model.generate(
        pixel_values=pv,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs.get("image_embeds_position_mask"),
        use_cache=True,
        max_new_tokens=64,
    )
    new_ids = generated[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(new_ids, skip_special_tokens=False)[0]
    caption, entities = processor.post_process_generation(raw)

    basket_kw = ("basket", "gray box", "container", "bin", "laundry")
    for entity_name, _span, boxes in entities:
        if any(k in entity_name.lower() for k in basket_kw) and boxes:
            x1, y1, x2, y2 = [float(v) for v in boxes[0]]
            if max(x1, y1, x2, y2) > 1.5:
                x1, y1, x2, y2 = x1/1000, y1/1000, x2/1000, y2/1000
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            return cx, cy, area

    return 0.5, 0.5, 0.0  # fallback


@torch.no_grad()
def extract_vis_feat(vision_model, processor, pil_image: Image.Image,
                     device: torch.device) -> torch.Tensor:
    inputs = processor(images=pil_image, return_tensors="pt")
    pv = inputs["pixel_values"].to(
        device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    out = vision_model(pixel_values=pv)
    return out.last_hidden_state.mean(dim=1).float().squeeze(0)  # (1024,)


def build_bbox_feat(frames, t):
    bbox = []
    for k in range(WINDOW):
        idx = max(0, t - (WINDOW - 1 - k))
        fr = frames[idx]
        bbox.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])
    return np.array(bbox, dtype=np.float32)


# ──────────────────────────────────────────────
# 오프라인 테스트 (H5 에피소드 기반)
# ──────────────────────────────────────────────
def run_offline_testcases(prompt_type: str, n_episodes_per_dir: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}")

    from transformers import AutoModelForVision2Seq
    base_model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    processor, vision_model, mlp = load_model(device)

    # 데이터셋 (Exp53 전용 있으면 우선, 없으면 Exp46 150ep)
    exp53_data = EXP53_DIR / "bbox_dataset.json"
    data_path = exp53_data if exp53_data.exists() else EXP46_DIR / "bbox_dataset_full.json"
    print(f"\n[DATA] {data_path}")
    bbox_data = json.loads(data_path.read_text())

    # 방향별 에피소드 샘플링
    dir_eps = defaultdict(list)
    for ep in bbox_data:
        direction = PATH_TO_DIRECTION.get(ep["path_type"])
        if direction:
            dir_eps[direction].append(ep)

    results = []
    print(f"\n[TEST] prompt_type = {prompt_type}\n")

    for direction in ["left", "center", "right"]:
        eps = dir_eps[direction][:n_episodes_per_dir]
        if not eps:
            print(f"  [{direction}] 에피소드 없음 — 건너뜀")
            continue

        if prompt_type == "directional":
            grounding_prompt = DIRECTIONAL_PROMPTS[direction]
        else:
            grounding_prompt = NEUTRAL_PROMPT

        ep_correct = []
        for ep in eps:
            frames = ep["frames"]
            h5_path = ep["episode"]

            try:
                with h5py.File(h5_path, "r") as f:
                    images = [
                        Image.fromarray(f["observations"]["images"][i])
                        for i in range(len(frames))
                    ]
            except Exception as e:
                print(f"    [SKIP] {h5_path}: {e}")
                continue

            # goal: 첫 프레임 grounding
            cx0, cy0, area0 = ground_image(
                processor, vision_model, images[0], grounding_prompt, base_model, device
            )
            goal = torch.tensor([cx0, cy0, area0], dtype=torch.float32, device=device)

            correct, total = 0, 0
            for t, (frame, pil) in enumerate(zip(frames, images)):
                vis_feat = extract_vis_feat(vision_model, processor, pil, device)
                bbox = torch.tensor(build_bbox_feat(frames, t), device=device)
                feat = torch.cat([bbox, vis_feat, goal]).unsqueeze(0)
                pred = mlp(feat).argmax(1).item()
                correct += int(pred == frame["gt_class"])
                total += 1

            pm = correct / total if total > 0 else 0.0
            ep_correct.append(pm)
            success = pm >= 0.8
            print(f"  [{direction}] {Path(h5_path).stem[:40]}  PM={pm:.1%}  {'✅' if success else '❌'}")

        avg_pm = np.mean(ep_correct) if ep_correct else 0.0
        success_rate = np.mean([pm >= 0.8 for pm in ep_correct]) if ep_correct else 0.0
        results.append({
            "prompt_type": prompt_type,
            "direction": direction,
            "n": len(ep_correct),
            "avg_pm": avg_pm,
            "success_rate": success_rate,
        })

    _print_table(results)
    return results


def _print_table(results):
    print("\n" + "=" * 60)
    print("Exp53 테스트케이스 결과")
    print("=" * 60)
    print(f"{'프롬프트':>10} {'방향':>8} {'시도':>4} {'평균PM':>8} {'성공률':>8}")
    print("-" * 60)
    for r in results:
        pt_label = "방향어포함" if r["prompt_type"] == "directional" else "방향어없음"
        print(f"{pt_label:>10} {r['direction']:>8} {r['n']:>4} {r['avg_pm']:>7.1%} {r['success_rate']:>7.1%}")
    print("=" * 60)

    # 판별 논리
    dir_results  = [r for r in results if r["prompt_type"] == "directional"]
    neu_results  = [r for r in results if r["prompt_type"] == "neutral"]
    if dir_results and neu_results:
        dir_avg = np.mean([r["success_rate"] for r in dir_results])
        neu_avg = np.mean([r["success_rate"] for r in neu_results])
        print(f"\n방향어포함 평균 성공률: {dir_avg:.1%}")
        print(f"방향어없음 평균 성공률: {neu_avg:.1%}")
        if dir_avg >= 0.8 and neu_avg >= 0.8:
            print("→ 판정: 박스 시각 인식 근거 (방향어 없어도 성공)")
        elif dir_avg >= 0.8 and neu_avg < 0.5:
            print("→ 판정: 텍스트 패턴 암기 의심 (방향어 있을 때만 성공)")
        else:
            print("→ 판정: 불명확 — 추가 테스트 필요")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-type",
        choices=["directional", "neutral", "both"],
        default="both",
        help="directional=방향어포함, neutral=방향어없음, both=둘 다 실행",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=3,
        help="방향당 테스트 에피소드 수 (기본 3)",
    )
    args = parser.parse_args()

    if not CKPT_PATH.exists():
        print(f"[WARNING] {CKPT_PATH} 없음.")
        print("먼저 학습 실행: python3 scripts/train_clip_lora_exp53.py")

    if args.prompt_type == "both":
        results_dir = run_offline_testcases("directional", args.n_episodes)
        results_neu = run_offline_testcases("neutral", args.n_episodes)
        _print_table(results_dir + results_neu)
    else:
        run_offline_testcases(args.prompt_type, args.n_episodes)


if __name__ == "__main__":
    main()
