#!/usr/bin/env python3
"""
Exp47 준비: 경로 유형별 instruction 후보 프롬프트 생성 및 점수화

방법:
1. 9개 path type별 행동 패턴 + bbox 위치 기반으로 7개 후보 프롬프트 생성
2. Kosmos-2 text encoder로 임베딩 추출
3. 판별 점수(다른 path type과 얼마나 구분되는가) 계산
4. 대표 프레임 grounding 검증 (basket 실제 검출 여부)
5. 종합 순위 출력 → docs/v5/instruction_candidates.json
"""
import json, sys, time
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_DIR   = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
BBOX_DATA  = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
OUT_PATH   = ROOT / "docs" / "v5" / "instruction_candidates.json"

PATH_TYPES = [
    "center_straight", "center_left",  "center_right",
    "left_straight",   "left_left",    "left_right",
    "right_straight",  "right_left",   "right_right",
]

# ──────────────────────────────────────────────
# 경로 유형별 후보 프롬프트 (행동 패턴 + bbox 위치 기반)
# ──────────────────────────────────────────────
CANDIDATES = {
    "center_straight": [
        "Navigate straight forward to the basket directly ahead of you.",
        "Move straight toward the gray basket in the center of the hallway.",
        "Go straight to reach the basket positioned directly in front.",
        "Drive forward along the center path to the basket ahead.",
        "Proceed straight to the gray basket in front without turning.",
        "Move directly forward to the basket at the end of the corridor.",
        "Navigate to the gray basket ahead by going straight.",
    ],
    "center_left": [
        "Navigate to the basket ahead by curving to the left.",
        "Move forward and veer left to reach the basket in front.",
        "Approach the basket by gradually turning left while moving forward.",
        "Navigate to the gray basket ahead, taking a left-curving path.",
        "Move toward the basket by swinging left from the center.",
        "Drive forward with a left curve to reach the basket ahead.",
        "Go to the basket in front by bearing left as you advance.",
    ],
    "center_right": [
        "Navigate to the basket ahead by curving to the right.",
        "Move forward and veer right to reach the basket in front.",
        "Approach the basket by gradually turning right while moving forward.",
        "Navigate to the gray basket ahead, taking a right-curving path.",
        "Move toward the basket by swinging right from the center.",
        "Drive forward with a right curve to reach the basket ahead.",
        "Go to the basket in front by bearing right as you advance.",
    ],
    "left_straight": [
        "The basket is to your left. Align yourself and navigate straight to it.",
        "Navigate to the gray basket on your left by adjusting right then going straight.",
        "The basket is on the left side. Rotate slightly right and drive straight to it.",
        "Approach the basket on your left by first turning to face it, then going straight.",
        "Move to the gray basket on the left side of the room straight ahead.",
        "The basket is ahead and to the left. Correct your heading and go straight.",
        "Navigate straight to the basket that is currently on your left side.",
    ],
    "left_left": [
        "The basket is to your left. Navigate there by moving further left.",
        "Move left and forward to reach the gray basket on the left side.",
        "Navigate to the basket on your left by curving further to the left.",
        "The basket is on your left. Approach it by continuing leftward.",
        "Move forward while bearing left to navigate to the basket on the left.",
        "Navigate to the gray basket on the left side with a left-curving path.",
        "Go to the basket on your left by moving diagonally left and forward.",
    ],
    "left_right": [
        "The basket is to your left. Navigate there by swinging to the right.",
        "Approach the basket on the left side by curving right to meet it.",
        "Navigate to the gray basket on the left by taking a right arc.",
        "The basket is on your left. Swing right to approach it from the side.",
        "Move to the basket on your left by first going right then turning toward it.",
        "Navigate to the left-side basket by curving to the right.",
        "Go to the basket on your left, approaching from the right side.",
    ],
    "right_straight": [
        "The basket is to your right. Align yourself and navigate straight to it.",
        "Navigate to the gray basket on your right by adjusting left then going straight.",
        "The basket is on the right side. Rotate slightly left and drive straight to it.",
        "Approach the basket on your right by first turning to face it, then going straight.",
        "Move to the gray basket on the right side of the room straight ahead.",
        "The basket is ahead and to the right. Correct your heading and go straight.",
        "Navigate straight to the basket that is currently on your right side.",
    ],
    "right_left": [
        "The basket is to your right. Navigate there by swinging to the left.",
        "Approach the basket on the right side by curving left to meet it.",
        "Navigate to the gray basket on the right by taking a left arc.",
        "The basket is on your right. Swing left to approach it from the side.",
        "Move to the basket on your right by curving left toward it.",
        "Navigate to the right-side basket by curving to the left.",
        "Go to the basket on your right, approaching from the left side.",
    ],
    "right_right": [
        "The basket is to your right. Navigate there by moving further right.",
        "Move right and forward to reach the gray basket on the right side.",
        "Navigate to the basket on your right by curving further to the right.",
        "The basket is on your right. Approach it by continuing rightward.",
        "Move forward while bearing right to navigate to the basket on the right.",
        "Navigate to the gray basket on the right side with a right-curving path.",
        "Go to the basket on your right by moving diagonally right and forward.",
    ],
}


def load_rep_frames(n_per_type=3):
    """각 path type별 대표 프레임 n개 로드."""
    files_by_type = defaultdict(list)
    for f in sorted(DATA_DIR.glob("*.h5")):
        for pt in PATH_TYPES:
            if pt in f.name:
                files_by_type[pt].append(f)
                break

    rep = {}
    for pt, files in files_by_type.items():
        frames = []
        for f_path in files[:n_per_type]:
            with h5py.File(f_path, "r") as f:
                imgs = f["observations"]["images"][:]
            frames.append(Image.fromarray(imgs[0].astype(np.uint8)))
        rep[pt] = frames
    return rep


def get_text_embedding(model, proc, text, device):
    """Kosmos-2 text encoder로 문장 임베딩 추출 (last hidden state mean-pool)."""
    inputs = proc(text=text, images=None, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        out = model.text_model(input_ids=input_ids, output_hidden_states=True)
        # hidden_states: tuple[n_layers+1] of (1, seq_len, hidden)
        emb = out.hidden_states[-1][0].mean(0)  # (hidden,)
    return emb.cpu().float().numpy()


def grounding_score(model, proc, prompt, frames, device):
    """grounding 프롬프트로 basket bbox 검출 성공률 반환."""
    full_prompt = f"<grounding>{prompt}"
    success = 0
    for img in frames:
        inputs = proc(text=full_prompt, images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48)
        decoded = proc.decode(out[0], skip_special_tokens=False)
        # bbox token 포함 여부로 판단
        has_bbox = "<phrase>" in decoded and "<box>" in decoded
        success += int(has_bbox)
    return success / max(len(frames), 1)


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1] 대표 프레임 로드...")
    rep_frames = load_rep_frames(n_per_type=3)
    print(f"    {sum(len(v) for v in rep_frames.values())}개 프레임 로드")

    print("\n[2] Kosmos-2 로딩...")
    proc  = AutoProcessor.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True
    ).to(device).eval()
    print("    완료")

    # ── 텍스트 임베딩 (모든 후보)
    print("\n[3] 텍스트 임베딩 추출...")
    all_embeddings = {}  # path_type → list of (prompt, emb)
    total_cands = sum(len(v) for v in CANDIDATES.values())
    done = 0
    for pt in PATH_TYPES:
        all_embeddings[pt] = []
        for prompt in CANDIDATES[pt]:
            emb = get_text_embedding(model, proc, prompt, device)
            all_embeddings[pt].append((prompt, emb))
            done += 1
            print(f"    [{done}/{total_cands}] {pt}: {prompt[:50]}...")

    # ── 판별 점수: 같은 path type 내 평균 sim - 다른 path type과의 평균 sim
    print("\n[4] 판별 점수 계산...")
    all_embs_flat = [(pt, p, e) for pt, pairs in all_embeddings.items() for p, e in pairs]

    discriminability = {}
    for pt in PATH_TYPES:
        discriminability[pt] = []
        for prompt, emb in all_embeddings[pt]:
            # 같은 path type 내 다른 후보들과 sim (intra)
            same_sims = [cosine_sim(emb, e2) for p2, e2 in all_embeddings[pt] if p2 != prompt]
            # 다른 path type 후보들과 sim (inter)
            other_sims = [cosine_sim(emb, e2) for pt2, pairs2 in all_embeddings.items()
                          if pt2 != pt for p2, e2 in pairs2]
            # 판별 점수: inter sim이 낮을수록 (다른 path와 구별 잘 됨), intra sim이 높을수록
            score = -np.mean(other_sims)  # 다른 path와 거리 클수록 ↑
            discriminability[pt].append((prompt, score))

    # ── Grounding 검증 (상위 4개 후보만)
    print("\n[5] Grounding 검증 (path type당 상위 4개 후보)...")
    grounding_results = {}
    for pt in PATH_TYPES:
        frames = rep_frames.get(pt, [])
        # 판별 점수 상위 4개만 grounding 테스트
        top4 = sorted(discriminability[pt], key=lambda x: -x[1])[:4]
        grounding_results[pt] = {}
        for prompt, _ in top4:
            gscore = grounding_score(model, proc, prompt, frames, device)
            grounding_results[pt][prompt] = gscore
            print(f"    [{pt}] gscore={gscore:.2f}  '{prompt[:55]}...'")

    # ── 종합 점수 & 순위
    print("\n[6] 종합 순위 계산...")
    results = {}
    for pt in PATH_TYPES:
        scored = []
        disc_map = dict(discriminability[pt])
        gr_map   = grounding_results.get(pt, {})

        for prompt in CANDIDATES[pt]:
            disc  = disc_map.get(prompt, 0.0)
            gr    = gr_map.get(prompt, None)
            # 정규화: disc는 이미 음수 sim 기반, gr은 0~1
            # grounding 검증 없으면 disc만 사용
            if gr is not None:
                combined = 0.5 * disc + 0.5 * gr
            else:
                combined = disc
            scored.append({
                "rank": None,
                "prompt": prompt,
                "disc_score": round(float(disc), 4),
                "grounding_rate": round(float(gr), 3) if gr is not None else None,
                "combined_score": round(float(combined), 4),
            })

        scored.sort(key=lambda x: -x["combined_score"])
        for i, s in enumerate(scored):
            s["rank"] = i + 1
        results[pt] = scored

    # ── 출력
    print("\n" + "="*60)
    print("=== 경로 유형별 추천 프롬프트 (상위 3개) ===")
    print("="*60)
    for pt in PATH_TYPES:
        print(f"\n[{pt}]")
        for s in results[pt][:3]:
            gr_str = f"  gr={s['grounding_rate']:.2f}" if s['grounding_rate'] is not None else ""
            print(f"  #{s['rank']}  disc={s['disc_score']:.4f}{gr_str}  combined={s['combined_score']:.4f}")
            print(f"       \"{s['prompt']}\"")

    # 저장
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    elapsed = time.time() - t_start
    print(f"\n저장: {OUT_PATH}")
    print(f"총 소요: {elapsed/60:.1f}분")


if __name__ == "__main__":
    main()
