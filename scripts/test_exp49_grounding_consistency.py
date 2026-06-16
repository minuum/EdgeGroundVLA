#!/usr/bin/env python3
"""
Exp49 Paraphrase-Grounding Consistency Test

같은 물체(바구니)를 다른 언어 표현으로 grounding했을 때
cx0가 얼마나 일치하는지 측정 → VLA paraphrase-robust 클레임 증명

테스트 구성:
  1. val 에피소드에서 첫 프레임 이미지 추출
  2. 다양한 언어 표현으로 Kosmos-2 grounding 실행
  3. cx0 분산 측정 (낮을수록 paraphrase-robust)
  4. cx0 → Exp49 MLP action 예측 → 행동 일관성 확인
"""
import json, sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoProcessor, AutoModelForVision2Seq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_KOSMOS  = ROOT / ".vlms" / "kosmos-2-patch14-224"
EXP46_DIR  = ROOT / "docs" / "v5" / "bbox_nav_exp46"
EXP49_DIR  = ROOT / "docs" / "v5" / "bbox_nav_exp49"
MLP49_DIR  = ROOT / "runs" / "v5_nav" / "mlp" / "exp49"
DATA_DIR   = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OUT_PATH   = EXP49_DIR / "grounding_consistency_results.json"

WINDOW      = 8
VIS_DIM     = 1024
GOAL_DIM    = 3
NUM_CLASSES = 8
CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]

# ── 테스트할 언어 표현 ──
# 모두 "바구니"를 가리키지만 표현이 다름
PROMPTS = {
    "original":    "<grounding>The gray basket is at",
    "paraphrase_1": "<grounding>The gray box is at",
    "paraphrase_2": "<grounding>The container is at",
    "paraphrase_3": "<grounding>The target object is at",
    "paraphrase_4": "<grounding>The basket in the scene is at",
    "wrong_object": "<grounding>The red chair is at",   # 존재하지 않는 물체 (sanity check)
}


def build_mlp(d_in):
    return nn.Sequential(
        nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64),   nn.ReLU(),
        nn.Linear(64, NUM_CLASSES),
    )


def load_exp49():
    ckpt = torch.load(str(MLP49_DIR / "exp49_mlp.pt"), map_location="cpu", weights_only=False)
    d_in = ckpt["d_in"]
    net  = build_mlp(d_in)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


def run_grounding(processor, model, image, prompt, device):
    """단일 이미지 + 프롬프트 → (cx, cy, area) 또는 None"""
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    pv = inputs["pixel_values"].to(torch.float16).to(device)
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    emb_pos_mask = inputs.get("image_embeds_position_mask")
    if emb_pos_mask is not None:
        emb_pos_mask = emb_pos_mask.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pv,
            input_ids=input_ids,
            attention_mask=attn_mask,
            image_embeds=None,
            image_embeds_position_mask=emb_pos_mask,
            use_cache=True,
            max_new_tokens=64,
        )

    new_ids = generated_ids[:, input_ids.shape[1]:]
    raw_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    _, entities = processor.post_process_generation(raw_text)

    best = None
    for _, _span, boxes in entities:
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > 0.85:  # fullscreen 필터
                continue
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if best is None or area < best["area"]:
                best = {"cx": float(cx), "cy": float(cy), "area": float(area)}

    return best


def predict_action(net, bbox_feats, vis_feat, goal_cx, goal_cy, goal_area, device):
    """Exp49 MLP로 action 예측"""
    goal = np.array([goal_cx, goal_cy, goal_area], dtype=np.float32)
    feat = np.concatenate([bbox_feats, vis_feat, goal])
    x = torch.tensor([feat], dtype=torch.float32).to(device)
    with torch.no_grad():
        cls = int(net(x).argmax(1).item())
    return cls


def main():
    print("=" * 65)
    print("Exp49 Paraphrase-Grounding Consistency Test")
    print("=" * 65)

    # ── 데이터 로드 ──
    bbox_data = json.loads((EXP46_DIR / "bbox_dataset_full.json").read_text())
    vis_npz   = np.load(str(EXP46_DIR / "vision_features.npz"))
    vis_idx   = json.loads((EXP46_DIR / "vision_features_index.json").read_text())
    vis_cache = {ep: vis_npz[f"ep_{i}"] for ep, i in vis_idx.items()}

    # val split (Exp49 학습과 동일)
    ep_labels = [ep["path_type"] for ep in bbox_data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(sss.split(np.zeros(len(bbox_data)), ep_labels))
    val_eps = [bbox_data[i] for i in val_idx]
    print(f"Val 에피소드: {len(val_eps)}개")

    # ── 모델 로드 ──
    net    = load_exp49()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = net.to(device)

    print(f"\n[MODEL] Pure HF Kosmos-2 grounding 모델 로드...")
    proc = AutoProcessor.from_pretrained(str(HF_KOSMOS), trust_remote_code=True)
    vlm  = AutoModelForVision2Seq.from_pretrained(
        str(HF_KOSMOS), trust_remote_code=True, torch_dtype=torch.float16
    ).to(device).eval()
    print("  ✅ 로드 완료")

    # ── path_type별 대표 에피소드 선택 (9개) ──
    seen_types = set()
    test_eps = []
    for ep in val_eps:
        if ep["path_type"] not in seen_types:
            seen_types.add(ep["path_type"])
            test_eps.append(ep)
        if len(seen_types) == 9:
            break
    print(f"\n테스트 에피소드: {len(test_eps)}개 (path_type 1개씩)")

    results = []
    print("\n" + "=" * 65)

    for ep_data in test_eps:
        pt       = ep_data["path_type"]
        ep_path  = ep_data["episode"]
        frames   = ep_data["frames"]
        vis_feats = vis_cache.get(ep_path)

        # 첫 프레임 이미지 로드
        with h5py.File(ep_path, "r") as f:
            if "observations" in f and "images" in f["observations"]:
                img_arr = f["observations"]["images"][0]
            else:
                img_arr = f["images"][0]
        image = Image.fromarray(img_arr)

        # 캐시된 원본 cx0 (Exp46 grounding 결과)
        fr0 = frames[0]
        cached_cx0 = fr0["cx"] if fr0["has_bbox"] else 0.5

        print(f"\n[{pt}]  cached_cx0={cached_cx0:.3f}")

        # bbox_history: 첫 프레임이므로 t=0 전부 frame 0
        bbox_feats = []
        for _ in range(WINDOW):
            bbox_feats.extend([fr0["cx"], fr0["cy"], fr0["area"], float(fr0["has_bbox"])])
        bbox_feats = np.array(bbox_feats, dtype=np.float32)
        vis_feat = vis_feats[0] if vis_feats is not None else np.zeros(VIS_DIM, dtype=np.float32)

        # 원본 goal로 예측한 baseline action
        baseline_action = predict_action(
            net, bbox_feats, vis_feat,
            cached_cx0, fr0["cy"] if fr0["has_bbox"] else 0.5,
            fr0["area"] if fr0["has_bbox"] else 0.0,
            device
        )

        ep_result = {
            "path_type": pt,
            "cached_cx0": cached_cx0,
            "baseline_action": CLASS_NAMES[baseline_action],
            "prompts": {},
        }

        cx_values = []
        for prompt_name, prompt in PROMPTS.items():
            grounded = run_grounding(proc, vlm, image, prompt, device)
            if grounded:
                cx = grounded["cx"]
                cy = grounded["cy"]
                area = grounded["area"]
                action = predict_action(net, bbox_feats, vis_feat, cx, cy, area, device)
                action_name = CLASS_NAMES[action]
                match = (action == baseline_action)
                cx_values.append(cx)
                print(f"  {prompt_name:<18}: cx={cx:.3f}  → {action_name:<8}  {'✅' if match else '❌'}")
                ep_result["prompts"][prompt_name] = {
                    "cx": cx, "cy": cy, "area": area,
                    "action": action_name, "matches_baseline": match,
                }
            else:
                print(f"  {prompt_name:<18}: grounding 실패")
                ep_result["prompts"][prompt_name] = {"cx": None, "action": None, "matches_baseline": None}

        # cx 분산 계산 (wrong_object 제외)
        valid_cxs = [ep_result["prompts"][k]["cx"]
                     for k in PROMPTS if k != "wrong_object"
                     and ep_result["prompts"][k]["cx"] is not None]
        if valid_cxs:
            cx_std = float(np.std(valid_cxs))
            ep_result["cx_std_excl_wrong"] = cx_std
            print(f"  → paraphrase cx std (wrong 제외): {cx_std:.4f}")

        results.append(ep_result)

    del vlm
    torch.cuda.empty_cache()

    # ── 요약 ──
    print("\n" + "=" * 65)
    print("최종 요약")
    print("=" * 65)

    paraphrase_keys = [k for k in PROMPTS if k not in ("original", "wrong_object")]
    total_match, total_tested = 0, 0
    cx_stds = []

    for r in results:
        pt = r["path_type"]
        baseline = r["baseline_action"]
        matches = []
        for pk in paraphrase_keys:
            pr = r["prompts"].get(pk, {})
            if pr.get("action") is not None:
                matches.append(pr["matches_baseline"])
                total_tested += 1
                if pr["matches_baseline"]:
                    total_match += 1
        if r.get("cx_std_excl_wrong") is not None:
            cx_stds.append(r["cx_std_excl_wrong"])
        match_str = f"{sum(matches)}/{len(matches)}" if matches else "—"
        print(f"  {pt:<18}: baseline={baseline:<8}  paraphrase_match={match_str}  cx_std={r.get('cx_std_excl_wrong', 0):.4f}")

    overall_match_rate = total_match / total_tested if total_tested > 0 else 0
    mean_cx_std = float(np.mean(cx_stds)) if cx_stds else 0

    print(f"\n  paraphrase action 일치율: {overall_match_rate:.1%}  ({total_match}/{total_tested})")
    print(f"  평균 cx std (paraphrase): {mean_cx_std:.4f}")
    print(f"  (낮을수록 grounding이 표현에 무관하게 일관됨)")

    verdict = "PASS" if overall_match_rate >= 0.7 and mean_cx_std < 0.10 else "PARTIAL" if overall_match_rate >= 0.5 else "FAIL"
    print(f"\n  Verdict: {verdict}")
    print("  (≥70% 일치 + cx_std<0.10 → PASS)")

    OUT_PATH.write_text(json.dumps({
        "n_episodes": len(results),
        "overall_match_rate": overall_match_rate,
        "mean_cx_std": mean_cx_std,
        "verdict": verdict,
        "results": results,
    }, indent=2))
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
