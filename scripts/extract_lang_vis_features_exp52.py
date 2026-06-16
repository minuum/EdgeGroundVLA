#!/usr/bin/env python3
"""
Exp52: Language-Conditioned Visual Feature 추출

Kosmos-2 joint forward(image + text) → LM last hidden state의 image token 추출
→ mean pool → 2048-dim (언어가 이미지 처리에 영향을 준 feature)

Step 0 (선행 검증): 두 instruction으로 같은 이미지를 넣었을 때 cosine sim 비교
Step 1: 전체 150 ep × all frames feature 추출 → lang_vis_features.npz

Usage:
  .venv/bin/python3 scripts/extract_lang_vis_features_exp52.py
  .venv/bin/python3 scripts/extract_lang_vis_features_exp52.py --skip_verify  # 검증 스킵
"""
import argparse, gc, json, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_KOSMOS_PATH = ROOT / ".vlms" / "kosmos-2-patch14-224"
EXP46_DIR      = ROOT / "docs" / "v5" / "bbox_nav_exp46"
OUT_DIR        = ROOT / "docs" / "v5" / "bbox_nav_exp52"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANG_VIS_CACHE = OUT_DIR / "lang_vis_features.npz"
LANG_VIS_INDEX = OUT_DIR / "lang_vis_features_index.json"

INSTRUCTIONS = {
    "center_straight": "Navigate straight ahead to the basket in the center",
    "center_left":     "Navigate to the basket on the left",
    "center_right":    "Navigate to the basket on the right",
    "left_straight":   "Turn left and navigate straight to the basket",
    "left_left":       "Turn left and go to the basket on the left side",
    "left_right":      "Turn left then right to reach the basket",
    "right_straight":  "Turn right and navigate straight to the basket",
    "right_left":      "Turn right then left to reach the basket",
    "right_right":     "Turn right and go to the basket on the right side",
}

PARAPHRASES = {
    "center_left":  "Go to the container on the left side",
    "center_right": "Head toward the container on the right",
}


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

def load_model():
    from transformers import AutoModelForVision2Seq, AutoProcessor
    proc  = AutoProcessor.from_pretrained(str(HF_KOSMOS_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(HF_KOSMOS_PATH), torch_dtype=torch.float16
    ).cuda().eval()
    return proc, model


def extract_feat(proc, model, pil_img, instruction):
    """joint forward → image token hidden states mean (2048-dim float32)"""
    inputs = proc(text=instruction, images=pil_img, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    # pixel_values → float16
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    hs   = out.hidden_states[-1]                              # (1, seq_len, 2048)
    mask = inputs["image_embeds_position_mask"][0].bool()     # (seq_len,)
    feat = hs[0][mask].mean(0).float().cpu().numpy()          # (2048,)
    return feat


def load_episode_images(ep_path):
    with h5py.File(ep_path, "r") as f:
        if "observations" in f and "images" in f["observations"]:
            return f["observations"]["images"][:]
        return f["images"][:]


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


# ──────────────────────────────────────────────
# Step 0: 선행 검증
# ──────────────────────────────────────────────

def run_verification():
    print("\n" + "=" * 60)
    print("Step 0: 선행 검증 — instruction이 visual feature에 영향을 주는가?")
    print("=" * 60)

    bbox_data = json.loads((EXP46_DIR / "bbox_dataset_full.json").read_text())

    # center_left 에피소드에서 첫 프레임 사용
    ep_left = next(e for e in bbox_data if e["path_type"] == "center_left")
    imgs = load_episode_images(ep_left["episode"])
    pil_img = Image.fromarray(imgs[0].astype(np.uint8)).convert("RGB")

    proc, model = load_model()

    instr_left   = INSTRUCTIONS["center_left"]
    instr_right  = INSTRUCTIONS["center_right"]
    instr_para   = PARAPHRASES["center_left"]

    print(f"\n이미지: {Path(ep_left['episode']).name} frame 0")
    print(f"  A: {instr_left}")
    print(f"  B: {instr_right}")
    print(f"  C: {instr_para}  (A paraphrase)")

    feat_a = extract_feat(proc, model, pil_img, instr_left)
    feat_b = extract_feat(proc, model, pil_img, instr_right)
    feat_c = extract_feat(proc, model, pil_img, instr_para)

    sim_ab = cosine_sim(feat_a, feat_b)
    sim_ac = cosine_sim(feat_a, feat_c)

    print(f"\n  cos_sim(A, B) [다른 지시] = {sim_ab:.4f}")
    print(f"  cos_sim(A, C) [같은 의미] = {sim_ac:.4f}")

    if sim_ac > sim_ab:
        diff = sim_ac - sim_ab
        print(f"\n  ✅ 언어가 feature에 차별적으로 영향을 줌 (gap={diff:+.4f})")
        print("  → Exp52 진행 가치 있음")
    else:
        diff = sim_ab - sim_ac
        print(f"\n  ❌ 언어 영향 약함 (gap={diff:+.4f} 반대방향)")
        print("  → feature가 instruction에 무감각할 수 있음. 계속 진행은 가능하나 주의 필요")

    result = {
        "feat_dim": 2048,
        "instr_left":  instr_left,
        "instr_right": instr_right,
        "instr_para":  instr_para,
        "cos_sim_different":  round(sim_ab, 6),
        "cos_sim_paraphrase": round(sim_ac, 6),
        "gap": round(sim_ac - sim_ab, 6),
        "verdict": "language_sensitive" if sim_ac > sim_ab else "language_insensitive",
    }
    (OUT_DIR / "verify_cosine_sim.json").write_text(json.dumps(result, indent=2))
    print(f"\n  결과 저장: {OUT_DIR}/verify_cosine_sim.json")

    del model; gc.collect(); torch.cuda.empty_cache()
    return result


# ──────────────────────────────────────────────
# Step 1: 전체 feature 추출
# ──────────────────────────────────────────────

def run_extraction():
    print("\n" + "=" * 60)
    print("Step 1: lang-vis feature 추출 (150 ep × all frames, ~22분 예상)")
    print("=" * 60)

    bbox_data = json.loads((EXP46_DIR / "bbox_dataset_full.json").read_text())
    proc, model = load_model()

    lang_vis_cache = {}  # ep_path → np.ndarray (N_frames, 2048)
    index = {}           # ep_path → int (for npz key)

    t0 = time.time()
    total_frames = 0
    for i, ep_data in enumerate(bbox_data):
        ep_path  = ep_data["episode"]
        pt       = ep_data["path_type"]
        instr    = INSTRUCTIONS[pt]
        imgs     = load_episode_images(ep_path)

        feats = []
        for fi in range(len(imgs)):
            pil = Image.fromarray(imgs[fi].astype(np.uint8)).convert("RGB")
            feat = extract_feat(proc, model, pil, instr)
            feats.append(feat)

        arr = np.stack(feats)  # (N_frames, 2048)
        lang_vis_cache[ep_path] = arr
        index[ep_path] = i
        total_frames += len(feats)

        elapsed = time.time() - t0
        fps = total_frames / elapsed
        remaining = (sum(len(e["frames"]) for e in bbox_data[i+1:]) / fps) if fps > 0 else 0
        print(
            f"  [{i+1:>3}/150] {Path(ep_path).name}  "
            f"frames={len(feats)}  {fps:.1f}fps  남은시간~{remaining/60:.1f}분",
            flush=True,
        )

    del model; gc.collect(); torch.cuda.empty_cache()

    print(f"\n총 프레임: {total_frames}  소요: {(time.time()-t0)/60:.1f}분")

    np.savez_compressed(
        str(LANG_VIS_CACHE),
        **{f"ep_{i}": lang_vis_cache[ep] for ep, i in index.items()}
    )
    LANG_VIS_INDEX.write_text(json.dumps(index, indent=2))

    print(f"  저장: {LANG_VIS_CACHE}")
    print(f"  저장: {LANG_VIS_INDEX}")
    return lang_vis_cache, index


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_verify",    action="store_true", help="선행 검증 스킵")
    parser.add_argument("--skip_extract",   action="store_true", help="feature 추출 스킵 (이미 있을 때)")
    args = parser.parse_args()

    if not args.skip_verify:
        result = run_verification()
        if result["verdict"] == "language_insensitive":
            ans = input("\n계속 추출 진행하시겠습니까? [y/N]: ").strip().lower()
            if ans != "y":
                print("중단.")
                return

    if not args.skip_extract:
        if LANG_VIS_CACHE.exists() and LANG_VIS_INDEX.exists():
            print(f"\n캐시 이미 존재: {LANG_VIS_CACHE}")
            ans = input("재추출하시겠습니까? [y/N]: ").strip().lower()
            if ans != "y":
                print("추출 스킵.")
                return
        run_extraction()
    else:
        print(f"\n추출 스킵 (--skip_extract). 기존 캐시 사용: {LANG_VIS_CACHE}")

    print("\n완료. 다음 단계: .venv/bin/python3 scripts/train_v5_exp52_true_vla.py")


if __name__ == "__main__":
    main()
