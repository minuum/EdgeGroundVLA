#!/usr/bin/env python3
"""
Exp49 이미지 로버스트니스 테스트

1. Brightness/Contrast 변동  — 조명 변화 시뮬레이션
2. Spatial crop/shift        — 카메라 위치 미세 이동 시뮬레이션
3. Gaussian blur             — 카메라 흔들림/초점 불량
4. Color jitter              — 색조·채도 변화
5. Horizontal flip (★)      — 좌우 대칭 → action도 반전되어야 진짜 이해
   정답: LEFT↔RIGHT, FWD+L↔FWD+R, ROT_L↔ROT_R, FORWARD/STOP 유지

각 augmentation에서:
  - Kosmos-2 vision feature 재추출 (augmented image)
  - grounding cx0 재계산
  - Exp49 MLP action 예측 → 원본 action과 비교
"""
import gc, json, sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_KOSMOS  = ROOT / ".vlms" / "kosmos-2-patch14-224"
EXP46_DIR  = ROOT / "docs" / "v5" / "bbox_nav_exp46"
EXP49_DIR  = ROOT / "docs" / "v5" / "bbox_nav_exp49"
MLP49_DIR  = ROOT / "runs" / "v5_nav" / "mlp" / "exp49"
OUT_PATH   = EXP49_DIR / "image_robustness_results.json"

WINDOW      = 8
VIS_DIM     = 1024
NUM_CLASSES = 8
CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]

FLIP_ACTION_MAP = {
    0: 0,   # STOP → STOP
    1: 1,   # FORWARD → FORWARD
    2: 3,   # LEFT → RIGHT
    3: 2,   # RIGHT → LEFT
    4: 5,   # FWD+L → FWD+R
    5: 4,   # FWD+R → FWD+L
    6: 7,   # ROT_L → ROT_R
    7: 6,   # ROT_R → ROT_L
}

GROUNDING_PROMPT = "<grounding>The gray basket is at"


# ─────────────────────────────────────────────
# Augmentation 정의
# ─────────────────────────────────────────────

AUGMENTATIONS = {
    # (이름, PIL 변환 함수, cx0 변환 함수)
    "original":        (lambda img: img,                        lambda cx: cx),
    "bright+40%":      (lambda img: ImageEnhance.Brightness(img).enhance(1.4), lambda cx: cx),
    "bright-40%":      (lambda img: ImageEnhance.Brightness(img).enhance(0.6), lambda cx: cx),
    "contrast+40%":    (lambda img: ImageEnhance.Contrast(img).enhance(1.4),   lambda cx: cx),
    "contrast-40%":    (lambda img: ImageEnhance.Contrast(img).enhance(0.6),   lambda cx: cx),
    "blur_sigma3":     (lambda img: img.filter(ImageFilter.GaussianBlur(3)),   lambda cx: cx),
    "blur_sigma6":     (lambda img: img.filter(ImageFilter.GaussianBlur(6)),   lambda cx: cx),
    "crop_left10%":    (lambda img: _crop_shift(img, "left",   0.10),          lambda cx: max(0.0, cx - 0.10)),
    "crop_right10%":   (lambda img: _crop_shift(img, "right",  0.10),          lambda cx: min(1.0, cx + 0.10)),
    "crop_center90%":  (lambda img: _crop_center(img, 0.90),                   lambda cx: cx),
    "color_jitter":    (lambda img: _color_jitter(img),                        lambda cx: cx),
    "flip_horizontal": (lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),      lambda cx: 1.0 - cx),
}


def _crop_shift(img, direction, ratio):
    w, h = img.size
    shift = int(w * ratio)
    if direction == "left":
        return img.crop((shift, 0, w, h)).resize((w, h), Image.BILINEAR)
    else:
        return img.crop((0, 0, w - shift, h)).resize((w, h), Image.BILINEAR)


def _crop_center(img, ratio):
    w, h = img.size
    dw, dh = int(w * (1 - ratio) / 2), int(h * (1 - ratio) / 2)
    return img.crop((dw, dh, w - dw, h - dh)).resize((w, h), Image.BILINEAR)


def _color_jitter(img):
    img = ImageEnhance.Color(img).enhance(np.random.uniform(0.6, 1.4))
    img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.8, 1.2))
    return img


# ─────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────

def build_mlp(d_in):
    return nn.Sequential(
        nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64),   nn.ReLU(),
        nn.Linear(64, NUM_CLASSES),
    )


def load_model(ckpt_path=None):
    if ckpt_path is None:
        ckpt_path = MLP49_DIR / "exp49_mlp.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    net  = build_mlp(ckpt["d_in"])
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


def extract_vision_feat(proc, vlm, image, device):
    inputs = proc(text="<grounding>", images=image, return_tensors="pt")
    pv = inputs["pixel_values"].to(device, dtype=torch.float16)
    with torch.no_grad():
        out = vlm.vision_model(pv)
        feat = out.last_hidden_state[0].mean(0).float().cpu().numpy()
    return feat


def run_grounding(proc, vlm, image, device):
    inputs = proc(text=GROUNDING_PROMPT, images=image, return_tensors="pt")
    pv     = inputs["pixel_values"].to(device, dtype=torch.float16)
    iids   = inputs["input_ids"].to(device)
    amask  = inputs["attention_mask"].to(device)
    epm    = inputs.get("image_embeds_position_mask")
    if epm is not None:
        epm = epm.to(device)
    with torch.no_grad():
        gen = vlm.generate(
            pixel_values=pv, input_ids=iids, attention_mask=amask,
            image_embeds=None, image_embeds_position_mask=epm,
            use_cache=True, max_new_tokens=64,
        )
    new_ids  = gen[:, iids.shape[1]:]
    raw_text = proc.batch_decode(new_ids, skip_special_tokens=True)[0]
    _, entities = proc.post_process_generation(raw_text)

    best = None
    for _, _span, boxes in entities:
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2-x1)*(y2-y1)
            if area > 0.85:
                continue
            cx = (x1+x2)/2
            if best is None or area < best["area"]:
                best = {"cx": float(cx), "cy": float((y1+y2)/2), "area": float(area)}
    return best


def predict_action(net, bbox_feat_t0, vis_feat, goal_cx, goal_cy, goal_area, device):
    goal = np.array([goal_cx, goal_cy, goal_area], np.float32)
    feat = np.concatenate([bbox_feat_t0, vis_feat, goal])
    x = torch.tensor([feat], dtype=torch.float32, device=device)
    with torch.no_grad():
        return int(net(x).argmax(1).item())


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="MLP checkpoint path")
    parser.add_argument("--out",  type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    out_path  = Path(args.out)  if args.out  else OUT_PATH

    np.random.seed(42)
    print("=" * 65)
    print("이미지 로버스트니스 테스트")
    print("=" * 65)

    bbox_data = json.loads((EXP46_DIR / "bbox_dataset_full.json").read_text())
    vis_npz   = np.load(str(EXP46_DIR / "vision_features.npz"))
    vis_idx   = json.loads((EXP46_DIR / "vision_features_index.json").read_text())
    vis_cache = {ep: vis_npz[f"ep_{i}"] for ep, i in vis_idx.items()}

    ep_labels = [ep["path_type"] for ep in bbox_data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(sss.split(np.zeros(len(bbox_data)), ep_labels))

    # path_type별 대표 1개씩 (9개)
    seen, test_eps = set(), []
    for i in te_idx:
        ep = bbox_data[i]
        if ep["path_type"] not in seen:
            seen.add(ep["path_type"])
            test_eps.append(ep)
        if len(seen) == 9:
            break
    print(f"테스트 에피소드: {len(test_eps)}개")

    net    = load_model(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = net.to(device)

    print("\n[MODEL] Kosmos-2 로드...")
    proc = AutoProcessor.from_pretrained(str(HF_KOSMOS), trust_remote_code=True)
    vlm  = AutoModelForVision2Seq.from_pretrained(
        str(HF_KOSMOS), trust_remote_code=True, torch_dtype=torch.float16
    ).to(device).eval()
    print("  ✅ 완료")

    all_results = {}
    aug_summary = {aug: {"match": 0, "total": 0} for aug in AUGMENTATIONS}
    flip_correct = 0  # flip 기대 action과 일치한 수

    for ep_data in test_eps:
        pt       = ep_data["path_type"]
        ep_path  = ep_data["episode"]
        frames   = ep_data["frames"]
        vis_feats = vis_cache.get(ep_path)

        with h5py.File(ep_path, "r") as f:
            if "observations" in f and "images" in f["observations"]:
                img_arr = f["observations"]["images"][0]
            else:
                img_arr = f["images"][0]
        orig_img = Image.fromarray(img_arr.astype(np.uint8)).convert("RGB")

        fr0 = frames[0]
        orig_goal_cx   = fr0["cx"]   if fr0["has_bbox"] else 0.5
        orig_goal_cy   = fr0["cy"]   if fr0["has_bbox"] else 0.5
        orig_goal_area = fr0["area"] if fr0["has_bbox"] else 0.0

        # bbox_history @ t=0
        bbox_feat_t0 = []
        for _ in range(WINDOW):
            bbox_feat_t0.extend([fr0["cx"], fr0["cy"], fr0["area"], float(fr0["has_bbox"])])
        bbox_feat_t0 = np.array(bbox_feat_t0, np.float32)

        orig_vis_feat   = vis_feats[0] if vis_feats is not None else np.zeros(VIS_DIM, np.float32)
        orig_action     = predict_action(net, bbox_feat_t0, orig_vis_feat,
                                         orig_goal_cx, orig_goal_cy, orig_goal_area, device)
        ep_results = {"path_type": pt, "orig_action": CLASS_NAMES[orig_action], "augmentations": {}}

        print(f"\n[{pt}]  orig={CLASS_NAMES[orig_action]}  cx0={orig_goal_cx:.3f}")

        for aug_name, (aug_fn, cx_fn) in AUGMENTATIONS.items():
            aug_img = aug_fn(orig_img)

            # vision feature 재추출
            aug_vis_feat = extract_vision_feat(proc, vlm, aug_img, device)

            # grounding 재실행
            grounded = run_grounding(proc, vlm, aug_img, device)
            if grounded:
                aug_cx = grounded["cx"]
                aug_cy = grounded["cy"]
                aug_area = grounded["area"]
            else:
                aug_cx   = cx_fn(orig_goal_cx)   # fallback: 기하학적 예측
                aug_cy   = orig_goal_cy
                aug_area = orig_goal_area

            aug_action = predict_action(net, bbox_feat_t0, aug_vis_feat,
                                        aug_cx, aug_cy, aug_area, device)

            if aug_name == "flip_horizontal":
                expected = FLIP_ACTION_MAP[orig_action]
                is_correct = (aug_action == expected)
                flip_correct += int(is_correct)
                marker = f"{'✅' if is_correct else '❌'} (기대={CLASS_NAMES[expected]})"
            else:
                is_correct = (aug_action == orig_action)
                marker = "✅" if is_correct else "❌"

            aug_summary[aug_name]["total"] += 1
            aug_summary[aug_name]["match"] += int(is_correct if aug_name != "flip_horizontal" else aug_action == FLIP_ACTION_MAP[orig_action])

            cx_delta = abs(aug_cx - orig_goal_cx)
            print(f"  {aug_name:<18}: cx={aug_cx:.3f}(Δ{cx_delta:.3f})  → {CLASS_NAMES[aug_action]:<8} {marker}")

            ep_results["augmentations"][aug_name] = {
                "aug_cx": float(aug_cx), "cx_delta": float(cx_delta),
                "aug_action": CLASS_NAMES[aug_action],
                "matches_orig": bool(aug_action == orig_action),
                "expected_flip": CLASS_NAMES[FLIP_ACTION_MAP[orig_action]] if aug_name == "flip_horizontal" else None,
                "flip_correct": bool(aug_action == FLIP_ACTION_MAP[orig_action]) if aug_name == "flip_horizontal" else None,
            }

        all_results[pt] = ep_results

    del vlm
    torch.cuda.empty_cache()

    # ── 요약 ──
    print("\n" + "=" * 65)
    print("Augmentation별 action 일치율 요약")
    print("=" * 65)
    for aug_name, stat in aug_summary.items():
        n, m = stat["total"], stat["match"]
        if aug_name == "flip_horizontal":
            label = f"flip (기대 반전 일치)"
        else:
            label = "원본과 일치"
        bar = "█" * int(m/n*20) + "░" * (20 - int(m/n*20))
        print(f"  {aug_name:<18}: {m}/{n} = {m/n*100:.0f}%  [{bar}]")

    print(f"\n★ Flip 대칭 검증: {flip_correct}/{len(test_eps)} 에피소드에서 action이 올바르게 반전")
    print(f"  (LEFT↔RIGHT, FWD+L↔FWD+R, ROT_L↔ROT_R 으로 뒤집혀야 함)")

    aug_rows = []
    for aug_name, stat in aug_summary.items():
        n, m = stat["total"], stat["match"]
        aug_rows.append({"aug": aug_name, "match": m, "total": n, "rate": m/n})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "flip_correct": flip_correct,
        "n_episodes": len(test_eps),
        "aug_summary": aug_rows,
        "per_episode": all_results,
    }, indent=2))
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
