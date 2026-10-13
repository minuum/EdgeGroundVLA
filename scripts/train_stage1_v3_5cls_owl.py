#!/usr/bin/env python3
"""
Stage 1 v3: 5-class (강좌/약좌/중앙/약우/강우) + OWL-v2 라벨 + 225ep 재학습

train_exp54_stage1_v2_frame_level.py 복제본. 변경점:
  - 데이터: bbox_dataset_frame_level.json(150ep, HSV 라벨) → bbox_dataset_v6_owl.json
    (225ep, OWL-v2 라벨 — 서빙 검출기와 일치. label 필드는 PaliGemma2 버전과 100% 동일 확인됨)
  - 클래스: 3-class(left/center/right) → 5-class(strong_left/weak_left/center/
    weak_right/strong_right) — 원본 데이터셋이 이 구조로 수집되어 합치지 않고 그대로 사용
  - 나머지(구조/하이퍼파라미터/학습 루프)는 원본과 동일

plan: docs/plans/plan_20260806_stage1_image_proj_5cls_retrain.md

Usage:
  .venv/bin/python3 scripts/train_stage1_v3_5cls_owl.py
"""

import json, sys, time, warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH = ROOT / "docs" / "v5" / "bbox_nav_owl" / "bbox_dataset_v6_owl.json"
OUT_DIR   = ROOT / "runs" / "v5_nav" / "mlp" / "stage1_v3_5cls"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROJ_DIM   = 256
LM_DIM     = 2048
VIS_DIM    = 1024
EPOCHS     = 30
BATCH_SIZE = 16
LR         = 3e-4
TEMPERATURE = 0.07

DIR_IDX = {
    "strong_left": 0, "weak_left": 1, "center": 2,
    "weak_right": 3, "strong_right": 4,
}
IDX_TO_DIR = {v: k for k, v in DIR_IDX.items()}
N_CLASSES = len(DIR_IDX)

ANCHOR_TEXTS = {
    "strong_left":  "The gray basket is strongly on the left side of the image",
    "weak_left":    "The gray basket is slightly on the left side of the image",
    "center":       "The gray basket is in the center of the image",
    "weak_right":   "The gray basket is slightly on the right side of the image",
    "strong_right": "The gray basket is strongly on the right side of the image",
}
ANCHOR_ORDER = ["strong_left", "weak_left", "center", "weak_right", "strong_right"]


# ─────────────────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────────────────

def load_base_model(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    return processor, model


@torch.no_grad()
def compute_text_anchors(model, processor, device):
    """5개 방향 텍스트 앵커 사전 계산 (frozen)."""
    text_model = model.text_model
    anchors = []
    for d in ANCHOR_ORDER:
        text = ANCHOR_TEXTS[d]
        inp = processor.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = text_model(
            input_ids=inp.input_ids,
            attention_mask=inp.attention_mask,
            output_hidden_states=True,
        )
        feat = out.hidden_states[-1][:, -1, :].float()  # (1, 2048)
        anchors.append(feat)
    return torch.cat(anchors, dim=0)  # (5, 2048)


# ─────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────

def load_frame_level_data():
    """consistent=True 프레임만 추출. 에피소드 단위 split을 위해 구조 유지."""
    raw = json.loads(DATA_PATH.read_text())
    episodes = []
    for ep in raw:
        frames = [f for f in ep["frames"] if f["consistent"] and f["label"] is not None]
        if frames:
            episodes.append({
                "episode":   ep["episode"],
                "direction": ep["direction"],   # 에피소드 방향 (split 기준)
                "frames":    frames,
            })
    return episodes


_IMG_CACHE = {}  # (h5_path, frame_idx) -> RGB uint8 np.ndarray (BGR 반전 이미 적용됨)


def load_image(h5_path, frame_idx):
    arr = _IMG_CACHE[(h5_path, frame_idx)]
    return Image.fromarray(arr)


def preload_all_episodes(episodes):
    """실제로 쓰이는 프레임만 미리 읽어서 BGR→RGB 반전까지 끝낸 뒤 RAM에 캐싱.
    (30에폭 동안 매번 h5 read + 반전을 반복하면 그게 GPU 연산보다 더 큰 병목이 됨 —
    한 번만 디스크 I/O·반전하고 이후 에폭은 캐시에서 바로 읽는다. 96GB RAM 여유로 가능,
    16,599프레임 × 720×1280×3 uint8 ≈ 46GB 추정)"""
    total = sum(len(ep["frames"]) for ep in episodes)
    print(f"[PRELOAD] {total}개 프레임 미리 읽고 BGR→RGB 반전해서 RAM에 캐싱 중...")
    t0 = time.time()
    n = 0
    for ep in episodes:
        path = ep["episode"]
        with h5py.File(path, "r") as f:
            images = f["images"]
            for fr in ep["frames"]:
                idx = fr["frame_idx"]
                arr = images[idx][:, :, ::-1].astype("uint8").copy()
                _IMG_CACHE[(path, idx)] = arr
                n += 1
        if n % 3000 == 0:
            print(f"  ... {n}/{total} ({time.time()-t0:.0f}s)")
    print(f"[PRELOAD] 완료 — {n}개 프레임 캐싱됨 ({time.time()-t0:.0f}초 소요)")


# ─────────────────────────────────────────────────────────
# 학습
# ─────────────────────────────────────────────────────────

def encode_images(vision_model, image_proj, processor, images, device):
    inputs = processor(images=images, return_tensors="pt")
    pv = inputs["pixel_values"].to(
        device, dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    with torch.no_grad():
        out = vision_model(pixel_values=pv)
    feat = out.last_hidden_state.mean(dim=1).float()
    return F.normalize(image_proj(feat), dim=-1)  # (N, 256)


def evaluate(vision_model, image_proj, processor, anchor_proj, val_eps, device):
    image_proj.eval()
    correct = total = 0
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for ep in val_eps:
        for fr in ep["frames"]:
            try:
                img = load_image(ep["episode"], fr["frame_idx"])
            except Exception:
                continue
            feat = encode_images(vision_model, image_proj, processor, [img], device)
            pred = (feat @ anchor_proj.T).argmax(dim=1).item()
            gt   = DIR_IDX[fr["label"]]
            confusion[gt][pred] += 1
            correct += int(pred == gt)
            total   += 1
    acc = correct / total if total > 0 else 0.0
    return acc, confusion


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # ── 데이터 로드
    all_eps = load_frame_level_data()
    preload_all_episodes(all_eps)
    ep_dirs = [ep["direction"] for ep in all_eps]

    # 방향별 프레임 통계
    from collections import Counter
    frame_label_counts = Counter(
        fr["label"] for ep in all_eps for fr in ep["frames"]
    )
    print(f"[DATA] 에피소드: {len(all_eps)}")
    print(f"       프레임: " + " ".join(
        f"{d}={frame_label_counts[d]}" for d in ANCHOR_ORDER
    ))

    # 에피소드 단위 train/val split (80/20, stratify by direction)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(np.zeros(len(all_eps)), ep_dirs))
    tr_eps = [all_eps[i] for i in tr_idx]
    val_eps = [all_eps[i] for i in te_idx]
    print(f"       train={len(tr_eps)} ep / val={len(val_eps)} ep")

    # ── 모델
    print("[MODEL] 로드 중...")
    processor, base_model = load_base_model(device)

    vision_model = base_model.vision_model.to(device)
    for p in vision_model.parameters():
        p.requires_grad = False  # frozen — proj만 학습

    # text anchor 사전 계산
    anchor_raw = compute_text_anchors(base_model, processor, device)  # (5, 2048)
    print(f"[MODEL] text anchor 계산 완료 ({N_CLASSES}-class)")

    # proj layers
    image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
    text_proj  = nn.Linear(LM_DIM,  PROJ_DIM).to(device)

    # anchor 투영 (고정)
    with torch.no_grad():
        anchor_proj = F.normalize(text_proj(anchor_raw), dim=-1)  # (5, 256)

    # ── class weight (빈도 역수 정규화, N_CLASSES로 일반화)
    counts = np.array([frame_label_counts[d] for d in ANCHOR_ORDER], dtype=float)
    weights = (counts.sum() / (N_CLASSES * counts))
    weights = weights / weights.sum() * N_CLASSES
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    print(f"[LOSS] class weight: " + " ".join(
        f"{d}={w:.2f}" for d, w in zip(ANCHOR_ORDER, weights)
    ))

    optimizer = torch.optim.AdamW(
        list(image_proj.parameters()) + list(text_proj.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_state = None

    print(f"\n{'epoch':>6} {'val_acc':>9} {'best':>9}")
    print("-" * 30)

    for epoch in range(1, EPOCHS + 1):
        image_proj.train()
        text_proj.train()
        np.random.shuffle(tr_eps)

        batch_feats, batch_labels = [], []

        for ep in tr_eps:
            images = []
            labels = []
            for fr in ep["frames"]:
                try:
                    images.append(load_image(ep["episode"], fr["frame_idx"]))
                    labels.append(DIR_IDX[fr["label"]])
                except Exception:
                    pass

            if not images:
                continue

            feats = encode_images(vision_model, image_proj, processor, images, device)
            batch_feats.append(feats)
            batch_labels.extend(labels)

            if len(batch_labels) >= BATCH_SIZE:
                x = torch.cat(batch_feats, dim=0)
                y = torch.tensor(batch_labels, dtype=torch.long, device=device)
                ap = F.normalize(text_proj(anchor_raw), dim=-1)
                logits = (x @ ap.T) / TEMPERATURE
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_feats, batch_labels = [], []

        if batch_feats:
            x = torch.cat(batch_feats, dim=0)
            y = torch.tensor(batch_labels, dtype=torch.long, device=device)
            ap = F.normalize(text_proj(anchor_raw), dim=-1)
            logits = (x @ ap.T) / TEMPERATURE
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        with torch.no_grad():
            anchor_proj = F.normalize(text_proj(anchor_raw), dim=-1)

        acc, confusion = evaluate(vision_model, image_proj, processor, anchor_proj, val_eps, device)

        if acc > best_acc:
            best_acc = acc
            best_state = {
                "image_proj": {k: v.cpu().clone() for k, v in image_proj.state_dict().items()},
                "text_proj":  {k: v.cpu().clone() for k, v in text_proj.state_dict().items()},
                "anchor_raw": anchor_raw.cpu().clone(),
                "val_acc":    best_acc,
                "dir_idx":    DIR_IDX,
            }

        print(f"{epoch:>6}  {acc:>8.4f}  {best_acc:>8.4f}")

    # ── 결과 출력
    print(f"\n{'='*50}")
    print(f"  Stage 1 v3 (5-class, OWL-v2, 225ep) 완료")
    print(f"  val_acc: {best_acc:.4f}")
    print(f"  v2(3-class, 150ep, HSV): 0.9811  →  v3(5-class, 225ep, OWL): {best_acc:.4f}")
    print(f"{'='*50}")

    if best_state is None:
        print("[ERROR] val_acc가 한 번도 개선되지 않음 — 저장 중단")
        return

    # 혼동 행렬 (best epoch 기준)
    image_proj.load_state_dict(best_state["image_proj"])
    text_proj.load_state_dict(best_state["text_proj"])
    with torch.no_grad():
        anchor_proj = F.normalize(text_proj(anchor_raw), dim=-1)
    _, confusion = evaluate(vision_model, image_proj, processor, anchor_proj, val_eps, device)

    print(f"\n혼동 행렬 (행=실제, 열=예측):")
    header = "".join(f"{d[:8]:>10}" for d in ANCHOR_ORDER)
    print(f"{'':>12}{header}  {'정확도':>8}")
    for i, d in enumerate(ANCHOR_ORDER):
        row = confusion[i]
        acc_d = row[i] / row.sum() * 100 if row.sum() > 0 else 0
        row_str = "".join(f"{v:>10}" for v in row)
        print(f"  {d[:10]:>10}{row_str}  {acc_d:>7.1f}%")

    # strong_left / weak_left 간 혼동 별도 강조 (실기 좌측 재검증 대상)
    sl, wl = DIR_IDX["strong_left"], DIR_IDX["weak_left"]
    print(f"\n[좌측 혼동] strong_left→weak_left: {confusion[sl][wl]} / "
          f"weak_left→strong_left: {confusion[wl][sl]}")

    # 저장
    ckpt_path = OUT_DIR / "stage1_v3_5cls_owl_projs.pt"
    torch.save(best_state, str(ckpt_path))
    print(f"\n[SAVE] {ckpt_path}")
    print(f"소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
