#!/usr/bin/env python3
"""
Exp53: CLIP LoRA Visual Grounding

교수님 지시 (5/15 미팅):
  - CLIP 비전 인코더 16~23번 레이어에만 LoRA (r=16, alpha=32, q_proj+v_proj)
  - LM(언어) 쪽 완전 frozen
  - 박스 시각 인식 vs 텍스트 패턴 암기 검증

아키텍처:
  이미지 → Kosmos-2 CLIP[0-15 frozen | 16-23 LoRA] → feat(1024)
  feat(1024) + bbox_history(32) + goal(3) → MLP → 8-class action

Exp49 대비 변경:
  - 사전 추출 vis_cache 사용 안 함 → 이미지 on-the-fly forward
  - CLIP LoRA 파라미터 공동 학습 (524K trainable)

Usage:
  python3 scripts/train_clip_lora_exp53.py
  python3 scripts/train_clip_lora_exp53.py --data docs/v5/bbox_nav_exp53/bbox_dataset.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
VLM_PATH    = ROOT / ".vlms" / "kosmos-2-patch14-224"
EXP46_DIR   = ROOT / "docs" / "v5" / "bbox_nav_exp46"
OUT_DIR     = ROOT / "docs" / "v5" / "bbox_nav_exp53"
MLP_DIR     = ROOT / "runs" / "v5_nav" / "mlp" / "exp53"
CKPT_PATH   = MLP_DIR / "exp53_clip_lora.pt"
LORA_DIR    = MLP_DIR / "clip_lora_adapter"

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
GOAL_DIM    = 3      # (cx0, cy0, area0)
D_IN        = WINDOW * 4 + VIS_DIM + GOAL_DIM  # 1059

LORA_R      = 16
LORA_ALPHA  = 32
LORA_LAYERS = list(range(16, 24))  # 16~23 (0-indexed)
LORA_TARGET = ["q_proj", "v_proj"]

BATCH_SIZE  = 32
EPOCHS      = 300
LR_LORA     = 2e-4  # LoRA는 낮게
LR_MLP      = 1e-3  # MLP는 Exp49와 동일


# ──────────────────────────────────────────────
# MLP 헤드 (Exp49와 동일 구조)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
# Vision 백본 (CLIP LoRA)
# ──────────────────────────────────────────────
class CLIPLoRABackbone(nn.Module):
    """Kosmos-2 비전 인코더 + LoRA (layers 16-23만 학습)."""

    def __init__(self, model_path: Path, device: torch.device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from peft import LoraConfig, get_peft_model

        self.processor = AutoProcessor.from_pretrained(str(model_path))
        base = AutoModelForVision2Seq.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )

        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET,
            layers_to_transform=LORA_LAYERS,
            layers_pattern="layers",
            lora_dropout=0.05,
            bias="none",
        )
        self.vision_model = get_peft_model(base.vision_model, lora_cfg)
        # 나머지 전체(LM 등)는 완전 frozen — LoRA 파라미터만 학습
        self.vision_model.print_trainable_parameters()

    def encode(self, pil_images: list[Image.Image], device: torch.device) -> torch.Tensor:
        """PIL 이미지 리스트 → (N, 1024) float32 feature."""
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt",
        )
        pv = inputs["pixel_values"].to(
            device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        out = self.vision_model(pixel_values=pv)
        # last_hidden_state: (N, seq_len, 1024) → mean pool → (N, 1024)
        feat = out.last_hidden_state.mean(dim=1).float()
        return feat


# ──────────────────────────────────────────────
# 이미지 로드 유틸
# ──────────────────────────────────────────────
def load_images_from_h5(h5_path: str, frame_indices: list[int]) -> list[Image.Image]:
    with h5py.File(h5_path, "r") as f:
        imgs = []
        for idx in frame_indices:
            arr = f["observations"]["images"][idx]  # (H, W, 3) uint8
            imgs.append(Image.fromarray(arr))
    return imgs


# ──────────────────────────────────────────────
# 데이터셋 빌드 (이미지 경로 + bbox/gt 정보)
# ──────────────────────────────────────────────
def build_episode_records(bbox_data: list[dict]) -> list[dict]:
    """에피소드별 레코드: H5 경로, 프레임 목록, goal, gt_classes."""
    records = []
    for ep in bbox_data:
        frames = ep["frames"]
        fr0 = frames[0]
        goal = np.array(
            [fr0["cx"], fr0["cy"], fr0["area"]] if fr0["has_bbox"] else [0.5, 0.5, 0.0],
            dtype=np.float32,
        )
        records.append({
            "h5_path": ep["episode"],
            "path_type": ep["path_type"],
            "frames": frames,
            "goal": goal,
        })
    return records


def build_bbox_feats(frames: list[dict], t: int) -> np.ndarray:
    bbox = []
    for k in range(WINDOW):
        idx = max(0, t - (WINDOW - 1 - k))
        fr = frames[idx]
        bbox.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])
    return np.array(bbox, dtype=np.float32)


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}")

    # ── 데이터 로드 ──
    data_path = Path(args.data) if args.data else EXP46_DIR / "bbox_dataset_full.json"
    print(f"[DATA] {data_path}")
    bbox_data = json.loads(data_path.read_text())
    print(f"에피소드: {len(bbox_data)}")

    records = build_episode_records(bbox_data)

    # ── Train / Val 분할 (에피소드 단위) ──
    ep_labels = [r["path_type"] for r in records]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(np.zeros(len(records)), ep_labels))
    tr_records = [records[i] for i in tr_idx]
    te_records  = [records[i] for i in te_idx]
    print(f"Train: {len(tr_records)} ep  Val: {len(te_records)} ep")

    # ── 모델 초기화 ──
    print("\n[MODEL] CLIP LoRA 초기화...")
    clip_backbone = CLIPLoRABackbone(VLM_PATH, device).to(device)
    mlp = GoalNavMLP(D_IN).to(device)

    # 옵티마이저: LoRA와 MLP를 다른 lr로
    lora_params = [p for p in clip_backbone.parameters() if p.requires_grad]
    mlp_params  = list(mlp.parameters())
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA},
        {"params": mlp_params,  "lr": LR_MLP},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 클래스 가중치 (train 전체 레이블로 계산)
    all_tr_labels = [fr["gt_class"] for r in tr_records for fr in r["frames"]]
    counts = np.bincount(all_tr_labels, minlength=NUM_CLASSES).astype(float)
    weights = np.where(counts > 0, 1.0 / (counts + 1e-6), 0.0)
    weights /= weights.sum() / NUM_CLASSES
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device)
    )

    # ── 학습 루프 ──
    best_acc, best_state_lora, best_state_mlp = 0.0, None, None

    for epoch in range(1, EPOCHS + 1):
        clip_backbone.train()
        mlp.train()

        np.random.shuffle(tr_records)

        # 에피소드 단위 미니배치 처리
        batch_feats, batch_labels = [], []

        for rec in tr_records:
            try:
                images = load_images_from_h5(
                    rec["h5_path"],
                    list(range(len(rec["frames"]))),
                )
            except Exception as e:
                print(f"  [SKIP] {rec['h5_path']}: {e}")
                continue

            with torch.set_grad_enabled(True):
                vis_feats = clip_backbone.encode(images, device)  # (T, 1024)

            goal = torch.tensor(rec["goal"], dtype=torch.float32, device=device)

            for t, frame in enumerate(rec["frames"]):
                bbox = torch.tensor(build_bbox_feats(rec["frames"], t), device=device)
                feat = torch.cat([bbox, vis_feats[t], goal])
                batch_feats.append(feat)
                batch_labels.append(frame["gt_class"])

            # 배치 크기에 도달하면 업데이트
            if len(batch_feats) >= BATCH_SIZE:
                _update_batch(batch_feats, batch_labels, mlp, optimizer, criterion, device)
                batch_feats, batch_labels = [], []

        # 남은 배치
        if batch_feats:
            _update_batch(batch_feats, batch_labels, mlp, optimizer, criterion, device)

        scheduler.step()

        # ── 검증 ──
        if epoch % 30 == 0 or epoch == EPOCHS:
            acc = evaluate_epoch(clip_backbone, mlp, te_records, device)
            if acc > best_acc:
                best_acc = acc
                best_state_lora = {k: v.cpu().clone() for k, v in clip_backbone.state_dict().items()}
                best_state_mlp  = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            print(f"  epoch {epoch:>3}  val_acc={acc:.4f}  best={best_acc:.4f}")

    # ── 저장 ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MLP_DIR.mkdir(parents=True, exist_ok=True)

    mlp.load_state_dict(best_state_mlp)
    clip_backbone.load_state_dict(best_state_lora)

    torch.save({"mlp": best_state_mlp, "val_acc": best_acc}, str(CKPT_PATH))
    print(f"\n[SAVE] MLP 체크포인트: {CKPT_PATH}")

    # LoRA 어댑터 별도 저장 (peft 형식)
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    clip_backbone.vision_model.save_pretrained(str(LORA_DIR))
    print(f"[SAVE] LoRA 어댑터: {LORA_DIR}")

    print(f"\n최종 val_acc: {best_acc:.4f}")
    return best_acc


def _update_batch(feats, labels, mlp, optimizer, criterion, device):
    x = torch.stack(feats)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    optimizer.zero_grad()
    loss = criterion(mlp(x), y)
    loss.backward()
    optimizer.step()


@torch.no_grad()
def evaluate_epoch(clip_backbone, mlp, records, device) -> float:
    clip_backbone.eval()
    mlp.eval()
    correct, total = 0, 0
    for rec in records:
        try:
            images = load_images_from_h5(
                rec["h5_path"],
                list(range(len(rec["frames"]))),
            )
        except Exception:
            continue
        vis_feats = clip_backbone.encode(images, device)
        goal = torch.tensor(rec["goal"], dtype=torch.float32, device=device)
        for t, frame in enumerate(rec["frames"]):
            bbox = torch.tensor(build_bbox_feats(rec["frames"], t), device=device)
            feat = torch.cat([bbox, vis_feats[t], goal]).unsqueeze(0)
            pred = mlp(feat).argmax(1).item()
            correct += int(pred == frame["gt_class"])
            total += 1
    return correct / total if total > 0 else 0.0


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=None,
        help="bbox_dataset JSON 경로. 미지정 시 Exp46 기존 150ep 사용.",
    )
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Exp53: CLIP LoRA Visual Grounding")
    print(f"D_IN={D_IN}  LoRA layers={LORA_LAYERS[0]}~{LORA_LAYERS[-1]}")
    print("=" * 60)

    val_acc = train(args)

    elapsed = time.time() - t0
    print(f"\n완료: {elapsed/60:.1f}분")
    print(f"결과: val_acc = {val_acc:.4f}")
    print(f"참고: Exp49 baseline = 0.9640")


if __name__ == "__main__":
    main()
