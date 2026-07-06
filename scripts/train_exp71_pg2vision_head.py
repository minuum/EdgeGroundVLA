#!/usr/bin/env python3
"""
2026-07-07 사용자 요청: "PG2쪽 액션헤드 방식도 적용해보자, KS2(Kosmos-2)도 했으니까"

지금까지 exp71 계열은 전부 FrozenCLIPV2(실제로는 Kosmos-2 vision encoder, 256d)를
이미지 feature로 썼다. 이 스크립트는 같은 Transformer 액션헤드 구조에 **PG2
(PaliGemma2-448)의 SigLIP vision tower(1152d, L2 정규화)**를 이미지 feature 소스로
붙여서 Kosmos-2 버전과 나란히 비교한다.

- PG2 vision tower는 완전 frozen. Kosmos-2 버전처럼 미리 학습된 256d projection이
  없으므로, 별도 projection 학습(detach 문제 회피) 없이 SigLIP raw feature(1152d)를
  L2 정규화해서 그대로 헤드에 투입 — 헤드의 입력 Linear/LayerNorm이 차원 축소를 맡음
- 나머지(bbox 4dim, window=6, 150ep pg448 데이터, Transformer 구조, has_bbox 등)는
  exp71과 동일. vis_feat 정규화는 2026-07-07 발견한 운영 서버 컨벤션(F.normalize)과 일치
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANN_PATH = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_pg448_cx.json"
PG2_PATH = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-448" \
    / "snapshots/1406c92ec87d32cc6b983239278901b904ba7a51"
CACHE_FILE = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp71_pg2vision_cache.pt"
OUT_FILE = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp71_pg2vision_vs_ks2.json"

WINDOW = 6
NUM_CLASSES = 8
BBOX_DIM = 4
VIS_DIM_PG2 = 1152
FRAME_DIM = BBOX_DIM + VIS_DIM_PG2


class FrozenPG2Vision(nn.Module):
    """PG2(PaliGemma2-448) SigLIP vision tower, 완전 frozen (projection 없음)."""

    def __init__(self, pg2_path, device):
        super().__init__()
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
        self.processor = PaliGemmaProcessor.from_pretrained(str(pg2_path))
        base = PaliGemmaForConditionalGeneration.from_pretrained(
            str(pg2_path), torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.vision_tower = base.vision_tower.to(device).eval()
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        self.device = device
        del base

    @torch.no_grad()
    def encode_batch(self, pil_imgs):
        results = []
        for img in pil_imgs:
            inp = self.processor(images=[img], text="", return_tensors="pt")
            pv = inp["pixel_values"].to(self.device, dtype=torch.float16)
            out = self.vision_tower(pixel_values=pv).last_hidden_state.mean(1).float()
            results.append(out.squeeze(0))
        feat = torch.stack(results)
        return F.normalize(feat, dim=-1)  # (N, 1152)


class TransformerActionHead(nn.Module):
    """bbox(4) + raw PG2 vision(1152) 입력 -> 학습 가능한 vision projection(1152->256)을
    forward 안에서 직접 수행(gradient 정상 흐름) -> Kosmos-2 버전과 동일하게 260d로 축소
    후 Transformer 처리. 1152d를 그대로 태워서 학습이 붕괴했던 1차 시도의 수정판."""

    def __init__(self, bbox_dim=BBOX_DIM, vis_dim_in=VIS_DIM_PG2, proj_dim=256,
                 window=WINDOW, nhead=4, num_layers=2):
        super().__init__()
        frame_dim = bbox_dim + proj_dim
        self.bbox_dim = bbox_dim
        self.vis_proj = nn.Sequential(nn.LayerNorm(vis_dim_in), nn.Linear(vis_dim_in, proj_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, frame_dim))
        self.pos_emb = nn.Embedding(window + 1, frame_dim)
        el = nn.TransformerEncoderLayer(d_model=frame_dim, nhead=nhead, dim_feedforward=512,
                                         dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, 128), nn.ReLU(),
                                   nn.Dropout(0.1), nn.Linear(128, NUM_CLASSES))

    def forward(self, x):
        # x: (B, window, bbox_dim + vis_dim_in)
        bbox = x[..., :self.bbox_dim]
        vis_raw = x[..., self.bbox_dim:]
        vis = self.vis_proj(vis_raw)  # (B, window, 256) -- gradient가 여기로 정상 유입됨
        frame = torch.cat([bbox, vis], dim=-1)
        B = frame.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, frame], dim=1)
        pos = torch.arange(seq.size(1), device=seq.device)
        seq = seq + self.pos_emb(pos)
        return self.head(self.encoder(seq)[:, 0])


def build_episode_cache(enc):
    with open(ANN_PATH) as f:
        ann = json.load(f)
    episodes = []
    for i, ep in enumerate(ann):
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            continue
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        if not frames:
            continue
        with h5py.File(str(h5_path), "r") as f:
            imgs_np = f["observations"]["images"][:]
        pil_imgs = [Image.fromarray(imgs_np[fr["frame_idx"]].astype("uint8")) for fr in frames]
        vis = enc.encode_batch(pil_imgs).cpu()
        bboxes = [(fr.get("cx_det", 0.5), fr.get("cy_det", 0.5),
                   fr.get("area_det", 0.05), float(fr.get("has_bbox", False))) for fr in frames]
        gts = [fr["gt_class"] for fr in frames]
        episodes.append({"stem": h5_path.stem, "bboxes": bboxes, "vis": vis, "gts": gts})
        if (i + 1) % 20 == 0:
            print(f"  encoded {i+1}/{len(ann)}")
    torch.save(episodes, CACHE_FILE)
    return episodes


def build_windows(episodes, window=WINDOW):
    X, y = [], []
    for ep in episodes:
        bboxes, vis, gts = ep["bboxes"], ep["vis"], ep["gts"]
        n = len(gts)
        for t in range(n):
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                seq.append(list(bboxes[idx]) + vis[idx].tolist())
            X.append(seq)
            y.append(gts[t])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def main():
    print(f"device={DEVICE}")
    if CACHE_FILE.exists():
        print(f"vis feature 캐시 로드: {CACHE_FILE}")
        episodes = torch.load(CACHE_FILE, weights_only=False)
    else:
        print("PG2 SigLIP vision tower 로드...")
        enc = FrozenPG2Vision(PG2_PATH, DEVICE)
        print("150ep vision feature 인코딩 중 (1회만)...")
        episodes = build_episode_cache(enc)
        del enc
        torch.cuda.empty_cache()
    print(f"episodes={len(episodes)}")

    rng = np.random.default_rng(42)
    idx = list(range(len(episodes)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * 0.15))
    val_eps = [episodes[i] for i in idx[:n_val]]
    train_eps = [episodes[i] for i in idx[n_val:]]
    print(f"train={len(train_eps)} val={len(val_eps)}")

    X_tr, y_tr = build_windows(train_eps)
    X_va, y_va = build_windows(val_eps)
    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE)
    y_va_t = torch.tensor(y_va, device=DEVICE)

    model = TransformerActionHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    epochs = 300
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            logits = model(X_tr_t[b])
            loss = F.cross_entropy(logits, y_tr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 50 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                acc = (model(X_va_t).argmax(1) == y_va_t).float().mean().item()
            print(f"  epoch {ep:4d}/{epochs}  val_acc={acc:.1%}")
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\n=== PG2 vision(SigLIP 1152d) 액션헤드 결과 ===")
    print(f"val_acc: {best_acc:.1%}  (window={WINDOW}, 150ep, train={len(train_eps)}/val={len(val_eps)})")
    print(f"참고(CH61): Kosmos-2 vision(256d) 동일 window6 baseline val_acc = 97.0~98.4%")

    OUT_FILE.write_text(json.dumps({
        "pg2_vision_val_acc": best_acc,
        "train_eps": len(train_eps), "val_eps": len(val_eps),
        "ks2_vision_val_acc_ref_ch61": {"baseline_w6": 0.970, "window3": 0.948},
        "note": "ks2 참고값은 CH61 truth_mini 격리재학습 split 기준(동일 split 아님, 참고용)",
    }, ensure_ascii=False, indent=2))
    print(f"saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
