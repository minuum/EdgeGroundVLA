#!/usr/bin/env python3
"""
2026-07-07: 배포용 최종 체크포인트 학습 — window=6, bbox_scale=3.0 (오늘 확인된
최선 설정: val_acc 72.3%->84.6%, 표준편차 27%p->4%p로 안정화, ablate_diagweight_
bboxscale_multiseed.json 참고).

전체 150ep(pg448) 사용, 5-seed 학습 후 최고 val_acc 시드의 state_dict 저장.
체크포인트 포맷은 운영 규격과 동일 + bbox_scale 메타데이터 추가
(stage2_v2_inference_server.py가 2026-07-07 패치로 이 필드를 읽어 자동 적용).

주의: 이 체크포인트는 반드시 서버가 bbox_scale 지원 패치(오늘 monavla-driving에
푸시한 커밋) 이후 버전에서만 사용할 것 — 구버전 서버는 bbox_scale을 무시하고
학습/추론 불일치가 재발한다.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = Path("docs/v5/closed_loop_eval/exp71_vis_cache_normfixed.pt")
OUT_DIR = Path("runs/v5_nav/mlp/exp71_window6_bboxscale3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CKPT = OUT_DIR / "action_transformer.pt"
OUT_LOG = Path("docs/v5/closed_loop_eval/exp71_window6_bboxscale3_final.json")

WINDOW = 6
BBOX_SCALE = 3.0
DIAG_MULT = 1.0
DIAG_CLASSES = [4, 5]
FRAME_DIM = 4 + 256
NUM_CLASSES = 8
SEEDS = [0, 1, 2, 3, 4]
VAL_RATIO = 0.15


class TransformerActionHead(nn.Module):
    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW, nhead=4, num_layers=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, frame_dim))
        self.pos_emb = nn.Embedding(window + 1, frame_dim)
        el = nn.TransformerEncoderLayer(d_model=frame_dim, nhead=nhead, dim_feedforward=512,
                                         dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, 128), nn.ReLU(),
                                   nn.Dropout(0.1), nn.Linear(128, NUM_CLASSES))

    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.size(1), device=x.device)
        return self.head(self.encoder(x + self.pos_emb(pos))[:, 0])


def build_windows(eps, window=WINDOW, bbox_scale=BBOX_SCALE):
    X, y = [], []
    for ep in eps:
        bboxes, vis, gts = ep["bboxes"], ep["vis"], ep["gts"]
        n = len(gts)
        for t in range(n):
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                scaled_bbox = [v * bbox_scale for v in bboxes[idx]]
                seq.append(scaled_bbox + vis[idx].tolist())
            X.append(seq)
            y.append(gts[t])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def train_one(X_tr, y_tr, X_va, y_va, seed, epochs=300, lr=5e-4):
    torch.manual_seed(seed)
    cls_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = 1.0 / cls_counts
    for c in DIAG_CLASSES:
        weights[c] *= DIAG_MULT
    weights = weights / weights.sum() * NUM_CLASSES
    weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE)
    y_va_t = torch.tensor(y_va, device=DEVICE)

    model = TransformerActionHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            logits = model(X_tr_t[b])
            loss = F.cross_entropy(logits, y_tr_t[b], weight=weights_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 50 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(X_va_t).argmax(1) == y_va_t).float().mean().item()
            if acc >= best_acc:
                best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    return best_acc, best_state


def main():
    episodes = torch.load(CACHE, weights_only=False)
    print(f"episodes={len(episodes)} (전체 150ep 중 로컬 h5 있는 것)")

    rng = np.random.default_rng(42)
    idx = list(range(len(episodes)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO))
    val_eps = [episodes[i] for i in idx[:n_val]]
    train_eps = [episodes[i] for i in idx[n_val:]]
    print(f"train={len(train_eps)} val={len(val_eps)}  window={WINDOW}  bbox_scale={BBOX_SCALE}")

    X_tr, y_tr = build_windows(train_eps)
    X_va, y_va = build_windows(val_eps)

    results = []
    best_overall_acc, best_overall_state, best_seed = 0.0, None, None
    for seed in SEEDS:
        acc, state = train_one(X_tr, y_tr, X_va, y_va, seed=seed)
        results.append({"seed": seed, "val_acc": acc})
        print(f"  seed={seed} val_acc={acc:.1%}")
        if acc > best_overall_acc:
            best_overall_acc, best_overall_state, best_seed = acc, state, seed

    print(f"\n최고 시드: seed={best_seed}  val_acc={best_overall_acc:.1%}")
    mean_acc = float(np.mean([r["val_acc"] for r in results]))
    std_acc = float(np.std([r["val_acc"] for r in results]))
    print(f"5-seed 평균: {mean_acc:.1%} ± {std_acc:.1%}")

    torch.save({
        "model": best_overall_state,
        "val_acc": best_overall_acc,
        "source": "pg448",
        "exp": "exp71_window6_bboxscale3",
        "head": "transformer",
        "window": WINDOW,
        "bbox_scale": BBOX_SCALE,
        "diag_mult": DIAG_MULT,
        "seed": best_seed,
        "seed_results": results,
        "seed_mean_acc": mean_acc,
        "seed_std_acc": std_acc,
        "note": "2026-07-07 CH61 후속 — bbox_scale=3.0 서버지원 패치 필요(구버전 서버 사용 금지)",
    }, str(OUT_CKPT))
    print(f"체크포인트 저장 -> {OUT_CKPT}")

    OUT_LOG.write_text(json.dumps({
        "window": WINDOW, "bbox_scale": BBOX_SCALE, "diag_mult": DIAG_MULT,
        "best_seed": best_seed, "best_val_acc": best_overall_acc,
        "seed_mean_acc": mean_acc, "seed_std_acc": std_acc,
        "all_seeds": results, "checkpoint_path": str(OUT_CKPT),
    }, ensure_ascii=False, indent=2))
    print(f"로그 저장 -> {OUT_LOG}")


if __name__ == "__main__":
    main()
