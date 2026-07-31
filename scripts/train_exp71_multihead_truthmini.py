#!/usr/bin/env python3
"""
1) 실제 exp71 레시피(FrozenCLIPV2 + 150ep pg448)로 헤드 구조(MLP/LSTM/Transformer) 비교,
   window=3(지난 ablation 1등) 고정.
2) 사람이 직접 라벨링한 bbox_truth_mini.json(72프레임, 18에피소드, 실제 bbox_xyxy_norm +
   gt_action_class)을 "완전히 깨끗한 bbox를 줬을 때도 헤드가 맞게 예측하는가" 검증용
   held-out set으로 사용 — 모델 검출 노이즈를 배제한 상한선 확인.
"""
import importlib.util
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

spec = importlib.util.spec_from_file_location("exp71", ROOT / "scripts" / "train_exp71_stage2_transformer.py")
exp71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp71)

flicker_spec = importlib.util.spec_from_file_location("flickertrain", ROOT / "scripts" / "train_exp71_flicker_robustness.py")
flickertrain = importlib.util.module_from_spec(flicker_spec)
flicker_spec.loader.exec_module(flickertrain)

CACHE_FILE = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp71_vis_cache.pt"
TRUTHMINI = ROOT / "docs" / "v5" / "bbox_truth_mini.json"
DATA_DIR = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OUT_FILE = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp71_multihead_truthmini.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 3  # 지난 ablation 1등 설정
NUM_CLASSES = 8
BBOX_DIM = 4
VIS_DIM = 256
FRAME_DIM = BBOX_DIM + VIS_DIM


class MLPHead(nn.Module):
    def __init__(self, window=WINDOW):
        super().__init__()
        d_in = FRAME_DIM * window
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):  # (B, window, FRAME_DIM)
        return self.net(x.reshape(x.size(0), -1))


class LSTMHead(nn.Module):
    def __init__(self, window=WINDOW):
        super().__init__()
        self.lstm = nn.LSTM(FRAME_DIM, 256, 2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def make_head(name, window=WINDOW):
    if name == "mlp":
        return MLPHead(window).to(DEVICE)
    if name == "lstm":
        return LSTMHead(window).to(DEVICE)
    return exp71.TransformerActionHead(window=window).to(DEVICE)


def build_windows(episodes, window):
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


def train_head(name, train_eps, val_eps, window=WINDOW, epochs=300, lr=5e-4, seed=42):
    torch.manual_seed(seed)
    X_tr, y_tr = build_windows(train_eps, window)
    X_va, y_va = build_windows(val_eps, window)
    X_tr_t = torch.from_numpy(X_tr).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).to(DEVICE)
    X_va_t = torch.from_numpy(X_va).to(DEVICE)
    y_va_t = torch.from_numpy(y_va).to(DEVICE)

    model = make_head(name, window)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            logits = model(X_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 50 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                acc = (model(X_va_t).argmax(1) == y_va_t).float().mean().item()
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def xyxy_to_cx_cy_area(box):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return cx, cy, area


def build_truthmini_windows(enc, pg_stems, window=WINDOW):
    """사람이 검증한 clean bbox로 현재 프레임을 덮어쓰고, 나머지 window는 원본 pg448
    검출 이력을 그대로 사용 (완전히 새 시퀀스를 만들지 않고 '현재 프레임만 깨끗하게
    교체했을 때' 예측이 맞는지 보는 현실적인 세팅)."""
    tm = json.loads(TRUTHMINI.read_text())
    anns = tm["annotations"]
    by_ep = {}
    for a in anns:
        by_ep.setdefault(a["episode"], []).append(a)

    X, y, meta = [], [], []
    for ep_stem, items in by_ep.items():
        pg_ep = pg_stems.get(ep_stem)
        if pg_ep is None:
            continue
        frames = pg_ep["frames"]
        h5_path = DATA_DIR / f"{ep_stem}.h5"
        with h5py.File(h5_path, "r") as f:
            imgs_np = f["observations"]["images"][:]

        # 프레임별 원본 검출 bbox (window 이력용)
        orig_bbox = {fr["frame_idx"]: (fr.get("cx_det", 0.5), fr.get("cy_det", 0.5),
                                        fr.get("area_det", 0.05), float(fr.get("has_bbox", False)))
                     for fr in frames}
        for item in items:
            t = item["frame_idx"]
            cx, cy, area = xyxy_to_cx_cy_area(item["bbox_xyxy_norm"])
            clean_bbox = (cx, cy, area, 1.0)

            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                if idx == t:
                    bbox = clean_bbox
                else:
                    bbox = orig_bbox.get(idx, (0.5, 0.5, 0.05, 0.0))
                pil = Image.fromarray(imgs_np[idx].astype("uint8"))
                vis = enc.encode_batch([pil], DEVICE).cpu().squeeze(0).tolist()
                seq.append(list(bbox) + vis)
            X.append(seq)
            y.append(item["gt_action_class"])
            meta.append({"episode": ep_stem, "frame_idx": t, "path_type": item["path_type"]})
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), meta


def main():
    print(f"device={DEVICE}")
    cache = torch.load(CACHE_FILE, weights_only=False)
    train_eps, val_eps = cache["train"], cache["val"]
    print(f"train episodes={len(train_eps)}  val episodes={len(val_eps)}")

    results = {"window": WINDOW}
    models = {}
    for name in ["mlp", "lstm", "transformer"]:
        model, acc = train_head(name, train_eps, val_eps, window=WINDOW)
        models[name] = model
        results[f"{name}_val_acc"] = float(acc)
        print(f"[{name:12s}] val_acc={acc:.1%}")

    print("\nFrozenCLIPV2 재로드 (truth_mini 인코딩용)...")
    enc = exp71.FrozenCLIPV2(exp71.VLM_PATH, exp71.STAGE1_PT, DEVICE).eval()

    with open(exp71.ANN_PATH) as f:
        pg = json.load(f)
    pg_stems = {Path(ep["episode"]).stem: ep for ep in pg}

    print("truth_mini(사람 검증 clean bbox) 윈도우 구성 중...")
    X_tm, y_tm, meta_tm = build_truthmini_windows(enc, pg_stems, window=WINDOW)
    print(f"truth_mini samples: {len(X_tm)}")
    X_tm_t = torch.from_numpy(X_tm).to(DEVICE)
    y_tm_t = torch.from_numpy(y_tm).to(DEVICE)

    for name, model in models.items():
        with torch.no_grad():
            preds = model(X_tm_t).argmax(dim=-1)
            acc = (preds == y_tm_t).float().mean().item()
        results[f"{name}_truthmini_acc"] = float(acc)
        print(f"[{name:12s}] truth_mini(clean bbox) acc={acc:.1%}  (n={len(y_tm)})")

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
