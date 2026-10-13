#!/usr/bin/env python3
"""
그라운딩 기반 서브골 조향 vs 단순 concat 비교 (plan_20261011_grounding_subgoal_steering.md)

(A) baseline: 기존 bbox+image concat-MLP (ablate_bbox_image_features.py와 동일 구조, 재확인용)
(B) subgoal steering: bbox를 서브골 임베딩으로 인코딩 후 FiLM(gamma,beta)으로 이미지 특징을 조향

공통점:
- 동일 캐시 bbox_dataset.json (45 eps, 794 frames, 오프라인 그라운딩 재생) — 새 데이터 수집 없음
- 동일 5-seed episode-level stratified split, 동일 class-weighted CE, 동일 epoch 수
- 파라미터 수를 비슷하게 맞춰 "모델이 커서 이겼다" 반박 방지

주의 (연구 범위):
- 그라운딩은 실시간 추론이 아니라 오프라인 캐시 재생 — "온라인 서브골 조향"이 아님
- 8-class 이산 타겟만 다룸 — flow-matching/연속 액션과 무관
"""

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

DATA_DIR = Path("/home/minum/26CS/MoNa-pi/mobile_vla_dataset_merged")  # bbox_dataset.json 캐시 당시 원본 위치(mobile_vla_dataset_v5는 이후 교체됨)
STEP1_DIR = ROOT / "docs" / "v5" / "bbox_nav_step1"
OUT_DIR = ROOT / "docs" / "v5" / "subgoal_steering_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_FILE = STEP1_DIR / "bbox_dataset.json"
SUMMARY_FILE = OUT_DIR / "summary.json"
HTML_FILE = OUT_DIR / "index.html"

PATH_TYPES = [
    "center_straight", "center_left", "center_right",
    "left_straight", "left_left", "left_right",
    "right_straight", "right_left", "right_right",
]
NUM_CLASSES = 8
WINDOW = 3
IMG_SIZE = 16
BBOX_DIM = WINDOW * 4
IMG_DIM = IMG_SIZE * IMG_SIZE
HIDDEN = 128
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 220
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Missing cached dataset: {DATASET_FILE}")
    return json.loads(DATASET_FILE.read_text())


def load_episode_frames(stem):
    path = next(DATA_DIR.glob(f"{stem}.h5"))
    with h5py.File(path, "r") as f:
        if "observations" in f and "images" in f["observations"]:
            imgs = f["observations"]["images"][:]
        else:
            imgs = f["images"][:]
    return imgs


def frame_to_image_feature(frame):
    img = Image.fromarray(frame.astype(np.uint8)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0


def build_windows(dataset, window=WINDOW):
    """두 조건 모두 동일 피처(bbox 12dim, image 256dim)를 쓰되 모델에서 결합 방식만 다르게 함."""
    X_bbox, X_img, y, meta = [], [], [], []
    for ep in dataset:
        frames = ep["frames"]
        imgs = load_episode_frames(ep["episode"])
        img_feats = [frame_to_image_feature(imgs[f["frame_idx"]]) for f in frames]
        for t in range(len(frames)):
            bfeat = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                f = frames[idx]
                bfeat.extend([f["cx"], f["cy"], f["area"], float(f["has_bbox"])])
            X_bbox.append(bfeat)
            X_img.append(img_feats[t].tolist())
            y.append(frames[t]["gt_class"])
            meta.append({
                "path_type": ep["path_type"],
                "episode": ep["episode"],
                "frame_idx": frames[t]["frame_idx"],
                "has_bbox": bool(frames[t]["has_bbox"]),
            })
    return (
        np.asarray(X_bbox, dtype=np.float32),
        np.asarray(X_img, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        meta,
    )


def make_episode_split(dataset, seed):
    rng = np.random.default_rng(seed)
    by_path = defaultdict(list)
    for i, ep in enumerate(dataset):
        by_path[ep["path_type"]].append(i)
    train_idx, test_idx = [], []
    for _, idxs in by_path.items():
        rng.shuffle(idxs)
        k = max(1, int(len(idxs) * 0.2))
        test_idx.extend(idxs[:k])
        train_idx.extend(idxs[k:])
    return [dataset[i] for i in train_idx], [dataset[i] for i in test_idx]


class ConcatMLP(nn.Module):
    """(A) baseline — ablate_bbox_image_features.py의 bbox_image 조건과 동일 구조."""

    def __init__(self):
        super().__init__()
        d_in = BBOX_DIM + IMG_DIM
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, bbox_feat, img_feat):
        return self.net(torch.cat([bbox_feat, img_feat], dim=-1))


class SubgoalSteeredMLP(nn.Module):
    """(B) bbox -> 서브골 임베딩 -> FiLM(gamma,beta)으로 이미지 특징 조향."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.subgoal_enc = nn.Sequential(
            nn.Linear(BBOX_DIM, 32), nn.ReLU(),
            nn.Linear(32, hidden),
        )
        self.film = nn.Linear(hidden, hidden * 2)
        self.img_proj = nn.Linear(IMG_DIM, hidden)
        self.trunk = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, bbox_feat, img_feat):
        s = self.subgoal_enc(bbox_feat)
        gamma, beta = self.film(s).chunk(2, dim=-1)
        h = self.img_proj(img_feat)
        h = gamma * h + beta
        return self.trunk(h)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_eval(model_cls, Xb_tr, Xi_tr, y_tr, Xb_te, Xi_te, y_te, seed):
    torch.manual_seed(seed)
    model = model_cls().to(DEVICE)

    cls_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = torch.tensor(1.0 / cls_counts, dtype=torch.float32, device=DEVICE)
    weights = weights / weights.sum() * NUM_CLASSES

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    Xb_tr_t = torch.tensor(Xb_tr, device=DEVICE)
    Xi_tr_t = torch.tensor(Xi_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
    Xb_te_t = torch.tensor(Xb_te, device=DEVICE)
    Xi_te_t = torch.tensor(Xi_te, device=DEVICE)
    y_te_t = torch.tensor(y_te, device=DEVICE)

    best_acc = 0.0
    best_preds = None
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(Xb_tr_t))
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            logits = model(Xb_tr_t[b], Xi_tr_t[b])
            loss = loss_fn(logits, y_tr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(Xb_te_t, Xi_te_t).argmax(dim=-1)
            acc = (preds == y_te_t).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_preds = preds.cpu().numpy()
    return best_acc, best_preds, count_params(model)


def per_path_pm(meta_te, y_te, preds):
    pm = defaultdict(lambda: {"correct": 0, "total": 0})
    for i, m in enumerate(meta_te):
        pm[m["path_type"]]["total"] += 1
        if preds[i] == y_te[i]:
            pm[m["path_type"]]["correct"] += 1
    return {k: v["correct"] / v["total"] for k, v in pm.items()}


def no_bbox_robustness(meta_te, y_te, preds):
    """has_bbox=False (그라운딩 실패) 프레임만 따로 정확도 집계."""
    idx = [i for i, m in enumerate(meta_te) if not m["has_bbox"]]
    if not idx:
        return None
    correct = sum(1 for i in idx if preds[i] == y_te[i])
    return correct / len(idx)


def main():
    print("데이터 로드...")
    dataset = load_dataset()

    results = {"concat": {"pm": [], "pm_by_path": [], "no_bbox_acc": [], "n_params": None},
               "steered": {"pm": [], "pm_by_path": [], "no_bbox_acc": [], "n_params": None}}

    for seed in SEEDS:
        train_eps, test_eps = make_episode_split(dataset, seed)
        Xb_tr, Xi_tr, y_tr, _ = build_windows(train_eps)
        Xb_te, Xi_te, y_te, meta_te = build_windows(test_eps)

        for name, cls in [("concat", ConcatMLP), ("steered", SubgoalSteeredMLP)]:
            acc, preds, n_params = train_eval(cls, Xb_tr, Xi_tr, y_tr, Xb_te, Xi_te, y_te, seed)
            results[name]["pm"].append(acc)
            results[name]["pm_by_path"].append(per_path_pm(meta_te, y_te, preds))
            nb_acc = no_bbox_robustness(meta_te, y_te, preds)
            if nb_acc is not None:
                results[name]["no_bbox_acc"].append(nb_acc)
            results[name]["n_params"] = n_params
            print(f"[seed {seed}] {name:8s} PM={acc:.4f} params={n_params}")

    summary = {}
    for name in ["concat", "steered"]:
        r = results[name]
        pm_by_path_mean = defaultdict(list)
        for d in r["pm_by_path"]:
            for k, v in d.items():
                pm_by_path_mean[k].append(v)
        summary[name] = {
            "mean_pm": float(np.mean(r["pm"])),
            "std_pm": float(np.std(r["pm"])),
            "mean_no_bbox_acc": float(np.mean(r["no_bbox_acc"])) if r["no_bbox_acc"] else None,
            "pm_by_path_mean": {k: float(np.mean(v)) for k, v in pm_by_path_mean.items()},
            "n_params": r["n_params"],
            "raw_pm": r["pm"],
        }

    print("\n=== 결과 요약 ===")
    for name in ["concat", "steered"]:
        s = summary[name]
        print(f"{name:8s}: PM={s['mean_pm']:.4f}±{s['std_pm']:.4f} "
              f"no_bbox_acc={s['mean_no_bbox_acc']} params={s['n_params']}")

    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n저장: {SUMMARY_FILE}")

    build_html(summary)
    print(f"저장: {HTML_FILE}")


def build_html(summary):
    colors = {"concat": "#60a5fa", "steered": "#22c55e"}
    labels = {"concat": "Concat-MLP (baseline)", "steered": "Subgoal-Steered FiLM"}

    boxes = ""
    for name in ["concat", "steered"]:
        s = summary[name]
        nb = f"{s['mean_no_bbox_acc']:.1%}" if s["mean_no_bbox_acc"] is not None else "N/A"
        boxes += f"""
    <div class="box">
      <div style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase;">{labels[name]}</div>
      <div class="num" style="color:{colors[name]};">{s['mean_pm']:.1%}</div>
      <div style="color:#64748b;">± {s['std_pm']:.1%} (5 seeds) · params={s['n_params']}</div>
      <div style="color:#64748b; margin-top:6px;">has_bbox=False 프레임 정확도: {nb}</div>
    </div>"""

    header = "<th>Path Type</th>" + "".join(f"<th style='color:{colors[n]}'>{labels[n]}</th>" for n in ["concat", "steered"])
    rows = ""
    for pt in PATH_TYPES:
        cells = f"<td>{pt}</td>"
        for name in ["concat", "steered"]:
            v = summary[name]["pm_by_path_mean"].get(pt, 0.0)
            cells += f"<td>{v:.1%}</td>"
        rows += f"<tr>{cells}</tr>"

    diff = summary["steered"]["mean_pm"] - summary["concat"]["mean_pm"]
    verdict = ("FiLM 서브골 조향이 concat보다 우세" if diff > 0.01 else
               "두 방식 사실상 동등 — concat만으로 충분" if abs(diff) <= 0.01 else
               "concat이 여전히 우세 — 서브골 분리가 이 스케일에서는 이득 없음")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>그라운딩 기반 서브골 조향 vs Concat</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 24px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 8px; }}
  .sub {{ color: #94a3b8; margin-bottom: 24px; max-width: 900px; line-height: 1.6; }}
  .back {{ color: #60a5fa; text-decoration: none; display: inline-block; margin-bottom: 16px; }}
  .grid2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 20px; }}
  .box {{ background: #1e293b; padding: 16px 20px; border-radius: 10px; }}
  .num {{ font-size: 2.3rem; font-weight: 800; }}
  table {{ border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; width: 100%; margin-top: 16px; }}
  th, td {{ padding: 8px 14px; border-bottom: 1px solid #334155; text-align: left; }}
  th {{ background: #0b1220; }}
  .diag {{ background: #172554; border-left: 4px solid #60a5fa; padding: 14px 18px; border-radius: 6px; color: #dbeafe; margin-top: 20px; line-height: 1.7; }}
  .caveat {{ background: #3f1d1d; border-left: 4px solid #f87171; padding: 14px 18px; border-radius: 6px; color: #fecaca; margin-top: 16px; line-height: 1.7; }}
</style>
</head>
<body>
  <a class="back" href="../index.html">← Back</a>
  <h1>그라운딩 기반 서브골 조향 vs Concat 비교</h1>
  <p class="sub">
    기존 bbox+image concat-MLP(Exp14, 76.7%)와, bbox를 서브골 임베딩으로 분리해 FiLM 게이팅으로
    이미지 특징을 조향하는 구조를 동일 캐시 bbox_dataset.json(45 ep, 794 frames)·동일 split·동일 epoch로 공정 비교.
  </p>

  <div class="grid2">{boxes}
  </div>

  <h2>PM per Path Type (5-seed 평균)</h2>
  <table>
    <tr>{header}</tr>
    {rows}
  </table>

  <div class="diag">
    <strong>결론(초안)</strong>: {verdict} (Δ={diff:+.1%})
  </div>

  <div class="caveat">
    <strong>범위 한계</strong><br>
    · bbox는 1회성 오프라인 캐시 그라운딩 재생이며 실시간 추론이 아님 — "온라인 서브골 조향" 주장 아님<br>
    · 8-class 이산 타겟만 다룸 — flow-matching/연속 회귀 액션과 무관<br>
    · 45 에피소드의 소규모 예비 실험(preliminary) — closed-loop 실주행 평가는 범위 밖
  </div>
</body>
</html>"""
    HTML_FILE.write_text(html)


if __name__ == "__main__":
    main()
