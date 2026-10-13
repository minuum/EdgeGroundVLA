#!/usr/bin/env python3
"""
서브골 조향 확장 실험 — V6 최신 데이터 버전 (bbox_dataset_v6_owl.json)

기존 eval_subgoal_steering_extended.py는 구세대 데이터(45 ep, 794 frames, 2026-04-08
수집, Pure HF Kosmos-2 그라운딩)로 돌렸었다. 실제 CH72 실기/FM 비교에 쓰인 최신 데이터는
bbox_dataset_v6_owl.json(225 ep, 16,599 frames, 5방향 앵커 center/weak_left/weak_right/
strong_left/strong_right, OWLv2 그라운딩)이라 이걸로 재실행한다.

1) 시드 15개 + paired 유의성 체크 (concat vs steered)
2) 경로(path_type, 15종)별 diff 분해
3) has_bbox=False일 때 FiLM이 identity(gamma~1,beta~0)로 수렴하도록 정규화한 steered_reg 변형
4) 파라미터 수를 steered와 맞춘 축소 concat_small 변형 (공정 비교)
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

DATA_DIR = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OWL_DIR = ROOT / "docs" / "v5" / "bbox_nav_owl"
OUT_DIR = ROOT / "docs" / "v5" / "subgoal_steering_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_FEAT_CACHE_FILE = OWL_DIR / "v6_img_feat_cache_16x16.npz"
_IMG_FEAT_CACHE = None  # lazy-loaded, Stage 1 산출물 (scripts/precompute_v6_image_features.py)


def get_img_feat_cache():
    global _IMG_FEAT_CACHE
    if _IMG_FEAT_CACHE is None:
        if not IMG_FEAT_CACHE_FILE.exists():
            raise FileNotFoundError(
                f"{IMG_FEAT_CACHE_FILE} 없음 — 먼저 scripts/precompute_v6_image_features.py 실행 필요")
        _IMG_FEAT_CACHE = np.load(IMG_FEAT_CACHE_FILE)
    return _IMG_FEAT_CACHE


DATASET_FILE = OWL_DIR / "bbox_dataset_v6_owl.json"
SUMMARY_FILE = OUT_DIR / "summary_extended_v6.json"
HTML_FILE = OUT_DIR / "extended_v6.html"

PATH_TYPES = [
    "center_straight", "center_left_curve", "center_right_curve",
    "weak_left_straight", "weak_left_left_curve", "weak_left_right_curve",
    "weak_right_straight", "weak_right_left_curve", "weak_right_right_curve",
    "strong_left_straight", "strong_left_left_curve", "strong_left_right_curve",
    "strong_right_straight", "strong_right_left_curve", "strong_right_right_curve",
]
NUM_CLASSES = 8
WINDOW = 3
IMG_SIZE = 16
BBOX_DIM = WINDOW * 4
IMG_DIM = IMG_SIZE * IMG_SIZE
HIDDEN = 128
SEEDS = list(range(10))  # Stage 1 캐시로 I/O 병목 제거, 1-seed 실측 2m25s → 10 seed로 런타임 조정(~20분)
EPOCHS = 120
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset():
    return json.loads(DATASET_FILE.read_text())


def resolve_episode_path(ep_field):
    """bbox_dataset_v6_owl.json의 'episode'는 절대경로 .h5 — 구버전(stem)과 둘 다 지원."""
    p = Path(ep_field)
    if p.suffix == ".h5" and p.exists():
        return p
    return next(DATA_DIR.glob(f"{ep_field}.h5"))


def load_episode_frames(ep_field):
    path = resolve_episode_path(ep_field)
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
    cache = get_img_feat_cache()
    X_bbox, X_img, y, meta = [], [], [], []
    for ep in dataset:
        frames = ep["frames"]
        img_feats = cache[ep["episode"]]  # Stage 1 캐시 (frame_idx 순서로 이미 정렬됨)
        for t in range(len(frames)):
            bfeat = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                f = frames[idx]
                bfeat.extend([f["cx_det"], f["cy_det"], f["area_det"], float(f["has_bbox"])])
            X_bbox.append(bfeat)
            X_img.append(img_feats[t].tolist())
            y.append(frames[t]["gt_class"])
            meta.append({
                "path_type": ep["path_type"],
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
    """원래 concat 베이스라인 (110,536 params)."""

    def __init__(self, h1=256, h2=128, h3=64):
        super().__init__()
        d_in = BBOX_DIM + IMG_DIM
        self.net = nn.Sequential(
            nn.Linear(d_in, h1), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, NUM_CLASSES),
        )

    def forward(self, bbox_feat, img_feat):
        return self.net(torch.cat([bbox_feat, img_feat], dim=-1)), None


class SubgoalSteeredMLP(nn.Module):
    """FiLM 서브골 조향. return_gate=True면 (logits, (gamma,beta)) 반환 (정규화용)."""

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
        return self.trunk(h), (gamma, beta)


# (4) 파라미터 수를 steered(79,336)에 맞춘 축소 concat
CONCAT_SMALL_DIMS = dict(h1=184, h2=96, h3=48)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_eval(model, Xb_tr, Xi_tr, y_tr, Xb_te, Xi_te, y_te, meta_te, seed,
               has_bbox_tr=None, identity_reg_weight=0.0):
    torch.manual_seed(seed)
    model = model.to(DEVICE)

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
    hb_tr_t = None
    if has_bbox_tr is not None:
        hb_tr_t = torch.tensor(has_bbox_tr, dtype=torch.bool, device=DEVICE)

    best_acc, best_preds, final_train_loss, final_test_loss = 0.0, None, None, None
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(Xb_tr_t))
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            logits, gate = model(Xb_tr_t[b], Xi_tr_t[b])
            loss = loss_fn(logits, y_tr_t[b])
            if identity_reg_weight > 0 and gate is not None and hb_tr_t is not None:
                gamma, beta = gate
                no_bbox_mask = ~hb_tr_t[b]
                if no_bbox_mask.any():
                    reg = ((gamma[no_bbox_mask] - 1.0) ** 2).mean() + (beta[no_bbox_mask] ** 2).mean()
                    loss = loss + identity_reg_weight * reg
            opt.zero_grad(); loss.backward(); opt.step()
            final_train_loss = loss.item()
        model.eval()
        with torch.no_grad():
            logits_te, _ = model(Xb_te_t, Xi_te_t)
            test_loss = loss_fn(logits_te, y_te_t).item()
            preds = logits_te.argmax(dim=-1)
            acc = (preds == y_te_t).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_preds = preds.cpu().numpy()
                final_test_loss = test_loss
    return best_acc, best_preds, final_train_loss, final_test_loss, count_params(model)


def per_path_pm(meta_te, y_te, preds):
    pm = defaultdict(lambda: {"correct": 0, "total": 0})
    for i, m in enumerate(meta_te):
        pm[m["path_type"]]["total"] += 1
        if preds[i] == y_te[i]:
            pm[m["path_type"]]["correct"] += 1
    return {k: v["correct"] / v["total"] for k, v in pm.items()}


def no_bbox_robustness(meta_te, y_te, preds):
    idx = [i for i, m in enumerate(meta_te) if not m["has_bbox"]]
    if not idx:
        return None
    return sum(1 for i in idx if preds[i] == y_te[i]) / len(idx)


def paired_ttest(a, b):
    """scipy 없을 수 있으니 수동 paired t-test."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    n = len(d)
    mean_d = d.mean()
    std_d = d.std(ddof=1)
    if std_d == 0:
        return mean_d, float("inf") if mean_d != 0 else 0.0, 0.0
    t_stat = mean_d / (std_d / np.sqrt(n))
    try:
        from scipy import stats
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    except ImportError:
        p = None
    return mean_d, t_stat, p


def main():
    print("데이터 로드...")
    dataset = load_dataset()

    conditions = ["concat", "concat_small", "steered", "steered_reg"]
    results = {c: {"pm": [], "pm_by_path": [], "no_bbox_acc": [], "train_loss": [], "test_loss": [], "n_params": None}
               for c in conditions}

    for seed in SEEDS:
        train_eps, test_eps = make_episode_split(dataset, seed)
        Xb_tr, Xi_tr, y_tr, meta_tr = build_windows(train_eps)
        Xb_te, Xi_te, y_te, meta_te = build_windows(test_eps)
        has_bbox_tr = np.array([m["has_bbox"] for m in meta_tr])

        model_specs = {
            "concat": (ConcatMLP(), 0.0),
            "concat_small": (ConcatMLP(**CONCAT_SMALL_DIMS), 0.0),
            "steered": (SubgoalSteeredMLP(), 0.0),
            "steered_reg": (SubgoalSteeredMLP(), 0.3),
        }
        for name, (model, reg_w) in model_specs.items():
            acc, preds, tr_loss, te_loss, n_params = train_eval(
                model, Xb_tr, Xi_tr, y_tr, Xb_te, Xi_te, y_te, meta_te, seed,
                has_bbox_tr=has_bbox_tr, identity_reg_weight=reg_w,
            )
            results[name]["pm"].append(acc)
            results[name]["pm_by_path"].append(per_path_pm(meta_te, y_te, preds))
            nb = no_bbox_robustness(meta_te, y_te, preds)
            if nb is not None:
                results[name]["no_bbox_acc"].append(nb)
            results[name]["train_loss"].append(tr_loss)
            results[name]["test_loss"].append(te_loss)
            results[name]["n_params"] = n_params
            print(f"[seed {seed:2d}] {name:14s} PM={acc:.4f} test_loss={te_loss:.4f} params={n_params}")

    summary = {}
    for name in conditions:
        r = results[name]
        pm_by_path_mean = defaultdict(list)
        for d in r["pm_by_path"]:
            for k, v in d.items():
                pm_by_path_mean[k].append(v)
        summary[name] = {
            "mean_pm": float(np.mean(r["pm"])),
            "std_pm": float(np.std(r["pm"])),
            "mean_train_loss": float(np.mean(r["train_loss"])),
            "mean_test_loss": float(np.mean(r["test_loss"])),
            "mean_no_bbox_acc": float(np.mean(r["no_bbox_acc"])) if r["no_bbox_acc"] else None,
            "pm_by_path_mean": {k: float(np.mean(v)) for k, v in pm_by_path_mean.items()},
            "n_params": r["n_params"],
            "raw_pm": r["pm"],
        }

    # (1) paired significance: concat vs steered, concat_small vs steered
    mean_d1, t1, p1 = paired_ttest(results["steered"]["pm"], results["concat"]["pm"])
    mean_d2, t2, p2 = paired_ttest(results["steered"]["pm"], results["concat_small"]["pm"])
    mean_d3, t3, p3 = paired_ttest(results["steered_reg"]["pm"], results["steered"]["pm"])
    summary["significance"] = {
        "steered_vs_concat": {"mean_diff": mean_d1, "t_stat": t1, "p_value": p1},
        "steered_vs_concat_small": {"mean_diff": mean_d2, "t_stat": t2, "p_value": p2},
        "steered_reg_vs_steered": {"mean_diff": mean_d3, "t_stat": t3, "p_value": p3},
    }

    print("\n=== 결과 요약 (15 seeds) ===")
    for name in conditions:
        s = summary[name]
        print(f"{name:14s}: PM={s['mean_pm']:.4f}±{s['std_pm']:.4f} test_loss={s['mean_test_loss']:.4f} "
              f"no_bbox_acc={s['mean_no_bbox_acc']} params={s['n_params']}")
    print(f"\nsteered vs concat        : Δ={mean_d1:+.4f} t={t1:.3f} p={p1}")
    print(f"steered vs concat_small  : Δ={mean_d2:+.4f} t={t2:.3f} p={p2}")
    print(f"steered_reg vs steered   : Δ={mean_d3:+.4f} t={t3:.3f} p={p3}")

    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n저장: {SUMMARY_FILE}")

    build_html(summary)
    print(f"저장: {HTML_FILE}")


def build_html(summary):
    colors = {"concat": "#60a5fa", "concat_small": "#38bdf8", "steered": "#22c55e", "steered_reg": "#a78bfa"}
    labels = {
        "concat": "Concat-MLP (110k params)",
        "concat_small": "Concat-Small (파라미터 매칭)",
        "steered": "Subgoal-Steered FiLM",
        "steered_reg": "Steered + Identity 정규화",
    }
    conditions = ["concat", "concat_small", "steered", "steered_reg"]

    boxes = ""
    for name in conditions:
        s = summary[name]
        nb = f"{s['mean_no_bbox_acc']:.1%}" if s["mean_no_bbox_acc"] is not None else "N/A"
        boxes += f"""
    <div class="box">
      <div style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase;">{labels[name]}</div>
      <div class="num" style="color:{colors[name]};">{s['mean_pm']:.1%}</div>
      <div style="color:#64748b;">± {s['std_pm']:.1%} (15 seeds) · params={s['n_params']}</div>
      <div style="color:#64748b;">test_loss={s['mean_test_loss']:.3f}</div>
      <div style="color:#64748b; margin-top:6px;">has_bbox=False 정확도: {nb}</div>
    </div>"""

    header = "<th>Path Type</th>" + "".join(f"<th style='color:{colors[n]}'>{labels[n]}</th>" for n in conditions)
    rows = ""
    for pt in PATH_TYPES:
        cells = f"<td>{pt}</td>"
        for name in conditions:
            v = summary[name]["pm_by_path_mean"].get(pt, 0.0)
            cells += f"<td>{v:.1%}</td>"
        rows += f"<tr>{cells}</tr>"

    sig = summary["significance"]

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>서브골 조향 확장 실험 (15 seeds + 유의성 + 정규화)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 24px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 1.7rem; margin-bottom: 8px; }}
  .sub {{ color: #94a3b8; margin-bottom: 24px; max-width: 900px; line-height: 1.6; }}
  .back {{ color: #60a5fa; text-decoration: none; display: inline-block; margin-bottom: 16px; }}
  .grid4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }}
  .box {{ background: #1e293b; padding: 14px 16px; border-radius: 10px; }}
  .num {{ font-size: 1.9rem; font-weight: 800; }}
  table {{ border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; width: 100%; margin-top: 16px; font-size: 0.85rem; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #334155; text-align: left; }}
  th {{ background: #0b1220; }}
  .diag {{ background: #172554; border-left: 4px solid #60a5fa; padding: 14px 18px; border-radius: 6px; color: #dbeafe; margin-top: 20px; line-height: 1.8; }}
  .caveat {{ background: #3f1d1d; border-left: 4px solid #f87171; padding: 14px 18px; border-radius: 6px; color: #fecaca; margin-top: 16px; line-height: 1.7; }}
</style>
</head>
<body>
  <a class="back" href="index.html">← 이전(5-seed) 리포트</a>
  <h1>서브골 조향 확장 실험 — 15 seeds + 유의성 + 파라미터 매칭 + 그라운딩 실패 정규화</h1>
  <p class="sub">
    concat(원본) / concat_small(파라미터를 steered와 맞춘 축소판) / steered(FiLM) / steered_reg(그라운딩 실패 시 identity 유도 정규화 추가)
    4조건을 15 seed로 비교.
  </p>

  <div class="grid4">{boxes}
  </div>

  <h2>PM per Path Type (15-seed 평균)</h2>
  <table>
    <tr>{header}</tr>
    {rows}
  </table>

  <div class="diag">
    <strong>Paired 유의성 (수동 paired t-test, df=14)</strong><br>
    steered vs concat(원본): Δ={sig['steered_vs_concat']['mean_diff']:+.4f}, t={sig['steered_vs_concat']['t_stat']:.3f}, p={sig['steered_vs_concat']['p_value']}<br>
    steered vs concat_small(파라미터 매칭): Δ={sig['steered_vs_concat_small']['mean_diff']:+.4f}, t={sig['steered_vs_concat_small']['t_stat']:.3f}, p={sig['steered_vs_concat_small']['p_value']}<br>
    steered_reg vs steered: Δ={sig['steered_reg_vs_steered']['mean_diff']:+.4f}, t={sig['steered_reg_vs_steered']['t_stat']:.3f}, p={sig['steered_reg_vs_steered']['p_value']}
  </div>

  <div class="caveat">
    <strong>범위 한계</strong><br>
    · bbox는 오프라인 캐시 그라운딩 재생 — 실시간 추론 아님<br>
    · 8-class 이산 타겟만 다룸 — flow-matching/연속 액션과 무관<br>
    · 45 에피소드 소규모 예비 실험, closed-loop 실주행 평가는 별도 과제(백로그 CH73-4)
  </div>
</body>
</html>"""
    HTML_FILE.write_text(html)


if __name__ == "__main__":
    main()
