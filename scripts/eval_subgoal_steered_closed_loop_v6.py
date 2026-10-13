#!/usr/bin/env python3
"""
Task 6 — V6 최신 데이터 버전: 서브골 조향(FiLM) 헤드를 offline-replay "closed-loop" 평가에 연결

구세대 bbox_dataset.json(45 ep) 대신 실제 CH72 실기/FM 비교에 쓰인 최신
bbox_dataset_v6_owl.json(225 ep, 16,599 frames, 5방향 앵커, OWLv2 그라운딩)로 재실행.

주의(용어 정정): evaluate_closed_loop_v5.py와 동일하게, 여기서의 "closed-loop"는
실제 closed-loop(예측이 다음 관측에 피드백)가 아니라 녹화된 expert 궤적과
프레임별로 비교하는 teacher-forced offline replay다. exp67/step2와 동일 프로토콜.

파이프라인(exp67 패턴 재사용):
  1) concat / steered_reg 두 모델을 학습해 체크포인트로 저장
  2) 테스트 에피소드마다 프레임별 클래스 예측 → continuous_to_class(expert)와 함께
     build_trajectory → compute_metrics(FPE, TLD, success)로 집계
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

from scripts.sim.rollout_core import build_trajectory, continuous_to_class, compute_metrics

DATA_DIR = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OWL_DIR = ROOT / "docs" / "v5" / "bbox_nav_owl"
CKPT_DIR = ROOT / "runs" / "v5_nav" / "mlp" / "subgoal_steered_v6"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "docs" / "v5" / "subgoal_steering_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_FILE = OWL_DIR / "bbox_dataset_v6_owl.json"
SUMMARY_FILE = OUT_DIR / "summary_closed_loop_v6.json"
HTML_FILE = OUT_DIR / "closed_loop_v6.html"

IMG_FEAT_CACHE_FILE = OWL_DIR / "v6_img_feat_cache_16x16.npz"
_IMG_FEAT_CACHE = None


def get_img_feat_cache():
    global _IMG_FEAT_CACHE
    if _IMG_FEAT_CACHE is None:
        if not IMG_FEAT_CACHE_FILE.exists():
            raise FileNotFoundError(
                f"{IMG_FEAT_CACHE_FILE} 없음 — 먼저 scripts/precompute_v6_image_features.py 실행 필요")
        _IMG_FEAT_CACHE = np.load(IMG_FEAT_CACHE_FILE)
    return _IMG_FEAT_CACHE

NUM_CLASSES = 8
WINDOW = 3
IMG_SIZE = 16
BBOX_DIM = WINDOW * 4
IMG_DIM = IMG_SIZE * IMG_SIZE
HIDDEN = 128
EPOCHS = 80  # 데이터 20배 증가로 수렴 빨라짐 — 220→80 (런타임 고려)
DT = 0.1
SUCCESS_FPE = 0.5
SEED = 0  # 단일 split으로 대표 체크포인트 저장 (exp67/step2와 동일 관례)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset():
    return json.loads(DATASET_FILE.read_text())


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


def resolve_episode_path(ep_field):
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
    X_bbox, X_img, y = [], [], []
    for ep in dataset:
        frames = ep["frames"]
        img_feats = cache[ep["episode"]]
        for t in range(len(frames)):
            bfeat = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                f = frames[idx]
                bfeat.extend([f["cx_det"], f["cy_det"], f["area_det"], float(f["has_bbox"])])
            X_bbox.append(bfeat)
            X_img.append(img_feats[t].tolist())
            y.append(frames[t]["gt_class"])
    return np.asarray(X_bbox, dtype=np.float32), np.asarray(X_img, dtype=np.float32), np.asarray(y, dtype=np.int64)


class ConcatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BBOX_DIM + IMG_DIM, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, bbox_feat, img_feat):
        return self.net(torch.cat([bbox_feat, img_feat], dim=-1)), None


class SubgoalSteeredMLP(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.subgoal_enc = nn.Sequential(nn.Linear(BBOX_DIM, 32), nn.ReLU(), nn.Linear(32, hidden))
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


def train_and_save(model, Xb, Xi, y, has_bbox, ckpt_path, identity_reg_weight=0.0, seed=SEED):
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    cls_counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = torch.tensor(1.0 / cls_counts, dtype=torch.float32, device=DEVICE)
    weights = weights / weights.sum() * NUM_CLASSES
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    Xb_t = torch.tensor(Xb, device=DEVICE)
    Xi_t = torch.tensor(Xi, device=DEVICE)
    y_t = torch.tensor(y, device=DEVICE)
    hb_t = torch.tensor(has_bbox, dtype=torch.bool, device=DEVICE)

    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(Xb_t))
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            logits, gate = model(Xb_t[b], Xi_t[b])
            loss = loss_fn(logits, y_t[b])
            if identity_reg_weight > 0 and gate is not None:
                gamma, beta = gate
                mask = ~hb_t[b]
                if mask.any():
                    reg = ((gamma[mask] - 1.0) ** 2).mean() + (beta[mask] ** 2).mean()
                    loss = loss + identity_reg_weight * reg
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 55 == 0 or ep == EPOCHS - 1:
            print(f"  ep{ep:3d} loss={loss.item():.4f}")
    model.eval()
    torch.save(model.state_dict(), ckpt_path)
    print(f"  저장: {ckpt_path}")
    return model


@torch.no_grad()
def predict_episode(model, ep_entry, window=WINDOW):
    frames = ep_entry["frames"]
    img_feats = get_img_feat_cache()[ep_entry["episode"]]
    pred_classes = []
    for t in range(len(frames)):
        bfeat = []
        for k in range(window):
            idx = max(0, t - (window - 1 - k))
            f = frames[idx]
            bfeat.extend([f["cx_det"], f["cy_det"], f["area_det"], float(f["has_bbox"])])
        bx = torch.tensor([bfeat], dtype=torch.float32, device=DEVICE)
        ix = torch.tensor([img_feats[t]], dtype=torch.float32, device=DEVICE)
        logits, _ = model(bx, ix)
        pred_classes.append(min(int(logits.argmax(1).item()), NUM_CLASSES - 1))

    path = resolve_episode_path(ep_entry["episode"])
    with h5py.File(path, "r") as f:
        expert_actions = f["actions"][:]
    return pred_classes, expert_actions[:len(frames)]


def eval_model_offline_replay(model, test_eps, tag):
    per_path = defaultdict(list)
    all_metrics = []
    for ep in test_eps:
        pred_classes, expert_actions = predict_episode(model, ep)
        expert_cls = [continuous_to_class(*a[:3]) for a in expert_actions]
        expert_traj = build_trajectory(expert_cls, DT)
        pred_traj = build_trajectory(pred_classes, DT)
        m = compute_metrics(expert_traj, pred_traj, SUCCESS_FPE)
        m["path_type"] = ep["path_type"]
        m["episode"] = ep["episode"]
        all_metrics.append(m)
        per_path[ep["path_type"]].append(m)
        print(f"  [{tag}] {ep['path_type']:16s} FPE={m['fpe']:.3f} TLD={m['tld']:.3f} success={m['success']}")

    success_rate = float(np.mean([m["success"] for m in all_metrics]))
    mean_fpe = float(np.mean([m["fpe"] for m in all_metrics]))
    mean_tld = float(np.mean([m["tld"] for m in all_metrics]))
    by_path_success = {k: float(np.mean([m["success"] for m in v])) for k, v in per_path.items()}
    return {
        "success_rate": success_rate,
        "mean_fpe": mean_fpe,
        "mean_tld": mean_tld,
        "by_path_success": by_path_success,
        "n_episodes": len(all_metrics),
        "raw": all_metrics,
    }


def main():
    print("데이터 로드...")
    dataset = load_dataset()
    train_eps, test_eps = make_episode_split(dataset, SEED)
    Xb_tr, Xi_tr, y_tr = build_windows(train_eps)
    has_bbox_tr = np.array([f["has_bbox"] for ep in train_eps for f in ep["frames"]])

    print("\n[1/2] Concat-MLP 학습+저장...")
    concat_model = train_and_save(ConcatMLP(), Xb_tr, Xi_tr, y_tr, has_bbox_tr,
                                   CKPT_DIR / "concat_mlp.pt")

    print("\n[2/2] Subgoal-Steered(FiLM + identity 정규화) 학습+저장...")
    steered_model = train_and_save(SubgoalSteeredMLP(), Xb_tr, Xi_tr, y_tr, has_bbox_tr,
                                    CKPT_DIR / "subgoal_steered_reg.pt", identity_reg_weight=0.3)

    print("\n=== Offline-replay 'closed-loop' 평가 (teacher-forced, 실제 closed-loop 아님) ===")
    print("-- concat --")
    r_concat = eval_model_offline_replay(concat_model, test_eps, "concat")
    print("-- steered_reg --")
    r_steered = eval_model_offline_replay(steered_model, test_eps, "steered_reg")

    print(f"\nconcat      : success={r_concat['success_rate']:.1%} FPE={r_concat['mean_fpe']:.3f} TLD={r_concat['mean_tld']:.3f}")
    print(f"steered_reg : success={r_steered['success_rate']:.1%} FPE={r_steered['mean_fpe']:.3f} TLD={r_steered['mean_tld']:.3f}")

    summary = {"concat": r_concat, "steered_reg": r_steered,
               "note": "offline teacher-forced replay (실제 closed-loop 아님), n_test_episodes=%d" % len(test_eps)}
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\n저장: {SUMMARY_FILE}")
    build_html(summary)
    print(f"저장: {HTML_FILE}")


def build_html(summary):
    r1, r2 = summary["concat"], summary["steered_reg"]
    path_types = sorted(set(list(r1["by_path_success"].keys()) + list(r2["by_path_success"].keys())))
    rows = ""
    for pt in path_types:
        c = r1["by_path_success"].get(pt, 0.0)
        s = r2["by_path_success"].get(pt, 0.0)
        rows += f"<tr><td>{pt}</td><td>{c:.0%}</td><td>{s:.0%}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head><meta charset="UTF-8"><title>Subgoal Steering — Offline Replay Eval</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin:0; padding:24px; background:#0f172a; color:#e2e8f0; }}
  h1 {{ font-size:1.6rem; }}
  .sub {{ color:#94a3b8; max-width:880px; line-height:1.6; }}
  .back {{ color:#60a5fa; text-decoration:none; display:inline-block; margin-bottom:16px; }}
  .grid2 {{ display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin:20px 0; }}
  .box {{ background:#1e293b; padding:16px 20px; border-radius:10px; }}
  .num {{ font-size:2.2rem; font-weight:800; }}
  table {{ border-collapse:collapse; background:#1e293b; border-radius:8px; overflow:hidden; width:100%; }}
  th,td {{ padding:8px 14px; border-bottom:1px solid #334155; text-align:left; }}
  th {{ background:#0b1220; }}
  .caveat {{ background:#3f1d1d; border-left:4px solid #f87171; padding:14px 18px; border-radius:6px; color:#fecaca; margin-top:20px; line-height:1.7; }}
</style></head>
<body>
  <a class="back" href="extended_v6.html">← 5-seed 확장 리포트(V6)</a>
  <h1>서브골 조향 — Offline Replay "Closed-Loop" 평가</h1>
  <p class="sub">exp67/step2와 동일 프로토콜(teacher-forced replay)로 concat vs steered_reg 체크포인트를 비교.
  success = FPE&lt;{SUCCESS_FPE}m AND 0.7≤TLD≤1.5.</p>

  <div class="grid2">
    <div class="box"><div style="color:#94a3b8">Concat-MLP</div>
      <div class="num" style="color:#60a5fa">{r1['success_rate']:.0%}</div>
      <div style="color:#64748b">FPE={r1['mean_fpe']:.3f}m · TLD={r1['mean_tld']:.3f} · n={r1['n_episodes']}</div></div>
    <div class="box"><div style="color:#94a3b8">Subgoal-Steered+Reg</div>
      <div class="num" style="color:#22c55e">{r2['success_rate']:.0%}</div>
      <div style="color:#64748b">FPE={r2['mean_fpe']:.3f}m · TLD={r2['mean_tld']:.3f} · n={r2['n_episodes']}</div></div>
  </div>

  <table><tr><th>Path Type</th><th>Concat success</th><th>Steered+Reg success</th></tr>{rows}</table>

  <div class="caveat">
    <strong>용어 정정</strong>: 이 평가는 실제 closed-loop(예측→다음 관측 피드백)가 아니라
    녹화된 expert 궤적 대비 teacher-forced offline replay다 — evaluate_closed_loop_v5.py의
    기존 step2/exp67 모드와 동일 정의. 진짜 closed-loop 실주행 평가는 범위 밖(별도 과제).<br>
    9개 path_type 중 1개 split(seed={SEED})만 평가한 대표 체크포인트 — 통계적 유의성은
    <a href="extended_v6.html" style="color:#fca5a5">5-seed 오프라인 PM 비교(V6)</a>를 참조.
  </div>
</body></html>"""
    HTML_FILE.write_text(html)


if __name__ == "__main__":
    main()
