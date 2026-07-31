#!/usr/bin/env python3
"""CH60: closed-loop 오프라인 리플레이 — 기존 Step2(PG2, w3) vs OWL(w6) vs OWL(w6)+text.

evaluate_closed_loop_v5.py의 step2 평가 흐름을 그대로 따르되:
  - 공통 43 에피소드로 통일 (billy 전용 2개 제외), episode명 기준 동일 split
  - 3 variant: pg2_w3(기존 재실측) / owl_w6 / owl_w6_text
  - text emb는 path_type instruction(사용자가 주는 입력에 해당 — leakage 아님)

Usage: .venv/bin/python3 scripts/sim/evaluate_closed_loop_owl_text.py
출력: docs/v5/closed_loop_eval/owl_text_metrics.json
"""
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.sim.rollout_core import (
    build_trajectory, continuous_to_class, compute_metrics,
)

spec = importlib.util.spec_from_file_location("instr", ROOT / "scripts" / "train_step2_instr_head.py")
instr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(instr)

DATA_DIR = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
PG2_DS = ROOT / "docs" / "v5" / "bbox_nav_step1" / "bbox_dataset.json"
OWL_DS = ROOT / "docs" / "v5" / "bbox_nav_owl" / "bbox_dataset_owl.json"
OUT = ROOT / "docs" / "v5" / "closed_loop_eval" / "owl_text_metrics.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 8
IMG_SIZE = 16
DT = 0.1          # rollout_core.DT_DEFAULT — 원본 closed-loop과 동일
SUCCESS_FPE = 0.5  # evaluate_closed_loop_v5.py 기본값과 동일
SEEDS = [0, 1, 2]

VARIANTS = {
    "pg2_w3":      {"ds": PG2_DS, "window": 3, "text": False},
    "owl_w6":      {"ds": OWL_DS, "window": 6, "text": False},
    "owl_w6_text": {"ds": OWL_DS, "window": 6, "text": True},
}


def img_feat_cache():
    cache = {}
    def get(ep_stem, frame_idx):
        k = (ep_stem, frame_idx)
        if k not in cache:
            with h5py.File(DATA_DIR / f"{ep_stem}.h5", "r") as f:
                frame = f["observations"]["images"][frame_idx]
            img = Image.fromarray(np.array(frame).astype(np.uint8)).convert("L").resize((IMG_SIZE, IMG_SIZE))
            cache[k] = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
        return cache[k]
    return get


def build_feats(eps, window, text, embs, get_img):
    X, y = [], []
    for ep in eps:
        frames = ep["frames"]
        tvec = embs[ep["path_type"]] if text else None
        for t in range(len(frames)):
            feat = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                f = frames[idx]
                feat.extend([f["cx"], f["cy"], f["area"], float(f["has_bbox"])])
            feat.extend(get_img(ep["episode"], frames[t]["frame_idx"]).tolist())
            if tvec is not None:
                feat.extend(tvec.tolist())
            X.append(feat)
            y.append(frames[t]["gt_class"])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def train_mlp(X, y, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.25),
        torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, NUM_CLASSES),
    ).to(DEVICE)
    w = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    w = np.where(w == 0, 1.0, w)
    wt = torch.tensor(1.0 / w, device=DEVICE)
    wt = wt / wt.sum() * NUM_CLASSES
    loss_fn = torch.nn.CrossEntropyLoss(weight=wt)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt = torch.tensor(X, device=DEVICE)
    yt = torch.tensor(y, device=DEVICE)
    for _ in range(220):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            loss = loss_fn(model(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def rollout_episode(ep, model, window, text, embs, get_img):
    frames = ep["frames"]
    tvec = embs[ep["path_type"]] if text else None
    preds = []
    for t in range(len(frames)):
        feat = []
        for k in range(window):
            idx = max(0, t - (window - 1 - k))
            f = frames[idx]
            feat.extend([f["cx"], f["cy"], f["area"], float(f["has_bbox"])])
        feat.extend(get_img(ep["episode"], frames[t]["frame_idx"]).tolist())
        if tvec is not None:
            feat.extend(tvec.tolist())
        with torch.no_grad():
            cls = int(model(torch.tensor([feat], dtype=torch.float32, device=DEVICE)).argmax(1).item())
        preds.append(min(cls, NUM_CLASSES - 1))
    with h5py.File(DATA_DIR / f"{ep['episode']}.h5", "r") as f:
        expert = f["actions"][:]
    return preds, expert[:len(frames)]


def main():
    pg2 = json.loads(PG2_DS.read_text())
    owl = json.loads(OWL_DS.read_text())
    common = {ep["episode"] for ep in owl} & {ep["episode"] for ep in pg2}
    pg2 = sorted([ep for ep in pg2 if ep["episode"] in common], key=lambda e: e["episode"])
    owl = sorted([ep for ep in owl if ep["episode"] in common], key=lambda e: e["episode"])
    print(f"공통 에피소드: {len(common)}")

    # episode명 기준 동일 split (path_type별 20% test, rng 42)
    rng = np.random.default_rng(42)
    by_path = defaultdict(list)
    for ep in pg2:
        by_path[ep["path_type"]].append(ep["episode"])
    test_names = set()
    for _, names in sorted(by_path.items()):
        names = sorted(names)
        rng.shuffle(names)
        test_names.update(names[:max(1, int(len(names) * 0.2))])
    print(f"test 에피소드: {len(test_names)}")

    embs = instr.embed_instructions()
    get_img = img_feat_cache()

    out = {}
    for vname, cfg in VARIANTS.items():
        ds = pg2 if cfg["ds"] == PG2_DS else owl
        train_eps = [ep for ep in ds if ep["episode"] not in test_names]
        test_eps = [ep for ep in ds if ep["episode"] in test_names]
        X_tr, y_tr = build_feats(train_eps, cfg["window"], cfg["text"], embs, get_img)
        srs, fpes, tlds = [], [], []
        per_path_last = None
        for seed in SEEDS:
            model = train_mlp(X_tr, y_tr, seed)
            all_m = []
            per_path = defaultdict(list)
            for ep in test_eps:
                preds, expert = rollout_episode(ep, model, cfg["window"], cfg["text"], embs, get_img)
                expert_cls = [continuous_to_class(*a[:3]) for a in expert]
                m = compute_metrics(build_trajectory(expert_cls, DT), build_trajectory(preds, DT), SUCCESS_FPE)
                m["episode"] = ep["episode"]
                all_m.append(m)
                per_path[ep["path_type"]].append(m)
            srs.append(sum(m["success"] for m in all_m) / len(all_m))
            fpes.append(float(np.mean([m["fpe"] for m in all_m])))
            tlds.append(float(np.mean([m["tld"] for m in all_m])))
            per_path_last = {pt: {"sr": sum(m["success"] for m in ms) / len(ms),
                                   "fpe": float(np.mean([m["fpe"] for m in ms]))}
                              for pt, ms in per_path.items()}
        out[vname] = {
            "sr_mean": float(np.mean(srs)), "sr_std": float(np.std(srs)), "sr_seeds": srs,
            "fpe_mean": float(np.mean(fpes)), "tld_mean": float(np.mean(tlds)),
            "per_path_lastseed": per_path_last, "n_test": len(test_eps),
        }
        print(f"[{vname}] SR {100*np.mean(srs):.1f}%±{100*np.std(srs):.1f}  "
              f"FPE {np.mean(fpes):.2f}m  TLD {np.mean(tlds):.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
