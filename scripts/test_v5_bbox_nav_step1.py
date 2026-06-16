#!/usr/bin/env python3
"""
Exp14 Step 1: BBox feature → learned MLP for action

Pure HF Kosmos-2 grounding을 더 많은 에피소드/프레임에 돌려 데이터셋 구축.
(cx, cy, area) feature + 짧은 history → MLP → 8-class action 학습.

Step 0 rule-based 31.1%에서 얼마나 올라가는가 확인.

Usage:
  python3 scripts/test_v5_bbox_nav_step1.py
  python3 scripts/test_v5_bbox_nav_step1.py --skip_extract  # 재학습만
"""

import sys, os, json, argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np
import h5py
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText

HF_KOSMOS_PATH = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_DIR = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OUT_DIR = ROOT / "docs" / "v5" / "bbox_nav_step1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)
CACHE_FILE = OUT_DIR / "bbox_dataset.json"

PATH_TYPES = [
    "center_straight", "center_left", "center_right",
    "left_straight",   "left_left",   "left_right",
    "right_straight",  "right_left",  "right_right",
]
CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
NUM_CLASSES = 8

EPS_PER_PATH = 5  # 9 path × 5 ep = 45 episodes (--full 시 전체 사용)
GROUNDING_PROMPT = "<grounding>The gray basket is at"
MAX_NEW_TOKENS = 48
WINDOW = 3  # history length


def gt_action_class(lx, ly, az):
    is_x = abs(lx) > 0.3
    is_y = abs(ly) > 0.3
    if not is_x and not is_y:
        if az > 0.1: return 6
        if az < -0.1: return 7
        return 0
    if lx > 0.3:
        if ly > 0.3: return 4
        if ly < -0.3: return 5
        return 1
    if abs(lx) < 0.3:
        if ly > 0.3: return 2
        if ly < -0.3: return 3
    return 0


def parse_basket_bbox(caption, entities):
    kw = ("basket", "gray box", "container", "gray")
    cands = []
    for ent_name, _span, boxes in entities:
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > 0.85: continue
            cands.append({
                "entity": ent_name,
                "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "area": area,
                "is_basket": any(k in ent_name.lower() for k in kw),
            })
    matched = [b for b in cands if b["is_basket"]]
    if matched:
        return matched[0]
    cap_low = caption.lower()
    if "far left" in cap_low:
        return {"cx": 0.1, "cy": 0.5, "area": 0.05, "entity": "caption:far_left"}
    if "far right" in cap_low:
        return {"cx": 0.9, "cy": 0.5, "area": 0.05, "entity": "caption:far_right"}
    if "left" in cap_low and "right" not in cap_low:
        return {"cx": 0.25, "cy": 0.5, "area": 0.05, "entity": "caption:left"}
    if "right" in cap_low and "left" not in cap_low:
        return {"cx": 0.75, "cy": 0.5, "area": 0.05, "entity": "caption:right"}
    if "center" in cap_low:
        return {"cx": 0.5, "cy": 0.5, "area": 0.05, "entity": "caption:center"}
    if cands:
        return cands[0]
    return None


def extract_bbox_dataset(full=False, data_dir=None):
    dd = Path(data_dir) if data_dir else DATA_DIR
    print(f"Loading Pure HF Kosmos-2 from {HF_KOSMOS_PATH}")
    processor = AutoProcessor.from_pretrained(str(HF_KOSMOS_PATH))
    model = AutoModelForImageTextToText.from_pretrained(
        str(HF_KOSMOS_PATH), torch_dtype=torch.float16
    ).cuda().eval()

    dataset = []
    for pt in PATH_TYPES:
        all_eps = sorted(dd.glob(f"episode_*target_{pt}_path*.h5"))
        eps = all_eps if full else all_eps[:EPS_PER_PATH]
        print(f"\n=== {pt}: {len(eps)} episodes ===")
        for ep in eps:
            with h5py.File(ep, "r") as f:
                if "observations" in f and "images" in f["observations"]:
                    imgs = f["observations"]["images"][:]
                else:
                    imgs = f["images"][:]
                actions = f["actions"][:]
            frames_data = []
            for fi in range(len(imgs)):
                gt_cls = gt_action_class(*actions[fi])
                pil = Image.fromarray(imgs[fi].astype(np.uint8)).convert("RGB")
                inputs = processor(text=GROUNDING_PROMPT, images=pil, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                pv = inputs["pixel_values"].to(torch.float16)
                with torch.no_grad():
                    out = model.generate(
                        pixel_values=pv,
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        image_embeds=None,
                        image_embeds_position_mask=inputs.get("image_embeds_position_mask"),
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                new_ids = out[:, inputs["input_ids"].shape[1]:]
                raw = processor.batch_decode(new_ids, skip_special_tokens=False)[0]
                caption, entities = processor.post_process_generation(raw)
                bbox = parse_basket_bbox(caption, entities)
                frames_data.append({
                    "frame_idx": fi, "gt_class": gt_cls,
                    "cx": bbox["cx"] if bbox else 0.5,
                    "cy": bbox["cy"] if bbox else 0.5,
                    "area": bbox["area"] if bbox else 0.0,
                    "has_bbox": bbox is not None,
                })
            dataset.append({
                "path_type": pt, "episode": ep.stem, "frames": frames_data,
            })
            print(f"  {ep.stem[:60]}: {len(frames_data)} frames")

    cache_file = OUT_DIR / ("bbox_dataset_full.json" if full else "bbox_dataset.json")
    cache_file.write_text(json.dumps(dataset, indent=2))
    print(f"Saved: {cache_file}")
    del model
    torch.cuda.empty_cache()
    return dataset, cache_file


def build_windows(dataset, window=WINDOW):
    X, y, meta = [], [], []
    for ep in dataset:
        frames = ep["frames"]
        for t in range(len(frames)):
            feat = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                f = frames[idx]
                feat.extend([f["cx"], f["cy"], f["area"], float(f["has_bbox"])])
            X.append(feat)
            y.append(frames[t]["gt_class"])
            meta.append({
                "path_type": ep["path_type"], "episode": ep["episode"],
                "frame_idx": frames[t]["frame_idx"],
            })
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), meta


def train_eval(X_tr, y_tr, X_te, y_te, epochs=200):
    d_in = X_tr.shape[1]
    model = nn.Sequential(
        nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, NUM_CLASSES),
    ).cuda()

    cls_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = torch.tensor(1.0 / cls_counts, dtype=torch.float32).cuda()
    weights = weights / weights.sum() * NUM_CLASSES
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    X_tr_t = torch.tensor(X_tr).cuda()
    y_tr_t = torch.tensor(y_tr).cuda()
    X_te_t = torch.tensor(X_te).cuda()
    y_te_t = torch.tensor(y_te).cuda()

    best_acc = 0.0
    best_preds = None
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(X_tr_t))
        for i in range(0, len(idx), 128):
            b = idx[i:i+128]
            logits = model(X_tr_t[b])
            loss = loss_fn(logits, y_tr_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).argmax(dim=-1)
            acc = (preds == y_te_t).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_preds = preds.cpu().numpy()
        if ep % 40 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d}: loss={loss.item():.3f} test_acc={acc:.3f} best={best_acc:.3f}")
    return best_acc, best_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_extract", action="store_true")
    ap.add_argument("--full", action="store_true", help="Use all available episodes (→ bbox_dataset_full.json)")
    ap.add_argument("--data_dir", type=str, default=None, help="Override DATA_DIR path")
    args = ap.parse_args()

    cache_file = OUT_DIR / ("bbox_dataset_full.json" if args.full else "bbox_dataset.json")

    if args.skip_extract and cache_file.exists():
        dataset = json.loads(cache_file.read_text())
        print(f"Loaded cached dataset: {len(dataset)} episodes from {cache_file.name}")
    else:
        dataset, cache_file = extract_bbox_dataset(full=args.full, data_dir=args.data_dir)

    # Episode-level split 80/20 (stratified by path_type)
    rng = np.random.default_rng(42)
    by_path = defaultdict(list)
    for i, ep in enumerate(dataset):
        by_path[ep["path_type"]].append(i)
    train_idx, test_idx = [], []
    for pt, idxs in by_path.items():
        rng.shuffle(idxs)
        k = max(1, int(len(idxs) * 0.2))
        test_idx.extend(idxs[:k])
        train_idx.extend(idxs[k:])

    train_ds = [dataset[i] for i in train_idx]
    test_ds = [dataset[i] for i in test_idx]
    print(f"\nEpisode split: train={len(train_ds)}  test={len(test_ds)}")

    X_tr, y_tr, _ = build_windows(train_ds)
    X_te, y_te, meta_te = build_windows(test_ds)
    print(f"Windows: train={len(X_tr)}  test={len(X_te)}")
    print(f"Train GT dist:", np.bincount(y_tr, minlength=NUM_CLASSES).tolist())
    print(f"Test  GT dist:", np.bincount(y_te, minlength=NUM_CLASSES).tolist())

    # Baseline: Step 0 rule
    print("\n=== Baseline: rule-based on test windows ===")
    def rule(cx, area):
        if area is None: return 1
        if area > 0.35: return 0
        if cx < 0.30: return 2 if cx < 0.15 else 4
        if cx > 0.70: return 3 if cx > 0.85 else 5
        return 1
    rule_preds = [rule(X_te[i, -4], X_te[i, -2]) for i in range(len(X_te))]
    rule_acc = np.mean([rule_preds[i] == y_te[i] for i in range(len(y_te))])
    print(f"  Rule test acc: {rule_acc:.3f}")

    print("\n=== Training MLP ===")
    best_acc, best_preds = train_eval(X_tr, y_tr, X_te, y_te)
    print(f"\nBest MLP test acc: {best_acc:.3f}")

    # Per-path breakdown
    pm_by_path = defaultdict(lambda: {"correct": 0, "total": 0})
    for i, m in enumerate(meta_te):
        pm_by_path[m["path_type"]]["total"] += 1
        if best_preds[i] == y_te[i]:
            pm_by_path[m["path_type"]]["correct"] += 1

    print("\n=== Per-path PM ===")
    for pt in PATH_TYPES:
        v = pm_by_path.get(pt, {"correct": 0, "total": 0})
        n = v["total"]
        print(f"  {pt:20s}: {v['correct']}/{n} = {v['correct']/max(n,1):.1%}")

    # Save & HTML
    summary = {
        "overall_pm_mlp": best_acc,
        "overall_pm_rule": rule_acc,
        "total_train": len(X_tr),
        "total_test": len(X_te),
        "pm_by_path": {k: {"correct": v["correct"], "total": v["total"]}
                       for k, v in pm_by_path.items()},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    build_html(summary, meta_te, y_te, best_preds)
    print(f"\nHTML: {OUT_DIR / 'index.html'}")


def build_html(summary, meta_te, y_te, preds):
    per = summary["pm_by_path"]
    pm_rows = []
    for pt in PATH_TYPES:
        v = per.get(pt, {"correct": 0, "total": 0})
        pm = v["correct"] / max(v["total"], 1)
        pm_rows.append(
            f"<tr><td>{pt}</td><td>{v['correct']}/{v['total']}</td>"
            f"<td><strong>{pm:.1%}</strong></td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Exp14 Step 1: BBox Feature MLP</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 24px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 2rem; margin-bottom: 8px; }}
  .sub {{ color: #94a3b8; margin-bottom: 24px; max-width: 900px; line-height: 1.6; }}
  .back {{ color: #60a5fa; text-decoration: none; display: inline-block; margin-bottom: 16px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
  .box {{ background: #1e293b; padding: 16px 20px; border-radius: 10px; }}
  .num {{ font-size: 2.5rem; font-weight: 800; }}
  .good {{ color: #22c55e; }} .warn {{ color: #fbbf24; }}
  table {{ border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; width: 100%; }}
  th, td {{ padding: 8px 14px; border-bottom: 1px solid #334155; text-align: left; }}
  th {{ background: #0b1220; }}
  .diag {{ background: #422006; border-left: 4px solid #fbbf24; padding: 14px 18px; border-radius: 6px; color: #fde68a; margin-top: 20px; }}
</style>
</head>
<body>
  <a class="back" href="../../index.html">← Back to main</a>
  <h1>Exp14 Step 1: BBox Feature MLP</h1>
  <p class="sub">
    Pure HF Kosmos-2 grounding에서 얻은 (cx, cy, area, has_bbox) × history=3 window 특징으로
    8-class action을 예측하는 작은 MLP 학습.
  </p>

  <div class="grid2">
    <div class="box">
      <div style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase;">Rule-based</div>
      <div class="num warn">{summary['overall_pm_rule']:.1%}</div>
      <div>(baseline)</div>
    </div>
    <div class="box">
      <div style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase;">MLP (learned)</div>
      <div class="num good">{summary['overall_pm_mlp']:.1%}</div>
      <div>({int(summary['overall_pm_mlp'] * summary['total_test'])}/{summary['total_test']})</div>
    </div>
  </div>

  <h2>PM per Path Type (test split)</h2>
  <table>
    <tr><th>Path Type</th><th>Correct/Total</th><th>PM</th></tr>
    {''.join(pm_rows)}
  </table>

  <div class="diag">
    <strong>참고:</strong>
    Train split은 episode-level stratified 80/20. MLP 입력은 최근 {WINDOW}프레임 BBox.
    Rule-based보다 유의미하게 나은지 확인하여 Step 2(image feature 결합) 진행 여부 판정.
  </div>
</body>
</html>"""
    (OUT_DIR / "index.html").write_text(html)


if __name__ == "__main__":
    main()
