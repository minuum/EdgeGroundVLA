#!/usr/bin/env python3
"""exp77 미러 증강 재학습 — Florence-2(phrase bbox+vis) 완전 통합판에 좌우 미러 쌍 추가.

배경 (CH69-7/69-9):
  exp77(Florence-2 bbox+vis)는 무작위 split에서는 exp73(OWL bbox+Kosmos-2 vis)을
  이겼지만(75.65% vs 74.13%), leave-one-direction-out(완전히 처음 보는 방향)에서는
  거꾸로 진다(54.0% vs 60.85%) — 특히 우측(weak_right/strong_right)에서 F→FR 오판이
  exp73보다 훨씬 심하게 증폭된다(69-9: 53.3%/79.8% vs exp73 7.6%/36.8%).

이 실험 (exp73_mirror_aug_train.py와 동일한 방법론을 Florence-2 파이프라인에 포팅):
  모든 학습 윈도에 좌우 미러 쌍을 추가해 좌우 대칭을 강제한다.
    · 비전 피처: 이미지 좌우 반전 후 Florence-2 vision_tower로 재추출 (1024d→proj 256d)
    · bbox: cx → 1 − cx (cy·area·has_bbox 불변)
    · 클래스 라벨: LEFT↔RIGHT, FWD+L↔FWD+R, ROT_L↔ROT_R 스왑
    · path_type(방향·커브 둘 다): weak_left↔weak_right, strong_left↔strong_right,
      left_curve↔right_curve (center/straight는 그대로)
  실기 데이터 추가 없이, 기존 좌측 에피소드의 거울상으로 "가상의 우측 데이터"를 만들어
  leave-one-direction-out 우측 결함이 완화되는지 오프라인으로 먼저 검증한다
  (Phase 1 — 10월 계획, EGVLA 2/Florence 우측 보강 + FM 연동 방향의 1단계).

재사용: exp77 빌드 캐시(exp77_florence2_phrase_full_vis_cache.pt)를 그대로 쓰지 않고
  ANN_FLORENCE에서 다시 프레임을 뽑아 미러 캐시를 새로 만든다(미러 이미지는 원본
  캐시에 없으므로 재인코딩 필수). 원본(비미러) 캐시는 그대로 재사용.

출력:
  docs/v5/closed_loop_eval/exp77_mirror_vis_cache.pt   (미러 전용 캐시)
  docs/v5/closed_loop_eval/exp77_mirror_aug.json
  runs/v5_nav/mlp/exp77_florence2_phrase_full/exp77_florence2_phrase_full_v6_mlp_mirroraug_seed{0,1,2}.pt

Usage:
  .venv/bin/python3 scripts/exp77_mirror_aug_train.py
  .venv/bin/python3 scripts/exp77_mirror_aug_train.py --loo   # leave-one-direction-out로도 재검증
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
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import (          # noqa: E402
    MLPActionHead, build_windows, train_one,
    WINDOW, BBOX_SCALE, NUM_CLASSES, VIS_DIM, PROJ_DIM,
    VAL_RATIO, SPLIT_SEED, DEVICE,
)
from scripts.train_exp77_florence2_phrase_full_stage2 import (  # noqa: E402
    FLOR, STAGE1_F2, ANN_FLORENCE, CACHE_OUT, OUT_DIR,
)

MIRROR_CACHE = ROOT / "docs/v5/closed_loop_eval/exp77_mirror_vis_cache.pt"
OUT_JSON = ROOT / "docs/v5/closed_loop_eval/exp77_mirror_aug.json"
LOO_OUT = ROOT / "docs/v5/closed_loop_eval/exp77_mirroraug_leave_one_direction_out.json"
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
MIRROR_CLS = [0, 1, 3, 2, 5, 4, 7, 6]      # STOP FWD | LEFT↔RIGHT | FWD+L↔FWD+R | ROT_L↔ROT_R
DIRECTIONS = ["center", "weak_left", "weak_right", "strong_left", "strong_right"]
SEEDS = [0, 1, 2]

DIR_MIRROR = {
    "center": "center", "weak_left": "weak_right", "weak_right": "weak_left",
    "strong_left": "strong_right", "strong_right": "strong_left",
}
CURVE_MIRROR = {"left_curve": "right_curve", "right_curve": "left_curve", "straight": "straight"}


def mirror_path_type(path_type):
    for d in sorted(DIR_MIRROR, key=len, reverse=True):
        if path_type.startswith(d + "_"):
            curve = path_type[len(d) + 1:]
            return f"{DIR_MIRROR[d]}_{CURVE_MIRROR.get(curve, curve)}"
    return path_type


def build_mirror_cache():
    if MIRROR_CACHE.exists():
        eps = torch.load(str(MIRROR_CACHE), weights_only=False)
        print(f"[CACHE] 미러 캐시 재사용: {MIRROR_CACHE.name} ({len(eps)}ep)")
        return eps

    from transformers import AutoModelForCausalLM, AutoProcessor

    ck = torch.load(str(STAGE1_F2), map_location="cpu", weights_only=False)
    proj = nn.Linear(VIS_DIM, PROJ_DIM)
    proj.load_state_dict(ck["image_proj"])
    proj = proj.to(DEVICE).eval()

    print("[FLORENCE-2] 미러 이미지 재인코딩용 vision_tower 로드...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        FLOR, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE).eval()
    enc = m.vision_tower
    proc = AutoProcessor.from_pretrained(FLOR, trust_remote_code=True)

    ann = json.loads(ANN_FLORENCE.read_text())
    episodes = []
    t0 = time.time()
    for i, ep in enumerate(ann):
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            continue
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        if not frames:
            continue

        with h5py.File(str(h5_path), "r") as f:
            imgs = f["observations"]["images"] if "observations" in f else f["images"]
            acts_np = f["actions"][:] if "actions" in f else None
            raw = np.zeros((len(frames), VIS_DIM), dtype=np.float32)
            for j, fr in enumerate(frames):
                im = np.array(imgs[fr["frame_idx"]])[:, :, ::-1].astype("uint8")  # BGR→RGB
                pil = Image.fromarray(im).transpose(Image.FLIP_LEFT_RIGHT)
                pv = proc(images=pil, text="<OD>",
                          return_tensors="pt")["pixel_values"].to(DEVICE, torch.float16)
                with torch.no_grad():
                    o = enc.forward_features_unpool(pv)[0]
                raw[j] = o.mean(0).float().cpu().numpy()

        with torch.no_grad():
            vis = F.normalize(proj(torch.tensor(raw, device=DEVICE)), dim=-1).cpu()

        bboxes = [(1.0 - fr.get("cx_det", 0.5), fr.get("cy_det", 0.5),
                   fr.get("area_det", 0.05), float(fr.get("has_bbox", False)))
                  for fr in frames]
        gts = [MIRROR_CLS[fr["gt_class"]] for fr in frames]
        acts = ([tuple(float(v) for v in acts_np[fr["frame_idx"]]) for fr in frames]
                if acts_np is not None else None)
        episodes.append({"stem": h5_path.stem + "_mirror", "path_type": mirror_path_type(ep["path_type"]),
                         "bboxes": bboxes, "vis": vis, "gts": gts, "acts": acts})
        if (i + 1) % 40 == 0:
            print(f"  미러 인코딩 {i+1}/{len(ann)}ep ({time.time()-t0:.0f}s)", flush=True)

    torch.save(episodes, str(MIRROR_CACHE))
    print(f"[CACHE] 저장 {MIRROR_CACHE.name} — {len(episodes)}ep · {time.time()-t0:.0f}s")
    return episodes


def val_split(n):
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    nv = max(1, int(n * VAL_RATIO))
    return set(idx[:nv])


def run_random_split():
    base = torch.load(str(CACHE_OUT), weights_only=False)
    mir = build_mirror_cache()
    assert len(base) == len(mir), f"원본/미러 에피소드 수 불일치: {len(base)} vs {len(mir)}"
    val_idx = val_split(len(base))

    tr_o = [e for i, e in enumerate(base) if i not in val_idx]
    va_o = [e for i, e in enumerate(base) if i in val_idx]
    tr_m = [e for i, e in enumerate(mir) if i not in val_idx]
    # 평가는 원본 val만 사용(미러 val은 cx→1-cx라 평가 자체가 다른 분포 비교가 됨) —
    # exp73_mirror_aug_train.py와 동일 원칙.
    print(f"train {len(tr_o)}ep(+미러 {len(tr_m)}ep) / val {len(va_o)}ep (원본만)")

    Xo, yo, _ = build_windows(tr_o)
    Xm, ym, _ = build_windows(tr_m)
    Xva, yva, _ = build_windows(va_o)
    Xtr = np.concatenate([Xo, Xm]); ytr = np.concatenate([yo, ym])
    cL = int(sum((ytr == c).sum() for c in (2, 4, 6)))
    cR = int(sum((ytr == c).sum() for c in (3, 5, 7)))
    print(f"학습 윈도: 원본 {len(Xo)} + 미러 {len(Xm)} = {len(Xtr)} / val {len(Xva)}")
    print(f"학습 라벨 좌계열 {cL} vs 우계열 {cR} → 차 {abs(cL-cR)} (0이면 완전 대칭)")

    res = []
    for seed in SEEDS:
        acc, state, per_class = train_one(MLPActionHead, Xtr, ytr, Xva, yva, seed)
        p = OUT_DIR / f"exp77_florence2_phrase_full_v6_mlp_mirroraug_seed{seed}.pt"
        torch.save({"model": state, "val_acc": acc, "head": "mlp", "window": WINDOW,
                    "bbox_scale": BBOX_SCALE, "exp": "exp77_mirroraug",
                    "backbone": FLOR, "grounder": "florence2"}, str(p))
        print(f"  seed{seed}: val_acc {acc*100:.2f}%  → {p.name}")
        res.append(dict(seed=seed, val_acc=float(acc),
                         per_class={CLASS_NAMES[c]: v for c, v in (per_class or {}).items()}))

    accs = [r["val_acc"] for r in res]
    OUT_JSON.write_text(json.dumps(dict(
        seeds=res, val_acc_mean=float(np.mean(accs)), val_acc_std=float(np.std(accs)),
        val_acc_best=float(max(accs)),
        baseline_exp77_mean=0.7565, baseline_exp77_best=0.7565,
        train_label_L=cL, train_label_R=cR,
        n_train_windows_orig=int(len(Xo)), n_train_windows_mirror=int(len(Xm))),
        indent=2, ensure_ascii=False))
    print(f"\n무작위 split 결과 — val_acc {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%p "
          f"(exp77 원본 75.65% 대비)")
    print(f"저장: {OUT_JSON}")


def run_leave_one_direction_out():
    """69-6/69-7과 동일 방법론 — 미러 증강이 '처음 보는 방향'에서도 통하는지가 진짜 검증."""
    base = torch.load(str(CACHE_OUT), weights_only=False)
    mir = build_mirror_cache()
    for e in base:
        e["direction"] = _get_direction(e["path_type"])
    for e in mir:
        e["direction"] = _get_direction(e["path_type"])

    results = {}
    for held in DIRECTIONS:
        va_eps = [e for e in base if e["direction"] == held]
        tr_eps = [e for e in base if e["direction"] != held]
        # 미러 증강 중 held와 같은 방향이 "되는" 것들은 제외해야 held-out 취지가 유지된다
        # (예: held=strong_right면 strong_left의 미러=strong_right가 되므로 이것도 빼야 함).
        tr_mir = [e for e in mir if e["direction"] != held]
        X_tr_o, y_tr_o, _ = build_windows(tr_eps)
        X_tr_m, y_tr_m, _ = build_windows(tr_mir)
        X_tr = np.concatenate([X_tr_o, X_tr_m]); y_tr = np.concatenate([y_tr_o, y_tr_m])
        X_va, y_va, _ = build_windows(va_eps)

        accs, best_overall, best_per_class = [], 0.0, None
        for seed in SEEDS:
            acc, state, per_class = train_one(MLPActionHead, X_tr, y_tr, X_va, y_va, seed)
            accs.append(acc)
            if acc > best_overall:
                best_overall, best_per_class = acc, per_class

        r_acc = best_per_class.get(3)
        fr_acc = best_per_class.get(5)
        results[held] = dict(
            n_train_ep=len(tr_eps) + len(tr_mir), n_val_ep=len(va_eps),
            n_train_windows=int(len(X_tr)), n_val_windows=int(len(X_va)),
            val_acc_mean=float(np.mean(accs)), val_acc_std=float(np.std(accs)),
            val_acc_best=float(best_overall),
            per_class_best={CLASS_NAMES[c]: v for c, v in best_per_class.items()},
            r_class_acc=r_acc, fr_class_acc=fr_acc,
        )
        print(f"[{held:14s}] held-out val={len(va_eps)}ep({len(X_va)}win) "
              f"acc mean={np.mean(accs)*100:.2f}% best={best_overall*100:.2f}% "
              f"R={r_acc*100 if r_acc else float('nan'):.1f}% "
              f"FR={fr_acc*100 if fr_acc else float('nan'):.1f}%", flush=True)

    LOO_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    overall_mean = np.mean([r["val_acc_best"] for r in results.values()])
    print(f"\n5개 방향 평균 held-out best acc: {overall_mean*100:.2f}% "
          f"(exp77 원본 54.00% / exp73 60.85% 대비)")
    print(f"저장: {LOO_OUT}")


def _get_direction(path_type):
    for d in sorted(DIRECTIONS, key=len, reverse=True):
        if path_type.startswith(d + "_"):
            return d
    return "center"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loo", action="store_true", help="leave-one-direction-out로 재검증")
    a = ap.parse_args()
    if a.loo:
        run_leave_one_direction_out()
    else:
        run_random_split()


if __name__ == "__main__":
    main()
