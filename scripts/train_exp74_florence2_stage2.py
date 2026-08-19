#!/usr/bin/env python3
"""exp74 — Florence-2 백본 위에서 Stage 2 행동 헤드 재학습 (2026-08-17, 계획서 2''-b 전반부).

목적:
  Stage 1에서 Florence-2 백본이 Kosmos-2를 이겼다(val_acc 94.92% vs 94.09%,
  최저클래스 +5.5%p, 편차 −8.1%p). 그 image_proj 위에 **동일한 Stage 2 헤드**를 올려
  8-class 행동 정확도를 베이스라인과 비교한다.

바꾸는 것 (딱 하나):
  비전 캐시의 `vis` 채널 출처
    베이스라인: Kosmos-2 vision → image_proj(stage1_v3_5cls_owl_projs.pt)      → 256d
    이번:       Florence-2 vision → image_proj(stage1_florence2_5cls_projs.pt) → 256d

바꾸지 않는 것:
  · 헤드 구조·학습 루프: train_exp73_stage1v3_heads.py의 MLPActionHead / train_one 직접 import
  · bbox 주석: bbox_dataset_v6_owl.json (동일)
  · window=6, bbox_scale=3.0, 300 epoch, seeds 0/1/2, 분할 seed 42 — 전부 동일

베이스라인 (exp73_stage1v3_trackA_heads.json · owl_stage1v3/v6/mlp):
  val_acc mean 0.7387 ± 0.0020 · best 0.7413

주의: 실기 검증은 별건이다. 이 프로젝트의 확정 발견 6번 — "val 지표와 실기 성능은
직결되지 않음"(val 74.1% 헤드가 실기 95/100) — 이므로 여기 결과로 실기 우세를 단정하지 않는다.

출력:
  docs/v5/closed_loop_eval/exp74_florence2_vis_cache.pt
  runs/v5_nav/mlp/exp74_florence2/exp74_florence2_v6_mlp.pt
  docs/v5/closed_loop_eval/exp74_florence2_stage2.json
"""
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

FLOR        = "microsoft/Florence-2-base"
STAGE1_F2   = ROOT / "runs/v5_nav/mlp/stage1_florence2_5cls/stage1_florence2_5cls_projs.pt"
ANN_OWL     = ROOT / "docs/v5/bbox_nav_owl/bbox_dataset_v6_owl.json"
FEAT_NPZ    = ROOT / "docs/v5/detector/stage1_florence2_feats.npz"   # 1024d 재사용
CACHE_OUT   = ROOT / "docs/v5/closed_loop_eval/exp74_florence2_vis_cache.pt"
OUT_DIR     = ROOT / "runs/v5_nav/mlp/exp74_florence2"
REPORT      = ROOT / "docs/v5/closed_loop_eval/exp74_florence2_stage2.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MEAN, BASE_STD, BASE_BEST = 0.7386535207430521, 0.002024509914618209, 0.7412587404251099
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
SEEDS = [0, 1, 2]


def build_cache():
    """Florence-2 vision → 새 image_proj → 256d 로 Stage 2 캐시 구성.

    1024d 피처는 Stage 1에서 만든 npz를 재사용하고, 거기 없는 프레임만 새로 인코딩한다.
    (Stage 1은 `consistent & label` 필터, Stage 2는 `gt_class` 필터 — 집합이 다를 수 있으므로
     가정하지 않고 키 단위로 조회한다.)
    """
    if CACHE_OUT.exists():
        eps = torch.load(str(CACHE_OUT), weights_only=False)
        print(f"[CACHE] 재사용 {CACHE_OUT.name} ({len(eps)}ep)")
        return eps

    ck = torch.load(str(STAGE1_F2), map_location="cpu", weights_only=False)
    proj = nn.Linear(VIS_DIM, PROJ_DIM)
    proj.load_state_dict(ck["image_proj"])
    proj = proj.to(DEVICE).eval()
    print(f"[STAGE1] image_proj 로드 (Florence-2 백본, Stage1 val_acc {ck['val_acc']:.4f})")

    # ⚠️ npz는 lazy 접근이라 z["feats"]를 반복문 안에서 쓰면 매번 전체 배열을 다시
    # 압축 해제하고, 각 행 슬라이스가 부모 배열을 붙잡아 메모리가 폭발한다(122GB OOM 경험).
    # 배열은 **한 번만** 실체화하고 dict에는 행 인덱스만 담는다.
    pre_arr, pre_idx = None, {}
    if FEAT_NPZ.exists():
        z = np.load(FEAT_NPZ, allow_pickle=True)
        pre_arr = np.asarray(z["feats"])                  # 1회 실체화
        keys = z["keys"]
        pre_idx = {(str(keys[i][0]), int(keys[i][1])): i for i in range(len(keys))}
        del z
        print(f"[FEAT] 1024d 캐시 재사용 가능 프레임 {len(pre_idx)} "
              f"({pre_arr.nbytes/1e6:.0f}MB 상주)")

    ann = json.loads(ANN_OWL.read_text())
    enc = None   # 필요할 때만 Florence-2 로드
    proc = None
    episodes, n_hit, n_miss = [], 0, 0
    t0 = time.time()

    for i, ep in enumerate(ann):
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            continue
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        if not frames:
            continue

        need = [fr for fr in frames if (str(h5_path), int(fr["frame_idx"])) not in pre_idx]
        raw = np.zeros((len(frames), VIS_DIM), dtype=np.float32)

        if need:
            if enc is None:
                from transformers import AutoModelForCausalLM, AutoProcessor
                print(f"[FLORENCE-2] 미보유 프레임 {len(need)}건 → 인코더 로드", flush=True)
                m = AutoModelForCausalLM.from_pretrained(
                    FLOR, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE).eval()
                enc = m.vision_tower
                proc = AutoProcessor.from_pretrained(FLOR, trust_remote_code=True)

        with h5py.File(str(h5_path), "r") as f:
            imgs = f["observations"]["images"] if "observations" in f else f["images"]
            acts_np = f["actions"][:] if "actions" in f else None
            for j, fr in enumerate(frames):
                key = (str(h5_path), int(fr["frame_idx"]))
                if key in pre_idx:
                    raw[j] = pre_arr[pre_idx[key]].astype(np.float32); n_hit += 1
                else:
                    im = np.array(imgs[fr["frame_idx"]])[:, :, ::-1].astype("uint8")
                    pv = proc(images=Image.fromarray(im), text="<OD>",
                              return_tensors="pt")["pixel_values"].to(DEVICE, torch.float16)
                    with torch.no_grad():
                        o = enc.forward_features_unpool(pv)[0]
                    raw[j] = o.mean(0).float().cpu().numpy(); n_miss += 1

        with torch.no_grad():
            vis = F.normalize(proj(torch.tensor(raw, device=DEVICE)), dim=-1).cpu()

        bboxes = [(fr.get("cx_det", 0.5), fr.get("cy_det", 0.5),
                   fr.get("area_det", 0.05), float(fr.get("has_bbox", False)))
                  for fr in frames]
        gts = [fr["gt_class"] for fr in frames]
        acts = ([tuple(float(v) for v in acts_np[fr["frame_idx"]]) for fr in frames]
                if acts_np is not None else None)
        episodes.append({"stem": h5_path.stem, "path_type": ep["path_type"],
                         "bboxes": bboxes, "vis": vis, "gts": gts, "acts": acts})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(ann)}ep  ({time.time()-t0:.0f}s)", flush=True)

    torch.save(episodes, str(CACHE_OUT))
    print(f"[CACHE] 저장 {CACHE_OUT.name} — {len(episodes)}ep · "
          f"1024d 재사용 {n_hit} / 신규인코딩 {n_miss} · {time.time()-t0:.0f}s")
    return episodes


def main():
    t_start = time.time()
    eps = build_cache()

    # 분할: 베이스라인과 동일
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(eps)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO))
    val_eps = [eps[i] for i in idx[:n_val]]
    tr_eps = [eps[i] for i in idx[n_val:]]
    print(f"[SPLIT] train={len(tr_eps)} / val={len(val_eps)}ep")

    X_tr, y_tr, _ = build_windows(tr_eps)
    X_va, y_va, _ = build_windows(val_eps)
    print(f"[WINDOW] train {X_tr.shape} / val {X_va.shape} "
          f"(window={WINDOW}, bbox_scale={BBOX_SCALE})")

    accs, best_overall, best_state, best_per_class = [], 0.0, None, None
    for seed in SEEDS:
        acc, state, per_class = train_one(MLPActionHead, X_tr, y_tr, X_va, y_va, seed)
        accs.append(acc)
        if acc > best_overall:
            best_overall, best_state, best_per_class = acc, state, per_class
        print(f"  [exp74/mlp] seed={seed} val_acc={acc*100:.2f}%", flush=True)

    mean, std = float(np.mean(accs)), float(np.std(accs))
    d_mean, d_best = mean - BASE_MEAN, best_overall - BASE_BEST

    ckpt = OUT_DIR / "exp74_florence2_v6_mlp.pt"
    torch.save({"model": best_state, "val_acc": best_overall, "head": "mlp",
                "window": WINDOW, "bbox_scale": BBOX_SCALE, "exp": "exp74",
                "backbone": FLOR, "stage1": str(STAGE1_F2)}, str(ckpt))

    rep = {
        "exp": "exp74", "backbone": FLOR,
        "baseline_exp": "exp73 (owl_stage1v3/v6/mlp)",
        "baseline_backbone": "kosmos-2-patch14-224",
        "val_acc_mean": mean, "val_acc_std": std, "val_acc_best": best_overall,
        "seeds": accs,
        "baseline_val_acc_mean": BASE_MEAN, "baseline_val_acc_std": BASE_STD,
        "baseline_val_acc_best": BASE_BEST,
        "delta_mean": d_mean, "delta_best": d_best,
        "per_class_best": {CLASS_NAMES[c]: v for c, v in (best_per_class or {}).items()},
        "n_train_windows": int(len(X_tr)), "n_val_windows": int(len(X_va)),
        "window": WINDOW, "bbox_scale": BBOX_SCALE, "epochs": 300,
        "checkpoint": str(ckpt), "cache": str(CACHE_OUT),
        "elapsed_min": (time.time() - t_start) / 60,
        "caveat": ("실기 검증 별건. 확정 발견 6번 — val 지표와 실기 성능은 직결되지 않음"
                   "(val 74.1% 헤드가 실기 95/100) — 이므로 이 수치로 실기 우세를 단정하지 않는다."),
    }
    REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    print("\n" + "=" * 66)
    print("  exp74 — Florence-2 백본 + 동일 Stage 2 MLP 헤드")
    print(f"  val_acc  mean {mean*100:.2f}% ± {std*100:.2f}%p   best {best_overall*100:.2f}%")
    print(f"  베이스라인(exp73/Kosmos-2)  mean {BASE_MEAN*100:.2f}% ± {BASE_STD*100:.2f}%p   "
          f"best {BASE_BEST*100:.2f}%")
    print(f"  차이  mean {d_mean*100:+.2f}%p   best {d_best*100:+.2f}%p")
    print("=" * 66)
    if best_per_class:
        print("\n클래스별 정확도 (best seed):")
        for c, v in sorted(best_per_class.items()):
            print(f"  {CLASS_NAMES[c]:>7s}  {v*100:5.1f}%")
    print(f"\n[SAVE] {ckpt}")
    print(f"[SAVE] {REPORT}")
    print(f"소요: {(time.time()-t_start)/60:.1f}분")


if __name__ == "__main__":
    main()
