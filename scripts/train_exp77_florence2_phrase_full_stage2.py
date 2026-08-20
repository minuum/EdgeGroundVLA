#!/usr/bin/env python3
"""exp77 — Florence-2 완전 통합 Stage 2, phrase 그라운딩 버전 (2026-08-21).

배경: exp75(완전 통합, 2026-08-19)는 그라운더로 열린 질문 방식(OD∪DENSE 키워드
매칭, V6 재현율 미검증·0807 최선이라도 34.7%)을 썼다. 2026-08-20~21에 발견한
명시적 phrase 그라운딩(`<CAPTION_TO_PHRASE_GROUNDING>`+"gray basket")이
0807 재현율 84.96%, V6 사람 검증(n=97) 100%로 압도적으로 나은 것으로 확인되어
(CH69, `docs/v5/detector/florence2_phrase_grounding_0807.json`,
`scripts/label/serve_v6_phrase_grounding_verify.py` 라벨 결과), 이 새 그라운더
주석(`gen_v6_florence2_phrase_annotation.py`)으로 exp75를 다시 만든다.

바꾸는 것 (exp75와의 차이는 bbox 생성 방식 하나뿐, 나머지 전부 동일):
  베이스라인(exp73)    : bbox=OWL-v2                    · vis=Kosmos-2
  exp74               : bbox=OWL-v2                    · vis=Florence-2
  그라운더 스왑(구)     : bbox=Florence-2(열린질문)       · vis=Kosmos-2
  exp75(구 완전통합)    : bbox=Florence-2(열린질문)       · vis=Florence-2
  exp76(신 그라운더 스왑): bbox=Florence-2(phrase)        · vis=Kosmos-2
  exp77(이 스크립트)    : bbox=Florence-2(phrase)        · vis=Florence-2   ← 완전 통합, 최신 방법

재사용(재인코딩 없음, 캐시 전부 기존 자산):
  1024d 원시 특징      : docs/v5/detector/stage1_florence2_feats.npz (Stage1에서 생성)
  bbox 주석            : docs/v5/bbox_nav_florence2/bbox_dataset_v6_florence2_phrase.json (2026-08-21 신규)
  image_proj 가중치    : runs/v5_nav/mlp/stage1_florence2_5cls/stage1_florence2_5cls_projs.pt

비교 기준:
  exp73(베이스라인)          val_acc mean 73.87%±0.20%p · best 74.13%
  exp74(비전만)              val_acc mean 75.15%±0.09%p · best 75.24%
  그라운더 스왑(구, bbox만)   val_acc mean 73.26%±0.29%p · best 73.59% (L/ROT_L 회귀)
  exp75(구 완전 통합)         val_acc mean 73.52%±0.25%p · best 73.84%

주의: 실기 검증은 별건이다. 확정 발견 6번 — val 지표와 실기 성능은 직결되지 않음.

출력:
  docs/v5/closed_loop_eval/exp77_florence2_phrase_full_vis_cache.pt
  runs/v5_nav/mlp/exp77_florence2_phrase_full/exp77_florence2_phrase_full_v6_mlp.pt
  docs/v5/closed_loop_eval/exp77_florence2_phrase_full_stage2.json
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
ANN_FLORENCE = ROOT / "docs/v5/bbox_nav_florence2/bbox_dataset_v6_florence2_phrase.json"
FEAT_NPZ    = ROOT / "docs/v5/detector/stage1_florence2_feats.npz"
CACHE_OUT   = ROOT / "docs/v5/closed_loop_eval/exp77_florence2_phrase_full_vis_cache.pt"
OUT_DIR     = ROOT / "runs/v5_nav/mlp/exp77_florence2_phrase_full"
REPORT      = ROOT / "docs/v5/closed_loop_eval/exp77_florence2_phrase_full_stage2.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MEAN, BASE_STD, BASE_BEST = 0.7386535207430521, 0.002024509914618209, 0.7412587404251099
EXP74_MEAN, EXP74_BEST = 0.7515, 0.7524
GROUNDER_SWAP_MEAN, GROUNDER_SWAP_BEST = 0.7326203385988871, 0.7359111905097961
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
SEEDS = [0, 1, 2]


def build_cache():
    if CACHE_OUT.exists():
        eps = torch.load(str(CACHE_OUT), weights_only=False)
        print(f"[CACHE] 재사용 {CACHE_OUT.name} ({len(eps)}ep)")
        return eps

    ck = torch.load(str(STAGE1_F2), map_location="cpu", weights_only=False)
    proj = nn.Linear(VIS_DIM, PROJ_DIM)
    proj.load_state_dict(ck["image_proj"])
    proj = proj.to(DEVICE).eval()
    print(f"[STAGE1] image_proj 로드 (Florence-2 백본, Stage1 val_acc {ck['val_acc']:.4f})")

    pre_arr, pre_idx = None, {}
    if FEAT_NPZ.exists():
        z = np.load(FEAT_NPZ, allow_pickle=True)
        pre_arr = np.asarray(z["feats"])                  # 1회 실체화 (OOM 방지, exp74와 동일 패턴)
        keys = z["keys"]
        pre_idx = {(str(keys[i][0]), int(keys[i][1])): i for i in range(len(keys))}
        del z
        print(f"[FEAT] 1024d 캐시 재사용 가능 프레임 {len(pre_idx)} "
              f"({pre_arr.nbytes/1e6:.0f}MB 상주)")

    ann = json.loads(ANN_FLORENCE.read_text())
    enc = None
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

        # ★ exp74와의 유일한 차이: bbox를 Florence-2 그라운더 주석에서 가져옴
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
        print(f"  [exp77/mlp] seed={seed} val_acc={acc*100:.2f}%", flush=True)

    mean, std = float(np.mean(accs)), float(np.std(accs))

    ckpt = OUT_DIR / "exp77_florence2_phrase_full_v6_mlp.pt"
    torch.save({"model": best_state, "val_acc": best_overall, "head": "mlp",
                "window": WINDOW, "bbox_scale": BBOX_SCALE, "exp": "exp77",
                "backbone": FLOR, "grounder": "florence2", "stage1": str(STAGE1_F2)}, str(ckpt))

    rep = {
        "exp": "exp77", "backbone": FLOR, "grounder": "florence2 (CAPTION_TO_PHRASE_GROUNDING + \"gray basket\")",
        "note": "그라운더+비전인코더+프로젝션 셋 다 Florence-2 계열로 교체한 완전 통합 버전",
        "val_acc_mean": mean, "val_acc_std": std, "val_acc_best": best_overall,
        "seeds": accs,
        "baseline_exp73_val_acc_mean": BASE_MEAN, "baseline_exp73_val_acc_best": BASE_BEST,
        "exp74_vis_only_val_acc_mean": EXP74_MEAN, "exp74_vis_only_val_acc_best": EXP74_BEST,
        "grounder_swap_only_val_acc_mean": GROUNDER_SWAP_MEAN, "grounder_swap_only_val_acc_best": GROUNDER_SWAP_BEST,
        "delta_vs_exp73_mean": mean - BASE_MEAN, "delta_vs_exp73_best": best_overall - BASE_BEST,
        "per_class_best": {CLASS_NAMES[c]: v for c, v in (best_per_class or {}).items()},
        "n_train_windows": int(len(X_tr)), "n_val_windows": int(len(X_va)),
        "window": WINDOW, "bbox_scale": BBOX_SCALE, "epochs": 300,
        "checkpoint": str(ckpt), "cache": str(CACHE_OUT),
        "elapsed_min": (time.time() - t_start) / 60,
        "caveat": ("실기 검증 별건. 확정 발견 6번 — val 지표와 실기 성능은 직결되지 않음. "
                   "또한 그라운더 스왑 단독 실험에서 L/ROT_L 클래스 회귀가 확인됐으므로 "
                   "완전 통합 버전에서도 클래스별 분해가 필요."),
    }
    REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("  exp77 — Florence-2 완전 통합, phrase 그라운딩 (그라운더+비전인코더+프로젝션)")
    print(f"  val_acc  mean {mean*100:.2f}% ± {std*100:.2f}%p   best {best_overall*100:.2f}%")
    print(f"  exp73(베이스라인)         mean {BASE_MEAN*100:.2f}%   best {BASE_BEST*100:.2f}%")
    print(f"  exp74(비전만 교체)        mean {EXP74_MEAN*100:.2f}%   best {EXP74_BEST*100:.2f}%")
    print(f"  그라운더 스왑(bbox만)     mean {GROUNDER_SWAP_MEAN*100:.2f}%   best {GROUNDER_SWAP_BEST*100:.2f}%")
    print(f"  차이(vs exp73)  mean {(mean-BASE_MEAN)*100:+.2f}%p   best {(best_overall-BASE_BEST)*100:+.2f}%p")
    print("=" * 70)
    if best_per_class:
        print("\n클래스별 정확도 (best seed):")
        for c, v in sorted(best_per_class.items()):
            print(f"  {CLASS_NAMES[c]:>7s}  {v*100:5.1f}%")


if __name__ == "__main__":
    main()
