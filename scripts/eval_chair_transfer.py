#!/usr/bin/env python3
"""CH68 68-6 우선순위 3번 — 객체 전이 실험 (기존 chair 28ep, 신규 수집 없음).

docs/plans/plan_20260830_ch68_language_conditioning_l0l4.md 3단계.
질문: basket 전용으로 학습된 액션 헤드(exp77, Florence-2 phrase grounding +
Florence-2 vision)가 chair 장면에 얼마나 전이되는가?

- chair 28ep(`ROS_action/mobile_vla_dataset_v5_2/`, instruction="Navigate until
  the chair is centered..." — 진짜 타겟 접근 궤적, 장애물 아님. 라벨 확인 완료)
  에 Florence-2 phrase grounding(phrase="chair")으로 새로 bbox 주석 생성
  (기존 docs/v5/bbox_chair/의 PG2 기반 주석은 그라운더가 달라 재사용 안 함 —
  exp77과 apples-to-apples 비교하려면 같은 그라운더로 다시 떠야 함).
- ① zero-shot: exp77 mlp 체크포인트(재학습 없이) 그대로 chair에 추론
- ② few-shot: chair만으로 처음부터 소규모 재학습, zero-shot과 비교

출력:
  docs/v5/bbox_frame_level/bbox_dataset_chair_florence2_phrase.json (신규 주석)
  docs/v5/closed_loop_eval/chair_transfer_eval.json
"""
import glob
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

from scripts.train_exp73_stage1v3_heads import (
    MLPActionHead, build_windows, train_one, cont_to_class_t,
    WINDOW, BBOX_SCALE, VAL_RATIO, SPLIT_SEED, DEVICE, VIS_DIM, PROJ_DIM,
)

CHAIR_DIR = ROOT / "ROS_action/mobile_vla_dataset_v5_2"
STAGE1_F2 = ROOT / "runs/v5_nav/mlp/stage1_florence2_5cls/stage1_florence2_5cls_projs.pt"
EXP77_CKPT = ROOT / "runs/v5_nav/mlp/exp77_florence2_phrase_full/exp77_florence2_phrase_full_v6_mlp.pt"
ANN_OUT = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_chair_florence2_phrase.json"
CACHE_OUT = ROOT / "docs/v5/closed_loop_eval/chair_transfer_vis_cache.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/chair_transfer_eval.json"
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
FLOR = "microsoft/Florence-2-base"
PHRASE_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"


def ground_and_encode():
    """chair 28ep 전부에 Florence-2 phrase grounding(phrase="chair") + vision
    인코딩을 새로 실행 — exp77과 동일 규격(cx,cy,area,has_bbox / 256d vis)."""
    if CACHE_OUT.exists():
        print(f"[CACHE] 재사용 {CACHE_OUT.name}")
        return torch.load(str(CACHE_OUT), weights_only=False)

    print("[Florence-2] 로딩...", flush=True)
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(FLOR, trust_remote_code=True,
                                                  torch_dtype=torch.float16).to(DEVICE).eval()
    proc = AutoProcessor.from_pretrained(FLOR, trust_remote_code=True)

    ck = torch.load(str(STAGE1_F2), map_location="cpu", weights_only=False)
    proj = nn.Linear(VIS_DIM, PROJ_DIM)
    proj.load_state_dict(ck["image_proj"])
    proj = proj.to(DEVICE).eval()

    files = sorted(glob.glob(str(CHAIR_DIR / "*chair*.h5")))
    print(f"[DATA] chair 파일 {len(files)}개", flush=True)

    ann = []
    episodes = []
    for fi, fpath in enumerate(files):
        with h5py.File(fpath, "r") as h:
            imgs = h["observations"]["images"][:]
            acts = h["actions"][:]
        n = len(imgs)
        bboxes, raw_feats = [], []
        for t in range(n):
            pil = Image.fromarray(imgs[t].astype(np.uint8)).convert("RGB")
            W, H = pil.width, pil.height
            text = PHRASE_TASK + "chair"
            inp = proc(text=text, images=pil, return_tensors="pt")
            with torch.no_grad():
                ids = model.generate(
                    input_ids=inp["input_ids"].to(DEVICE),
                    pixel_values=inp["pixel_values"].to(DEVICE, torch.float16),
                    max_new_tokens=128, num_beams=3, do_sample=False)
            txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = proc.post_process_generation(txt, task=PHRASE_TASK, image_size=(W, H))[PHRASE_TASK]
            boxes = parsed.get("bboxes", []) or []
            if boxes:
                x1, y1, x2, y2 = boxes[0]
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                area = (x2 - x1) * (y2 - y1) / (W * H)
                has_bbox = True
            else:
                cx, cy, area, has_bbox = 0.5, 0.5, 0.05, False
            bboxes.append((cx, cy, area, float(has_bbox)))

            # vision tower raw 1024d feature(exp77과 동일 경로: <OD> 프롬프트로 vision만)
            pv = proc(images=pil, text="<OD>", return_tensors="pt")["pixel_values"].to(DEVICE, torch.float16)
            with torch.no_grad():
                o = model.vision_tower.forward_features_unpool(pv)[0]
            raw_feats.append(o.mean(0).float().cpu().numpy())

        raw = np.stack(raw_feats)
        with torch.no_grad():
            vis = F.normalize(proj(torch.tensor(raw, device=DEVICE)), dim=-1).cpu()
        gts = [int(cont_to_class_t(torch.tensor(acts[t:t+1], dtype=torch.float32)).item()) for t in range(n)]
        episodes.append(dict(stem=Path(fpath).stem, path_type="chair_target",
                              bboxes=bboxes, vis=vis, gts=gts,
                              acts=[tuple(float(v) for v in acts[t]) for t in range(n)]))
        ann.append(dict(episode=fpath, frames=[dict(frame_idx=t, cx_det=bboxes[t][0], cy_det=bboxes[t][1],
                                                     area_det=bboxes[t][2], has_bbox=bool(bboxes[t][3]),
                                                     gt_class=gts[t]) for t in range(n)]))
        print(f"  {fi+1}/{len(files)} {Path(fpath).stem} ({n}프레임)", flush=True)

    ANN_OUT.parent.mkdir(parents=True, exist_ok=True)
    ANN_OUT.write_text(json.dumps(ann, indent=2, ensure_ascii=False))
    torch.save(episodes, str(CACHE_OUT))
    print(f"[CACHE] 저장 → {CACHE_OUT}, 주석 → {ANN_OUT}")
    return episodes


def main():
    episodes = ground_and_encode()
    X, y, _ = build_windows(episodes)
    print(f"\n[WINDOW] chair 전체 {X.shape}")

    results = {}

    # ① zero-shot — exp77 체크포인트 그대로
    ck = torch.load(str(EXP77_CKPT), map_location=DEVICE, weights_only=False)
    model = MLPActionHead().to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, device=DEVICE)).argmax(1).cpu().numpy()
    acc = float((pred == y).mean())
    per_class = {CLASS_NAMES[c]: float((pred[y == c] == c).mean()) for c in range(8) if (y == c).sum() > 0}
    results["zero_shot"] = dict(acc=acc, n=len(y), per_class=per_class,
                                 class_dist={CLASS_NAMES[c]: int((y == c).sum()) for c in range(8)})
    print(f"\n[zero-shot] acc={acc*100:.1f}% (n={len(y)})")
    print("  클래스분포:", results["zero_shot"]["class_dist"])
    print("  per_class:", {k: f"{v*100:.1f}%" for k, v in per_class.items()})

    # ② few-shot — chair만으로 처음부터 소규모 재학습(에피소드 단위 80/20 split)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(episodes)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO))
    va_eps = [episodes[i] for i in idx[:n_val]]
    tr_eps = [episodes[i] for i in idx[n_val:]]
    X_tr, y_tr, _ = build_windows(tr_eps)
    X_va, y_va, _ = build_windows(va_eps)
    print(f"\n[few-shot] train {X_tr.shape} / val {X_va.shape} ({len(tr_eps)}/{len(va_eps)}ep)")

    accs, best_acc, best_per_class = [], 0.0, None
    for seed in [0, 1, 2]:
        acc, _, per_class = train_one(MLPActionHead, X_tr, y_tr, X_va, y_va, seed, epochs=200)
        accs.append(acc)
        if acc > best_acc:
            best_acc, best_per_class = acc, per_class
        print(f"  seed={seed} val_acc={acc*100:.1f}%")
    results["few_shot"] = dict(
        val_acc_mean=float(np.mean(accs)), val_acc_std=float(np.std(accs)), val_acc_best=best_acc,
        n_train=len(X_tr), n_val=len(X_va),
        per_class_best={CLASS_NAMES[c]: v for c, v in (best_per_class or {}).items()})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n=== 요약 ===")
    print(f"zero-shot(exp77 그대로): {acc*100:.1f}% <- 재확인용, 위 zero_shot 값 참고")
    print(f"few-shot(chair 전용 재학습): mean={np.mean(accs)*100:.1f}±{np.std(accs)*100:.1f}% best={best_acc*100:.1f}%")
    print(f"저장 → {OUT}")


if __name__ == "__main__":
    main()
