#!/usr/bin/env python3
"""Stage 1 — 비전 백본을 Florence-2로 교체 (2026-08-17, 계획서 2'' 단계).

목적:
  `train_stage1_v3_5cls_owl.py`(Kosmos-2 vision_model, val_acc 94.09%)와
  **비전 인코더 하나만** 바꿔 apples-to-apples 비교한다.

바꾸는 것 (딱 하나):
  Kosmos-2 vision_model → Florence-2 vision_tower.forward_features_unpool
    Kosmos-2  : last_hidden_state.mean(1)        → 1024d
    Florence-2: forward_features_unpool[0].mean(0) → 1024d   ← 차원 동일, image_proj 그대로

바꾸지 않는 것 (전부 베이스라인과 동일):
  · 데이터: bbox_dataset_v6_owl.json, consistent & label 있는 프레임만
  · 분할: StratifiedShuffleSplit(test_size=0.2, random_state=42), 에피소드 단위, direction stratify
  · 텍스트 앵커: **Kosmos-2 text_model 그대로** (5문장 → 2048d → text_proj 2048→256)
      ↳ 앵커를 Florence-2 것으로 바꾸면 변수가 2개가 되어 원인 분리가 불가능해진다.
        앵커는 학습 후 고정값으로 저장되어 배포 시 재호출되지 않으므로 이 선택은 배포 비용과 무관.
  · 하이퍼파라미터: 30 epoch, batch 16, AdamW lr 3e-4, cosine, temperature 0.07,
    class weight = 빈도 역수 정규화
  · 학습 대상: image_proj(1024→256) + text_proj(2048→256)

속도 최적화 (수치에는 영향 없음):
  베이스라인은 frozen 인코더를 30에폭 내내 재실행해 276.9분이 걸렸다. 인코더가 frozen이고
  이미지 증강이 없으므로 **피처를 1회만 추출해 캐시**하면 수학적으로 등가이며 훨씬 빠르다.

출력:
  runs/v5_nav/mlp/stage1_florence2_5cls/stage1_florence2_5cls_projs.pt
  docs/v5/detector/stage1_florence2_5cls.json
"""
import json
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH   = ROOT / ".vlms" / "kosmos-2-patch14-224"      # 텍스트 앵커 전용
FLOR       = "microsoft/Florence-2-base"                   # 비전 백본
DATA_PATH  = ROOT / "docs" / "v5" / "bbox_nav_owl" / "bbox_dataset_v6_owl.json"
OUT_DIR    = ROOT / "runs" / "v5_nav" / "mlp" / "stage1_florence2_5cls"
FEAT_CACHE = ROOT / "docs" / "v5" / "detector" / "stage1_florence2_feats.npz"
REPORT     = ROOT / "docs" / "v5" / "detector" / "stage1_florence2_5cls.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROJ_DIM, LM_DIM, VIS_DIM = 256, 2048, 1024
EPOCHS, BATCH_SIZE, LR, TEMPERATURE = 30, 16, 3e-4, 0.07
BASELINE_VAL_ACC = 0.9408895265423243     # Kosmos-2 백본 (train_stage1_v3_5cls_owl.py)

DIR_IDX = {"strong_left": 0, "weak_left": 1, "center": 2,
           "weak_right": 3, "strong_right": 4}
N_CLASSES = len(DIR_IDX)
ANCHOR_TEXTS = {
    "strong_left":  "The gray basket is strongly on the left side of the image",
    "weak_left":    "The gray basket is slightly on the left side of the image",
    "center":       "The gray basket is in the center of the image",
    "weak_right":   "The gray basket is slightly on the right side of the image",
    "strong_right": "The gray basket is strongly on the right side of the image",
}
ANCHOR_ORDER = ["strong_left", "weak_left", "center", "weak_right", "strong_right"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_frame_level_data():
    """베이스라인 load_frame_level_data()와 동일 필터."""
    raw = json.loads(DATA_PATH.read_text())
    eps = []
    for ep in raw:
        frames = [f for f in ep["frames"] if f["consistent"] and f["label"] is not None]
        if frames:
            eps.append({"episode": ep["episode"], "direction": ep["direction"],
                        "frames": frames})
    return eps


@torch.no_grad()
def extract_features(eps):
    """Florence-2 vision_tower로 프레임별 1024d mean-pooled 피처 1회 추출 후 캐시."""
    flat = [(ep["episode"], f["frame_idx"]) for ep in eps for f in ep["frames"]]
    if FEAT_CACHE.exists():
        z = np.load(FEAT_CACHE, allow_pickle=True)
        if len(z["keys"]) == len(flat):
            print(f"[CACHE] 재사용 {FEAT_CACHE.name} ({len(flat)} 프레임)")
            # npz lazy 접근 주의 — 배열을 1회만 실체화해야 한다(반복 접근 시 OOM).
            arr = np.asarray(z["feats"])
            keys = z["keys"]
            return {(str(keys[i][0]), int(keys[i][1])): arr[i] for i in range(len(keys))}
        print("[CACHE] 프레임 수 불일치 — 재추출")

    from transformers import AutoModelForCausalLM, AutoProcessor
    print(f"[FLORENCE-2] 로딩 ({FLOR})")
    m = AutoModelForCausalLM.from_pretrained(
        FLOR, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(FLOR, trust_remote_code=True)
    vt = m.vision_tower

    print(f"[EXTRACT] {len(flat)} 프레임 · 1회만 추출 (frozen)")
    feats = np.zeros((len(flat), VIS_DIM), dtype=np.float16)
    t0 = time.time()
    cur, hf = None, None
    for i, (path, fi) in enumerate(flat):
        if path != cur:
            if hf is not None:
                hf.close()
            hf = h5py.File(path, "r"); cur = path
        # 베이스라인과 동일한 BGR→RGB 반전
        im = np.array(hf["images"][fi])[:, :, ::-1].astype("uint8")
        pv = proc(images=Image.fromarray(im), text="<OD>",
                  return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
        o = vt.forward_features_unpool(pv)[0]          # (576, 1024)
        feats[i] = o.mean(0).cpu().numpy().astype(np.float16)
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(flat)}  {el:.0f}s (ETA {el/(i+1)*(len(flat)-i-1):.0f}s)",
                  flush=True)
    if hf is not None:
        hf.close()
    del m
    torch.cuda.empty_cache()

    keys = np.array(flat, dtype=object)
    np.savez(FEAT_CACHE, feats=feats, keys=keys)
    print(f"[EXTRACT] 완료 {time.time()-t0:.0f}s → {FEAT_CACHE}")
    return {(str(k[0]), int(k[1])): feats[i] for i, k in enumerate(flat)}


@torch.no_grad()
def compute_text_anchors():
    """베이스라인과 동일 — Kosmos-2 text_model로 5문장 인코딩 (학습 후 고정 저장)."""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("[KOSMOS-2] 텍스트 앵커 계산 (앵커 전용 · 배포엔 미사용)")
    proc = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16 if DEV.type == "cuda" else torch.float32
    ).to(DEV)
    tm = model.text_model
    out = []
    for d in ANCHOR_ORDER:
        inp = proc.tokenizer(ANCHOR_TEXTS[d], return_tensors="pt",
                            add_special_tokens=True).to(DEV)
        o = tm(input_ids=inp.input_ids, attention_mask=inp.attention_mask,
               output_hidden_states=True)
        out.append(o.hidden_states[-1][:, -1, :].float())
    anchors = torch.cat(out, dim=0)      # (5, 2048)
    del model
    torch.cuda.empty_cache()
    return anchors


def main():
    t_start = time.time()
    print(f"[DEVICE] {DEV}")

    eps = load_frame_level_data()
    from collections import Counter
    cnt = Counter(f["label"] for ep in eps for f in ep["frames"])
    total = sum(cnt.values())
    print(f"[DATA] 에피소드 {len(eps)} · 프레임 {total}")
    print("       " + " ".join(f"{d}={cnt[d]}" for d in ANCHOR_ORDER))

    feat_map = extract_features(eps)
    anchor_raw = compute_text_anchors()

    # ── 분할: 베이스라인과 동일 ──────────────────────────────────────
    ep_dirs = [ep["direction"] for ep in eps]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(np.zeros(len(eps)), ep_dirs))
    tr_eps = [eps[i] for i in tr_idx]
    val_eps = [eps[i] for i in te_idx]
    print(f"       train={len(tr_eps)} ep / val={len(val_eps)} ep")

    def pack(ep_list):
        X, y = [], []
        for ep in ep_list:
            for f in ep["frames"]:
                k = (str(ep["episode"]), int(f["frame_idx"]))
                if k in feat_map:
                    X.append(feat_map[k]); y.append(DIR_IDX[f["label"]])
        return (torch.tensor(np.array(X), dtype=torch.float32, device=DEV),
                torch.tensor(y, dtype=torch.long, device=DEV))

    Xtr, ytr = pack(tr_eps)
    Xva, yva = pack(val_eps)
    print(f"       train {tuple(Xtr.shape)} / val {tuple(Xva.shape)}")

    image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(DEV)
    text_proj  = nn.Linear(LM_DIM,  PROJ_DIM).to(DEV)

    counts = np.array([cnt[d] for d in ANCHOR_ORDER], dtype=float)
    w = counts.sum() / (N_CLASSES * counts)
    w = w / w.sum() * N_CLASSES
    class_weight = torch.tensor(w, dtype=torch.float32, device=DEV)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    print("[LOSS] class weight: " + " ".join(f"{d}={x:.2f}" for d, x in zip(ANCHOR_ORDER, w)))

    opt = torch.optim.AdamW(list(image_proj.parameters()) + list(text_proj.parameters()),
                            lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc, best_state = 0.0, None
    print(f"\n{'epoch':>6} {'val_acc':>9} {'best':>9}")
    print("-" * 30)
    for epoch in range(1, EPOCHS + 1):
        image_proj.train(); text_proj.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(perm), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            f_img = F.normalize(image_proj(Xtr[b]), dim=-1)
            ap = F.normalize(text_proj(anchor_raw), dim=-1)
            loss = criterion((f_img @ ap.T) / TEMPERATURE, ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        image_proj.eval(); text_proj.eval()
        with torch.no_grad():
            ap = F.normalize(text_proj(anchor_raw), dim=-1)
            pred = (F.normalize(image_proj(Xva), dim=-1) @ ap.T).argmax(1)
            acc = (pred == yva).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {
                "image_proj": {k: v.detach().cpu().clone() for k, v in image_proj.state_dict().items()},
                "text_proj":  {k: v.detach().cpu().clone() for k, v in text_proj.state_dict().items()},
                "anchor_raw": anchor_raw.detach().cpu().clone(),
                "val_acc": best_acc, "dir_idx": DIR_IDX,
                "backbone": FLOR, "vis_dim": VIS_DIM,
            }
        print(f"{epoch:>6}  {acc:>8.4f}  {best_acc:>8.4f}", flush=True)

    # ── 혼동행렬 (best) ─────────────────────────────────────────────
    image_proj.load_state_dict({k: v.to(DEV) for k, v in best_state["image_proj"].items()})
    text_proj.load_state_dict({k: v.to(DEV) for k, v in best_state["text_proj"].items()})
    image_proj.eval(); text_proj.eval()
    with torch.no_grad():
        ap = F.normalize(text_proj(anchor_raw), dim=-1)
        pred = (F.normalize(image_proj(Xva), dim=-1) @ ap.T).argmax(1).cpu().numpy()
    gt = yva.cpu().numpy()
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(gt, pred):
        cm[t, p] += 1

    delta = best_acc - BASELINE_VAL_ACC
    print("\n" + "=" * 62)
    print("  Stage 1 — Florence-2 백본 (5-class, OWL 라벨, 225ep)")
    print(f"  val_acc: {best_acc:.4f}")
    print(f"  Kosmos-2 베이스라인: {BASELINE_VAL_ACC:.4f}  →  차이 {delta:+.4f} ({delta*100:+.2f}%p)")
    print("=" * 62)
    print("\n혼동 행렬 (행=실제, 열=예측):")
    hdr = "".join(f"{d[:8]:>10s}" for d in ANCHOR_ORDER)
    print(" " * 12 + hdr + "      정확도")
    for i, d in enumerate(ANCHOR_ORDER):
        row = "".join(f"{cm[i][j]:>10d}" for j in range(N_CLASSES))
        acc_i = cm[i][i] / max(cm[i].sum(), 1) * 100
        print(f"  {d[:10]:>10s}" + row + f"    {acc_i:5.1f}%")

    ckpt = OUT_DIR / "stage1_florence2_5cls_projs.pt"
    torch.save(best_state, str(ckpt))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "backbone": FLOR, "baseline_backbone": "kosmos-2-patch14-224",
        "val_acc": best_acc, "baseline_val_acc": BASELINE_VAL_ACC,
        "delta": delta,
        "n_episodes": len(eps), "n_frames": total,
        "n_train_ep": len(tr_eps), "n_val_ep": len(val_eps),
        "n_train_frames": int(len(Xtr)), "n_val_frames": int(len(Xva)),
        "confusion_matrix": cm.tolist(), "class_order": ANCHOR_ORDER,
        "per_class_acc": [float(cm[i][i] / max(cm[i].sum(), 1)) for i in range(N_CLASSES)],
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
        "temperature": TEMPERATURE, "checkpoint": str(ckpt),
        "elapsed_min": (time.time() - t_start) / 60,
    }, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] {ckpt}")
    print(f"[SAVE] {REPORT}")
    print(f"소요: {(time.time()-t_start)/60:.1f}분")


if __name__ == "__main__":
    main()
