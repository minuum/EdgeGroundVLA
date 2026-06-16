#!/usr/bin/env python3
"""
Exp55: Stage 2 v2 + Free 21 에피소드 (캐싱 기반 빠른 학습)

기존 150 에피소드 + 신규 21 free 에피소드를 포함해 MLP 재학습.
Stage 1 v2 frozen encoder 재사용 + 특징 캐싱으로 30분 내 완료.

구조:
  Phase 1: 21 free 에피소드 Kosmos-2 bbox 추출 (10~15분)
  Phase 2: 21 free 에피소드 CLIP 특징 추출 + 전체 256-dim 캐싱 (5~10분)
  Phase 3: 캐시에서 MLP 학습, 300 epochs (2~3분)

Usage:
  .venv/bin/python3 scripts/train_exp55_free_episodes.py
  .venv/bin/python3 scripts/train_exp55_free_episodes.py --skip_phase1  # bbox 재사용
  .venv/bin/python3 scripts/train_exp55_free_episodes.py --skip_phase1 --skip_phase2  # 캐시 재사용
"""

import argparse, json, sys, gc, time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH    = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_DIR    = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OLD_BBOX    = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
OLD_VIS_NPZ = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "vision_features.npz"
STAGE1_V2   = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
OUT_DIR     = ROOT / "docs" / "v5" / "bbox_nav_exp55"
FREE_BBOX   = OUT_DIR / "bbox_dataset_free.json"
FEAT_CACHE  = OUT_DIR / "features_256.npz"
CKPT_DIR    = ROOT / "runs" / "v5_nav" / "mlp" / "exp55"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES  = 8
WINDOW       = 8
VIS_DIM      = 1024
PROJ_DIM     = 256
D_IN         = WINDOW * 4 + PROJ_DIM   # 288

GROUNDING_PROMPT = "<grounding>The gray basket is at"
MAX_NEW_TOKENS   = 48


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────

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
            cands.append({"cx": (x1+x2)/2, "cy": (y1+y2)/2, "area": area,
                           "is_basket": any(k in ent_name.lower() for k in kw)})
    matched = [b for b in cands if b["is_basket"]]
    if matched: return matched[0]
    cap_low = caption.lower()
    if "far left"  in cap_low: return {"cx": 0.10, "cy": 0.5, "area": 0.05}
    if "far right" in cap_low: return {"cx": 0.90, "cy": 0.5, "area": 0.05}
    if "left"  in cap_low and "right" not in cap_low: return {"cx": 0.25, "cy": 0.5, "area": 0.05}
    if "right" in cap_low and "left"  not in cap_low: return {"cx": 0.75, "cy": 0.5, "area": 0.05}
    if "center" in cap_low: return {"cx": 0.5, "cy": 0.5, "area": 0.05}
    if cands: return cands[0]
    return None


def load_h5_images(h5_path):
    with h5py.File(str(h5_path), "r") as f:
        imgs = f["observations"]["images"][:]
        actions = f["actions"][:]
    return imgs, actions


def free_ep_path_type(h5_name):
    """episode_260522_112550_free_left__xxx → free_left"""
    name = Path(h5_name).stem
    parts = name.split("_")
    for i, p in enumerate(parts):
        if p == "free" and i + 1 < len(parts):
            return f"free_{parts[i+1]}"
    return "free_unknown"


# ── Phase 1: 21 free 에피소드 bbox 추출 ─────────────────────────────────────

def phase1_bbox_extraction(device):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    free_files = sorted(DATA_DIR.glob("episode_*free*.h5"))
    print(f"\n[Phase 1] Free 에피소드 bbox 추출: {len(free_files)}개")

    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForImageTextToText.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16
    ).to(device).eval()

    dataset = []
    for idx, h5_path in enumerate(free_files):
        pt = free_ep_path_type(h5_path.name)
        imgs_arr, actions = load_h5_images(h5_path)
        frames_data = []
        for fi in range(len(imgs_arr)):
            gt_cls = gt_action_class(*actions[fi])
            pil = Image.fromarray(imgs_arr[fi].astype(np.uint8)).convert("RGB")
            inputs = processor(text=GROUNDING_PROMPT, images=pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
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
                "frame_idx": fi,
                "gt_class":  gt_cls,
                "cx":        float(bbox["cx"])   if bbox else 0.5,
                "cy":        float(bbox["cy"])   if bbox else 0.5,
                "area":      float(bbox["area"]) if bbox else 0.0,
                "has_bbox":  bbox is not None,
            })
        dataset.append({"path_type": pt, "episode": str(h5_path), "frames": frames_data})
        n_bbox = sum(1 for f in frames_data if f["has_bbox"])
        print(f"  [{idx+1:2d}/{len(free_files)}] {h5_path.name[:50]}  bbox={n_bbox}/{len(frames_data)}", flush=True)

    del model; gc.collect(); torch.cuda.empty_cache()
    FREE_BBOX.write_text(json.dumps(dataset, indent=2))
    print(f"[Phase 1] 저장: {FREE_BBOX}")
    return dataset


# ── Phase 2: 특징 추출 + 256-dim 캐싱 ────────────────────────────────────────

def phase2_feature_caching(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"\n[Phase 2] 특징 추출 및 256-dim 캐싱")

    # Stage 1 v2 image_proj 로드
    ckpt = torch.load(str(STAGE1_V2), map_location=device, weights_only=False)
    print(f"  Stage1 v2 val_acc={ckpt['val_acc']:.4f}")
    image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
    image_proj.load_state_dict(ckpt["image_proj"])
    image_proj.eval()
    for p in image_proj.parameters(): p.requires_grad = False

    cache = {}

    # (A) 기존 150 에피소드: npz → image_proj 적용
    print(f"  (A) 기존 150 에피소드: vision_features.npz → 256-dim 변환")
    old_data = json.loads(OLD_BBOX.read_text())
    old_npz  = np.load(str(OLD_VIS_NPZ))
    with torch.no_grad():
        for ep_idx, ep in enumerate(old_data):
            key   = f"ep_{ep_idx}"
            feat1024 = torch.tensor(old_npz[key], dtype=torch.float32).to(device)  # (T, 1024)
            feat256  = F.normalize(image_proj(feat1024), dim=-1)                    # (T, 256)
            cache[f"old_{ep_idx}"] = feat256.cpu().numpy()
    print(f"  → {len(old_data)} 에피소드 변환 완료")
    del old_npz; gc.collect()

    # (B) 신규 21 free 에피소드: H5 → vision_model → image_proj
    print(f"  (B) 신규 21 free 에피소드: H5 → CLIP 인코딩")
    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    base = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16)
    vision_model = base.vision_model.to(device).eval()
    for p in vision_model.parameters(): p.requires_grad = False
    del base; gc.collect()

    free_files = sorted(DATA_DIR.glob("episode_*free*.h5"))
    with torch.no_grad():
        for idx, h5_path in enumerate(free_files):
            imgs_arr, _ = load_h5_images(h5_path)
            pil_imgs = [Image.fromarray(imgs_arr[i].astype(np.uint8)) for i in range(len(imgs_arr))]
            # 배치 인코딩
            BATCH = 16
            feats = []
            for b in range(0, len(pil_imgs), BATCH):
                batch_imgs = pil_imgs[b:b+BATCH]
                inputs = processor(images=batch_imgs, return_tensors="pt")
                pv = inputs["pixel_values"].to(device, dtype=torch.float16)
                out = vision_model(pixel_values=pv)
                feat1024 = out.last_hidden_state.mean(dim=1).float()   # (B, 1024)
                feat256  = F.normalize(image_proj(feat1024), dim=-1)   # (B, 256)
                feats.append(feat256.cpu().numpy())
            cache[f"free_{idx}"] = np.concatenate(feats, axis=0)
            print(f"  [{idx+1:2d}/{len(free_files)}] {h5_path.name[:50]}  frames={len(imgs_arr)}", flush=True)

    del vision_model, image_proj; gc.collect(); torch.cuda.empty_cache()
    np.savez_compressed(str(FEAT_CACHE), **cache)
    print(f"[Phase 2] 저장: {FEAT_CACHE}  ({len(cache)} entries)")
    return cache


# ── Phase 3: MLP 학습 ─────────────────────────────────────────────────────────

class ActionMLP(nn.Module):
    def __init__(self, d_in=D_IN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES),
        )
    def forward(self, x): return self.net(x)


def bbox_feat(frames, t):
    arr = []
    for k in range(WINDOW):
        fr = frames[max(0, t - (WINDOW - 1 - k))]
        arr.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])
    return np.array(arr, dtype=np.float32)


def phase3_train(args, device):
    oversample = args.oversample_free
    print(f"\n[Phase 3] 캐시 기반 MLP 학습 (epochs={args.epochs}, oversample_free={oversample}x)")

    # 데이터 로드
    old_data  = json.loads(OLD_BBOX.read_text())
    free_data = json.loads(FREE_BBOX.read_text())
    # free 오버샘플링: free 에피소드를 N배 반복
    all_data  = old_data + free_data * oversample
    feat_cache = np.load(str(FEAT_CACHE))

    # 특징 매핑 구성 (에피소드 경로 → cache key)
    key_map = {}
    for idx, ep in enumerate(old_data):
        key_map[ep["episode"]] = f"old_{idx}"
    free_files = sorted(DATA_DIR.glob("episode_*free*.h5"))
    for idx, h5_path in enumerate(free_files):
        key_map[str(h5_path)] = f"free_{idx}"

    # X, y 구성
    X_list, y_list, ep_labels = [], [], []
    for ep in all_data:
        cache_key = key_map.get(ep["episode"])
        if cache_key is None or cache_key not in feat_cache:
            print(f"  [SKIP] {ep['episode']}: 캐시 없음")
            continue
        feats = feat_cache[cache_key]   # (T, 256)
        frames = ep["frames"]
        for t, fr in enumerate(frames):
            if t >= len(feats): break
            bf = bbox_feat(frames, t)
            x  = np.concatenate([bf, feats[t]])
            X_list.append(x)
            y_list.append(fr["gt_class"])
        ep_labels.append(ep["path_type"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"  총 샘플: {len(X)} (old+free)  D_IN={X.shape[1]}")

    # Train/val split (에피소드 단위)
    all_ep_list = [ep for ep in all_data if key_map.get(ep["episode"]) in feat_cache.files]
    ep_path_labels = [ep["path_type"] for ep in all_ep_list]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    ep_idx_arr = np.zeros(len(all_ep_list))
    tr_idx, te_idx = next(sss.split(ep_idx_arr, ep_path_labels))
    tr_eps = [all_ep_list[i] for i in tr_idx]
    te_eps = [all_ep_list[i] for i in te_idx]

    def build_tensors(eps):
        Xl, yl = [], []
        for ep in eps:
            ck = key_map.get(ep["episode"])
            if ck not in feat_cache: continue
            feats = feat_cache[ck]
            frames = ep["frames"]
            for t, fr in enumerate(frames):
                if t >= len(feats): break
                bf = bbox_feat(frames, t)
                Xl.append(np.concatenate([bf, feats[t]]))
                yl.append(fr["gt_class"])
        return (torch.tensor(np.array(Xl, dtype=np.float32), device=device),
                torch.tensor(np.array(yl, dtype=np.int64), device=device))

    Xtr, ytr = build_tensors(tr_eps)
    Xte, yte = build_tensors(te_eps)
    print(f"  Train: {len(Xtr)} samples  Val: {len(Xte)} samples")

    counts  = np.bincount(ytr.cpu().numpy(), minlength=NUM_CLASSES).astype(float)
    weights = np.where(counts > 0, 1.0 / (counts + 1e-6), 0.0)
    weights /= weights.sum() / NUM_CLASSES
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    mlp = ActionMLP(D_IN).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_state = 0.0, None

    print(f"\n{'epoch':>6} {'val_acc':>9} {'best':>9}")
    print("-" * 30)

    for epoch in range(1, args.epochs + 1):
        mlp.train()
        idx = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(idx), args.batch_size):
            b = idx[i:i+args.batch_size]
            opt.zero_grad()
            criterion(mlp(Xtr[b]), ytr[b]).backward()
            opt.step()
        sched.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            mlp.eval()
            with torch.no_grad():
                acc = (mlp(Xte).argmax(1) == yte).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            print(f"{epoch:>6}  {acc:>8.4f}  {best_acc:>8.4f}", flush=True)

    # 최종 평가
    mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        preds = mlp(Xte).argmax(1)
    per_class = {}
    for c in range(NUM_CLASSES):
        mask = yte == c
        if mask.sum() > 0:
            per_class[c] = (preds[mask] == c).float().mean().item()

    print(f"\n{'='*55}")
    print(f"  Exp55 결과")
    print(f"  val_acc: {best_acc:.4f}  ({len(tr_eps)} train ep / {len(te_eps)} val ep)")
    print(f"  참고: Exp54 Stage2 v2 target=96%+")
    print(f"{'='*55}")
    print("\n클래스별:")
    for c, a in per_class.items():
        print(f"  {CLASS_NAMES[c]:<8}: {a*100:>5.1f}%")

    suffix = f"_os{oversample}" if oversample > 1 else ""
    ckpt_path = CKPT_DIR / f"exp55_mlp{suffix}.pt"
    torch.save({"mlp": best_state, "val_acc": best_acc, "d_in": D_IN,
                "n_eps": len(all_ep_list), "includes_free": True,
                "oversample_free": oversample}, str(ckpt_path))
    print(f"\n[SAVE] {ckpt_path}")
    return best_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_phase1",  action="store_true", help="Phase 1 건너뜀 (bbox 재사용)")
    p.add_argument("--skip_phase2",  action="store_true", help="Phase 2 건너뜀 (캐시 재사용)")
    p.add_argument("--epochs",          type=int,   default=300)
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--oversample_free", type=int,   default=1, help="free 에피소드 반복 배수 (3 권장)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    t0 = time.time()
    print("="*60)
    print(f"Exp55: Stage 2 + 21 Free Episodes  (D_IN={D_IN}, epochs={args.epochs})")
    print("="*60)

    if not args.skip_phase1:
        if not FREE_BBOX.exists():
            phase1_bbox_extraction(device)
        else:
            print(f"[Phase 1] 기존 파일 재사용: {FREE_BBOX}")
    else:
        if not FREE_BBOX.exists():
            print("[ERROR] --skip_phase1 설정이지만 bbox 파일 없음")
            return

    if not args.skip_phase2:
        if not FEAT_CACHE.exists():
            phase2_feature_caching(device)
        else:
            print(f"[Phase 2] 기존 캐시 재사용: {FEAT_CACHE}")
    else:
        if not FEAT_CACHE.exists():
            print("[ERROR] --skip_phase2 설정이지만 캐시 파일 없음")
            return

    phase3_train(args, device)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
