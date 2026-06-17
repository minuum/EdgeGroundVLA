#!/usr/bin/env python3
"""
STOP-Weighted Training Ablation (Layer 3)
──────────────────────────────────────────
현재 bbox_dataset_full.json에는 STOP(gt_class=0) 프레임이 0개.
→ 각 에피소드 마지막 프레임을 STOP으로 합성 주입 후 재학습.

stop_weight_mult in [1.0, 2.0, 5.0, 10.0]:
  STOP 클래스 CrossEntropy 가중치를 N배 증폭.
  나머지 클래스는 역빈도 기본값 유지.

각 variant:
  train (300 epoch) → save → CL eval via subprocess

저장: runs/v5_nav/mlp/stop_weighted/stop_wM.pt
결과: docs/v5/closed_loop_eval/stop_weighted_results.json

Usage:
  .venv/bin/python3 scripts/ablate_stop_weighted_train.py
  .venv/bin/python3 scripts/ablate_stop_weighted_train.py --weights 2.0 5.0
"""
import sys, json, time, argparse, subprocess, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

VLM_PATH    = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_CKPT = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
DATA_PATH   = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
OUT_DIR     = ROOT / "runs" / "v5_nav" / "mlp" / "stop_weighted"
RESULT_PATH = ROOT / "docs" / "v5" / "closed_loop_eval" / "stop_weighted_results.json"
EVAL_SCRIPT = ROOT / "scripts" / "eval_exp54_stage2_v2_closedloop.py"
PYTHON      = sys.executable

NUM_CLASSES  = 8
WINDOW       = 8
VIS_DIM      = 1024
PROJ_DIM     = 256
D_IN         = WINDOW * 4 + PROJ_DIM   # 288
EPOCHS       = 300
BATCH_SIZE   = 32
LR           = 2e-3
SEED         = 42

STOP_WEIGHT_MULTS = [1.0, 2.0, 5.0, 10.0]


# ── 모델 (train_exp54_stage2_v2_action.py 동일) ────────────────────────────
class FrozenCLIPV2(nn.Module):
    def __init__(self, vlm_path, ckpt_path, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        print(f"[MODEL] Stage1 v2 val_acc={ckpt['val_acc']:.4f}", flush=True)
        self.processor  = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vision_model = base.vision_model.to(device)
        self.image_proj   = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.image_proj.load_state_dict(ckpt["image_proj"])
        for p in self.vision_model.parameters(): p.requires_grad = False
        for p in self.image_proj.parameters():   p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, pil_images, device, batch=32):
        all_feats = []
        for i in range(0, len(pil_images), batch):
            imgs = pil_images[i:i+batch]
            inputs = self.processor(images=imgs, return_tensors="pt")
            pv  = inputs["pixel_values"].to(device, dtype=torch.float16)
            out = self.vision_model(pixel_values=pv)
            feat = out.last_hidden_state.mean(dim=1).float()
            all_feats.append(F.normalize(self.image_proj(feat), dim=-1))
        return torch.cat(all_feats, dim=0)


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


# ── 이미지 로드 ────────────────────────────────────────────────────────────
def load_images(h5_path, indices):
    import io as _io
    with h5py.File(str(h5_path), "r") as f:
        imgs_ds = f["observations"]["images"]
        result = []
        for i in indices:
            raw = imgs_ds[min(i, len(imgs_ds)-1)]
            if hasattr(raw, "dtype") and raw.dtype != object and raw.ndim >= 2:
                result.append(Image.fromarray(raw.astype("uint8")))
            else:
                arr = np.frombuffer(bytes(raw), dtype=np.uint8)
                result.append(Image.open(_io.BytesIO(arr)).convert("RGB"))
        return result


def precompute_features(enc, eps, device, label):
    cache = {}
    n = len(eps)
    print(f"[CACHE] {label} {n} eps feature 추출 중...", flush=True)
    t0 = time.time()
    for i, ep in enumerate(eps):
        try:
            imgs = load_images(ep["episode"], [fr["frame_idx"] for fr in ep["frames"]])
        except Exception as e:
            print(f"  skip {ep['episode']}: {e}", flush=True)
            cache[ep["episode"]] = None
            continue
        feats = enc.encode_batch(imgs, device)
        cache[ep["episode"]] = feats.cpu()
        if (i+1) % 20 == 0 or (i+1) == n:
            print(f"  {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[CACHE] {label} 완료 {time.time()-t0:.1f}s", flush=True)
    return cache


# ── 합성 STOP 프레임 주입 ──────────────────────────────────────────────────
def inject_stop_frames(eps, cache):
    """
    각 에피소드의 마지막 프레임을 STOP(gt_class=0)으로 복제 주입.
    반환: (synth_feats, synth_bbox) 리스트 — 학습 시 정규 프레임과 mix.
    """
    synth = []  # list of (feat_tensor[PROJ_DIM], bbox_arr[WINDOW*4], 0)
    for ep in eps:
        feats = cache.get(ep["episode"])
        if feats is None:
            continue
        frames = ep["frames"]
        t_last = len(frames) - 1
        # bbox feature for last frame (window-padded)
        arr = []
        for k in range(WINDOW):
            fr = frames[max(0, t_last - (WINDOW - 1 - k))]
            cx   = fr.get("cx",   fr.get("cx_det",   0.5))
            cy   = fr.get("cy",   fr.get("cy_det",   0.5))
            area = fr.get("area", fr.get("area_det", 0.05))
            has  = float(fr.get("has_bbox", fr.get("detected", False)))
            arr.extend([cx, cy, area, has])
        bbox = torch.tensor(arr, dtype=torch.float32)
        vis  = feats[t_last]
        synth.append((vis, bbox))
    print(f"[SYNTH] {len(synth)} STOP 프레임 주입 (각 ep 마지막 프레임 → gt_class=0)", flush=True)
    return synth


# ── 학습 루프 (단일 variant) ────────────────────────────────────────────────
def train_one(tr_eps, te_eps, tr_cache, te_cache, synth_stops,
              stop_weight_mult, device, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 정규 학습 데이터 레이블 통계
    all_labels = [fr["gt_class"] for ep in tr_eps for fr in ep["frames"]
                  if tr_cache.get(ep["episode"]) is not None]
    n_synth    = len(synth_stops)
    # STOP 합성 추가
    all_labels += [0] * n_synth
    counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    weights = np.where(counts > 0, 1.0 / (counts + 1e-6), 0.0)
    weights /= weights.sum() / NUM_CLASSES
    # STOP 가중치 승수 적용
    weights[0] *= stop_weight_mult
    print(f"  [LOSS] stop_wt_mult={stop_weight_mult:.1f}x  "
          f"STOP_weight={weights[0]:.3f}  "
          f"STOP_count={int(counts[0])}", flush=True)

    mlp = ActionMLP().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt   = torch.optim.AdamW(mlp.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # 학습 배치 빌더
    def iter_batches(shuffle_eps):
        bf_batch, lb_batch = [], []
        # 정규 프레임
        for ep in shuffle_eps:
            feats = tr_cache.get(ep["episode"])
            if feats is None:
                continue
            frames = ep["frames"]
            for t, fr in enumerate(frames):
                arr = []
                for k in range(WINDOW):
                    f2 = frames[max(0, t - (WINDOW-1-k))]
                    arr.extend([f2.get("cx", f2.get("cx_det", 0.5)),
                                 f2.get("cy", f2.get("cy_det", 0.5)),
                                 f2.get("area", f2.get("area_det", 0.05)),
                                 float(f2.get("has_bbox", f2.get("detected", False)))])
                bf = torch.tensor(arr, dtype=torch.float32)
                bf_batch.append(torch.cat([bf, feats[t]]))
                lb_batch.append(fr["gt_class"])
                if len(lb_batch) >= BATCH_SIZE:
                    yield bf_batch, lb_batch
                    bf_batch, lb_batch = [], []
        # 합성 STOP 프레임
        for vis, bbox in synth_stops:
            bf_batch.append(torch.cat([bbox, vis]))
            lb_batch.append(0)
            if len(lb_batch) >= BATCH_SIZE:
                yield bf_batch, lb_batch
                bf_batch, lb_batch = [], []
        if lb_batch:
            yield bf_batch, lb_batch

    @torch.no_grad()
    def evaluate():
        mlp.eval()
        correct = total = 0
        for ep in te_eps:
            feats = te_cache.get(ep["episode"])
            if feats is None:
                continue
            frames = ep["frames"]
            for t, fr in enumerate(frames):
                arr = []
                for k in range(WINDOW):
                    f2 = frames[max(0, t - (WINDOW-1-k))]
                    arr.extend([f2.get("cx", f2.get("cx_det", 0.5)),
                                 f2.get("cy", f2.get("cy_det", 0.5)),
                                 f2.get("area", f2.get("area_det", 0.05)),
                                 float(f2.get("has_bbox", f2.get("detected", False)))])
                x = torch.cat([torch.tensor(arr), feats[t]]).unsqueeze(0).to(device)
                pred = mlp(x).argmax(1).item()
                correct += int(pred == fr["gt_class"])
                total   += 1
        return correct / max(total, 1)

    best_acc, best_state = 0.0, None
    eps_list = list(tr_eps)
    for epoch in range(1, EPOCHS + 1):
        mlp.train()
        np.random.shuffle(eps_list)
        for bf_batch, lb_batch in iter_batches(eps_list):
            x = torch.stack(bf_batch).to(device)
            y = torch.tensor(lb_batch, dtype=torch.long, device=device)
            opt.zero_grad(); criterion(mlp(x), y).backward(); opt.step()
        sched.step()
        if epoch % 50 == 0 or epoch == EPOCHS:
            acc = evaluate()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            print(f"  ep{epoch:4d}  val_acc={acc:.4f}  best={best_acc:.4f}", flush=True)

    mlp.load_state_dict(best_state)
    return mlp, best_acc


# ── CL eval via subprocess ─────────────────────────────────────────────────
def run_cl_eval(ckpt_path, tag):
    cmd = [
        PYTHON, "-u", str(EVAL_SCRIPT),
        "--ckpt", str(ckpt_path),
        "--tag",  tag,
    ]
    log_path = ROOT / "logs" / "stop_ablation" / f"L3_eval_{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [CL EVAL] {ckpt_path.name} → {log_path.name}", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(ROOT)
        )
        log_path.write_text(result.stdout + "\n---STDERR---\n" + result.stderr)
        # parse success_rate from stdout
        for line in result.stdout.splitlines():
            if "success" in line.lower() and "%" in line:
                print(f"    {line.strip()}", flush=True)
        print(f"  [CL EVAL] 완료 ({time.time()-t0:.0f}s)", flush=True)
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"  [CL EVAL] TIMEOUT (600s)", flush=True)
        return ""


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default=str(DATA_PATH))
    p.add_argument("--stage1",  default=str(STAGE1_CKPT))
    p.add_argument("--epochs",  type=int,   default=EPOCHS)
    p.add_argument("--weights", type=float, nargs="+", default=STOP_WEIGHT_MULTS,
                   help="STOP class weight multipliers (default: 1 2 5 10)")
    p.add_argument("--skip_eval", action="store_true",
                   help="CL eval 건너뜀 (학습만)")
    p.add_argument("--tag",     default="stop_weighted")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}", flush=True)
    print(f"[CONFIG] epochs={args.epochs}  stop_weight_mults={args.weights}", flush=True)

    data      = json.loads(Path(args.data).read_text())
    ep_labels = [ep.get("path_type", "unknown") for ep in data]
    print(f"[DATA]   {len(data)} episodes", flush=True)

    # ── train/val split (seed=42 고정) ──
    from collections import Counter
    can_strat = all(c >= 2 for c in Counter(ep_labels).values())
    if can_strat:
        sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=SEED)
        tr_idx, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
    else:
        from sklearn.model_selection import ShuffleSplit
        ss = ShuffleSplit(1, test_size=0.2, random_state=SEED)
        tr_idx, te_idx = next(ss.split(np.zeros(len(data))))
    tr_eps = [data[i] for i in tr_idx]
    te_eps = [data[i] for i in te_idx]
    print(f"[SPLIT]  train={len(tr_eps)}  val={len(te_eps)}", flush=True)

    # ── CLIP feature 한 번만 추출 ──
    print("\n[MODEL] Stage1 로드...", flush=True)
    enc = FrozenCLIPV2(VLM_PATH, Path(args.stage1), device).to(device).eval()
    tr_cache = precompute_features(enc, tr_eps, device, "train")
    te_cache = precompute_features(enc, te_eps, device, "val")
    del enc
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print("[CACHE] VLM 해제 — MLP 학습만 남음\n", flush=True)

    # ── 합성 STOP 주입 ──
    synth_stops = inject_stop_frames(tr_eps, tr_cache)

    # ── 각 stop_weight_mult 학습 ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for mult in args.weights:
        tag = f"sw{mult:.0f}x" if mult == int(mult) else f"sw{mult}"
        print(f"\n{'='*60}", flush=True)
        print(f"  STOP weight ×{mult}  ({tag})", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()

        mlp, best_acc = train_one(
            tr_eps, te_eps, tr_cache, te_cache, synth_stops,
            stop_weight_mult=mult, device=device
        )

        ckpt_path = OUT_DIR / f"stop_wt_{tag}.pt"
        torch.save({
            "mlp":             mlp.state_dict(),
            "val_acc":         best_acc,
            "stop_weight_mult": mult,
            "n_synth_stop":    len(synth_stops),
            "window":          WINDOW,
            "head":            "mlp",
        }, str(ckpt_path))
        print(f"  [SAVE] {ckpt_path.name}  val_acc={best_acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

        cl_result = ""
        if not args.skip_eval:
            cl_result = run_cl_eval(ckpt_path, tag)

        results.append({
            "tag":              tag,
            "stop_weight_mult": mult,
            "val_acc":          best_acc,
            "n_synth_stop":     len(synth_stops),
            "ckpt":             str(ckpt_path),
            "cl_log":           str(ROOT / "logs" / "stop_ablation" / f"L3_eval_{tag}.log"),
        })

    # ── 최종 요약 ──
    print(f"\n{'='*60}", flush=True)
    print(f"  L3 STOP-Weighted Training 완료", flush=True)
    print(f"  {'tag':<12} {'val_acc':>9} {'stop_wt':>9}", flush=True)
    print(f"  {'-'*34}", flush=True)
    for r in results:
        print(f"  {r['tag']:<12} {r['val_acc']:>8.4f}  {r['stop_weight_mult']:>8.1f}x", flush=True)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps({
        "tag":    args.tag,
        "data":   str(args.data),
        "epochs": args.epochs,
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\n[SAVED] {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
