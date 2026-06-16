# -*- coding: utf-8 -*-
"""
STOP 학습 유도 Stage2 MLP 재학습.

B1 config(PG2 grounding, 243ep, no-flip) + class-weighted CrossEntropy.
학습 데이터: build_stop_annotation.py 산출 (mid-stop 재라벨 + 도착 plateau STOP 합성).
목표: 규칙 override 없이 MLP가 "도착(area↑·중앙) → STOP"을 직접 예측.

Usage:
  .venv/bin/python3 scripts/train_stop_mlp.py
  .venv/bin/python3 scripts/train_stop_mlp.py --stop-weight 3.0 --epochs 300
"""
import json, random, warnings, argparse, sys
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path("/home/minum/26CS/MoNaVLA")
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image

VLM = ROOT / ".vlms/kosmos-2-patch14-224"
S1  = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
ANN = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx_stop.json"
OUT = ROOT / "runs/v5_nav/mlp/exp60"
WINDOW = 8
NUM_CLASSES = 8
CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ck = torch.load(str(S1), map_location="cuda", weights_only=False)
        self.proc = AutoProcessor.from_pretrained(str(VLM))
        base = AutoModelForVision2Seq.from_pretrained(str(VLM), torch_dtype=torch.float16)
        self.vm = base.vision_model.to("cuda").eval()
        self.proj = nn.Linear(1024, 256).to("cuda")
        self.proj.load_state_dict(ck["image_proj"]); self.proj.eval()

    @torch.no_grad()
    def encode(self, pil):
        inp = self.proc(images=[pil], return_tensors="pt")
        pv = inp["pixel_values"].to("cuda", dtype=torch.float16)
        return self.proj(self.vm(pixel_values=pv).last_hidden_state.mean(1).float()).squeeze(0)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(288, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 8))
    def forward(self, x): return self.net(x)


def build(eps, enc):
    X, y = [], []
    for ep in eps:
        h5 = Path(ep["episode"])
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        if not frames or not h5.exists():
            continue
        try:
            with h5py.File(str(h5), "r") as f:
                imgs = f["observations"]["images"][:]
        except Exception:
            continue
        vo = torch.stack([enc.encode(Image.fromarray(imgs[fr["frame_idx"]].astype("uint8"))) for fr in frames])
        for t, fr in enumerate(frames):
            ho = []
            for k in range(WINDOW):
                f2 = frames[max(0, t - (WINDOW - 1 - k))]
                cx = f2.get("cx_det") or 0.5
                cy = f2.get("cy_det") or 0.5
                ar = f2.get("area_det") or 0.05
                hb = float(f2.get("has_bbox", f2.get("detected", False)))
                ho.extend([cx, cy, ar, hb])
            X.append(ho + vo[t].cpu().tolist())
            y.append(fr["gt_class"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


@torch.no_grad()
def stop_metrics(mlp, Xv, yv):
    """STOP(0) precision/recall + 전체 acc."""
    mlp.eval()
    pred = mlp(Xv).argmax(1)
    acc = (pred == yv).float().mean().item()
    p_stop = (pred == 0); g_stop = (yv == 0)
    tp = (p_stop & g_stop).sum().item()
    prec = tp / max(p_stop.sum().item(), 1)
    rec  = tp / max(g_stop.sum().item(), 1)
    return acc, prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--stop-weight", type=float, default=0.0,
                    help="STOP 클래스 추가 가중 배수 (0=역빈도 자동, >0=STOP에 곱)")
    ap.add_argument("--tag", default="stop")
    ap.add_argument("--ann", default=str(ANN))
    args = ap.parse_args()

    ann = json.loads(Path(args.ann).read_text())
    ann = [ep for ep in ann if ep.get("path_type", "") not in ("", "unknown")]  # free 포함
    random.seed(42); np.random.seed(42); random.shuffle(ann)
    n_val = max(1, int(len(ann) * 0.15))
    val_eps, train_eps = ann[:n_val], ann[n_val:]
    print(f"[DATA] train {len(train_eps)} / val {len(val_eps)} ep")

    enc = Enc().to("cuda").eval()
    Xtr, ytr = build(train_eps, enc)
    Xva, yva = build(val_eps, enc)
    print(f"[DATA] train {len(Xtr)} / val {len(Xva)} frames")
    import collections
    print(f"[DATA] train gt_class: {dict(sorted(collections.Counter(ytr.tolist()).items()))}")

    Xtr_t = torch.from_numpy(Xtr).cuda(); ytr_t = torch.from_numpy(ytr).cuda()
    Xva_t = torch.from_numpy(Xva).cuda(); yva_t = torch.from_numpy(yva).cuda()

    # class weights — 역빈도 (균형), STOP 추가 가중 옵션
    counts = np.bincount(ytr, minlength=NUM_CLASSES).astype(float)
    w = np.where(counts > 0, 1.0 / (counts + 1e-6), 0.0)
    w = w / w.sum() * NUM_CLASSES
    if args.stop_weight > 0:
        w[0] *= args.stop_weight
    wt = torch.tensor(w, dtype=torch.float32, device="cuda")
    print(f"[LOSS] class weights: {[f'{x:.2f}' for x in w]}")

    mlp = MLP().cuda()
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_rec, best_state, best_row = -1, None, None
    print(f"\n{'epoch':>6} {'acc':>7} {'STOP_P':>8} {'STOP_R':>8}")
    for ep in range(1, args.epochs + 1):
        mlp.train()
        perm = torch.randperm(len(Xtr_t), device="cuda")
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            loss = F.cross_entropy(mlp(Xtr_t[idx]), ytr_t[idx], weight=wt)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 20 == 0 or ep == args.epochs:
            acc, prec, rec = stop_metrics(mlp, Xva_t, yva_t)
            print(f"{ep:>6} {acc*100:>6.1f}% {prec*100:>7.1f}% {rec*100:>7.1f}%")
            # STOP recall*acc 기준 best
            score = rec + acc
            if score > best_rec:
                best_rec = score
                best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
                best_row = (acc, prec, rec)

    acc, prec, rec = best_row
    OUT.mkdir(parents=True, exist_ok=True)
    fname = f"{args.tag}_mlp.pt"
    torch.save({"mlp": best_state, "val_acc": acc, "d_in": 288,
                "stop_prec": prec, "stop_rec": rec, "tag": args.tag,
                "stop_weight": args.stop_weight}, str(OUT / fname))
    print(f"\n[BEST] acc={acc*100:.1f}%  STOP precision={prec*100:.1f}%  recall={rec*100:.1f}%")
    print(f"[SAVE] {OUT / fname}")


if __name__ == "__main__":
    main()
