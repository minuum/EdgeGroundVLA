#!/usr/bin/env python3
"""
Exp54 Stage 2 v2 PM 평가 (confusion matrix + path_type별 정확도)

Stage 1 v2 (frozen base CLIP + image_proj, 256-dim) 위에 올린 ActionMLP 평가.
v1 eval과의 차이: FrozenCLIPLoRA(1024) → FrozenCLIPV2(256), D_IN=1056→288

Usage:
  .venv/bin/python3 scripts/eval_exp54_stage2_v2.py
  .venv/bin/python3 scripts/eval_exp54_stage2_v2.py --ckpt runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt
"""
import sys, json, warnings, argparse
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

VLM_PATH    = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH   = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
STAGE1_V2   = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
DEFAULT_CKPT= ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage2_v2" / "stage2_v2_mlp.pt"

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
PROJ_DIM    = 256
D_IN        = WINDOW * 4 + PROJ_DIM   # 288


class FrozenCLIPV2(nn.Module):
    """Stage 1 v2 encoder: frozen base Kosmos-2 + trained image_proj."""

    def __init__(self, vlm_path, ckpt_path, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        print(f"[MODEL] Stage1 v2 val_acc={ckpt['val_acc']:.4f}")

        self.processor  = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vision_model = base.vision_model.to(device)
        self.image_proj   = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.image_proj.load_state_dict(ckpt["image_proj"])

        for p in self.vision_model.parameters(): p.requires_grad = False
        for p in self.image_proj.parameters():   p.requires_grad = False

    @torch.no_grad()
    def encode(self, pil_images, device):
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pv  = inputs["pixel_values"].to(device, dtype=torch.float16)
        out = self.vision_model(pixel_values=pv)
        feat = out.last_hidden_state.mean(dim=1).float()
        return F.normalize(self.image_proj(feat), dim=-1)   # (N, 256)


class ActionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_IN, 256), nn.ReLU(), nn.Dropout(0.25),
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[ERROR] 체크포인트 없음: {ckpt_path}")
        print("학습이 완료될 때까지 기다려주세요.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[CKPT]   {ckpt_path}")

    data = json.loads(DATA_PATH.read_text())
    ep_labels = [ep["path_type"] for ep in data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
    val_eps = [data[i] for i in te_idx]
    print(f"Val: {len(val_eps)} episodes")

    print("[MODEL] Stage 1 v2 인코더 로드 중...")
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_V2, device).to(device).eval()

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    mlp  = ActionMLP().to(device)
    mlp.load_state_dict(ckpt["mlp"])
    mlp.eval()
    print(f"[MODEL] Stage 2 v2 MLP: best_val_acc={ckpt['val_acc']:.4f}")

    confusion    = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    path_correct = {}
    path_total   = {}

    with torch.no_grad():
        for ep in val_eps:
            try:
                with h5py.File(ep["episode"], "r") as f:
                    imgs = [Image.fromarray(f["observations"]["images"][i])
                            for i in range(len(ep["frames"]))]
            except Exception as e:
                print(f"[SKIP] {ep['episode']}: {e}")
                continue
            vis_feats = enc.encode(imgs, device)
            pt = ep.get("path_type", "unknown")
            path_correct.setdefault(pt, 0)
            path_total.setdefault(pt, 0)

            for t, fr in enumerate(ep["frames"]):
                bf   = torch.tensor(bbox_feat(ep["frames"], t), device=device)
                x    = torch.cat([bf, vis_feats[t]]).unsqueeze(0)
                pred = mlp(x).argmax(1).item()
                gt   = fr["gt_class"]
                confusion[gt][pred] += 1
                path_correct[pt] += int(pred == gt)
                path_total[pt]   += 1

    total   = confusion.sum()
    correct = np.diag(confusion).sum()

    print(f"\n{'='*55}")
    print(f"  Exp54 Stage 2 v2 PM 평가 결과")
    print(f"  전체 PM: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  참고: Exp49=96.4%  Exp53=94.7%  Exp54-S2-v1=?%")
    print(f"{'='*55}")

    print(f"\n{'클래스':<10} {'정답':>6} {'전체':>6} {'정확도':>8}")
    print("-" * 38)
    for c in range(NUM_CLASSES):
        row_t = confusion[c].sum()
        row_c = confusion[c][c]
        acc   = row_c / row_t * 100 if row_t > 0 else 0
        print(f"  {CLASS_NAMES[c]:<8} {row_c:>6} {row_t:>6} {acc:>7.1f}%")

    print(f"\n혼동 행렬 (행=정답, 열=예측):")
    print("       " + "".join(f"{n:>8}" for n in CLASS_NAMES))
    for c in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[c]:<7}" + "".join(f"{confusion[c][p]:>8}" for p in range(NUM_CLASSES)))

    print(f"\npath_type별 정확도:")
    for pt in sorted(path_total.keys()):
        acc = path_correct[pt] / path_total[pt] * 100 if path_total[pt] > 0 else 0
        print(f"  {pt:<22} {path_correct[pt]:>4}/{path_total[pt]:<4} = {acc:.1f}%")


if __name__ == "__main__":
    main()
