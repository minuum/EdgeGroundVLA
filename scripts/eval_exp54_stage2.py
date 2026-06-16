#!/usr/bin/env python3
"""
Exp54 Stage 2 per-class PM 평가 (confusion matrix 포함)
Usage: python3 scripts/eval_exp54_stage2.py
"""
import sys, json, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import h5py
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

VLM_PATH   = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH  = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
STAGE1_DIR = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage1"
STAGE2_DIR = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage2"
LORA_DIR   = STAGE1_DIR / "clip_lora_adapter"
CKPT_PATH  = STAGE2_DIR / "stage2_mlp.pt"

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES = 8
WINDOW = 8
D_IN = WINDOW * 4 + 1024  # 1056


class FrozenCLIPLoRA(nn.Module):
    def __init__(self, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from peft import PeftModel
        self.processor = AutoProcessor.from_pretrained(str(VLM_PATH))
        base = AutoModelForVision2Seq.from_pretrained(
            str(VLM_PATH),
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        self.vision_model = PeftModel.from_pretrained(base.vision_model, str(LORA_DIR))
        for p in self.vision_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, pil_images, device):
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pv = inputs["pixel_values"].to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        out = self.vision_model(pixel_values=pv)
        return out.last_hidden_state.mean(dim=1).float()


class ActionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_IN, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )
    def forward(self, x):
        return self.net(x)


def build_bbox_feat(frames, t):
    bbox = []
    for k in range(WINDOW):
        idx = max(0, t - (WINDOW - 1 - k))
        fr = frames[idx]
        bbox.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])
    return np.array(bbox, dtype=np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    data = json.loads(DATA_PATH.read_text())
    ep_labels = [ep["path_type"] for ep in data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
    val_eps = [data[i] for i in te_idx]
    print(f"Val: {len(val_eps)} episodes")

    print("[MODEL] Stage 1 LoRA 로드 중...")
    clip_enc = FrozenCLIPLoRA(device).to(device).eval()

    ckpt = torch.load(str(CKPT_PATH), map_location=device)
    mlp = ActionMLP().to(device)
    mlp.load_state_dict(ckpt["mlp"])
    mlp.eval()
    print(f"[MODEL] Stage 2 MLP 로드: val_acc_at_save={ckpt['val_acc']:.4f}")

    # 평가
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    path_correct = {}
    path_total = {}

    with torch.no_grad():
        for ep in val_eps:
            try:
                images = [Image.fromarray(h5py.File(ep["episode"], "r")["observations"]["images"][i])
                          for i in range(len(ep["frames"]))]
            except:
                continue
            vis_feats = clip_enc.encode(images, device)
            frames = ep["frames"]
            pt = ep.get("path_type", "unknown")

            for t, frame in enumerate(frames):
                bbox = torch.tensor(build_bbox_feat(frames, t), device=device)
                feat = torch.cat([bbox, vis_feats[t]]).unsqueeze(0)
                pred = mlp(feat).argmax(1).item()
                gt = frame["gt_class"]
                confusion[gt][pred] += 1

                if pt not in path_correct:
                    path_correct[pt] = 0
                    path_total[pt] = 0
                path_correct[pt] += int(pred == gt)
                path_total[pt] += 1

    # 결과 출력
    total = confusion.sum()
    correct = np.diag(confusion).sum()
    print(f"\n{'='*55}")
    print(f"  Exp54 Stage2 PM 평가")
    print(f"  전체 PM: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  (Exp53: 94.7%  Exp49: 96.4%)")
    print(f"{'='*55}")

    print(f"\n{'클래스':<10} {'정답':>6} {'전체':>6} {'정확도':>8}")
    print("-" * 35)
    for c in range(NUM_CLASSES):
        row_total = confusion[c].sum()
        row_correct = confusion[c][c]
        acc = row_correct / row_total * 100 if row_total > 0 else 0
        print(f"  {CLASS_NAMES[c]:<8} {row_correct:>6} {row_total:>6} {acc:>7.1f}%")

    print(f"\n혼동 행렬 (행=정답, 열=예측):")
    header = "       " + "".join(f"{n:>8}" for n in CLASS_NAMES)
    print(header)
    for c in range(NUM_CLASSES):
        row = f"{CLASS_NAMES[c]:<7}" + "".join(f"{confusion[c][p]:>8}" for p in range(NUM_CLASSES))
        print(row)

    print(f"\npath_type별 정확도:")
    for pt in sorted(path_total.keys()):
        acc = path_correct[pt] / path_total[pt] * 100 if path_total[pt] > 0 else 0
        print(f"  {pt:<20} {path_correct[pt]:>4}/{path_total[pt]:<4} = {acc:.1f}%")


if __name__ == "__main__":
    main()
