# -*- coding: utf-8 -*-
import json
import random
import warnings
import numpy as np
import sys
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path("/home/minum/26CS/MoNaVLA")
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image

# Kosmos-2 백본 및 이미지 프로젝션 레이어 경로 정의
VLM = ROOT/".vlms/kosmos-2-patch14-224"
S1  = ROOT/"runs/v5_nav/mlp/exp54/stage1_v2/stage1_v2_projs.pt"
ANN = ROOT/"docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"
OUT = ROOT/"runs/v5_nav/mlp/exp60"
WINDOW = 8
MIRROR = {0:0, 1:1, 2:3, 3:2, 4:5, 5:4, 6:7, 7:6}

# 테스트 모드 아규먼트 체크
TEST_RUN = "--test-run" in sys.argv
if TEST_RUN:
    sys.argv.remove("--test-run")

TAG       = sys.argv[1]   # A2 A3 B1 B2
USE_HSV   = sys.argv[2] == "hsv"
DO_FLIP   = sys.argv[3] == "flip"
print(f"=== {TAG}: hsv={USE_HSV} flip={DO_FLIP} (TestRun={TEST_RUN}) ===")

class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        # 스테이지 1의 이미지 프로젝션 가중치 로드
        ck = torch.load(str(S1), map_location="cuda", weights_only=False)
        self.proc = AutoProcessor.from_pretrained(str(VLM))
        base = AutoModelForVision2Seq.from_pretrained(str(VLM), torch_dtype=torch.float16)
        self.vm = base.vision_model.to("cuda").eval()
        self.proj = nn.Linear(1024, 256).to("cuda")
        self.proj.load_state_dict(ck["image_proj"])
        self.proj.eval()

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
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 8)
        )
    def forward(self, x): 
        return self.net(x)

# 데이터셋 로드
with open(ANN) as f: 
    ann = json.load(f)

# 유효한 에피소드만 필터링
ann = [ep for ep in ann if ep.get("path_type", "") not in ("", "free", "unknown")]
random.seed(42)
np.random.seed(42)
random.shuffle(ann)

n_val = max(1, int(len(ann) * 0.15))
val_eps, train_eps = ann[:n_val], ann[n_val:]
print(f"전체 필터링 전 - Train:{len(train_eps)} Val:{len(val_eps)}")

# 백본 모델 로드
enc = Enc().to("cuda").eval()

def make(eps):
    X, y = [], []
    for ep in eps:
        h5 = Path(ep["episode"])
        if not h5.exists(): 
            continue
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        if not frames: 
            continue
        
        # USE_HSV 모드일 때, HSV 관련 데이터가 누락되었거나 None인 프레임이 있으면 해당 에피소드를 통째로 제외
        if USE_HSV:
            invalid = False
            for fr in frames:
                if "cx_det_hsv" not in fr or fr["cx_det_hsv"] is None:
                    invalid = True
                    break
            if invalid:
                continue
                
        try:
            with h5py.File(str(h5), "r") as f: 
                imgs = f["observations"]["images"][:]
        except: 
            continue
            
        origs = [Image.fromarray(imgs[fr["frame_idx"]].astype("uint8")) for fr in frames]
        vo = torch.stack([enc.encode(p) for p in origs])
        vf = torch.stack([enc.encode(p.transpose(Image.FLIP_LEFT_RIGHT)) for p in origs]) if DO_FLIP else None
        
        for t, fr in enumerate(frames):
            gc = fr["gt_class"]
            ho = []
            hf = []
            for k in range(WINDOW):
                fi = max(0, t - (WINDOW - 1 - k))
                f2 = frames[fi]
                if USE_HSV:
                    cx = f2.get("cx_det_hsv")
                    cy = f2.get("cy_det_hsv")
                    ar = f2.get("area_det_hsv")
                else:
                    cx = f2.get("cx_det")
                    cy = f2.get("cy_det")
                    ar = f2.get("area_det")
                
                # None 값 방어 코드 추가 (nan loss 방지)
                if cx is None: cx = 0.5
                if cy is None: cy = 0.5
                if ar is None: ar = 0.05
                
                hb = float(f2.get("has_bbox", f2.get("detected", False)))
                
                ho.extend([cx, cy, ar, hb])
                if DO_FLIP: 
                    hf.extend([1 - cx, cy, ar, hb])
                    
            X.append(ho + vo[t].cpu().tolist())
            y.append(gc)
            if DO_FLIP: 
                X.append(hf + vf[t].cpu().tolist())
                y.append(MIRROR.get(gc, gc))
                
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

Xtr, ytr = make(train_eps)
Xva, yva = make(val_eps)
print(f"최종 매핑 후 - Train:{len(Xtr)} Val:{len(Xva)}")

Xtr_t = torch.from_numpy(Xtr).cuda()
ytr_t = torch.from_numpy(ytr).cuda()
Xva_t = torch.from_numpy(Xva).cuda()
yva_t = torch.from_numpy(yva).cuda()

mlp = MLP().cuda()
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

# 테스트 런인 경우 에포크 수 1로 제한
total_epochs = 1 if TEST_RUN else 300
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_epochs)
best = 0.0

for ep in range(1, total_epochs + 1):
    mlp.train()
    perm = torch.randperm(len(Xtr_t), device="cuda")
    ls = 0
    for i in range(0, len(perm), 256):
        idx = perm[i:i+256]
        l = F.cross_entropy(mlp(Xtr_t[idx]), ytr_t[idx])
        opt.zero_grad()
        l.backward()
        opt.step()
        ls += l.item()
    sched.step()
    
    # 주기적인 중간 평가 및 저장
    if ep % 100 == 0 or ep == total_epochs:
        mlp.eval()
        with torch.no_grad(): 
            acc = (mlp(Xva_t).argmax(1) == yva_t).float().mean().item()
        print(f"  epoch {ep:3d}  loss={ls:.2f}  val_acc={acc*100:.1f}%")
        if acc >= best:
            best = acc
            fname = f"abl_{TAG.lower()}_mlp.pt"
            torch.save({"mlp": mlp.state_dict(), "val_acc": acc, "d_in": 288, "tag": TAG}, str(OUT/fname))
            print(f"    [BEST] {acc*100:.1f}% → {fname}")
            
print(f"{TAG} 완료: {best*100:.1f}%")
