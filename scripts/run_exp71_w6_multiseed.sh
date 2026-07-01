#!/usr/bin/env bash
# exp71 Transformer WINDOW=6 × 5 seeds 학습 + CL eval
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENV="$ROOT/.venv/bin/python3"
LOG_DIR="$ROOT/logs/exp71_w6_multiseed"
mkdir -p "$LOG_DIR"
SEEDS=(0 1 2 3 4)

echo "======================================================"
echo " exp71 Transformer WINDOW=6 × 5 seeds"
echo "======================================================"

for SEED in "${SEEDS[@]}"; do
  OUT="$ROOT/runs/v5_nav/mlp/exp71_w6_seed${SEED}"
  LOG="$LOG_DIR/train_seed${SEED}.log"
  echo ""
  echo "[TRAIN] seed=$SEED → $OUT"
  $VENV scripts/train_exp71_stage2_transformer.py \
    --window 6 --seed "$SEED" --out-dir "$OUT" \
    2>&1 | tee "$LOG" | grep -E "BEST|val_acc|epoch 300|결과"
done

echo ""
echo "======================================================"
echo " CL Evaluation (WINDOW=6, 5 seeds)"
echo "======================================================"

$VENV - << 'PYEOF'
import sys, json, warnings, statistics
import numpy as np
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

import torch, torch.nn as nn, h5py
from PIL import Image
from scripts.sim.rollout_core import build_trajectory, continuous_to_class, compute_metrics, DT_DEFAULT

DATA_DIR  = ROOT / "ROS_action/mobile_vla_dataset_v5"
STEP1_DIR = ROOT / "docs/v5/bbox_nav_step1"
ANN_PG448 = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json"
VLM_PATH  = ROOT / ".vlms/kosmos-2-patch14-224"
STAGE1_PT = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
LOG_DIR   = ROOT / "logs/exp71_w6_multiseed"
WINDOW=6; CLIP_PROJ=256; CLIP_VIS=1024; NUM_CLASSES=8; FD=CLIP_PROJ+4

# 인코더 로드
from transformers import AutoModelForVision2Seq, AutoProcessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt0  = torch.load(str(STAGE1_PT), map_location=device, weights_only=False)
proc   = AutoProcessor.from_pretrained(str(VLM_PATH))
base   = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16)
vm     = base.vision_model.to(device).eval()
proj   = nn.Linear(CLIP_VIS, CLIP_PROJ).to(device)
proj.load_state_dict(ckpt0["image_proj"]); proj.eval()
print("[ENCODER] 로드 완료")

def enc(img_np):
    img = Image.fromarray(img_np.astype("uint8"))
    inp = proc(images=[img], return_tensors="pt")
    pv  = inp["pixel_values"].to(device, dtype=torch.float16)
    with torch.no_grad():
        f = vm(pixel_values=pv).last_hidden_state.mean(1).float()
        return proj(f).squeeze(0)

# 테스트 split
ann_pg448 = json.loads(ANN_PG448.read_text())
ep_stems  = {Path(ep["episode"]).stem: ep for ep in ann_pg448}
bbox_ds   = json.loads((STEP1_DIR/"bbox_dataset.json").read_text())
by_path   = defaultdict(list)
for i,ep in enumerate(bbox_ds): by_path[ep["path_type"]].append(i)
rng = np.random.default_rng(42); test_idx=[]
for _,idxs in by_path.items():
    rng.shuffle(idxs); test_idx.extend(idxs[:max(1,int(len(idxs)*0.2))])
test_eps=[ep_stems.get(bbox_ds[i]["episode"]) for i in test_idx
          if ep_stems.get(bbox_ds[i]["episode"]) and list(DATA_DIR.glob(f"{bbox_ds[i]['episode']}.h5"))]
print(f"[DATA] 테스트 {len(test_eps)}개")

class TH(nn.Module):
    def __init__(self,w):
        super().__init__(); self.w=w
        self.cls=nn.Parameter(torch.randn(1,1,FD))
        self.pe=nn.Embedding(w+1,FD)
        el=nn.TransformerEncoderLayer(d_model=FD,nhead=4,dim_feedforward=512,dropout=0.1,batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(el,num_layers=2)
        self.head=nn.Sequential(nn.LayerNorm(FD),nn.Linear(FD,128),nn.ReLU(),nn.Dropout(0.1),nn.Linear(128,NUM_CLASSES))
    def forward(self,x):
        B=x.size(0); c=self.cls.expand(B,-1,-1)
        x=torch.cat([c,x],1); x=x+self.pe(torch.arange(x.size(1),device=x.device))
        return self.head(self.enc(x)[:,0])

def eval_ep(ann_ep, head):
    frames=ann_ep["frames"]
    path=next(DATA_DIR.glob(f"{Path(ann_ep['episode']).stem}.h5"))
    with h5py.File(str(path),"r") as f:
        imgs=f["observations"]["images"][:]; expert=f["actions"][:]
    cache={}
    def gv(fi):
        if fi not in cache: cache[fi]=enc(imgs[fi])
        return cache[fi]
    preds=[]
    for t,fr in enumerate(frames):
        seq=[]
        for k in range(WINDOW):
            fi=max(0,t-(WINDOW-1-k)); f2=frames[fi]
            bbox=[f2.get("cx_det",.5),f2.get("cy_det",.5),f2.get("area_det",.05),float(f2.get("has_bbox",False))]
            seq.append(bbox+gv(frames[fi]["frame_idx"]).cpu().tolist())
        x=torch.tensor([seq],dtype=torch.float32,device=device)
        with torch.no_grad(): preds.append(min(int(head(x).argmax(1).item()),NUM_CLASSES-1))
    ec=[continuous_to_class(*a[:3]) for a in expert[:len(frames)]]
    return compute_metrics(build_trajectory(ec,DT_DEFAULT),build_trajectory(preds,DT_DEFAULT),0.5)

rows=[]; all_va=[]
for seed in range(5):
    pt=ROOT/f"runs/v5_nav/mlp/exp71_w6_seed{seed}/action_transformer.pt"
    if not pt.exists(): print(f"  [SKIP] seed={seed}"); continue
    ckpt=torch.load(str(pt),map_location=device,weights_only=False)
    head=TH(w=WINDOW).to(device); head.load_state_dict(ckpt["model"]); head.eval()
    va=ckpt.get("val_acc",0)*100
    srs=[]; fpes=[]
    for ann_ep in test_eps:
        try:
            m=eval_ep(ann_ep,head); srs.append(float(m["success"])); fpes.append(float(m["fpe"]))
        except: pass
    sr=np.mean(srs)*100; fpe=np.mean(fpes)
    rows.append({"seed":seed,"val_acc":va,"cl_sr":sr,"cl_fpe":fpe})
    all_va.append(va)
    print(f"  seed={seed}: val_acc={va:.1f}%  CL_SR={sr:.1f}%  FPE={fpe:.3f}m")

if rows:
    va_m=np.mean([r["val_acc"] for r in rows]); va_s=statistics.stdev([r["val_acc"] for r in rows])
    sr_m=np.mean([r["cl_sr"]  for r in rows]); sr_s=statistics.stdev([r["cl_sr"]  for r in rows])
    fp_m=np.mean([r["cl_fpe"] for r in rows]); fp_s=statistics.stdev([r["cl_fpe"] for r in rows])
    print(f"\n=== exp71 WINDOW=6 (5 seeds) ===")
    print(f"  val_acc = {va_m:.1f}±{va_s:.1f}%")
    print(f"  CL_SR   = {sr_m:.1f}±{sr_s:.1f}%")
    print(f"  FPE     = {fp_m:.3f}±{fp_s:.3f}m")
    summary={"window":6,"val_acc_mean":va_m,"val_acc_std":va_s,"cl_sr_mean":sr_m,"cl_sr_std":sr_s,"cl_fpe_mean":fp_m,"cl_fpe_std":fp_s,"raw":rows}
    out=LOG_DIR/"summary.json"
    with open(out,"w") as f: json.dump(summary,f,indent=2)
    print(f"  → {out}")
PYEOF

echo ""
echo "[DONE] exp71 WINDOW=6 × 5 seeds 완료"
