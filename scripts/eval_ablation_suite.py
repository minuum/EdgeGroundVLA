#!/usr/bin/env python3
"""
Exp66 기준 동일 val set으로 전체 ablation 모델 CL 평가.
출력: docs/v5/ablation_suite_results.json + 터미널 표
"""
import sys, json, warnings, re
import numpy as np
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch, torch.nn as nn, torch.nn.functional as F
import h5py
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

from scripts.sim.rollout_core import ACTION_VEL, DT_DEFAULT, build_trajectory, compute_metrics
from transformers import AutoModelForVision2Seq, AutoProcessor

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_PT = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
EVAL_DATA = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_base_pg2_cx.json"  # Exp66 기준 150ep
RUNS      = ROOT / "runs/v5_nav/mlp"
OUT       = ROOT / "docs/v5/ablation_suite_results.json"

NUM_CLASSES = 8; VIS_DIM = 1024; PROJ_DIM = 256

# ── 평가 대상 ────────────────────────────────────────────────────────
ABLATIONS = [
    # (label, group, ckpt, head, window)
    # Pipeline
    ("Exp65b (no L2, no aug)",  "pipeline", RUNS/"exp65b/stage2_pg2cx_mlp.pt",        "mlp", 8),
    ("Exp66 MLP w=8 ★",         "pipeline", RUNS/"exp66/action_mlp.pt",               "mlp", 8),
    # Head
    ("Linear",                  "head",     RUNS/"exp68/action_linear.pt",            "linear", 8),
    ("FCHead",                  "head",     RUNS/"exp69/action_fc.pt",                "fc",     8),
    ("LSTM w=8",                "head",     RUNS/"exp70/action_lstm_w8.pt",           "lstm",   8),
    ("MLP w=8 ★",               "head",     RUNS/"exp66/action_mlp.pt",               "mlp",    8),
    # Window — MLP
    ("MLP w=2",                 "window",   RUNS/"ablation_window/mlp_w2.pt",         "mlp",    2),
    ("MLP w=4",                 "window",   RUNS/"ablation_window/mlp_w4.pt",         "mlp",    4),
    ("MLP w=8 ★",               "window",   RUNS/"exp66/action_mlp.pt",               "mlp",    8),
    ("MLP w=16",                "window",   RUNS/"ablation_window/mlp_w16.pt",        "mlp",   16),
    # Window — LSTM
    ("LSTM w=4",                "window",   RUNS/"ablation_window/lstm_w4.pt",        "lstm",   4),
    ("LSTM w=8",                "window",   RUNS/"exp70/action_lstm_w8.pt",           "lstm",   8),
    ("LSTM w=16",               "window",   RUNS/"ablation_window/lstm_w16.pt",       "lstm",  16),
]

# ── Stage1 encoder ───────────────────────────────────────────────────
class FrozenCLIPV2(nn.Module):
    def __init__(self, device):
        super().__init__()
        ckpt = torch.load(str(STAGE1_PT), map_location=device, weights_only=False)
        self.proc = AutoProcessor.from_pretrained(str(VLM_PATH))
        base = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16)
        self.vm   = base.vision_model.to(device).eval()
        self.proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.proj.load_state_dict(ckpt["image_proj"])
        for p in self.parameters(): p.requires_grad_(False)
        self.device = device
    @torch.no_grad()
    def encode(self, pil):
        inp = self.proc(images=[pil], return_tensors="pt")
        pv  = inp["pixel_values"].to(self.device, dtype=torch.float16)
        feat = self.vm(pixel_values=pv).last_hidden_state.mean(1).float()
        return F.normalize(self.proj(feat), dim=-1)[0]

# ── Head models ──────────────────────────────────────────────────────
class ActionMLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in,256),nn.ReLU(),nn.Dropout(0.25),
                                  nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),
                                  nn.Linear(128,64), nn.ReLU(),nn.Dropout(0.1),
                                  nn.Linear(64,NUM_CLASSES))
    def forward(self,x): return self.net(x)

class LinearHead(nn.Module):
    def __init__(self, d_in): super().__init__(); self.net=nn.Linear(d_in,NUM_CLASSES)
    def forward(self,x): return self.net(x)

class FCHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_in,1024),nn.ReLU(),nn.Linear(1024,512),nn.ReLU(),
                                nn.Linear(512,256),nn.ReLU(),nn.Linear(256,NUM_CLASSES))
    def forward(self,x): return self.net(x)

class LSTMHead(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.lstm = nn.LSTM(PROJ_DIM+4, 256, 2, batch_first=True, dropout=0.1)
        self.classifier = nn.Linear(256, NUM_CLASSES)
    def forward(self,x): out,_=self.lstm(x); return self.classifier(out[:,-1])

HEAD_MAP = {"mlp": ActionMLP, "linear": LinearHead, "fc": FCHead, "lstm": LSTMHead}

def load_head(ckpt_path, head_name, window, device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    w    = ckpt.get("window", window)
    d_in = w*4 + PROJ_DIM
    H    = HEAD_MAP[head_name]
    head = (H(w) if head_name=="lstm" else H(d_in=d_in)).to(device)
    head.load_state_dict(ckpt["mlp"])
    return head.eval(), w, float(ckpt.get("val_acc",0))

# ── CL rollout for one episode ───────────────────────────────────────
def rollout(ep, enc, head, head_name, window, device):
    h5p = Path(ep["episode"])
    if not h5p.exists(): return None
    frames = ep["frames"]
    with h5py.File(str(h5p)) as f:
        imgs_np = f["observations"]["images"][:]

    history = []
    actions  = []
    for t, fr in enumerate(frames):
        pil  = Image.fromarray(imgs_np[fr["frame_idx"]].astype("uint8"))
        vf   = enc.encode(pil)
        cx   = fr.get("cx", fr.get("cx_det", 0.5))
        cy   = fr.get("cy", fr.get("cy_det", 0.5))
        area = fr.get("area", fr.get("area_det", 0.05))
        has  = float(fr.get("has_bbox", fr.get("detected", False)))
        history.append({"vf":vf,"cx":cx,"cy":cy,"area":area,"has":has})

        with torch.no_grad():
            if head_name == "lstm":
                seq=[]
                for k in range(window):
                    idx=max(0,t-(window-1-k)); h=history[idx]
                    seq.append(torch.cat([h["vf"],torch.tensor([h["cx"],h["cy"],h["area"],h["has"]],device=device)]))
                x=torch.stack(seq).unsqueeze(0)
            else:
                bbox=[]
                for k in range(window):
                    idx=max(0,t-(window-1-k)); h=history[idx]
                    bbox.extend([h["cx"],h["cy"],h["area"],h["has"]])
                x=torch.cat([torch.tensor(bbox,device=device,dtype=torch.float32),vf]).unsqueeze(0)
            pred=int(head(x).argmax(1).item())
        actions.append(pred)

    gt_acts  = [fr["gt_class"] for fr in frames]
    exp_traj = build_trajectory(gt_acts)
    pred_traj= build_trajectory(actions)
    return compute_metrics(exp_traj, pred_traj)

# ── Fixed val set (Exp66 split, seed=42) ────────────────────────────
def get_val_eps(data):
    labels = [e["path_type"] for e in data]
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        _, idx = next(sss.split(np.zeros(len(data)), labels))
    except:
        from sklearn.model_selection import ShuffleSplit
        ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        _, idx = next(ss.split(np.zeros(len(data))))
    return [data[i] for i in idx]

# ── Main ─────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = json.loads(EVAL_DATA.read_text())
    val_eps = get_val_eps(data)
    print(f"Fixed val set: {len(val_eps)} episodes (Exp66 기준, seed=42)\n")

    print("Stage1 로드 중...")
    enc = FrozenCLIPV2(device)

    results = []
    fmt = f"{'Label':<28} {'Group':<10} {'SR':>6} {'ok/n':>6} {'FPE':>8} {'val_acc':>8}"
    print(fmt); print("-"*70)

    for label, group, ckpt_path, head_name, window in ABLATIONS:
        if not Path(ckpt_path).exists():
            print(f"  SKIP {label} (ckpt 없음)")
            continue
        head, w, val_acc = load_head(ckpt_path, head_name, window, device)
        ok=0; fpes=[]
        for ep in val_eps:
            m = rollout(ep, enc, head, head_name, w, device)
            if m is None: continue
            if m["success"]: ok+=1
            fpes.append(m["fpe"])
        n   = len([e for e in val_eps if Path(e["episode"]).exists()])
        sr  = ok/n if n else 0
        fpe = float(np.mean(fpes)) if fpes else 0
        print(f"  {label:<28} {group:<10} {sr*100:5.1f}%  {ok:2d}/{n:2d}  {fpe:7.3f}m  {val_acc*100:6.1f}%")
        results.append({"label":label,"group":group,"sr":sr,"ok":ok,"n":n,"fpe":fpe,"val_acc":val_acc,"head":head_name,"window":int(w)})

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT}")

if __name__=="__main__":
    main()
