# -*- coding: utf-8 -*-
"""
Table 2 — 전체 파이프라인 CL ablation (Grounding × Control).

grounding 캐시(gen_grounding_ablation.py 산출) × Control MLP 조합으로 closed-loop 평가.
CLIP vis feature는 프레임별 1회 계산 후 재사용(메모리 캐시).

Grounding 소스: hsv(cx_det_hsv) · base · exp57 · exp58 · exp59
Control MLP   : exp54(HSV학습) · b1(pg2 243ep) · flip · stop65

산출: docs/v5/grounding_ablation/table2.json

Usage:
  .venv/bin/python3 scripts/eval_pipeline_ablation.py --success-fpe 0.5
"""
import sys, json, argparse, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, torch.nn as nn, h5py
from PIL import Image
from scripts.sim.rollout_core import build_trajectory, compute_metrics

VLM = ROOT / ".vlms/kosmos-2-patch14-224"
S1  = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
ANN = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"
GDIR = ROOT / "docs/v5/grounding_ablation"
WINDOW = 8

CONTROLS = {
    "exp54_HSV": ROOT / "runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt",
    "b1":        ROOT / "runs/v5_nav/mlp/exp60/abl_b1_mlp.pt",
    "flip":      ROOT / "runs/v5_nav/mlp/exp60/stage2_pg2cx_flip_mlp.pt",
    "stop65":    ROOT / "runs/v5_nav/mlp/exp60/stop65_mlp.pt",
}
GROUNDERS = ["hsv", "base", "exp57", "exp58", "exp59"]


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ck = torch.load(str(S1), map_location="cuda", weights_only=False)
        self.proc = AutoProcessor.from_pretrained(str(VLM))
        base = AutoModelForVision2Seq.from_pretrained(str(VLM), torch_dtype=torch.float16)
        self.vm = base.vision_model.to("cuda").eval()
        self.proj = nn.Linear(1024, 256).to("cuda"); self.proj.load_state_dict(ck["image_proj"]); self.proj.eval()
    @torch.no_grad()
    def encode(self, pil):
        inp = self.proc(images=[pil], return_tensors="pt")
        pv = inp["pixel_values"].to("cuda", dtype=torch.float16)
        return self.proj(self.vm(pixel_values=pv).last_hidden_state.mean(1).float()).squeeze(0)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(288,256),nn.ReLU(),nn.Dropout(0.25),
            nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.1),nn.Linear(64,8))
    def forward(self,x): return self.net(x)


def load_grounding_source(tag):
    """tag별 {episode: [ (cx,cy,area,hit,gt_class,frame_idx) ... ]} 반환."""
    if tag == "hsv":
        ann = json.loads(ANN.read_text())
        ann = [ep for ep in ann if ep.get("path_type","") not in ("","free","unknown")]
        import random; random.seed(42); np.random.seed(42); random.shuffle(ann)
        ann = ann[:max(1,int(len(ann)*0.15))]
        out = {}
        for ep in ann:
            rows = []
            for fr in ep["frames"]:
                if fr.get("gt_class") is None: continue
                cx = fr.get("cx_det_hsv"); cy = fr.get("cy_det_hsv"); ar = fr.get("area_det_hsv")
                hit = cx is not None
                rows.append((cx, cy, ar, hit, fr["gt_class"], fr["frame_idx"]))
            out[ep["episode"]] = rows
        return out
    cache = json.loads((GDIR / f"grounding_{tag}.json").read_text())
    out = {}
    for ep in cache:
        out[ep["episode"]] = [(f["cx"], f["cy"], f["area"], f["hit"], f["gt_class"], f["frame_idx"]) for f in ep["frames"]]
    return out


def bbox_window_from(rows, t):
    arr = []
    for k in range(WINDOW):
        i = max(0, t - (WINDOW - 1 - k))
        cx, cy, ar, hit, _, _ = rows[i]
        arr.extend([cx if cx is not None else 0.5, cy if cy is not None else 0.5,
                    ar if ar is not None else 0.05, 1.0 if hit else 0.0])
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--success-fpe", type=float, default=0.5)
    args = ap.parse_args()
    dev = torch.device("cuda")

    enc = Enc().to(dev).eval()
    # grounding 소스 로드
    sources = {}
    for g in GROUNDERS:
        try:
            sources[g] = load_grounding_source(g)
        except Exception as e:
            print(f"  [skip grounding {g}] {e}")
    # vis feat 캐시 (episode,frame_idx) → tensor
    print("[CACHE] CLIP vis feature 추출...", flush=True)
    visfeat = {}
    any_src = sources[next(iter(sources))]
    for epname, rows in any_src.items():
        h5 = Path(epname)
        if not h5.exists(): continue
        with h5py.File(str(h5), "r") as f: imgs = f["observations"]["images"][:]
        for (_,_,_,_,_,fi) in rows:
            if (epname, fi) not in visfeat and fi < len(imgs):
                visfeat[(epname, fi)] = enc.encode(Image.fromarray(imgs[fi].astype("uint8"))).cpu()
    print(f"[CACHE] {len(visfeat)} frames", flush=True)

    # control MLP 로드
    mlps = {}
    for cname, p in CONTROLS.items():
        if not p.exists(): print(f"  [skip control {cname}] no file"); continue
        ck = torch.load(str(p), map_location=dev, weights_only=False)
        m = MLP().to(dev); m.load_state_dict(ck["mlp"]); m.eval(); mlps[cname] = m

    table2 = {}
    for g, src in sources.items():
        table2[g] = {}
        for cname, mlp in mlps.items():
            ms = []
            with torch.no_grad():
                for epname, rows in src.items():
                    pred, expert = [], []
                    for t, (_,_,_,_,gt,fi) in enumerate(rows):
                        vf = visfeat.get((epname, fi))
                        if vf is None: continue
                        x = torch.tensor(bbox_window_from(rows, t) + vf.tolist(),
                                         dtype=torch.float32, device=dev).unsqueeze(0)
                        pred.append(int(mlp(x).argmax(1).item())); expert.append(gt)
                    if len(pred) >= 2:
                        ms.append(compute_metrics(build_trajectory(expert), build_trajectory(pred), args.success_fpe))
            sr = float(np.mean([m["success"] for m in ms])) if ms else 0.0
            fpe = float(np.mean([m["fpe"] for m in ms])) if ms else 0.0
            table2[g][cname] = {"success_rate": sr, "mean_fpe": fpe, "n": len(ms)}
            print(f"  {g:<6} × {cname:<10}: SR={sr*100:>5.1f}%  FPE={fpe:.3f}m  (n={len(ms)})", flush=True)

    (GDIR / "table2.json").write_text(json.dumps({"success_fpe": args.success_fpe, "table": table2}, indent=2, ensure_ascii=False))
    print("\n===== Table 2 (Grounding × Control, CL success) =====")
    hdr = "grounding".ljust(8) + "".join(c.rjust(12) for c in mlps)
    print(hdr); print("-"*len(hdr))
    for g in GROUNDERS:
        if g not in table2: continue
        row = g.ljust(8) + "".join(f"{table2[g][c]['success_rate']*100:>11.1f}%" for c in mlps)
        print(row)
    print(f"\n[SAVE] {GDIR}/table2.json")


if __name__ == "__main__":
    main()
