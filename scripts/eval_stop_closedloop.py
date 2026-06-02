# -*- coding: utf-8 -*-
"""
STOP 규칙 Closed-Loop 평가 (plan_20260602_stop_arrival_rule.md S2~S4)

precomputed pg2_cx.json(area_det/cx_det/gt_class)을 사용 — abl 학습 split과 동일.
4-cell 비교:
  expert: raw(gt_class 그대로) | synth(도착 area 규칙으로 STOP 래치)
  pred  : STOP override off | on

파이프라인: CLIP 인코딩(이미지) + bbox(area_det/cx_det) → MLP → 8-class
            → (옵션) STOP override → rollout_core로 궤적/FPE

Usage:
  .venv/bin/python3 scripts/eval_stop_closedloop.py
  .venv/bin/python3 scripts/eval_stop_closedloop.py --mlp runs/v5_nav/mlp/exp60/abl_b3_mlp.pt \
      --th-area 0.5 --th-cx 0.3 --window-avg 5 --min-steps 0
"""
import json, random, warnings, argparse, sys
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import h5py
from PIL import Image
from scripts.sim.rollout_core import build_trajectory, compute_metrics

VLM = ROOT / ".vlms/kosmos-2-patch14-224"
S1  = ROOT / "runs/v5_nav/mlp/exp54/stage1_v2/stage1_v2_projs.pt"
ANN = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"
OUTDIR = ROOT / "docs/v5/closed_loop_eval"
WINDOW = 8


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
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 8))
    def forward(self, x): return self.net(x)


def bbox_window(frames, t):
    """frame t 기준 WINDOW개 (cx,cy,area,has_bbox) → 32-vec (pg2 area_det)."""
    arr = []
    for k in range(WINDOW):
        f2 = frames[max(0, t - (WINDOW - 1 - k))]
        cx = f2.get("cx_det") or 0.5
        cy = f2.get("cy_det") or 0.5
        ar = f2.get("area_det") or 0.05
        hb = float(f2.get("has_bbox", f2.get("detected", False)))
        arr.extend([cx, cy, ar, hb])
    return arr


def stop_trigger_idx(frames, th_area, th_cx, W, min_steps):
    """도착 area 규칙 → 최초 STOP 트리거 frame idx (없으면 None). 출발스파이크 없어 rising 생략."""
    buf = []
    for i, fr in enumerate(frames):
        buf.append(fr.get("area_det") or 0.0)
        if len(buf) > W: buf = buf[-W:]
        area_avg = float(np.mean(buf))
        cx = fr.get("cx_det") or 0.5
        if i >= min_steps and area_avg > th_area and abs(cx - 0.5) < th_cx:
            return i
    return None


def apply_latched_stop(actions, trig):
    """trig 이후(포함) 전부 STOP(0)으로 래치."""
    if trig is None:
        return list(actions)
    return [a if i < trig else 0 for i, a in enumerate(actions)]


def expert_synth_stop(frames, th_area, th_cx, W, min_steps):
    """expert: gt_class 사용하되 도착 규칙으로 STOP 래치(이미 STOP인 ep는 그대로 유지)."""
    gt = [fr["gt_class"] for fr in frames]
    # gt에 이미 STOP 있으면 그 위치, 없으면 규칙 트리거
    gt_stop = next((i for i, c in enumerate(gt) if c == 0), None)
    trig = gt_stop if gt_stop is not None else stop_trigger_idx(frames, th_area, th_cx, W, min_steps)
    return apply_latched_stop(gt, trig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mlp", default=str(ROOT / "runs/v5_nav/mlp/exp60/abl_b3_mlp.pt"))
    p.add_argument("--th-area", type=float, default=0.5)
    p.add_argument("--th-cx",   type=float, default=0.3)
    p.add_argument("--window-avg", type=int, default=5, help="area 평균 윈도 W")
    p.add_argument("--min-steps",  type=int, default=0)
    p.add_argument("--success-fpe", type=float, default=0.5)
    p.add_argument("--out-tag", default="stop")
    args = p.parse_args()

    ann = json.loads(ANN.read_text())
    ann = [ep for ep in ann if ep.get("path_type", "") not in ("", "free", "unknown")]
    random.seed(42); np.random.seed(42); random.shuffle(ann)
    n_val = max(1, int(len(ann) * 0.15))
    val_eps = ann[:n_val]
    print(f"[DATA] val {len(val_eps)} ep  (split seed=42, 15%)")

    enc = Enc().to("cuda").eval()
    ck = torch.load(args.mlp, map_location="cuda", weights_only=False)
    mlp = MLP().cuda(); mlp.load_state_dict(ck["mlp"]); mlp.eval()
    print(f"[MLP] {Path(args.mlp).name}  val_acc={ck.get('val_acc',0)*100:.1f}%")
    print(f"[RULE] th_area={args.th_area} th_cx={args.th_cx} W={args.window_avg} min_steps={args.min_steps}\n")

    # cell 누적
    cells = {("raw", "off"): [], ("raw", "on"): [], ("synth", "off"): [], ("synth", "on"): []}

    for ep in val_eps:
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        h5 = Path(ep["episode"])
        if not frames or not h5.exists():
            continue
        try:
            with h5py.File(str(h5), "r") as f:
                imgs = f["observations"]["images"][:]
        except Exception:
            continue
        # MLP per-frame 예측
        pred = []
        with torch.no_grad():
            for t, fr in enumerate(frames):
                vis = enc.encode(Image.fromarray(imgs[fr["frame_idx"]].astype("uint8")))
                x = torch.tensor(bbox_window(frames, t) + vis.cpu().tolist(),
                                 dtype=torch.float32, device="cuda").unsqueeze(0)
                pred.append(int(mlp(x).argmax(1).item()))

        trig = stop_trigger_idx(frames, args.th_area, args.th_cx, args.window_avg, args.min_steps)
        pred_on  = apply_latched_stop(pred, trig)
        exp_raw   = [fr["gt_class"] for fr in frames]
        exp_synth = expert_synth_stop(frames, args.th_area, args.th_cx, args.window_avg, args.min_steps)

        variants = {
            ("raw",   "off"): (exp_raw,   pred),
            ("raw",   "on"):  (exp_raw,   pred_on),
            ("synth", "off"): (exp_synth, pred),
            ("synth", "on"):  (exp_synth, pred_on),
        }
        for key, (e, pr) in variants.items():
            m = compute_metrics(build_trajectory(e), build_trajectory(pr), args.success_fpe)
            cells[key].append(m)

    print(f"{'expert':<8}{'pred_stop':<11}{'CL success':<15}{'mean_FPE':<11}{'mean_TLD'}")
    print("-" * 56)
    summary = {}
    for key in [("raw","off"),("raw","on"),("synth","off"),("synth","on")]:
        ms = cells[key]
        if not ms: continue
        sr = np.mean([m["success"] for m in ms])
        fpe = np.mean([m["fpe"] for m in ms]); tld = np.mean([m["tld"] for m in ms])
        n = len(ms)
        print(f"{key[0]:<8}{key[1]:<11}{sr*100:>5.1f}% ({round(sr*n)}/{n})    {fpe:>7.3f}m   {tld:.3f}")
        summary[f"{key[0]}_{key[1]}"] = {"success_rate": float(sr), "mean_fpe": float(fpe),
                                          "mean_tld": float(tld), "n": n}

    out = OUTDIR / f"stop_closedloop_result_{args.out_tag}.json"
    out.write_text(json.dumps({
        "mlp": Path(args.mlp).name,
        "rule": {"th_area": args.th_area, "th_cx": args.th_cx,
                 "W": args.window_avg, "min_steps": args.min_steps},
        "cells": summary}, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] {out}")


if __name__ == "__main__":
    main()
