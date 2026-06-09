# -*- coding: utf-8 -*-
"""
학습된 STOP 모델의 추론 후처리 sweep — precision 보강으로 CL 향상.

학습 STOP은 recall↑ precision↓(접근 transient에 과발화). 후처리로 조기발화 억제:
  (나) confidence 임계  : STOP softmax prob > conf 일 때만 STOP 수용
  (가) latch + area gate: STOP이고 area_det>gate & cx 중앙이면 STOP 래치(이후 유지)
  조합 가능.

expert = 도착정지 목표(synth, area 규칙). pred = 학습 모델 후처리.
pg2_cx.json + CLIP feature 사용 (eval_stop_closedloop과 동일 split).

Usage:
  .venv/bin/python3 scripts/eval_learned_stop.py --mlp runs/v5_nav/mlp/exp60/stop65_mlp.pt
"""
import json, random, warnings, argparse, sys
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image
from scripts.sim.rollout_core import build_trajectory, compute_metrics
from scripts.eval_stop_closedloop import (Enc, MLP, bbox_window, ANN,
                                          stop_trigger_idx, apply_latched_stop,
                                          expert_synth_stop, WINDOW)


def postprocess(argmax, stop_prob, areas, cxs, conf, area_gate, th_cx, latch):
    """학습 STOP을 conf 임계 / area-gate / latch로 정제."""
    out = []
    stopped = False
    for i in range(len(argmax)):
        if stopped:
            out.append(0); continue
        c = argmax[i]
        is_stop = (c == 0)
        if conf > 0:                       # (나) confidence: 확신할 때만 STOP
            is_stop = (stop_prob[i] > conf)
        if is_stop and area_gate > 0:      # (가) area gate: 큰·중앙 basket일 때만 인정
            if not (areas[i] > area_gate and abs(cxs[i] - 0.5) < th_cx):
                is_stop = False
        if is_stop:
            if latch:
                stopped = True
            out.append(0)
        else:
            # STOP 거부 시 비-STOP 대체: argmax가 0이면 차순위
            out.append(c if c != 0 else 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mlp", default=str(ROOT / "runs/v5_nav/mlp/exp60/stop65_mlp.pt"))
    p.add_argument("--success-fpe", type=float, default=0.15)
    p.add_argument("--th-cx", type=float, default=0.3)
    p.add_argument("--out-tag", default="learnedpp")
    args = p.parse_args()

    ann = json.loads(ANN.read_text())
    ann = [ep for ep in ann if ep.get("path_type", "") not in ("", "free", "unknown")]
    random.seed(42); np.random.seed(42); random.shuffle(ann)
    val_eps = ann[:max(1, int(len(ann) * 0.15))]
    print(f"[DATA] val {len(val_eps)} ep   [MLP] {Path(args.mlp).name}")

    enc = Enc().to("cuda").eval()
    ck = torch.load(args.mlp, map_location="cuda", weights_only=False)
    mlp = MLP().cuda(); mlp.load_state_dict(ck["mlp"]); mlp.eval()

    # 에피소드별 per-frame 캐시
    episodes = []
    for ep in val_eps:
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        h5 = Path(ep["episode"])
        if not frames or not h5.exists(): continue
        try:
            with h5py.File(str(h5), "r") as f: imgs = f["observations"]["images"][:]
        except Exception: continue
        argmax, sprob, areas, cxs = [], [], [], []
        with torch.no_grad():
            for t, fr in enumerate(frames):
                vis = enc.encode(Image.fromarray(imgs[fr["frame_idx"]].astype("uint8")))
                x = torch.tensor(bbox_window(frames, t) + vis.cpu().tolist(),
                                 dtype=torch.float32, device="cuda").unsqueeze(0)
                prob = F.softmax(mlp(x), dim=-1)[0]
                argmax.append(int(prob.argmax().item()))
                sprob.append(float(prob[0].item()))         # STOP prob
                areas.append(fr.get("area_det") or 0.0)
                cxs.append(fr.get("cx_det") or 0.5)
        exp_synth = expert_synth_stop(frames, 0.5, 0.3, 5, 0)
        episodes.append((argmax, sprob, areas, cxs, exp_synth))

    # 후처리 config sweep
    configs = [
        ("raw (학습 그대로)",        dict(conf=0,   area_gate=0,   latch=False)),
        ("(나) conf>0.7",            dict(conf=0.7, area_gate=0,   latch=False)),
        ("(나) conf>0.9",            dict(conf=0.9, area_gate=0,   latch=False)),
        ("(가) latch+gate0.6",       dict(conf=0,   area_gate=0.6, latch=True)),
        ("(가) latch+gate0.7",       dict(conf=0,   area_gate=0.7, latch=True)),
        ("(가+나) conf0.7+gate0.6+latch", dict(conf=0.7, area_gate=0.6, latch=True)),
        ("(가+나) conf0.9+gate0.7+latch", dict(conf=0.9, area_gate=0.7, latch=True)),
    ]
    print(f"\n{'config':<34}{'CL success':<15}{'FPE':<10}{'TLD'}")
    print("-" * 66)
    results = {}
    for name, kw in configs:
        ms = []
        for (argmax, sprob, areas, cxs, exp) in episodes:
            pr = postprocess(argmax, sprob, areas, cxs, th_cx=args.th_cx, **kw)
            ms.append(compute_metrics(build_trajectory(exp), build_trajectory(pr), args.success_fpe))
        sr = np.mean([m["success"] for m in ms]); fpe = np.mean([m["fpe"] for m in ms])
        tld = np.mean([m["tld"] for m in ms]); n = len(ms)
        print(f"{name:<34}{sr*100:>5.1f}% ({round(sr*n)}/{n})    {fpe:>6.3f}m   {tld:.3f}")
        results[name] = {"success_rate": float(sr), "mean_fpe": float(fpe), "mean_tld": float(tld), "n": n}

    out = ROOT / "docs/v5/closed_loop_eval" / f"stop_closedloop_result_{args.out_tag}.json"
    out.write_text(json.dumps({"mlp": Path(args.mlp).name, "configs": results}, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] {out}")


if __name__ == "__main__":
    main()
