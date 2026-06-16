# -*- coding: utf-8 -*-
"""
학습된 STOP 모델의 Softmax 확률값에 시간적 스무딩(Temporal Window Smoothing)을 적용해
조기 정지를 억제하고 Closed-Loop 성공률을 검증하는 스크립트.

W (윈도우 크기), th_prob (평균 확률 임계값), latch 여부에 대해 sweep을 수행합니다.
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
                                          expert_synth_stop, WINDOW)


def postprocess_window(argmax, stop_prob, W, th_prob, latch):
    """STOP 클래스 확률의 W-프레임 이동평균이 th_prob을 넘을 때 STOP 판단."""
    out = []
    stopped = False
    buf = [0.0] * W  # 윈도우 버퍼 초기화
    
    for i in range(len(argmax)):
        if stopped:
            out.append(0)
            continue
            
        # 윈도우 업데이트 및 평균 계산
        buf.append(stop_prob[i])
        buf = buf[-W:]
        mean_p = sum(buf) / W
        
        is_stop = (mean_p > th_prob)
        
        if is_stop:
            if latch:
                stopped = True
            out.append(0)
        else:
            c = argmax[i]
            out.append(c if c != 0 else 1)  # STOP 거부 시 1(FORWARD)로 강제 대체
            
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mlp", default=str(ROOT / "runs/v5_nav/mlp/exp60/stop75_mlp.pt"))
    p.add_argument("--success-fpe", type=float, default=0.15)
    p.add_argument("--out-tag", default="window_sweep")
    args = p.parse_args()

    ann = json.loads(ANN.read_text())
    ann = [ep for ep in ann if ep.get("path_type", "") not in ("", "free", "unknown")]
    random.seed(42); np.random.seed(42); random.shuffle(ann)
    val_eps = ann[:max(1, int(len(ann) * 0.15))]
    print(f"[DATA] val {len(val_eps)} ep   [MLP] {Path(args.mlp).name}")

    enc = Enc().to("cuda").eval()
    ck = torch.load(args.mlp, map_location="cuda", weights_only=True)
    mlp = MLP().cuda(); mlp.load_state_dict(ck["mlp"]); mlp.eval()

    # 에피소드 프레임별 피처 및 추론값 미리 캐싱
    episodes = []
    for ep in val_eps:
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        h5 = Path(ep["episode"])
        if not frames or not h5.exists(): continue
        try:
            with h5py.File(str(h5), "r") as f: imgs = f["observations"]["images"][:]
        except Exception: continue
        argmax, sprob = [], []
        with torch.no_grad():
            for t, fr in enumerate(frames):
                vis = enc.encode(Image.fromarray(imgs[fr["frame_idx"]].astype("uint8")))
                x = torch.tensor(bbox_window(frames, t) + vis.cpu().tolist(),
                                 dtype=torch.float32, device="cuda").unsqueeze(0)
                prob = F.softmax(mlp(x), dim=-1)[0]
                argmax.append(int(prob.argmax().item()))
                sprob.append(float(prob[0].item()))  # index 0 is STOP
        exp_synth = expert_synth_stop(frames, 0.5, 0.3, 5, 0)
        episodes.append((argmax, sprob, exp_synth))

    # 하이퍼파라미터 sweep 조합 구성
    configs = []
    # 윈도우 W: 3, 5, 8
    # 임계값 th_prob: 0.5, 0.6, 0.7, 0.8
    # latch: True, False
    for W in [3, 5, 8]:
        for th in [0.5, 0.6, 0.7, 0.8]:
            for latch in [True, False]:
                name = f"W={W}, th={th:.1f}, latch={latch}"
                configs.append((name, dict(W=W, th_prob=th, latch=latch)))

    print(f"\n{'config':<34}{'CL success':<15}{'FPE':<10}{'TLD'}")
    print("-" * 66)
    results = {}
    
    for name, kw in configs:
        ms = []
        for (argmax, sprob, exp) in episodes:
            pr = postprocess_window(argmax, sprob, **kw)
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
