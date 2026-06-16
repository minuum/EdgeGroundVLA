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
S1  = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
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


def stop_trigger_idx(frames, th_area, th_cx, th_cy, W, min_steps):
    """도착 area + cy 규칙 → 최초 STOP 트리거 frame idx (없으면 None)."""
    buf_area = []
    buf_cy = []
    for i, fr in enumerate(frames):
        buf_area.append(fr.get("area_det") or 0.0)
        buf_cy.append(fr.get("cy_det") or 0.0)
        if len(buf_area) > W:
            buf_area = buf_area[-W:]
            buf_cy = buf_cy[-W:]
        area_avg = float(np.mean(buf_area))
        cy_avg = float(np.mean(buf_cy))
        cx = fr.get("cx_det") or 0.5
        if i >= min_steps and area_avg > th_area and abs(cx - 0.5) < th_cx and cy_avg > th_cy:
            return i
    return None


def apply_latched_stop(actions, trig):
    """trig 이후(포함) 전부 STOP(0)으로 래치."""
    if trig is None:
        return list(actions)
    return [a if i < trig else 0 for i, a in enumerate(actions)]


def expert_synth_stop(frames, th_area, th_cx, th_cy, W, min_steps):
    """expert: gt_class 사용하되 도착 규칙으로 STOP 래치(이미 STOP인 ep는 그대로 유지)."""
    gt = [fr["gt_class"] for fr in frames]
    # gt에 이미 STOP 있으면 그 위치, 없으면 규칙 트리거
    gt_stop = next((i for i, c in enumerate(gt) if c == 0), None)
    trig = gt_stop if gt_stop is not None else stop_trigger_idx(frames, th_area, th_cx, th_cy, W, min_steps)
    return apply_latched_stop(gt, trig)


def run_evaluation(all_preds, th_area, th_cx, th_cy, W, min_steps, success_fpe):
    """지정된 th_cy 값으로 4-cell 메트릭 평가를 실행하고 요약 딕셔너리를 반환한다."""
    cells = {("raw", "off"): [], ("raw", "on"): [], ("synth", "off"): [], ("synth", "on"): []}
    
    for ep, frames, pred in all_preds:
        # th_cy를 반영한 trigger index
        trig = stop_trigger_idx(frames, th_area, th_cx, th_cy, W, min_steps)
        pred_on = apply_latched_stop(pred, trig)
        exp_raw = [fr["gt_class"] for fr in frames]
        exp_synth = expert_synth_stop(frames, th_area, th_cx, th_cy, W, min_steps)
        
        variants = {
            ("raw",   "off"): (exp_raw,   pred),
            ("raw",   "on"):  (exp_raw,   pred_on),
            ("synth", "off"): (exp_synth, pred),
            ("synth", "on"):  (exp_synth, pred_on),
        }
        for key, (e, pr) in variants.items():
            m = compute_metrics(build_trajectory(e), build_trajectory(pr), success_fpe)
            cells[key].append(m)
            
    summary = {}
    for key in [("raw", "off"), ("raw", "on"), ("synth", "off"), ("synth", "on")]:
        ms = cells[key]
        if not ms: continue
        sr = np.mean([m["success"] for m in ms])
        fpe = np.mean([m["fpe"] for m in ms])
        tld = np.mean([m["tld"] for m in ms])
        n = len(ms)
        summary[f"{key[0]}_{key[1]}"] = {
            "success_rate": float(sr),
            "mean_fpe": float(fpe),
            "mean_tld": float(tld),
            "n": n
        }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mlp", default=str(ROOT / "runs/v5_nav/mlp/exp60/abl_b3_mlp.pt"))
    p.add_argument("--th-area", type=float, default=0.5)
    p.add_argument("--th-cx",   type=float, default=0.3)
    p.add_argument("--th-cy",   type=float, default=0.5, help="Y-Center 임계값 게이트")
    p.add_argument("--window-avg", type=int, default=5, help="area/cy 평균 윈도 W")
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
    print(f"[RULE] th_area={args.th_area} th_cx={args.th_cx} th_cy={args.th_cy} W={args.window_avg} min_steps={args.min_steps}\n")

    # 1단계: 모든 validation 에피소드에 대해 MLP 예측 캐싱 (연산 속도 단축)
    all_preds = []
    print("[RUN] MLP closed-loop 예측 feature 추출 중...")
    for idx, ep in enumerate(val_eps):
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        h5 = Path(ep["episode"])
        if not frames or not h5.exists():
            continue
        try:
            with h5py.File(str(h5), "r") as f:
                imgs = f["observations"]["images"][:]
        except Exception:
            continue
        
        pred = []
        with torch.no_grad():
            for t, fr in enumerate(frames):
                vis = enc.encode(Image.fromarray(imgs[fr["frame_idx"]].astype("uint8")))
                x = torch.tensor(bbox_window(frames, t) + vis.cpu().tolist(),
                                 dtype=torch.float32, device="cuda").unsqueeze(0)
                pred.append(int(mlp(x).argmax(1).item()))
        
        all_preds.append((ep, frames, pred))
        if (idx + 1) % 5 == 0 or (idx + 1) == len(val_eps):
            print(f"  - 진행 상황: {idx + 1}/{len(val_eps)} 에피소드 완료")

    print("\n[EVAL] Ablation 평가 수행 중 (No CY Gate vs With CY Gate)...")
    summary_no_cy = run_evaluation(all_preds, args.th_area, args.th_cx, 0.0, args.window_avg, args.min_steps, args.success_fpe)
    summary_with_cy = run_evaluation(all_preds, args.th_area, args.th_cx, args.th_cy, args.window_avg, args.min_steps, args.success_fpe)

    # 2단계: 결과 대조 테이블 출력
    def print_table(title, summary):
        print(f"\n[{title}]")
        print(f"{'expert':<8}{'pred_stop':<11}{'CL success':<15}{'mean_FPE':<11}{'mean_TLD'}")
        print("-" * 56)
        for key in [("raw","off"),("raw","on"),("synth","off"),("synth","on")]:
            k_str = f"{key[0]}_{key[1]}"
            if k_str not in summary: continue
            val = summary[k_str]
            sr = val["success_rate"]
            fpe = val["mean_fpe"]
            tld = val["mean_tld"]
            n = val["n"]
            print(f"{key[0]:<8}{key[1]:<11}{sr*100:>5.1f}% ({round(sr*n)}/{n})    {fpe:>7.3f}m   {tld:.3f}")

    print("=" * 60)
    print(" Ablation Study: 도착 STOP Y-Center Gate (th_cy) 비교 검증")
    print("=" * 60)
    print_table(f"No CY Gate (th_cy=0.0)", summary_no_cy)
    print_table(f"With CY Gate (th_cy={args.th_cy})", summary_with_cy)
    print("=" * 60)

    # 3단계: JSON으로 결과 저장
    out = OUTDIR / f"stop_closedloop_result_{args.out_tag}.json"
    out.write_text(json.dumps({
        "mlp": Path(args.mlp).name,
        "rule": {
            "th_area": args.th_area,
            "th_cx": args.th_cx,
            "th_cy": args.th_cy,
            "W": args.window_avg,
            "min_steps": args.min_steps
        },
        "ablation": {
            "no_cy_gate": summary_no_cy,
            "with_cy_gate": summary_with_cy
        }
    }, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] {out}")


if __name__ == "__main__":
    main()
