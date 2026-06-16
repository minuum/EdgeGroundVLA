#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp59 실시간 그라운딩(PaliGemma2) + 학습형 STOP 모델(stop65_mlp) Closed-Loop 결합 평가 스크립트.
주석은 한국어로 작성되었습니다.
"""
import sys, json, argparse, warnings, re
import numpy as np
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image

# ── 경로 설정 ─────────────────────────────────────────────────────────────
VLM_PATH    = ROOT / ".vlms" / "kosmos-2-patch14-224"          # Stage1 CLIP
PG2_PATH    = Path.home() / ".cache/huggingface/hub" \
              / "models--google--paligemma2-3b-mix-224" \
              / "snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
EXP59_PATH  = ROOT / "runs/v5_nav/grounding/exp59"             # PG2 Grounding LoRA
STAGE1_PT   = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
STAGE2_PT   = ROOT / "runs/v5_nav/mlp/exp60/stop65_mlp.pt"      # 학습형 STOP MLP 모델
ANN_JSON    = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"
OUT_DIR     = ROOT / "docs/v5/closed_loop_eval"
OUT_DIR.mkdir(exist_ok=True)

WINDOW      = 8
LOC_RE      = re.compile(r"<loc(\d{4})>")

# ── Stage1 v2: CLIP LoRA 인코더 ──────────────────────────────────────────
class Stage1Encoder(nn.Module):
    def __init__(self, vlm_path, stage1_pt, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ckpt = torch.load(str(stage1_pt), map_location=device, weights_only=True)
        self.processor = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vm = base.vision_model.to(device).eval()
        self.proj = nn.Linear(1024, 256).to(device)
        self.proj.load_state_dict(ckpt["image_proj"])
        self.proj.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, pil_img):
        inp = self.processor(images=[pil_img], return_tensors="pt")
        pv  = inp["pixel_values"].to(self.device, dtype=torch.float16)
        feat = self.vm(pixel_values=pv).last_hidden_state.mean(1).float()
        return self.proj(feat).squeeze(0)  # (256,)

# ── Stage2 v2: Action MLP (8-class, 0번은 STOP) ──────────────────────────
class ActionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(288, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 8),
        )
    def forward(self, x): return self.net(x)

# ── PaliGemma2 Exp59 그라운더 ─────────────────────────────────────────────
class Exp59Grounder:
    def __init__(self, pg2_path, adapter_path, device):
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
        from peft import PeftModel
        self.device = device
        dtype = torch.bfloat16
        self.dtype = dtype
        proc  = PaliGemmaProcessor.from_pretrained(str(pg2_path))
        base  = PaliGemmaForConditionalGeneration.from_pretrained(
                    str(pg2_path), torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
        model = PeftModel.from_pretrained(base, str(adapter_path)).eval()
        self.proc  = proc
        self.model = model

    @torch.no_grad()
    def detect(self, pil_img):
        inp = self.proc(text="<image>detect gray basket", images=pil_img,
                        return_tensors="pt").to(self.device)
        inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
        gen = self.model.generate(**inp, max_new_tokens=48, do_sample=False)
        raw = self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                     skip_special_tokens=False)[0]
        locs = [int(v) / 1023.0 for v in LOC_RE.findall(raw)]
        if len(locs) >= 4:
            y1, x1, y2, x2 = locs[:4]
            cx   = (x1 + x2) / 2
            cy   = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            return cx, cy, area
        return None

# ── 에피소드 평가 (확률 윈도우 스무딩 래치 포함) ─────────────────────────
def eval_episode(ep_entry, enc, mlp, grounder, device, w_stop, th_prob, ema_alpha=1.0):
    frames = ep_entry["frames"]
    h5_path = Path(ep_entry["episode"])
    if not h5_path.exists():
        return None, None, None, 0.0

    try:
        with h5py.File(str(h5_path), "r") as f:
            imgs_np = f["observations"]["images"][:]
    except Exception:
        return None, None, None, 0.0

    n = len(frames)
    pil_imgs = [Image.fromarray(imgs_np[fr["frame_idx"]].astype("uint8")) for fr in frames]

    # 1. PaliGemma2 라이브 그라운딩 수행
    detections = []
    for img in pil_imgs:
        det = grounder.detect(img)
        detections.append(det)  # (cx, cy, area) or None

    hit_n = sum(1 for d in detections if d is not None)

    # 2. Stage1 CLIP Feature 인코딩
    vis_feats = [enc.encode(img) for img in pil_imgs]

    # 3. Closed-Loop 주행 및 STOP 예측 시뮬레이션
    pred_raw = []
    pred_stop = []
    expert_raw = [fr["gt_class"] for fr in frames]

    # BBox 히스토리 윈도우 버퍼
    hist = [(0.5, 0.5, 0.05, 0.0)] * WINDOW
    
    # STOP 확률 윈도우 스무딩 버퍼
    stop_prob_history = [0.0] * w_stop
    is_latched = False

    smoothed_cx = None
    smoothed_cy = None
    smoothed_area = None

    with torch.no_grad():
        for t in range(n):
            det = detections[t]
            if det is not None:
                cx, cy, area = det
                if ema_alpha < 1.0:
                    if smoothed_cx is None:
                        smoothed_cx, smoothed_cy, smoothed_area = cx, cy, area
                    else:
                        smoothed_cx   = ema_alpha * cx + (1.0 - ema_alpha) * smoothed_cx
                        smoothed_cy   = ema_alpha * cy + (1.0 - ema_alpha) * smoothed_cy
                        smoothed_area = ema_alpha * area + (1.0 - ema_alpha) * smoothed_area
                    hist.append((smoothed_cx, smoothed_cy, smoothed_area, 1.0))
                else:
                    hist.append((cx, cy, area, 1.0))
            else:
                last_cx, last_cy, last_area, _ = hist[-1]
                hist.append((last_cx, last_cy, last_area, 0.0))
            
            hist = hist[-WINDOW:]
            bbox_vec = torch.tensor([v for item in hist for v in item],
                                     dtype=torch.float32, device=device)[:32]
            
            x = torch.cat([bbox_vec, vis_feats[t].to(device)]).unsqueeze(0)
            logits = mlp(x)
            
            # (가) 스무딩 이전의 순수 MLP Argmax 클래스 예측
            raw_cls = int(logits.argmax(1).item())
            pred_raw.append(raw_cls)
            
            # (나) 윈도우 스무딩 확률 계산
            probs = F.softmax(logits, dim=-1)
            p_stop = probs[0, 0].item() # 0번 클래스가 STOP
            stop_prob_history.append(p_stop)
            stop_prob_history = stop_prob_history[-w_stop:]
            
            mean_p_stop = sum(stop_prob_history) / w_stop
            
            if mean_p_stop > th_prob:
                is_latched = True
            
            # 래치 발화 시 STOP(0), 미발화 시 예측된 행동 클래스
            final_cls = 0 if is_latched else raw_cls
            pred_stop.append(final_cls)

    return pred_raw, pred_stop, expert_raw, hit_n / max(n, 1)

# ── 궤적 메트릭 계산 ──────────────────────────────────────────────────────
def compute_metrics(pred, expert, success_fpe=0.5):
    from scripts.sim.rollout_core import build_trajectory, compute_metrics as core_compute_metrics
    try:
        pred_traj   = build_trajectory(pred,   dt=0.5)
        expert_traj = build_trajectory(expert, dt=0.5)
        res = core_compute_metrics(expert_traj, pred_traj, success_fpe)
        return {"fpe": res["fpe"], "tld": res["tld"], "success": res["success"]}
    except Exception:
        return {"fpe": 9.9, "tld": 0.0, "success": False}

# ── 메인 실행부 ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-fpe", type=float, default=0.5)
    parser.add_argument("--n-eps",       type=int,   default=None, help="테스트 에피소드 수 (지정 없으면 전체)")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage2-pt",   default=str(STAGE2_PT), help="STOP MLP 가중치 pt")
    parser.add_argument("--w-stop",      type=int,   default=3, help="STOP 확률 스무딩 윈도우 크기")
    parser.add_argument("--th-prob",     type=float, default=0.8, help="STOP 발화 판단 임계 확률")
    parser.add_argument("--ema-alpha",   type=float, default=1.0)
    parser.add_argument("--out-tag",     default="exp59_stop")
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"[DEVICE] {device}")
    print(f"[STAGE2_PT] {Path(args.stage2_pt).name}")
    print(f"[STOP CONFIG] Window W={args.w_stop}, Th={args.th_prob}")
    
    print("\n[1/3] Stage1 CLIP 인코더 로드 중...")
    enc = Stage1Encoder(VLM_PATH, STAGE1_PT, device).eval()

    print("[2/3] Stage2 Action MLP 로드 중...")
    ckpt = torch.load(args.stage2_pt, map_location=device, weights_only=True)
    mlp = ActionMLP().to(device)
    mlp.load_state_dict(ckpt["mlp"])
    mlp.eval()
    print(f"  MLP val_acc: {ckpt.get('val_acc', 0)*100:.1f}%")

    print("[3/3] PaliGemma2 Exp59 LoRA 그라운더 로드 중 (GPU VRAM 필요)...")
    grounder = Exp59Grounder(PG2_PATH, EXP59_PATH, device)
    print("모든 모델 로드 완료.\n")

    # 고정 split 구성
    with open(ANN_JSON) as f:
        ann = json.load(f)
    ann = [ep for ep in ann if ep.get("path_type", "") not in ("", "free", "unknown")]
    import random; random.seed(42)
    random.shuffle(ann)
    val_n = max(1, int(len(ann) * 0.15))
    val_eps = ann[:val_n]
    if args.n_eps:
        val_eps = val_eps[:args.n_eps]

    print(f"[DATA] 테스트 에피소드 수: {len(val_eps)}개")

    cells = {
        "raw_off": [],    # 스무딩 없는 예측
        "raw_on": [],     # 확률 윈도우 스무딩 탑재 예측
        "synth_off": [],  # expert가 synth인 경우의 비교군 (W/O STOP)
        "synth_on": []    # expert가 synth인 경우의 비교군 (W/ STOP)
    }
    grnd_rates = []

    # 4-cell 검증용 expert synth STOP 구하는 헬퍼 함수
    # (여기서 룰 th_area=0.5, th_cx=0.3, W=5을 기본 적용하여 expert synth 종결점을 정의)
    from scripts.eval_stop_closedloop import expert_synth_stop

    for i, ep in enumerate(val_eps):
        pt = ep.get("path_type", "unknown")
        out = eval_episode(ep, enc, mlp, grounder, device, args.w_stop, args.th_prob, ema_alpha=args.ema_alpha)
        if out[0] is None:
            continue
        
        pred_raw, pred_stop, expert_raw, grnd_rate = out
        grnd_rates.append(grnd_rate)

        # expert synth 종결점 산출
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        expert_synth = expert_synth_stop(frames, th_area=0.5, th_cx=0.3, W=5, min_steps=0)

        # 4-cell 변종 비교 매칭
        variants = {
            "raw_off": (expert_raw,   pred_raw),
            "raw_on":  (expert_raw,   pred_stop),
            "synth_off": (expert_synth, pred_raw),
            "synth_on":  (expert_synth, pred_stop),
        }

        for key, (e, pr) in variants.items():
            m = compute_metrics(pr, e, success_fpe=args.success_fpe)
            m["path_type"] = pt
            cells[key].append(m)

        mark = "✅" if cells["synth_on"][-1]["success"] else "❌"
        print(f"  {mark} [{i+1:3d}/{len(val_eps)}] {pt:<22} "
              f"FPE={cells['synth_on'][-1]['fpe']:.3f}m TLD={cells['synth_on'][-1]['tld']:.2f} "
              f"grnd={grnd_rate*100:.0f}%")

    print(f"\n{'='*60}")
    print(f"  [RESULT] Live Grounding + Learned STOP 윈도우 스무딩 Closed-Loop")
    print(f"  평균 그라운딩 성공률: {np.mean(grnd_rates)*100:.1f}%\n")
    
    print(f"{'expert':<8}{'pred_stop':<11}{'CL success':<15}{'mean_FPE':<11}{'mean_TLD'}")
    print("-" * 56)
    summary = {}
    for key_name in ["raw_off", "raw_on", "synth_off", "synth_on"]:
        ms = cells[key_name]
        if not ms: continue
        sr = np.mean([m["success"] for m in ms])
        fpe = np.mean([m["fpe"] for m in ms])
        tld = np.mean([m["tld"] for m in ms])
        n_ms = len(ms)
        
        e_type, p_type = key_name.split("_")
        print(f"{e_type:<8}{p_type:<11}{sr*100:>5.1f}% ({round(sr*n_ms)}/{n_ms})    {fpe:>7.3f}m   {tld:.3f}")
        summary[key_name] = {
            "success_rate": float(sr),
            "mean_fpe": float(fpe),
            "mean_tld": float(tld),
            "n": n_ms
        }
    print(f"{'='*60}")

    # 결과 JSON 저장
    result_data = {
        "exp": "exp59_live_stop_closedloop",
        "stage2_pt": Path(args.stage2_pt).name,
        "smoothing": {"W": args.w_stop, "th_prob": args.th_prob},
        "mean_grounding_rate": float(np.mean(grnd_rates)),
        "cells": summary
    }
    fname = f"exp59_live_stop_result_{args.out_tag}.json"
    out_path = OUT_DIR / fname
    out_path.write_text(json.dumps(result_data, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] {out_path}")

if __name__ == "__main__":
    main()
