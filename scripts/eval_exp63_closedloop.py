#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp63 E2E Kosmos-2 VLA Closed-Loop 시뮬레이션 평가 스크립트
이미지와 텍스트 프롬프트를 입력받아 모델이 직접 액션을 생성하며 주행을 평가합니다.
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import h5py
from PIL import Image

# 경로 정의
VLM_PATH = ROOT / ".vlms" / "kosmos-2-patch14-224"
LORA_PATH = ROOT / "runs/v5_nav/e2e/exp63"
ANN_PG2 = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"
OUT_DIR = ROOT / "docs/v5/closed_loop_eval"
OUT_DIR.mkdir(exist_ok=True)

ACTION_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD_LEFT", "FWD_RIGHT", "ROT_L", "ROT_R"]
PROMPT = "Navigate to the gray basket. Robot action:"

def load_model(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel
    print(f"[LOAD] 백본 모델 로드 중: {VLM_PATH}")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    proc = AutoProcessor.from_pretrained(str(VLM_PATH))
    base_model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    
    print(f"[LOAD] LoRA 어댑터 가중치 로드 중: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, str(LORA_PATH)).eval()
    return proc, model, dtype

def predict_action(proc, model, img_np, device, dtype):
    """이미지 -> 액션 인덱스 예측"""
    pil = Image.fromarray(img_np).convert("RGB")
    inp = proc(text=PROMPT, images=pil, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    
    # generate 함수 호출 시 inp의 인자들을 언패킹하여 전달
    gen = model.generate(**inp, max_new_tokens=5, do_sample=False)
    raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    
    # 텍스트 결과에서 매핑되는 액션명 탐색
    raw_up = raw.upper().replace("+", "_").replace(" ", "_")
    for i, name in enumerate(ACTION_NAMES):
        if name in raw_up or raw_up in name:
            return i, raw
    return 1, raw  # 매핑 실패 시 기본 전진(FORWARD)

def eval_episode(ep_entry, proc, model, device, dtype):
    """단일 에피소드 시뮬레이션 평가 수행"""
    frames = ep_entry["frames"]
    h5_path = Path(ep_entry["episode"])
    if not h5_path.exists():
        return None, None

    try:
        with h5py.File(str(h5_path), "r") as f:
            imgs_np = f["observations"]["images"][:]
    except Exception as e:
        print(f"H5 파일 로드 에러 {h5_path.name}: {e}")
        return None, None

    n = len(frames)
    pred_classes = []
    expert_classes = [fr["gt_class"] for fr in frames]

    with torch.no_grad():
        for t in range(n):
            img_np = imgs_np[frames[t]["frame_idx"]].astype("uint8")
            pred, _ = predict_action(proc, model, img_np, device, dtype)
            pred_classes.append(pred)

    return pred_classes, expert_classes

def compute_metrics(pred, expert, dt=0.5, success_fpe=0.5):
    """지표(FPE, TLD, Success) 계산"""
    from scripts.sim.rollout_core import build_trajectory, compute_metrics as core_compute_metrics
    try:
        pred_traj   = build_trajectory(pred,   dt=dt)
        expert_traj = build_trajectory(expert, dt=dt)
        res = core_compute_metrics(expert_traj, pred_traj, success_fpe)
        fpe = res["fpe"]
        tld = res["tld"]
        success = res["success"]
    except Exception as e:
        print(f"지표 계산 실패: {e}")
        fpe, tld, success = 9.9, 0.0, False
    return {"fpe": fpe, "tld": tld, "success": success}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-fpe", type=float, default=0.5, help="성공 임계 FPE(m)")
    parser.add_argument("--n-eps",       type=int,   default=None,  help="평가할 에피소드 수")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"[DEVICE] {device}")
    
    proc, model, dtype = load_model(device)
    model.eval()
    print("모델 로드 완료\n")

    with open(ANN_PG2) as f:
        ann = json.load(f)
    
    # 유효 에피소드 필터링
    ann = [ep for ep in ann if ep.get("path_type","") not in ("","free","unknown")]
    
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    random.shuffle(ann)
    
    val_ratio = 0.15
    n_val = max(1, int(len(ann) * val_ratio))
    val_ep = ann[:n_val]
    
    if args.n_eps:
        val_ep = val_ep[:args.n_eps]

    print(f"테스트 검증 에피소드 수: {len(val_ep)}개\n")

    results_by_path = defaultdict(list)
    all_m = []

    for i, ep in enumerate(val_ep):
        pt = ep.get("path_type", "unknown")
        out = eval_episode(ep, proc, model, device, dtype)
        if out[0] is None:
            continue
        
        pred, expert = out
        m = compute_metrics(pred, expert, success_fpe=args.success_fpe)
        m["path_type"] = pt
        results_by_path[pt].append(m)
        all_m.append(m)
        
        mark = "✅" if m["success"] else "❌"
        print(f"  {mark} [{i+1:3d}/{len(val_ep)}] {pt:<22} "
              f"FPE={m['fpe']:.3f}m TLD={m['tld']:.2f}")

    total   = len(all_m)
    success = sum(1 for m in all_m if m["success"])
    success_rate = success / total if total > 0 else 0.0
    mean_fpe = float(np.mean([m["fpe"] for m in all_m])) if total > 0 else 9.9
    mean_tld = float(np.mean([m["tld"] for m in all_m])) if total > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"  Exp63 E2E Kosmos-2 Closed-Loop Evaluation 결과")
    print(f"  성공률: {success}/{total} = {success_rate*100:.1f}%")
    print(f"  평균 FPE: {mean_fpe:.3f}m")
    print(f"  평균 TLD: {mean_tld:.3f}")
    print(f"{'='*60}")

    print("\npath_type별 상세 결과:")
    for pt in sorted(results_by_path):
        ms = results_by_path[pt]
        sr  = sum(1 for m in ms if m["success"]) / len(ms)
        mfpe = np.mean([m["fpe"] for m in ms])
        print(f"  {pt:<22} {sum(1 for m in ms if m['success'])}/{len(ms)} SR={sr:.0%} FPE={mfpe:.3f}m")

    # 결과 JSON 저장
    result = {
        "exp": "exp63_closedloop",
        "success_rate": success_rate,
        "mean_fpe": mean_fpe,
        "mean_tld": mean_tld,
        "n_episodes": total,
    }
    
    out_path = OUT_DIR / "exp63_closedloop_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nJSON 저장 완료 -> {out_path}")

if __name__ == "__main__":
    main()
