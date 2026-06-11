#!/usr/bin/env python3
"""
Exp54 Stage 2 v2 Closed-Loop (Offline Replay) Evaluation

기존 evaluate_closed_loop_v5.py의 kinematic simulation 구조를 재사용해
FrozenCLIPV2 + ActionMLP(288)를 평가.

성공 기준: FPE < 0.5m AND TLD ∈ [0.7, 1.5]

Usage:
  .venv/bin/python3 scripts/eval_exp54_stage2_v2_closedloop.py
  .venv/bin/python3 scripts/eval_exp54_stage2_v2_closedloop.py --success_fpe 0.8
"""
import sys, json, argparse, warnings
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
from sklearn.model_selection import StratifiedShuffleSplit

from scripts.sim.rollout_core import (
    ACTION_VEL, CLASS_NAMES, DT_DEFAULT,
    build_trajectory, compute_metrics,
)

VLM_PATH    = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH   = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
STAGE1_V2   = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage1_v2" / "stage1_v2_projs.pt"
STAGE2_CKPT = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage2_v2" / "stage2_v2_mlp.pt"
DATA_DIR    = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OUT_DIR     = ROOT / "docs" / "v5" / "closed_loop_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES_8 = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES   = 8
WINDOW        = 8
VIS_DIM       = 1024
PROJ_DIM      = 256
D_IN          = WINDOW * 4 + PROJ_DIM   # 288


class FrozenCLIPV2(nn.Module):
    def __init__(self, vlm_path, ckpt_path, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        print(f"[MODEL] Stage1 v2 val_acc={ckpt['val_acc']:.4f}")
        self.processor  = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vision_model = base.vision_model.to(device)
        self.image_proj   = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.image_proj.load_state_dict(ckpt["image_proj"])
        for p in self.vision_model.parameters(): p.requires_grad = False
        for p in self.image_proj.parameters():   p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, pil_images, device, batch=16):
        all_feats = []
        for i in range(0, len(pil_images), batch):
            imgs = pil_images[i:i+batch]
            inputs = self.processor(images=imgs, return_tensors="pt")
            pv  = inputs["pixel_values"].to(device, dtype=torch.float16)
            out = self.vision_model(pixel_values=pv)
            feat = out.last_hidden_state.mean(dim=1).float()
            all_feats.append(F.normalize(self.image_proj(feat), dim=-1))
        return torch.cat(all_feats, dim=0)   # (N, 256)


class ActionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_IN, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES),
        )
    def forward(self, x): return self.net(x)


def bbox_feat(frames, t):
    arr = []
    for k in range(WINDOW):
        fr = frames[max(0, t - (WINDOW - 1 - k))]
        arr.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])
    return np.array(arr, dtype=np.float32)


def eval_episode(ep_entry, enc, mlp, device):
    """에피소드 1개 → (pred_classes, expert_actions)"""
    frames = ep_entry["frames"]
    ep_path = Path(ep_entry["episode"])
    try:
        # episode field가 절대경로면 직접 사용, 아니면 DATA_DIR에서 검색
        if ep_path.is_absolute() and ep_path.exists():
            h5_path = ep_path
        else:
            stem = ep_path.stem
            candidates = list(DATA_DIR.glob(f"{stem}.h5"))
            if not candidates:
                candidates = list(DATA_DIR.glob(f"**/{stem}.h5"))
            if not candidates:
                return None, None
            h5_path = candidates[0]
        with h5py.File(str(h5_path), "r") as f:
            imgs = [Image.fromarray(f["observations"]["images"][i])
                    for i in range(len(frames))]
    except Exception as e:
        print(f"  [SKIP] {ep_path}: {e}")
        return None, None

    # gt_class는 JSON frames에서 읽음 (H5 actions는 연속 속도벡터)
    expert_classes = [fr["gt_class"] for fr in frames]

    vis_feats = enc.encode_batch(imgs, device)
    pred_classes = []
    mlp.eval()
    with torch.no_grad():
        for t in range(len(frames)):
            bf = torch.tensor(bbox_feat(frames, t), device=device)
            x  = torch.cat([bf, vis_feats[t]]).unsqueeze(0)
            pred_classes.append(mlp(x).argmax(1).item())

    return pred_classes, expert_classes


def compute_episode_metrics(pred_classes, expert_classes, dt, success_fpe):
    from scripts.sim.rollout_core import Pose, pose_step, build_trajectory

    pred_traj   = build_trajectory(pred_classes, dt)
    expert_traj = build_trajectory(expert_classes, dt)

    pred_fp   = pred_traj.final_pos()
    expert_fp = expert_traj.final_pos()

    fpe = np.sqrt((pred_fp[0]-expert_fp[0])**2 + (pred_fp[1]-expert_fp[1])**2)
    pred_len   = pred_traj.total_length()
    expert_len = expert_traj.total_length()
    tld = pred_len / expert_len if expert_len > 1e-6 else float("inf")
    success = (fpe < success_fpe) and (0.7 <= tld <= 1.5)
    return {"fpe": fpe, "tld": tld, "success": success}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dt",          type=float, default=DT_DEFAULT)
    p.add_argument("--success_fpe", type=float, default=0.5)
    p.add_argument("--ckpt",        type=str,   default=str(STAGE2_CKPT))
    p.add_argument("--data",        type=str,   default=str(DATA_PATH),
                   help="CL 평가 데이터 (cx 소스 매칭용). 기본=HSV 벤치마크")
    p.add_argument("--tag",         type=str,   default="exp54_s2v2",
                   help="결과 JSON에 기록할 라벨")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[ERROR] 체크포인트 없음: {ckpt_path}")
        print("학습이 완료될 때까지 기다려주세요.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    data = json.loads(Path(args.data).read_text())
    print(f"[DATA] {args.data} (tag={args.tag})")
    ep_labels = [ep["path_type"] for ep in data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
    val_eps = [data[i] for i in te_idx]
    print(f"Val episodes: {len(val_eps)}")

    print("[MODEL] Stage 1 v2 인코더 로드 중...")
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_V2, device).to(device).eval()

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    mlp  = ActionMLP().to(device)
    mlp.load_state_dict(ckpt["mlp"])
    mlp.eval()
    print(f"[MODEL] Stage 2 v2 MLP loaded (val_acc={ckpt['val_acc']:.4f})")

    results_by_path = defaultdict(list)
    all_metrics = []

    for i, ep in enumerate(val_eps):
        pt = ep.get("path_type", "unknown")
        pred, expert = eval_episode(ep, enc, mlp, device)
        if pred is None:
            continue
        m = compute_episode_metrics(pred, expert, args.dt, args.success_fpe)
        m["path_type"] = pt
        results_by_path[pt].append(m)
        all_metrics.append(m)
        print(f"  [{i+1:3d}/{len(val_eps)}] {pt:<22} FPE={m['fpe']:.3f}m  TLD={m['tld']:.2f}  {'✅' if m['success'] else '❌'}")

    # 요약
    total   = len(all_metrics)
    success = sum(1 for m in all_metrics if m["success"])
    print(f"\n{'='*55}")
    print(f"  Exp54 Stage 2 v2 Closed-Loop 평가")
    print(f"  성공: {success}/{total} = {success/total*100:.1f}%  (FPE<{args.success_fpe}m & TLD∈[0.7,1.5])")
    print(f"  평균 FPE: {np.mean([m['fpe'] for m in all_metrics]):.3f}m")
    print(f"  평균 TLD: {np.mean([m['tld'] for m in all_metrics]):.3f}")
    print(f"  참고: step2=66.7%  Exp11=0%")
    print(f"{'='*55}")

    print(f"\npath_type별 성공률:")
    for pt in sorted(results_by_path.keys()):
        ms = results_by_path[pt]
        sr = sum(1 for m in ms if m["success"]) / len(ms)
        mfpe = np.mean([m["fpe"] for m in ms])
        print(f"  {pt:<22} {sum(1 for m in ms if m['success'])}/{len(ms)}  SR={sr:.0%}  FPE={mfpe:.3f}m")

    # JSON 저장 (기존 rollout_metrics.json에 병합)
    metrics_path = OUT_DIR / "rollout_metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
    else:
        existing = {"summary": {}, "per_path": {}}

    existing["summary"][args.tag] = {
        "success_rate": success / total if total > 0 else 0,
        "mean_fpe": float(np.mean([m["fpe"] for m in all_metrics])),
        "mean_tld": float(np.mean([m["tld"] for m in all_metrics])),
        "n_episodes": total,
    }
    existing["per_path"][args.tag] = {
        pt: [{"fpe": float(m["fpe"]), "tld": float(m["tld"]), "success": bool(m["success"])}
             for m in ms]
        for pt, ms in results_by_path.items()
    }
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.bool_): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    metrics_path.write_text(json.dumps(existing, indent=2, cls=NpEncoder))
    print(f"\n[SAVE] {metrics_path}")


if __name__ == "__main__":
    main()
