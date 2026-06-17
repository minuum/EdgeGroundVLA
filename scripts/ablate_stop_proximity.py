#!/usr/bin/env python3
"""
STOP Proximity Override Ablation
─────────────────────────────────
Stage2 v2 (Exp66) 모델 위에 proximity override 6 variant를 적용해
path_type별 CL 성공률을 비교한다. eval_exp54_stage2_v2_closedloop.py와
동일한 FrozenCLIPV2 / ActionMLP / eval_episode 구조 재사용.

Variant 정의:
  V0  no_override        모델 예측 그대로 (baseline)
  V1  current_server     area≥0.25 & |cx-0.5|≤0.35 & consec=2  (서버 기본값)
  V2  cx_removed         area≥0.25 & consec=2  (cx 조건 제거)
  V3  cx_rm_cy045        area≥0.25 & cy≥0.45 & consec=2
  V4  area_adaptive      area≥0.40이면 STOP (cx/cy 무시), else V1 조건
  V5  area_only_035      area≥0.35 & consec=2

사용법:
  .venv/bin/python3 scripts/ablate_stop_proximity.py
  .venv/bin/python3 scripts/ablate_stop_proximity.py \\
      --ckpt runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt \\
      --data docs/v5/bbox_nav_exp46/bbox_dataset_full.json \\
      --seeds 5 --tag my_ablation
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
    DT_DEFAULT, build_trajectory,
)

# ── 기본 경로 (eval_exp54_stage2_v2_closedloop.py 동일) ────────────────────
VLM_PATH    = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_CKPT = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
STAGE2_CKPT = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage2_v2" / "stage2_v2_mlp.pt"
DATA_PATH   = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
DATA_DIR    = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OUT_PATH    = ROOT / "docs" / "v5" / "closed_loop_eval" / "stop_ablation_results.json"

NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
PROJ_DIM    = 256
D_IN        = WINDOW * 4 + PROJ_DIM   # 288

PATH_TYPES = [
    "center_straight", "center_left",  "center_right",
    "left_straight",   "left_left",    "left_right",
    "right_straight",  "right_left",   "right_right",
]

# ── Proximity override variants ────────────────────────────────────────────
#  (name, area_th, cx_tol[None=ignore], cy_thr[None=ignore], consec, adaptive)
VARIANTS = [
    ("V0_no_override",    None,  None,  None,  2, False),
    ("V1_current_server", 0.25,  0.35,  None,  2, False),
    ("V2_cx_removed",     0.25,  None,  None,  2, False),
    ("V3_cx_rm_cy045",    0.25,  None,  0.45,  2, False),
    ("V4_area_adaptive",  0.25,  0.35,  None,  2, True),
    ("V5_area_only_035",  0.35,  None,  None,  2, False),
]


# ── 모델 구조 (eval_exp54_stage2_v2_closedloop.py 동일) ────────────────────
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
        return torch.cat(all_feats, dim=0)


class ActionMLP(nn.Module):
    def __init__(self, d_in=D_IN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES),
        )
    def forward(self, x): return self.net(x)


# ── 프레임 피처 (eval_exp54_stage2_v2_closedloop.py 동일) ──────────────────
def bbox_feat(frames, t, window=WINDOW):
    arr = []
    for k in range(window):
        fr = frames[max(0, t - (window - 1 - k))]
        cx   = fr.get("cx",   fr.get("cx_det",   0.5))
        cy   = fr.get("cy",   fr.get("cy_det",   0.5))
        area = fr.get("area", fr.get("area_det", 0.05))
        has  = float(fr.get("has_bbox", fr.get("detected", False)))
        arr.extend([cx, cy, area, has])
    return np.array(arr, dtype=np.float32)


# ── 에피소드 추론 (eval_exp54_stage2_v2_closedloop.py 동일) ───────────────
def eval_episode(ep_entry, enc, mlp, device, data_dir, window=WINDOW):
    frames   = ep_entry["frames"]
    ep_path  = Path(ep_entry["episode"])
    data_dir = Path(data_dir)
    try:
        if ep_path.is_absolute() and ep_path.exists():
            h5_path = ep_path
        else:
            stem = ep_path.stem
            candidates = list(data_dir.glob(f"{stem}.h5"))
            if not candidates:
                candidates = list(data_dir.glob(f"**/{stem}.h5"))
            if not candidates:
                return None, None
            h5_path = candidates[0]
        import io as _io
        with h5py.File(str(h5_path), "r") as f:
            imgs_ds = f["observations"]["images"]
            n_imgs  = len(imgs_ds)
            imgs = []
            for fr in frames:
                idx = min(fr["frame_idx"], n_imgs - 1)
                raw = imgs_ds[idx]
                if hasattr(raw, "dtype") and raw.dtype != object and raw.ndim >= 2:
                    imgs.append(Image.fromarray(raw.astype("uint8")))
                else:
                    arr = np.frombuffer(bytes(raw), dtype=np.uint8)
                    imgs.append(Image.open(_io.BytesIO(arr)).convert("RGB"))
    except Exception as e:
        print(f"  [SKIP] {ep_path}: {e}")
        return None, None

    expert_classes = [fr["gt_class"] for fr in frames]
    vis_feats = enc.encode_batch(imgs, device)

    pred_classes = []
    mlp.eval()
    with torch.no_grad():
        for t in range(len(frames)):
            bf = torch.tensor(bbox_feat(frames, t, window=window), device=device)
            x  = torch.cat([bf, vis_feats[t]]).unsqueeze(0)
            pred_classes.append(mlp(x).argmax(1).item())

    return pred_classes, expert_classes


# ── Proximity override (오프라인 replay, 래치 방식) ────────────────────────
def apply_override(frames, pred_classes, area_th, cx_tol, cy_thr, consec, adaptive):
    """
    pred_classes 복사 후 proximity 조건 충족 시점부터 래치(latch) STOP.
    area_th=None이면 override 없이 그대로 반환.
    adaptive=True: area>=0.40이면 cx/cy 조건 무시.
    """
    if area_th is None:
        return list(pred_classes)

    result  = list(pred_classes)
    latched = False

    for t in range(len(frames)):
        if latched:
            result[t] = 0
            continue

        start  = max(0, t - consec + 1)
        needed = t - start + 1
        ok_cnt = 0

        for i in range(start, t + 1):
            fr = frames[i]
            if not fr.get("has_bbox", fr.get("detected", False)):
                continue
            area = fr.get("area", fr.get("area_det", 0.0))
            cx   = fr.get("cx",   fr.get("cx_det",   0.5))
            cy   = fr.get("cy",   fr.get("cy_det",   0.5))

            area_ok = (area >= area_th)
            if adaptive and area >= 0.40:
                cx_ok = cy_ok = True
            else:
                cx_ok = (cx_tol is None) or (abs(cx - 0.5) <= cx_tol)
                cy_ok = (cy_thr is None) or (cy >= cy_thr)

            if area_ok and cx_ok and cy_ok:
                ok_cnt += 1

        if ok_cnt >= needed:
            result[t] = 0
            latched = True

    return result


# ── CL 메트릭 계산 ─────────────────────────────────────────────────────────
def episode_metrics(pred, expert, dt, success_fpe):
    pred_traj   = build_trajectory(pred,   dt)
    expert_traj = build_trajectory(expert, dt)
    pf = pred_traj.final_pos()
    ef = expert_traj.final_pos()
    fpe = float(np.sqrt((pf[0]-ef[0])**2 + (pf[1]-ef[1])**2))
    tld = pred_traj.total_length() / max(expert_traj.total_length(), 1e-6)
    return {
        "fpe": fpe, "tld": tld,
        "success": (fpe < success_fpe) and (0.7 <= tld <= 1.5),
    }


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        default=str(STAGE2_CKPT))
    p.add_argument("--stage1",      default=str(STAGE1_CKPT))
    p.add_argument("--data",        default=str(DATA_PATH))
    p.add_argument("--data_dir",    default=str(DATA_DIR))
    p.add_argument("--dt",          type=float, default=DT_DEFAULT)
    p.add_argument("--success_fpe", type=float, default=0.5)
    p.add_argument("--test_size",   type=float, default=0.2)
    p.add_argument("--seeds",       type=int,   default=5)
    p.add_argument("--tag",         default="stop_ablation")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # ── 데이터 ──
    data = json.loads(Path(args.data).read_text())
    print(f"[DATA]  n={len(data)} episodes  ({args.data})")
    ep_labels = [ep.get("path_type", "unknown") for ep in data]

    # ── 모델 ──
    print("[MODEL] Stage1 로드...")
    enc = FrozenCLIPV2(VLM_PATH, Path(args.stage1), device).to(device).eval()

    ckpt   = torch.load(args.ckpt, map_location=device, weights_only=False)
    window = ckpt.get("window", WINDOW)
    d_in   = window * 4 + PROJ_DIM
    mlp    = ActionMLP(d_in=d_in).to(device)
    mlp.load_state_dict(ckpt["mlp"])
    mlp.eval()
    print(f"[MODEL] Stage2  window={window}  val_acc={ckpt.get('val_acc', 0):.4f}")

    # ── 멀티시드 루프 ──
    variant_names  = [v[0] for v in VARIANTS]
    seed_sr_lists  = {n: [] for n in variant_names}
    seed_fpe_lists = {n: [] for n in variant_names}
    last_path_det  = {}

    for seed in range(args.seeds):
        print(f"\n{'─'*65}")
        print(f" Seed {seed+1}/{args.seeds}")

        from collections import Counter
        can_strat = all(c >= 2 for c in Counter(ep_labels).values())
        if can_strat:
            sss = StratifiedShuffleSplit(1, test_size=args.test_size, random_state=seed)
            _, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
        else:
            from sklearn.model_selection import ShuffleSplit
            ss = ShuffleSplit(1, test_size=args.test_size, random_state=seed)
            _, te_idx = next(ss.split(np.zeros(len(data))))
        val_eps = [data[i] for i in te_idx]

        seed_metrics = {n: [] for n in variant_names}
        path_detail  = defaultdict(lambda: {n: [] for n in variant_names})

        for ep in val_eps:
            pt = ep.get("path_type", "unknown")
            raw_pred, expert = eval_episode(ep, enc, mlp, device, args.data_dir, window)
            if raw_pred is None:
                continue

            for (name, area_th, cx_tol, cy_thr, consec, adaptive) in VARIANTS:
                pred = apply_override(ep["frames"], raw_pred,
                                      area_th, cx_tol, cy_thr, consec, adaptive)
                m = episode_metrics(pred, expert, args.dt, args.success_fpe)
                seed_metrics[name].append(m)
                path_detail[pt][name].append(m["success"])

        for name in variant_names:
            ms = seed_metrics[name]
            if not ms:
                continue
            sr  = sum(1 for m in ms if m["success"]) / len(ms)
            fpe = float(np.mean([m["fpe"] for m in ms]))
            seed_sr_lists[name].append(sr)
            seed_fpe_lists[name].append(fpe)
            ok = sum(1 for m in ms if m["success"])
            print(f"  {name:<25} SR={sr*100:5.1f}%  FPE={fpe:.3f}m  ({ok}/{len(ms)})")

        last_path_det = path_detail

    # ── 최종 집계 ──
    summary = {}
    for name in variant_names:
        sv  = seed_sr_lists[name]
        fv  = seed_fpe_lists[name]
        if not sv:
            continue
        summary[name] = {
            "sr_mean":  float(np.mean(sv)),
            "sr_std":   float(np.std(sv)),
            "fpe_mean": float(np.mean(fv)),
            "fpe_std":  float(np.std(fv)),
        }

    best_v = max(summary, key=lambda n: summary[n]["sr_mean"])

    print(f"\n{'='*65}")
    print(f"  STOP Proximity Override Ablation  ({args.seeds}seed 평균)")
    print(f"{'='*65}")
    print(f"  {'Variant':<25} {'SR mean':>8} {'±':>5} {'FPE mean':>9} {'±':>7}")
    print(f"  {'-'*57}")
    for name in variant_names:
        s = summary.get(name)
        if not s:
            continue
        marker = " ★" if name == best_v else ""
        print(f"  {name:<25} {s['sr_mean']*100:>7.1f}%"
              f" {s['sr_std']*100:>4.1f}%"
              f" {s['fpe_mean']:>8.3f}m"
              f" {s['fpe_std']:>6.3f}m{marker}")

    # ── path_type별 상세 ──
    print(f"\n  Path-type별 성공률 (마지막 seed, V0 vs {best_v[:20]})")
    print(f"  {'path_type':<22} {'V0_no_override':>16} {best_v[:22]:>22}")
    print(f"  {'-'*63}")
    for pt in PATH_TYPES:
        d  = last_path_det.get(pt, {})
        v0 = d.get("V0_no_override", [])
        bv = d.get(best_v, [])
        v0s = f"{sum(v0)}/{len(v0)} ({sum(v0)/len(v0)*100:.0f}%)" if v0 else "—"
        bvs = f"{sum(bv)}/{len(bv)} ({sum(bv)/len(bv)*100:.0f}%)" if bv else "—"
        print(f"  {pt:<22} {v0s:>16} {bvs:>22}")

    # ── JSON 저장 ──
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "tag":         args.tag,
        "ckpt":        str(args.ckpt),
        "data":        str(args.data),
        "seeds":       args.seeds,
        "success_fpe": args.success_fpe,
        "variants":    variant_names,
        "summary":     summary,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[SAVED] {OUT_PATH}")
    print(f"[BEST]  {best_v}  SR={summary[best_v]['sr_mean']*100:.1f}%"
          f"±{summary[best_v]['sr_std']*100:.1f}%  "
          f"FPE={summary[best_v]['fpe_mean']:.3f}m")


if __name__ == "__main__":
    main()
