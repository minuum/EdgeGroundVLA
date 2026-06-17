#!/usr/bin/env python3
"""
STOP CLIP-Similarity Ablation (Layer 2)
────────────────────────────────────────
재학습 없이 CLIP 시각 유사도 기반 STOP을 평가한다.

아이디어:
  학습 데이터에서 area > AREA_REF_TH인 프레임들 (= 바스켓 근접 프레임)의
  CLIP feature를 평균해 reference 벡터를 만든다.
  평가 시: cosim(frame_feat, ref) > threshold AND consec → STOP (래치)

5개 threshold (0.70~0.90) × 5 seeds → CL success rate 비교.
결과: docs/v5/closed_loop_eval/stop_clipsim_results.json

Usage:
  .venv/bin/python3 scripts/ablate_stop_clip_sim.py
  .venv/bin/python3 scripts/ablate_stop_clip_sim.py --area_ref_th 0.20 --seeds 3
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

from scripts.sim.rollout_core import DT_DEFAULT, build_trajectory

VLM_PATH    = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_CKPT = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
STAGE2_CKPT = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage2_v2" / "stage2_v2_mlp.pt"
DATA_PATH   = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
DATA_DIR    = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OUT_PATH    = ROOT / "docs" / "v5" / "closed_loop_eval" / "stop_clipsim_results.json"

NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
PROJ_DIM    = 256
D_IN        = WINDOW * 4 + PROJ_DIM

COSIM_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]
CONSEC = 2


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


def load_h5_images(ep_entry, data_dir):
    ep_path  = Path(ep_entry["episode"])
    data_dir = Path(data_dir)
    import io as _io
    if ep_path.is_absolute() and ep_path.exists():
        h5_path = ep_path
    else:
        stem = ep_path.stem
        cands = list(data_dir.glob(f"{stem}.h5"))
        if not cands:
            cands = list(data_dir.glob(f"**/{stem}.h5"))
        if not cands:
            return None
        h5_path = cands[0]
    frames = ep_entry["frames"]
    imgs = []
    with h5py.File(str(h5_path), "r") as f:
        imgs_ds = f["observations"]["images"]
        n_imgs  = len(imgs_ds)
        for fr in frames:
            idx = min(fr["frame_idx"], n_imgs - 1)
            raw = imgs_ds[idx]
            if hasattr(raw, "dtype") and raw.dtype != object and raw.ndim >= 2:
                imgs.append(Image.fromarray(raw.astype("uint8")))
            else:
                arr = np.frombuffer(bytes(raw), dtype=np.uint8)
                imgs.append(Image.open(_io.BytesIO(arr)).convert("RGB"))
    return imgs


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


def episode_metrics(pred, expert, dt, success_fpe):
    pred_traj   = build_trajectory(pred,   dt)
    expert_traj = build_trajectory(expert, dt)
    pf = pred_traj.final_pos()
    ef = expert_traj.final_pos()
    fpe = float(np.sqrt((pf[0]-ef[0])**2 + (pf[1]-ef[1])**2))
    tld = pred_traj.total_length() / max(expert_traj.total_length(), 1e-6)
    return {"fpe": fpe, "tld": tld,
            "success": (fpe < success_fpe) and (0.7 <= tld <= 1.5)}


def apply_clipsim_override(frames, vis_feats_cpu, ref_feat, threshold, consec):
    """
    cosim(vis_feat[t], ref_feat) > threshold AND consec 연속 → 래치 STOP.
    ref_feat: (PROJ_DIM,) tensor on cpu
    vis_feats_cpu: (T, PROJ_DIM) tensor on cpu
    """
    result  = list(range(len(frames)))  # placeholder; will fill with pred below
    latched = False
    for t in range(len(frames)):
        if latched:
            result[t] = 0
            continue
        start  = max(0, t - consec + 1)
        needed = t - start + 1
        ok_cnt = 0
        for i in range(start, t + 1):
            sim = float(F.cosine_similarity(vis_feats_cpu[i].unsqueeze(0),
                                            ref_feat.unsqueeze(0)).item())
            if sim > threshold:
                ok_cnt += 1
        if ok_cnt >= needed:
            result[t] = 0
            latched = True
    return result


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
    p.add_argument("--area_ref_th", type=float, default=0.25,
                   help="train set에서 ref 프레임으로 쓸 area threshold")
    p.add_argument("--tag",         default="clipsim_stop")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    data = json.loads(Path(args.data).read_text())
    print(f"[DATA]  n={len(data)} episodes")
    ep_labels = [ep.get("path_type", "unknown") for ep in data]

    print("[MODEL] Stage1 로드...")
    enc = FrozenCLIPV2(VLM_PATH, Path(args.stage1), device).to(device).eval()

    ckpt   = torch.load(args.ckpt, map_location=device, weights_only=False)
    window = ckpt.get("window", WINDOW)
    d_in   = window * 4 + PROJ_DIM
    mlp    = ActionMLP(d_in=d_in).to(device)
    mlp.load_state_dict(ckpt["mlp"])
    mlp.eval()
    print(f"[MODEL] Stage2  window={window}  val_acc={ckpt.get('val_acc',0):.4f}")

    variant_names  = [f"cosim_th{th:.2f}" for th in COSIM_THRESHOLDS]
    variant_names  = ["V_no_override"] + variant_names
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
            tr_idx = [i for i in range(len(data)) if i not in set(te_idx)]
        else:
            from sklearn.model_selection import ShuffleSplit
            ss = ShuffleSplit(1, test_size=args.test_size, random_state=seed)
            tr_idx_arr, te_idx = next(ss.split(np.zeros(len(data))))
            tr_idx = list(tr_idx_arr)
        tr_eps = [data[i] for i in tr_idx]
        val_eps = [data[i] for i in te_idx]

        # ── Reference embedding: train set에서 area > th인 프레임들 ──
        print(f"  [REF] 학습셋 area>{args.area_ref_th} 프레임에서 ref 임베딩 계산...")
        ref_imgs = []
        for ep in tr_eps:
            imgs = load_h5_images(ep, args.data_dir)
            if imgs is None:
                continue
            for i, fr in enumerate(ep["frames"]):
                area = fr.get("area", fr.get("area_det", 0.0))
                if area > args.area_ref_th and fr.get("has_bbox", fr.get("detected", False)):
                    ref_imgs.append(imgs[i])
        if not ref_imgs:
            print("  [WARN] ref_imgs 없음 → threshold 낮춤 (0.1)")
            for ep in tr_eps:
                imgs = load_h5_images(ep, args.data_dir)
                if imgs is None:
                    continue
                for i, fr in enumerate(ep["frames"]):
                    area = fr.get("area", fr.get("area_det", 0.0))
                    if area > 0.1:
                        ref_imgs.append(imgs[i])

        print(f"  [REF] {len(ref_imgs)} ref frames → encoding...")
        ref_feats = enc.encode_batch(ref_imgs, device)   # (N, PROJ_DIM)
        ref_feat  = ref_feats.mean(dim=0).cpu()           # (PROJ_DIM,)
        ref_feat  = F.normalize(ref_feat, dim=0)
        print(f"  [REF] ref_feat norm={ref_feat.norm():.4f}")

        # ── 에피소드 평가 ──
        seed_metrics = {n: [] for n in variant_names}
        path_detail  = defaultdict(lambda: {n: [] for n in variant_names})

        for ep in val_eps:
            pt = ep.get("path_type", "unknown")
            frames = ep["frames"]
            try:
                imgs = load_h5_images(ep, args.data_dir)
                if imgs is None:
                    continue
            except Exception as e:
                print(f"  [SKIP] {ep['episode']}: {e}")
                continue

            expert_classes = [fr["gt_class"] for fr in frames]
            vis_feats = enc.encode_batch(imgs, device)
            vis_feats_cpu = vis_feats.cpu()

            # raw MLP pred
            raw_pred = []
            mlp.eval()
            with torch.no_grad():
                for t in range(len(frames)):
                    bf = torch.tensor(bbox_feat(frames, t, window=window), device=device)
                    x  = torch.cat([bf, vis_feats[t]]).unsqueeze(0)
                    raw_pred.append(mlp(x).argmax(1).item())

            # no-override baseline
            m = episode_metrics(raw_pred, expert_classes, args.dt, args.success_fpe)
            seed_metrics["V_no_override"].append(m)
            path_detail[pt]["V_no_override"].append(m["success"])

            # CLIP-sim override variants
            for th, name in zip(COSIM_THRESHOLDS, variant_names[1:]):
                override = apply_clipsim_override(frames, vis_feats_cpu, ref_feat, th, CONSEC)
                # merge: cosim trigger replaces pred from that point (latch)
                pred = list(raw_pred)
                latched = False
                for t in range(len(frames)):
                    if latched:
                        pred[t] = 0
                    elif override[t] == 0:
                        pred[t] = 0
                        latched = True
                m = episode_metrics(pred, expert_classes, args.dt, args.success_fpe)
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
            print(f"  {name:<22} SR={sr*100:5.1f}%  FPE={fpe:.3f}m  ({ok}/{len(ms)})")

        last_path_det = path_detail

    # ── 결과 집계 ──
    summary = {}
    for name in variant_names:
        sv = seed_sr_lists[name]; fv = seed_fpe_lists[name]
        if not sv: continue
        summary[name] = {
            "sr_mean":  float(np.mean(sv)), "sr_std":   float(np.std(sv)),
            "fpe_mean": float(np.mean(fv)), "fpe_std":  float(np.std(fv)),
        }

    best_v = max(summary, key=lambda n: summary[n]["sr_mean"])

    print(f"\n{'='*65}")
    print(f"  STOP CLIP-Similarity Ablation  ({args.seeds}seed 평균)")
    print(f"{'='*65}")
    print(f"  {'Variant':<22} {'SR mean':>8} {'±':>5} {'FPE mean':>9} {'±':>7}")
    print(f"  {'-'*53}")
    for name in variant_names:
        s = summary.get(name); 
        if not s: continue
        marker = " ★" if name == best_v else ""
        print(f"  {name:<22} {s['sr_mean']*100:>7.1f}%"
              f" {s['sr_std']*100:>4.1f}%"
              f" {s['fpe_mean']:>8.3f}m"
              f" {s['fpe_std']:>6.3f}m{marker}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "tag": args.tag, "ckpt": str(args.ckpt), "data": str(args.data),
        "seeds": args.seeds, "success_fpe": args.success_fpe,
        "area_ref_th": args.area_ref_th,
        "thresholds": COSIM_THRESHOLDS, "summary": summary,
    }
    OUT_PATH.with_name("stop_clipsim_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[SAVED] {OUT_PATH.with_name('stop_clipsim_results.json')}")
    print(f"[BEST]  {best_v}  SR={summary[best_v]['sr_mean']*100:.1f}%"
          f"±{summary[best_v]['sr_std']*100:.1f}%")


if __name__ == "__main__":
    main()
