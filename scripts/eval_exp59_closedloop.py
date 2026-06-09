#!/usr/bin/env python3
"""
Exp59 Closed-Loop Evaluation
HSV bbox → PaliGemma2 Exp59 LoRA grounding으로 교체

파이프라인:
  이미지 → PaliGemma2 Exp59 LoRA → cx, cy (신경망 grounding)
          → Stage1 v2 CLIP LoRA  → visual_feat 256dim
          → Stage2 v2 MLP        → action class (8-class)

기준: FPE < 0.5m AND TLD ∈ [0.7, 1.5]

Usage:
  .venv/bin/python3 scripts/eval_exp59_closedloop.py
  .venv/bin/python3 scripts/eval_exp59_closedloop.py --success-fpe 0.8 --n-eps 20
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

# ── 경로 ──────────────────────────────────────────────────────────────────
VLM_PATH    = ROOT / ".vlms" / "kosmos-2-patch14-224"          # CLIP (Stage1)
PG2_PATH    = Path.home() / ".cache/huggingface/hub" \
              / "models--google--paligemma2-3b-mix-224" \
              / "snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
EXP59_PATH  = ROOT / "runs/v5_nav/grounding/exp59"
STAGE1_PT   = ROOT / "runs/v5_nav/mlp/exp54/stage1_v2/stage1_v2_projs.pt"
STAGE2_PT   = ROOT / "runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt"
ANN_JSON    = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_frame_level.json"
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


# ── Stage2 v2: Action MLP ────────────────────────────────────────────────

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


# ── PaliGemma2 Exp59 그라운딩 ─────────────────────────────────────────────

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
        """→ (cx, cy, area) or None if not detected"""
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


# ── 에피소드 평가 ────────────────────────────────────────────────────────

def eval_episode(ep_entry, enc, mlp, grounder, device, ema_alpha=1.0):
    frames = ep_entry["frames"]
    h5_path = Path(ep_entry["episode"])
    if not h5_path.exists():
        return None, None, 0.0

    try:
        with h5py.File(str(h5_path), "r") as f:
            imgs_np = f["observations"]["images"][:]
    except Exception as e:
        return None, None, 0.0

    n = len(frames)
    pil_imgs = [Image.fromarray(imgs_np[fr["frame_idx"]].astype("uint8")) for fr in frames]

    # PaliGemma2로 각 프레임 grounding
    detections = []
    for img in pil_imgs:
        det = grounder.detect(img)
        detections.append(det)  # (cx, cy, area) or None

    hit_n = sum(1 for d in detections if d is not None)

    # CLIP visual feature
    vis_feats = [enc.encode(img) for img in pil_imgs]

    pred_classes = []
    expert_classes = [fr["gt_class"] for fr in frames]

    # bbox_hist 구성 (WINDOW 이전 프레임까지 축적)
    # (cx, cy, area, has_bbox)
    hist = [(0.5, 0.5, 0.05, 0.0)] * WINDOW  # fallback: 중앙

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
                        smoothed_cx = cx
                        smoothed_cy = cy
                        smoothed_area = area
                    else:
                        smoothed_cx   = ema_alpha * cx + (1.0 - ema_alpha) * smoothed_cx
                        smoothed_cy   = ema_alpha * cy + (1.0 - ema_alpha) * smoothed_cy
                        smoothed_area = ema_alpha * area + (1.0 - ema_alpha) * smoothed_area
                    hist.append((smoothed_cx, smoothed_cy, smoothed_area, 1.0))
                else:
                    hist.append((cx, cy, area, 1.0))
            else:
                last_cx, last_cy, last_area, _ = hist[-1]
                hist.append((last_cx, last_cy, last_area, 0.0))  # 이전 좌표 유지하되, 미검출(0.0)
            
            hist = hist[-WINDOW:]

            bbox_vec = torch.tensor(
                [v for item in hist for v in item],
                dtype=torch.float32, device=device
            )[:32]  # 8×4 = 32

            x = torch.cat([bbox_vec, vis_feats[t].to(device)]).unsqueeze(0)
            pred_classes.append(mlp(x).argmax(1).item())

    return pred_classes, expert_classes, hit_n / max(n, 1)


# ── 시뮬레이션 지표 ──────────────────────────────────────────────────────

def compute_metrics(pred, expert, dt=0.5, success_fpe=0.5):
    from scripts.sim.rollout_core import build_trajectory, compute_metrics as core_compute_metrics
    try:
        pred_traj   = build_trajectory(pred,   dt=dt)
        expert_traj = build_trajectory(expert, dt=dt)  # note: expert here is class list
        res = core_compute_metrics(expert_traj, pred_traj, success_fpe)
        fpe = res["fpe"]
        tld = res["tld"]
        success = res["success"]
    except Exception:
        fpe, tld, success = 9.9, 0.0, False
    return {"fpe": fpe, "tld": tld, "success": success}


# ── 메인 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-fpe", type=float, default=0.5)
    parser.add_argument("--n-eps",       type=int,   default=None,
                        help="테스트 에피소드 수 (None=전체)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ema-alpha",   type=float, default=1.0,
                        help="BBox 스무딩용 EMA 가중치 (1.0=EMA 비활성, 0.0~1.0 사이 값)")
    parser.add_argument("--stage2-pt",   default=str(STAGE2_PT),
                        help="Stage2 MLP 가중치 (Exp60 aug 모델 평가용)")
    parser.add_argument("--out-tag",     default="",
                        help="결과 JSON 파일명 suffix (모델별 분리 저장)")
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"[DEVICE] {device}")
    print(f"[EMA-ALPHA] {args.ema_alpha if args.ema_alpha < 1.0 else 'Disabled'}")
    print("\n[1/3] Stage1 v2 CLIP 인코더 로드...")
    enc  = Stage1Encoder(VLM_PATH, STAGE1_PT, device).eval()

    print(f"[2/3] Stage2 v2 MLP 로드... ({Path(args.stage2_pt).name})")
    ckpt = torch.load(str(args.stage2_pt), map_location=device, weights_only=True)
    mlp  = ActionMLP().to(device)
    mlp.load_state_dict(ckpt["mlp"])
    mlp.eval()
    print(f"  val_acc={ckpt['val_acc']*100:.1f}%")

    print("[3/3] PaliGemma2 Exp59 LoRA 로드...")
    grounder = Exp59Grounder(PG2_PATH, EXP59_PATH, device)
    print("로드 완료\n")

    with open(ANN_JSON) as f:
        ann = json.load(f)

    # val set (고정 split)
    import random; random.seed(42)
    random.shuffle(ann)
    val_n  = max(1, int(len(ann) * 0.15))
    val_ep = ann[:val_n]
    if args.n_eps:
        val_ep = val_ep[:args.n_eps]

    print(f"테스트 에피소드: {len(val_ep)}개\n")

    results_by_path = defaultdict(list)
    all_m = []
    grnd_rates = []

    for i, ep in enumerate(val_ep):
        pt = ep.get("path_type", "unknown")
        out = eval_episode(ep, enc, mlp, grounder, device, ema_alpha=args.ema_alpha)
        if out[0] is None:
            continue
        pred, expert, grnd_rate = out
        m = compute_metrics(pred, expert, success_fpe=args.success_fpe)
        m["path_type"] = pt
        results_by_path[pt].append(m)
        all_m.append(m)
        grnd_rates.append(grnd_rate)
        mark = "✅" if m["success"] else "❌"
        print(f"  {mark} [{i+1:3d}/{len(val_ep)}] {pt:<22} "
              f"FPE={m['fpe']:.3f}m TLD={m['tld']:.2f} "
              f"grnd={grnd_rate*100:.0f}%")

    total   = len(all_m)
    success = sum(1 for m in all_m if m["success"])
    print(f"\n{'='*60}")
    print(f"  Exp59 Grounding → Stage2 MLP Closed-Loop")
    print(f"  성공: {success}/{total} = {success/total*100:.1f}%")
    print(f"  평균 FPE: {np.mean([m['fpe'] for m in all_m]):.3f}m")
    print(f"  평균 TLD: {np.mean([m['tld'] for m in all_m]):.3f}")
    print(f"  평균 grounding 성공률: {np.mean(grnd_rates)*100:.1f}%")
    print(f"  참고: Exp54 CL=96.7% (HSV bbox 사용)")
    print(f"{'='*60}")

    print("\npath_type별:")
    for pt in sorted(results_by_path):
        ms = results_by_path[pt]
        sr  = sum(1 for m in ms if m["success"]) / len(ms)
        mfpe = np.mean([m["fpe"] for m in ms])
        print(f"  {pt:<22} {sum(1 for m in ms if m['success'])}/{len(ms)} SR={sr:.0%} FPE={mfpe:.3f}m")

    result = {
        "exp": "exp59_closedloop",
        "ema_alpha": args.ema_alpha,
        "success_rate": success / total if total > 0 else 0,
        "mean_fpe": float(np.mean([m["fpe"] for m in all_m])),
        "mean_tld": float(np.mean([m["tld"] for m in all_m])),
        "mean_grounding_rate": float(np.mean(grnd_rates)),
        "n_episodes": total,
    }
    result["stage2_pt"] = Path(args.stage2_pt).name
    fname = f"exp59_closedloop_result{('_' + args.out_tag) if args.out_tag else ''}.json"
    out_path = OUT_DIR / fname
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nJSON → {out_path}")


if __name__ == "__main__":
    main()
