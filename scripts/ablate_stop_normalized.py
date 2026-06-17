#!/usr/bin/env python3
"""
STOP Normalization Ablation (CH37 추가 실험)
─────────────────────────────────────────────
시작 프레임 area variance 문제를 보정하는 3가지 variant + PG2 grounding 비교.

문제: bbox_dataset의 40% 에피소드가 시작부터 area≥0.25 → V1이 즉시 발동.
해결 아이디어:
  V6 min_steps_5    : 처음 5프레임은 STOP 금지 (warm-up guard)
  V7 delta_area     : area_t - area_0 > 0.15 (절대값이 아닌 성장량 기준)
  V8 relative_area  : area_t / area_0 > 2.5  (상대적 2.5배 성장)
  V9 min_steps+delta: V6 + V7 조합 (guard + delta)

그리고 PG2 grounding으로 같은 실험 재실행 (--data_pg2 flag).

Usage:
  .venv/bin/python3 scripts/ablate_stop_normalized.py
  .venv/bin/python3 scripts/ablate_stop_normalized.py --seeds 5
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
DATA_HSV    = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
DATA_PG2    = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_base_pg2_cx_243.json"
DATA_DIR    = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OUT_DIR     = ROOT / "docs" / "v5" / "closed_loop_eval"

NUM_CLASSES = 8; WINDOW = 8; VIS_DIM = 1024; PROJ_DIM = 256
D_IN = WINDOW * 4 + PROJ_DIM

PATH_TYPES = [
    "center_straight","center_left","center_right",
    "left_straight","left_left","left_right",
    "right_straight","right_left","right_right",
]


class FrozenCLIPV2(nn.Module):
    def __init__(self, vlm_path, ckpt_path, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        print(f"[MODEL] Stage1 val_acc={ckpt['val_acc']:.4f}")
        self.processor  = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vision_model = base.vision_model.to(device)
        self.image_proj   = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.image_proj.load_state_dict(ckpt["image_proj"])
        for p in self.parameters(): p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, pil_images, device, batch=16):
        all_feats = []
        for i in range(0, len(pil_images), batch):
            inp = self.processor(images=pil_images[i:i+batch], return_tensors="pt")
            pv  = inp["pixel_values"].to(device, dtype=torch.float16)
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


def get_area(fr):  return float(fr.get("area", fr.get("area_det", 0.05)))
def get_cx(fr):    return float(fr.get("cx",   fr.get("cx_det",   0.5)))
def get_cy(fr):    return float(fr.get("cy",   fr.get("cy_det",   0.5)))
def get_has(fr):   return bool(fr.get("has_bbox", fr.get("detected", False)))

def bbox_feat(frames, t, window=WINDOW):
    arr = []
    for k in range(window):
        fr = frames[max(0, t - (window-1-k))]
        arr.extend([get_cx(fr), get_cy(fr), get_area(fr), float(get_has(fr))])
    return np.array(arr, dtype=np.float32)


# ── 정규화 variant override ────────────────────────────────────────────────
def apply_normalized_override(frames, pred_classes, variant):
    """
    variant 딕셔너리:
      area_th    : 절대 area threshold (None = 사용 안 함)
      delta_th   : area_t - area_0 threshold (None = 사용 안 함)
      rel_th     : area_t / area_0 threshold (None = 사용 안 함)
      cx_tol     : |cx-0.5| 허용 범위 (None = 무시)
      min_steps  : 첫 N 프레임 동안 STOP 금지
      consec     : 연속 조건 충족 프레임 수
    """
    if variant.get("no_override"):
        return list(pred_classes)

    area_0     = get_area(frames[0]) if frames else 0.05
    area_th    = variant.get("area_th")
    delta_th   = variant.get("delta_th")
    rel_th     = variant.get("rel_th")
    cx_tol     = variant.get("cx_tol")
    min_steps  = variant.get("min_steps", 0)
    consec     = variant.get("consec", 2)

    result  = list(pred_classes)
    latched = False

    for t in range(len(frames)):
        if latched:
            result[t] = 0
            continue
        if t < min_steps:
            continue

        start  = max(0, t - consec + 1)
        needed = t - start + 1
        ok_cnt = 0

        for i in range(start, t + 1):
            fr = frames[i]
            if not get_has(fr):
                continue
            area = get_area(fr)
            cx   = get_cx(fr)

            # 절대 area 조건
            area_ok = (area_th is None) or (area >= area_th)
            # delta area 조건 (성장량)
            if delta_th is not None:
                area_ok = area_ok and (area - area_0 >= delta_th)
            # relative area 조건 (배율)
            if rel_th is not None:
                area_ok = area_ok and (area / max(area_0, 0.01) >= rel_th)
            # cx 조건
            cx_ok = (cx_tol is None) or (abs(cx - 0.5) <= cx_tol)

            if area_ok and cx_ok:
                ok_cnt += 1

        if ok_cnt >= needed:
            result[t] = 0
            latched = True

    return result


VARIANTS = [
    # baseline
    {"name": "V0_no_override",       "no_override": True},
    # 기존 서버
    {"name": "V1_current_server",    "area_th": 0.25, "cx_tol": 0.35, "consec": 2},
    # 시작 프레임 보정
    {"name": "V6_min_steps_5",       "area_th": 0.25, "cx_tol": 0.35, "consec": 2, "min_steps": 5},
    {"name": "V7_delta_area_015",    "delta_th": 0.15, "consec": 2},
    {"name": "V8_rel_area_25x",      "rel_th": 2.5,   "consec": 2},
    {"name": "V9_minstep_delta",     "delta_th": 0.15, "consec": 2, "min_steps": 3},
    # 참고: V3 (best of L1)
    {"name": "V3_cx_rm_cy045",       "area_th": 0.25, "consec": 2, "cy_thr_ref": True},  # special
]


def apply_v3(frames, pred_classes):
    result = list(pred_classes); latched = False
    for t in range(len(frames)):
        if latched: result[t] = 0; continue
        start = max(0, t-1); ok = 0
        for i in range(start, t+1):
            fr = frames[i]
            if get_has(fr) and get_area(fr)>=0.25 and get_cy(fr)>=0.45:
                ok += 1
        if ok >= (t-start+1): result[t]=0; latched=True
    return result


# ── 에피소드 eval ──────────────────────────────────────────────────────────
def eval_episode(ep_entry, enc, mlp, device, data_dir, window=WINDOW):
    frames  = ep_entry["frames"]
    ep_path = Path(ep_entry["episode"])
    data_dir = Path(data_dir)
    try:
        if ep_path.is_absolute() and ep_path.exists():
            h5_path = ep_path
        else:
            stem = ep_path.stem
            cands = list(data_dir.glob(f"{stem}.h5")) or list(data_dir.glob(f"**/{stem}.h5"))
            if not cands: return None, None
            h5_path = cands[0]
        import io as _io
        with h5py.File(str(h5_path), "r") as f:
            imgs_ds = f["observations"]["images"]; n_imgs = len(imgs_ds)
            imgs = []
            for fr in frames:
                idx = min(fr["frame_idx"], n_imgs-1); raw = imgs_ds[idx]
                if hasattr(raw,"dtype") and raw.dtype!=object and raw.ndim>=2:
                    imgs.append(Image.fromarray(raw.astype("uint8")))
                else:
                    arr = np.frombuffer(bytes(raw), dtype=np.uint8)
                    imgs.append(Image.open(_io.BytesIO(arr)).convert("RGB"))
    except Exception as e:
        return None, None

    expert = [fr["gt_class"] for fr in frames]
    vf = enc.encode_batch(imgs, device)
    mlp.eval()
    with torch.no_grad():
        preds = [mlp(torch.cat([torch.tensor(bbox_feat(frames,t,window),device=device),vf[t]]).unsqueeze(0)).argmax(1).item()
                 for t in range(len(frames))]
    return preds, expert


def episode_metrics(pred, expert, dt, success_fpe):
    pt = build_trajectory(pred, dt); et = build_trajectory(expert, dt)
    pf = pt.final_pos(); ef = et.final_pos()
    fpe = float(np.sqrt((pf[0]-ef[0])**2 + (pf[1]-ef[1])**2))
    tld = pt.total_length() / max(et.total_length(), 1e-6)
    return {"fpe": fpe, "tld": tld, "success": (fpe < success_fpe) and (0.7 <= tld <= 1.5)}


def run_on_dataset(data, enc, mlp, device, args, tag="hsv"):
    ep_labels = [ep.get("path_type","unknown") for ep in data]
    variant_names = [v["name"] for v in VARIANTS]
    seed_sr  = {n: [] for n in variant_names}
    seed_fpe = {n: [] for n in variant_names}

    for seed in range(args.seeds):
        from collections import Counter
        can_strat = all(c>=2 for c in Counter(ep_labels).values())
        if can_strat:
            sss = StratifiedShuffleSplit(1, test_size=args.test_size, random_state=seed)
            _, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
        else:
            from sklearn.model_selection import ShuffleSplit
            ss = ShuffleSplit(1, test_size=args.test_size, random_state=seed)
            _, te_idx = next(ss.split(np.zeros(len(data))))
        val_eps = [data[i] for i in te_idx]

        seed_metrics = {n: [] for n in variant_names}
        for ep in val_eps:
            raw, expert = eval_episode(ep, enc, mlp, device, args.data_dir, WINDOW)
            if raw is None: continue
            for v in VARIANTS:
                if v.get("cy_thr_ref"):
                    pred = apply_v3(ep["frames"], raw)
                else:
                    pred = apply_normalized_override(ep["frames"], raw, v)
                m = episode_metrics(pred, expert, args.dt, args.success_fpe)
                seed_metrics[v["name"]].append(m)

        for name in variant_names:
            ms = seed_metrics[name]
            if not ms: continue
            sr  = sum(1 for m in ms if m["success"]) / len(ms)
            fpe = float(np.mean([m["fpe"] for m in ms]))
            seed_sr[name].append(sr); seed_fpe[name].append(fpe)
        print(f"  [{tag}] seed={seed+1} V0={seed_sr['V0_no_override'][-1]*100:.0f}%"
              f"  V7={seed_sr.get('V7_delta_area_015',['?'])[-1]*100:.0f}%"
              f"  V8={seed_sr.get('V8_rel_area_25x',['?'])[-1]*100:.0f}%", flush=True)

    summary = {}
    for name in variant_names:
        sv = seed_sr[name]; fv = seed_fpe[name]
        if not sv: continue
        summary[name] = {"sr_mean": float(np.mean(sv)), "sr_std": float(np.std(sv)),
                         "fpe_mean": float(np.mean(fv)), "fpe_std": float(np.std(fv))}
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        default=str(STAGE2_CKPT))
    p.add_argument("--stage1",      default=str(STAGE1_CKPT))
    p.add_argument("--data_dir",    default=str(DATA_DIR))
    p.add_argument("--dt",          type=float, default=DT_DEFAULT)
    p.add_argument("--success_fpe", type=float, default=0.5)
    p.add_argument("--test_size",   type=float, default=0.2)
    p.add_argument("--seeds",       type=int,   default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    print("[MODEL] 로드...")
    enc = FrozenCLIPV2(VLM_PATH, Path(args.stage1), device).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    window = ckpt.get("window", WINDOW)
    mlp  = ActionMLP(d_in=window*4+PROJ_DIM).to(device)
    mlp.load_state_dict(ckpt["mlp"]); mlp.eval()
    print(f"[MODEL] val_acc={ckpt.get('val_acc',0):.4f}")

    results = {}

    # ── HSV grounding ──
    print("\n=== HSV Grounding (bbox_dataset_full.json) ===")
    data_hsv = json.loads(DATA_HSV.read_text())
    print(f"  {len(data_hsv)} episodes")
    results["hsv"] = run_on_dataset(data_hsv, enc, mlp, device, args, "HSV")

    # ── PG2 grounding ──
    if DATA_PG2.exists():
        print("\n=== PG2 Grounding (bbox_dataset_base_pg2_cx_243.json) ===")
        data_pg2 = json.loads(DATA_PG2.read_text())
        # PG2 has 'free' path type — filter to 9 standard types
        data_pg2_std = [ep for ep in data_pg2 if ep.get("path_type") in set(PATH_TYPES)]
        print(f"  {len(data_pg2_std)} episodes (9 standard path types)")
        results["pg2"] = run_on_dataset(data_pg2_std, enc, mlp, device, args, "PG2")
    else:
        print(f"\n[SKIP] PG2 data not found: {DATA_PG2}")

    # ── 결과 출력 ──
    for grnd, summary in results.items():
        print(f"\n{'='*70}")
        print(f"  [{grnd.upper()}] STOP Normalization Ablation  ({args.seeds}seeds)")
        print(f"{'='*70}")
        print(f"  {'Variant':<25} {'SR':>8} {'±':>5} {'FPE':>8} {'±':>6}")
        print(f"  {'-'*55}")
        best = max(summary, key=lambda n: summary[n]["sr_mean"])
        for name in [v["name"] for v in VARIANTS]:
            s = summary.get(name)
            if not s: continue
            mk = " ★" if name == best else ""
            print(f"  {name:<25} {s['sr_mean']*100:>7.1f}%"
                  f" {s['sr_std']*100:>4.1f}%"
                  f" {s['fpe_mean']:>7.3f}m"
                  f" {s['fpe_std']:>5.3f}m{mk}")

    # ── JSON 저장 ──
    out_path = OUT_DIR / "stop_normalized_results.json"
    out_path.write_text(json.dumps({
        "variants": [v["name"] for v in VARIANTS],
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
