#!/usr/bin/env python3
"""
VLM Grounding 기반 추론 비교

색상 임계값 bbox → Stage2 MLP (현재) 대신
VLM grounding bbox → Stage2 MLP 또는 직접 방향 판단으로 교체.

세 가지 방식 비교:
  A. 색상 임계값 bbox → Stage2 MLP          (현재, baseline)
  B. VLM grounding bbox → Stage2 MLP        (bbox 소스만 교체)
  C. VLM grounding cx → 직접 방향 결정       (MLP 없이)

결과가 같으면: Stage2 MLP가 bbox 무시, visual만 봄
결과가 다르면: bbox 소스 교체가 의미 있음

Usage:
  .venv/bin/python3 scripts/eval_vlm_grounding_inference.py
  .venv/bin/python3 scripts/eval_vlm_grounding_inference.py --phrase "gray target"
  .venv/bin/python3 scripts/eval_vlm_grounding_inference.py --n-episodes 20
"""

import argparse, json, sys, warnings, time
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH    = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH   = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
DATA_DIR    = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
STAGE1_V2   = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
STAGE2_CKPT = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage2_v2" / "stage2_v2_mlp.pt"

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
PROJ_DIM    = 256
D_IN        = WINDOW * 4 + PROJ_DIM  # 288


# ─── 모델 클래스 ─────────────────────────────────────────────────────────────

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
    def encode(self, img, device):
        inputs = self.processor(images=[img], return_tensors="pt")
        pv = inputs["pixel_values"].to(device, dtype=torch.float16)
        out = self.vision_model(pixel_values=pv)
        feat = out.last_hidden_state.mean(dim=1).float()
        return F.normalize(self.image_proj(feat), dim=-1).squeeze(0)  # (256,)


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


# ─── VLM Grounding ───────────────────────────────────────────────────────────

class VLMGrounder:
    def __init__(self, vlm_path, device):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.proc  = AutoProcessor.from_pretrained(str(vlm_path))
        self.model = AutoModelForVision2Seq.from_pretrained(
            str(vlm_path), torch_dtype=torch.float16
        ).to(device).eval()
        self.device = device

    @torch.no_grad()
    def ground(self, img, phrase="gray target"):
        """이미지에서 phrase를 grounding → (cx, cy, area, detected) 반환."""
        prompt = f"<grounding> An image of {phrase}."
        inputs = self.proc(text=prompt, images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v,"to") else v for k,v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        raw = self.proc.decode(out[0], skip_special_tokens=False)
        _, entities = self.proc.post_process_generation(raw, cleanup_and_extract=True)

        boxes = []
        for _, _span, ent_boxes in entities:
            boxes.extend(ent_boxes)

        if not boxes:
            return 0.5, 0.5, 0.0, False  # cx, cy, area, detected

        # 가장 큰 box 선택
        best = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        x1, y1, x2, y2 = best
        cx   = (x1 + x2) / 2
        cy   = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        return float(cx), float(cy), float(area), True


# ─── bbox_feat 빌더 ──────────────────────────────────────────────────────────

def build_bbox_feat_from_history(history):
    """history: list of (cx, cy, area, has_bbox) — WINDOW개."""
    arr = []
    for cx, cy, area, has in history:
        arr.extend([cx, cy, area, float(has)])
    return np.array(arr, dtype=np.float32)


# ─── 방향 직접 판단 (C 방식) ─────────────────────────────────────────────────

def cx_to_action(cx, detected):
    """VLM grounding cx만으로 단순 방향 결정."""
    if not detected:
        return 1  # FORWARD (basket 없으면 직진)
    if cx < 0.35:
        return 2   # LEFT
    elif cx > 0.65:
        return 3   # RIGHT
    else:
        return 1   # FORWARD


# ─── 에피소드 평가 ────────────────────────────────────────────────────────────

def load_images(ep):
    ep_path = Path(ep["episode"])
    if ep_path.is_absolute() and ep_path.exists():
        h5_path = ep_path
    else:
        cands = list(DATA_DIR.glob(f"{ep_path.stem}.h5"))
        if not cands:
            cands = list(DATA_DIR.glob(f"**/{ep_path.stem}.h5"))
        if not cands:
            return None
        h5_path = cands[0]
    try:
        with h5py.File(str(h5_path), "r") as f:
            return [Image.fromarray(f["observations"]["images"][fr["frame_idx"]])
                    for fr in ep["frames"]]
    except Exception:
        return None


def eval_episode_A(enc, mlp, imgs, frames, device):
    """A: 색상 임계값 bbox (원본) → Stage2 MLP."""
    hist = [(0.5, 0.5, 0.0, False)] * WINDOW
    preds = []
    with torch.no_grad():
        for img, fr in zip(imgs, frames):
            vis = enc.encode(img, device)
            cx, cy, area = fr["cx"], fr["cy"], fr["area"]
            has = bool(fr.get("has_bbox", False))
            hist.append((cx, cy, area, has))
            hist = hist[-WINDOW:]
            bf  = torch.tensor(build_bbox_feat_from_history(hist), device=device)
            x   = torch.cat([bf, vis]).unsqueeze(0)
            pred = mlp(x).argmax(1).item()
            preds.append(pred)
    return preds


def eval_episode_B(enc, mlp, grounder, imgs, frames, phrase, device):
    """B: VLM grounding bbox → Stage2 MLP."""
    hist = [(0.5, 0.5, 0.0, False)] * WINDOW
    preds = []
    with torch.no_grad():
        for img, fr in zip(imgs, frames):
            vis = enc.encode(img, device)
            # VLM grounding으로 bbox 추출
            cx, cy, area, detected = grounder.ground(img, phrase)
            hist.append((cx, cy, area, detected))
            hist = hist[-WINDOW:]
            bf  = torch.tensor(build_bbox_feat_from_history(hist), device=device)
            x   = torch.cat([bf, vis]).unsqueeze(0)
            pred = mlp(x).argmax(1).item()
            preds.append(pred)
    return preds


def eval_episode_C(grounder, imgs, frames, phrase):
    """C: VLM grounding cx → 직접 방향 결정 (MLP 없음)."""
    preds = []
    for img, fr in zip(imgs, frames):
        cx, cy, area, detected = grounder.ground(img, phrase)
        pred = cx_to_action(cx, detected)
        preds.append(pred)
    return preds


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phrase",      type=str, default="gray target",
                   help="VLM grounding phrase")
    p.add_argument("--n-episodes",  type=int, default=15)
    p.add_argument("--mode",
                   choices=["A","B","C","AB","AC","BC","all"], default="all")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[PHRASE] '{args.phrase}'")

    # 데이터
    data = json.loads(DATA_PATH.read_text())
    labels = [ep["path_type"] for ep in data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, va_idx = next(sss.split(np.zeros(len(data)), labels))
    val_eps = [data[i] for i in va_idx][:args.n_episodes]
    print(f"[DATA] val {len(val_eps)} episodes")

    # Stage1+2 로드
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_V2, device).to(device).eval()
    ckpt = torch.load(str(STAGE2_CKPT), map_location=device, weights_only=False)
    mlp  = ActionMLP().to(device)
    mlp.load_state_dict(ckpt["mlp"])
    mlp.eval()
    print(f"[MODEL] Stage2 v2 val_acc={ckpt['val_acc']:.4f}")

    run_modes = {"A","B","C"} if args.mode == "all" else set(args.mode)
    need_grounder = bool(run_modes & {"B","C"})

    # VLM Grounder (B, C에서만 필요)
    grounder = None
    if need_grounder:
        print(f"[MODEL] VLM Grounder 로드 중...")
        grounder = VLMGrounder(VLM_PATH, device)
        print(f"[MODEL] 완료")

    results = {m: {"correct":0, "total":0, "by_path": defaultdict(lambda:{"c":0,"t":0})}
               for m in ["A","B","C"]}

    for ei, ep in enumerate(val_eps):
        imgs = load_images(ep)
        if imgs is None:
            continue
        frames = ep["frames"]
        pt = ep.get("path_type","?")
        gt_classes = [fr["gt_class"] for fr in frames]

        preds = {}
        t0 = time.time()

        if "A" in run_modes:
            preds["A"] = eval_episode_A(enc, mlp, imgs, frames, device)
        if "B" in run_modes:
            preds["B"] = eval_episode_B(enc, mlp, grounder, imgs, frames, args.phrase, device)
        if "C" in run_modes:
            preds["C"] = eval_episode_C(grounder, imgs, frames, args.phrase)

        elapsed = time.time() - t0

        # 정확도 계산
        line_parts = [f"[{ei+1:2d}/{len(val_eps)}] {pt:<22}"]
        for m, pred_list in preds.items():
            correct = sum(p==g for p,g in zip(pred_list, gt_classes))
            total   = len(gt_classes)
            results[m]["correct"] += correct
            results[m]["total"]   += total
            results[m]["by_path"][pt]["c"] += correct
            results[m]["by_path"][pt]["t"] += total
            acc = correct/total
            line_parts.append(f"{m}={acc:.0%}")
        line_parts.append(f"({elapsed:.0f}s)")
        print("  " + "  ".join(line_parts))

    # 요약
    print(f"\n{'='*60}")
    print(f"SUMMARY  (phrase='{args.phrase}')")
    print(f"{'='*60}")
    desc = {
        "A": "색상 임계값 bbox → Stage2 MLP (baseline)",
        "B": f"VLM grounding '{args.phrase}' → Stage2 MLP",
        "C": f"VLM grounding cx → 직접 방향 결정 (MLP 없음)",
    }
    accs = {}
    for m in ["A","B","C"]:
        if m not in run_modes:
            continue
        r = results[m]
        acc = r["correct"]/r["total"] if r["total"]>0 else 0
        accs[m] = acc
        print(f"\n  [{m}] {desc[m]}")
        print(f"      전체 acc: {acc:.1%}  ({r['correct']}/{r['total']})")
        print(f"      path_type별:")
        for pt in sorted(r["by_path"].keys()):
            v = r["by_path"][pt]
            print(f"        {pt:<22} {v['c']/max(1,v['t']):.0%}  ({v['c']}/{v['t']})")

    print(f"\n{'='*60}")
    print("해석")
    print(f"{'='*60}")
    if "A" in accs and "B" in accs:
        diff = accs["B"] - accs["A"]
        if abs(diff) < 0.03:
            print(f"  A≈B ({diff:+.1%}): Stage2 MLP가 bbox 소스 구분 안 함 → visual이 주 신호")
        elif diff > 0:
            print(f"  B>A ({diff:+.1%}): VLM grounding bbox가 색상 감지보다 Stage2에 더 유용")
        else:
            print(f"  A>B ({diff:+.1%}): 색상 감지가 더 정확, VLM grounding이 노이즈")
    if "A" in accs and "C" in accs:
        diff = accs["C"] - accs["A"]
        print(f"  C vs A ({diff:+.1%}): VLM cx 직접 판단 {'유효' if diff > -0.1 else '불충분'}")

    print(f"\n[다음 단계]")
    print(f"  Option C 학습 시 '{args.phrase}' 프롬프트 사용 권장" if "B" in accs else "")
    print(f"  .venv/bin/python3 scripts/train_optionC_lora.py --prompt p2")


if __name__ == "__main__":
    main()
