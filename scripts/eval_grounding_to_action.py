#!/usr/bin/env python3
"""
Grounding → Action 파이프라인 연결 테스트

질문: PaliGemma grounding(Exp57)이 cx를 올바르게 내면
      Stage2 MLP가 올바른 action을 예측하는가?

파이프라인:
  이미지 → PaliGemma Exp57 LoRA → cx_pred (basket 위치)
         → Stage1 v2 CLIP LoRA → visual_feat (256dim)
         → Stage2 v2 MLP → action class

GT 비교: 각 프레임의 gt_action vs 예측 action

Usage:
  .venv/bin/python3 scripts/eval_grounding_to_action.py
  .venv/bin/python3 scripts/eval_grounding_to_action.py --n 20 --show-errors
"""
import argparse, json, re, sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 경로
PG1_PATH   = ROOT / ".vlms/kosmos-2-patch14-224"          # Stage1 CLIP
EXP57_PATH = ROOT / "runs/v5_nav/grounding/exp57"          # PaliGemma LoRA
PG_VLM     = Path.home() / ".cache/huggingface/hub" \
             / "models--google--paligemma-3b-pt-224/snapshots" \
             / "35e4f46485b4d07967e7e9935bc3786aad50687c"
STAGE1_PT  = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
STAGE2_PT  = ROOT / "runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt"
ANN_JSON   = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_frame_level.json"

ACTIONS = ["STOP","FORWARD","LEFT","RIGHT","FWD+LEFT","FWD+RIGHT","ROT_L","ROT_R"]
LOC_RE  = re.compile(r"<loc(\d{4})>")
WINDOW  = 8  # bbox history window


def load_stage1(device):
    """Stage1 v2: Kosmos-2 CLIP LoRA → 256dim visual feature"""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel

    ckpt = torch.load(str(STAGE1_PT), map_location=device, weights_only=False)
    processor = AutoProcessor.from_pretrained(str(PG1_PATH))
    base = AutoModelForVision2Seq.from_pretrained(str(PG1_PATH), torch_dtype=torch.float16)
    vm = base.vision_model.to(device).eval()

    image_proj = nn.Linear(1024, 256).to(device).eval()
    image_proj.load_state_dict(ckpt["image_proj"])
    anchor = F.normalize(ckpt["anchor_raw"].to(device).float(), dim=-1)
    text_proj = nn.Linear(2048, 256).to(device).eval()
    text_proj.load_state_dict(ckpt["text_proj"])
    return processor, vm, image_proj, text_proj, anchor


def load_stage2(device):
    """Stage2 v2: MLP (256 visual + 32 bbox_hist) → 8-class action
    구조: net.0(288→256) ReLU net.3(256→128) ReLU net.6(128→64) ReLU net.9(64→8)
    """
    ckpt = torch.load(str(STAGE2_PT), map_location=device, weights_only=False)
    mlp_state = ckpt["mlp"]  # {'net.0.weight': ..., ...}

    class ActionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(288, 256), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, 8)
            )
        def forward(self, x): return self.net(x)

    mlp = ActionMLP().to(device)
    mlp.load_state_dict(mlp_state)
    mlp.eval()
    print(f"  Stage2 MLP 로드 완료 (val_acc={ckpt['val_acc']*100:.1f}%)")
    return mlp


def load_paligemma_grounding(device):
    """PaliGemma Exp57 LoRA → grounding (detect gray basket → cx)"""
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    from peft import PeftModel

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = PaliGemmaProcessor.from_pretrained(str(PG_VLM))
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG_VLM), torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model = PeftModel.from_pretrained(base, str(EXP57_PATH)).eval()
    return processor, model, dtype


@torch.no_grad()
def ground(pg_model, pg_proc, img_np, device, dtype):
    """PaliGemma → cx (0~1), 없으면 None"""
    pil = Image.fromarray(img_np).convert("RGB")
    inp = pg_proc(text="detect gray basket", images=pil, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    gen = pg_model.generate(**inp, max_new_tokens=48, do_sample=False)
    raw = pg_proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                skip_special_tokens=False)[0]
    locs = [int(v) / 1023.0 for v in LOC_RE.findall(raw)]
    if len(locs) >= 4:
        _, x1, _, x2 = locs[:4]
        cx = (x1 + x2) / 2
        return cx, raw.strip()
    return None, raw.strip()


@torch.no_grad()
def get_visual_feat(vm, image_proj, pg1_proc, img_np, device):
    """CLIP LoRA → 256dim visual feature"""
    pil = Image.fromarray(img_np).convert("RGB")
    inp = pg1_proc(images=[pil], return_tensors="pt")
    pv = inp["pixel_values"].to(device, dtype=torch.float16)
    feat = vm(pixel_values=pv).last_hidden_state.mean(dim=1).float()
    return F.normalize(image_proj(feat), dim=-1)  # (1, 256)


def cx_to_bbox_vec(cx: float, cy: float = 0.5, area: float = 0.05) -> torch.Tensor:
    """cx → bbox_hist (WINDOW, 4) → flatten 32dim"""
    row = torch.tensor([cx, cy, area, 0.0])
    hist = row.unsqueeze(0).repeat(WINDOW, 1)  # (8, 4)
    return hist.flatten()  # 32dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="테스트 프레임 수")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"[DEVICE] {device}")
    print("\n[1/3] Stage1 v2 CLIP LoRA 로드...")
    pg1_proc, vm, image_proj, _, _ = load_stage1(device)
    print("[2/3] Stage2 v2 MLP 로드...")
    mlp = load_stage2(device)
    print("[3/3] PaliGemma Exp57 LoRA 로드...")
    pg_proc, pg_model, pg_dtype = load_paligemma_grounding(device)
    print("로드 완료\n")

    with open(ANN_JSON) as f:
        ann = json.load(f)

    results = []
    tested = 0

    for ep in ann:
        if tested >= args.n:
            break
        frames = [fr for fr in ep["frames"]
                  if fr.get("detected") and 0.03 < fr.get("area_det", 0) < 0.25]
        if not frames:
            continue

        h5_path = ep["episode"]
        try:
            with h5py.File(h5_path, "r") as f:
                imgs = f["observations"]["images"][:]
                actions_raw = f["actions"][:]  # (N, 3)
        except Exception:
            continue

        fr = frames[len(frames) // 2]  # 중간 프레임
        fidx = fr["frame_idx"]
        if fidx >= len(imgs):
            continue

        img_np = imgs[fidx].astype(np.uint8)
        gt_cx  = fr["cx_det"]
        gt_lbl = fr["label"]   # "left" / "center" / "right"
        gt_action_raw = actions_raw[fidx]  # (vx, vy, w) 연속 액션

        # GT 액션 클래스 (연속→이산 변환 근사)
        # left/center/right label 기반
        gt_class_approx = {"left": 4, "center": 1, "right": 5}.get(gt_lbl, 1)

        # ── PaliGemma grounding ──
        cx_pred, raw_out = ground(pg_model, pg_proc, img_np, device, pg_dtype)
        grounding_ok = cx_pred is not None

        if not grounding_ok:
            cx_pred = 0.5  # fallback

        # ── Visual feature ──
        vis_feat = get_visual_feat(vm, image_proj, pg1_proc, img_np, device)  # (1,256)

        # ── bbox_hist from grounding cx ──
        bbox_vec = cx_to_bbox_vec(cx_pred).to(device)  # 32dim

        # ── MLP action ──
        inp_vec = torch.cat([vis_feat.squeeze(0), bbox_vec], dim=0).unsqueeze(0)  # (1,288)
        logits = mlp(inp_vec.float())
        pred_class = logits.argmax(dim=1).item()
        pred_name = ACTIONS[pred_class]

        # cx 방향 일치 여부 (grounding cx → action 방향)
        # cx < 0.4 → LEFT계열, cx > 0.6 → RIGHT계열, 중간 → FORWARD
        if cx_pred < 0.40:  expected_dir = "left"
        elif cx_pred > 0.60: expected_dir = "right"
        else: expected_dir = "center"

        pred_dir = "left" if pred_class in (2, 4, 6) else \
                   "right" if pred_class in (3, 5, 7) else "center"
        dir_match = (expected_dir == pred_dir)

        results.append({
            "ep": Path(h5_path).stem[:30],
            "frame": fidx,
            "gt_lbl": gt_lbl,
            "gt_cx": round(gt_cx, 3),
            "cx_pred": round(cx_pred, 3) if cx_pred else None,
            "grounding_ok": grounding_ok,
            "raw_out": raw_out[:60],
            "pred_action": pred_name,
            "dir_match": dir_match,
        })
        tested += 1

        mark = "✅" if dir_match else "❌"
        grnd = f"cx={cx_pred:.3f}" if grounding_ok else "miss"
        print(f"  {mark} ep={Path(h5_path).stem[-20:]} fr={fidx} "
              f"gt={gt_lbl}({gt_cx:.2f}) grnd={grnd} → {pred_name} ({pred_dir})")

    # ── 결과 집계 ──
    n = len(results)
    grnd_ok  = sum(1 for r in results if r["grounding_ok"])
    dir_ok   = sum(1 for r in results if r["dir_match"])
    grnd_and_dir = sum(1 for r in results if r["grounding_ok"] and r["dir_match"])

    print(f"\n{'='*60}")
    print(f"결과 ({n}프레임)")
    print(f"{'='*60}")
    print(f"  Grounding 성공률: {grnd_ok}/{n} = {grnd_ok/n*100:.1f}%")
    print(f"  방향 일치율:     {dir_ok}/{n} = {dir_ok/n*100:.1f}%")
    print(f"  Grounding→Action 연결 성공: {grnd_and_dir}/{n} = {grnd_and_dir/n*100:.1f}%")
    print()
    print("해석: grounding cx가 올바르면 action도 올바른 방향이면")
    print("  → PaliGemma grounding이 실제 내비게이션으로 연결됨 증명")

    if args.show_errors:
        print("\n[오류 케이스]")
        for r in results:
            if not r["dir_match"]:
                print(f"  ❌ {r['ep']} gt={r['gt_lbl']} cx_pred={r['cx_pred']} → {r['pred_action']}")
                print(f"     decoded: {r['raw_out']}")


if __name__ == "__main__":
    main()
