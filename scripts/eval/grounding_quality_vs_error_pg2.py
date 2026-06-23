#!/usr/bin/env python3
"""
plan_20260622_grounding_quality_and_window_ablation.md §1 — "방향전환 실패가
그라운딩 신뢰도 낮은 프레임에서 더 많이 일어나는가?"를 진단한다.

closed_loop_eval_hidden_state.py와 같은 모델 로딩/추론 경로를 재사용하되,
집계(SR/FPE)만 내던 걸 프레임 단위 상세 레코드로 바꿔서 저장한다.
새 학습 없음 — 이미 끝난 3개 ckpt(baseline/add/replace)로 val 29개 에피소드를
다시 추론만 해서 로그를 남기는 읽기 전용 분석.

산출: docs/v5/closed_loop_eval/grounding_quality_vs_error.json
  레코드: {episode, path_type, t, mode, pred, gt, error, has_bbox, area, cx, cy}

Usage:
  .venv/bin/python3 scripts/eval/grounding_quality_vs_error.py
"""
import sys
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_CKPT = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
DATA_PATH = ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2.json"
HIDDEN_CACHE = ROOT / "docs/v5/hidden_state_cache/v5_hidden_states.npz"

CKPTS = {
    "baseline": (ROOT / "runs/v5_nav/mlp/exp_hidden_state/stage2_v2/stage2_hidden_none.pt", "none"),
    "add": (ROOT / "runs/v5_nav/mlp/exp_hidden_state/stage2_v2/stage2_hidden_add.pt", "add"),
    "replace": (ROOT / "runs/v5_nav/mlp/exp_hidden_state/stage2_v2/stage2_hidden_replace.pt", "replace"),
}

PROJ_DIM = 256
HIDDEN_DIM = 2304
WINDOW = 8


class FrozenCLIPV2(nn.Module):
    def __init__(self, vlm_path, ckpt_path, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        self.processor = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vision_model = base.vision_model.to(device)
        self.image_proj = nn.Linear(1024, PROJ_DIM).to(device)
        self.image_proj.load_state_dict(ckpt["image_proj"])
        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.image_proj.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, pil_images, device, batch=32):
        all_feats = []
        for i in range(0, len(pil_images), batch):
            imgs = pil_images[i:i+batch]
            inputs = self.processor(images=imgs, return_tensors="pt")
            pv = inputs["pixel_values"].to(device, dtype=torch.float16)
            out = self.vision_model(pixel_values=pv)
            feat = out.last_hidden_state.mean(dim=1).float()
            all_feats.append(F.normalize(self.image_proj(feat), dim=-1))
        return torch.cat(all_feats, dim=0)


class ActionMLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 8),
        )
    def forward(self, x): return self.net(x)


def bbox_feat(frames, t, window=WINDOW):
    arr = []
    for k in range(window):
        fr = frames[max(0, t - (window - 1 - k))]
        cx = fr.get("cx", 0.5); cy = fr.get("cy", 0.5)
        area = fr.get("area", 0.05); has = float(fr.get("has_bbox", False))
        arr.extend([cx, cy, area, has])
    return np.array(arr, dtype=np.float32)


def load_images(h5_path, indices):
    with h5py.File(h5_path, "r") as f:
        imgs_ds = f["observations"]["images"]
        return [Image.fromarray(imgs_ds[i].astype(np.uint8)).convert("RGB") for i in indices]


def build_x(mode, bf, img_feat, hv):
    if mode == "none":
        return torch.cat([bf, img_feat])
    h = torch.from_numpy(hv.astype(np.float32)) if hv is not None else torch.zeros(HIDDEN_DIM, dtype=torch.float32)
    if mode == "add":
        return torch.cat([bf, img_feat, h])
    return torch.cat([img_feat, h])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = json.loads(DATA_PATH.read_text())
    ep_labels = [ep["path_type"] for ep in data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
    te_eps_all = [data[i] for i in te_idx]
    print(f"[DATA] val episodes(전체)={len(te_eps_all)}")

    hidden_cache = {k: v for k, v in np.load(HIDDEN_CACHE).items()}
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_CKPT, device).to(device).eval()

    records = []
    for mode_name, (ckpt_path, mode) in CKPTS.items():
        if mode == "none":
            te_eps = te_eps_all
        else:
            te_eps = [ep for ep in te_eps_all
                      if any(f"{Path(ep['episode']).stem}__f{fr['frame_idx']}" in hidden_cache for fr in ep["frames"])]

        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        mlp = ActionMLP(d_in=ckpt["d_in"]).to(device)
        mlp.load_state_dict(ckpt["mlp"])
        mlp.eval()

        for ep in te_eps:
            try:
                imgs = load_images(ep["episode"], [fr["frame_idx"] for fr in ep["frames"]])
            except Exception:
                continue
            feats = enc.encode_batch(imgs, device)
            with torch.no_grad():
                for t, fr in enumerate(ep["frames"]):
                    bf = torch.tensor(bbox_feat(ep["frames"], t), dtype=torch.float32)
                    hv = hidden_cache.get(f"{Path(ep['episode']).stem}__f{fr['frame_idx']}") if mode != "none" else None
                    x = build_x(mode, bf, feats[t].cpu(), hv).unsqueeze(0).to(device)
                    pred = int(mlp(x).argmax(1).item())
                    gt = fr["gt_class"]
                    records.append({
                        "episode": Path(ep["episode"]).stem,
                        "path_type": ep["path_type"],
                        "t": t,
                        "mode": mode_name,
                        "pred": pred,
                        "gt": gt,
                        "error": int(pred != gt),
                        "has_bbox": bool(fr.get("has_bbox", False)),
                        "area": float(fr.get("area", 0.0)),
                        "cx": float(fr.get("cx", 0.5)),
                        "cy": float(fr.get("cy", 0.5)),
                    })
        print(f"[{mode_name}] {len(te_eps)} episodes 처리 완료, 누적 레코드={len(records)}")

    out_path = ROOT / "docs/v5/closed_loop_eval/grounding_quality_vs_error_pg2.json"
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\n[저장] {out_path} ({len(records)} 레코드)")

    # ── 간단 상관 분석 ──
    import statistics as st
    for mode_name in CKPTS:
        recs = [r for r in records if r["mode"] == mode_name]
        err_has = [r["error"] for r in recs if r["has_bbox"]]
        err_nohas = [r["error"] for r in recs if not r["has_bbox"]]
        areas_err = [r["area"] for r in recs if r["error"] == 1 and r["has_bbox"]]
        areas_ok = [r["area"] for r in recs if r["error"] == 0 and r["has_bbox"]]
        print(f"\n=== {mode_name} ===")
        print(f"  has_bbox=True  오류율: {st.mean(err_has)*100:.1f}% (n={len(err_has)})" if err_has else "  has_bbox=True 없음")
        print(f"  has_bbox=False 오류율: {st.mean(err_nohas)*100:.1f}% (n={len(err_nohas)})" if err_nohas else "  has_bbox=False 없음")
        if areas_err and areas_ok:
            print(f"  area 평균(오류 프레임): {st.mean(areas_err):.4f} vs (정답 프레임): {st.mean(areas_ok):.4f}")


if __name__ == "__main__":
    main()
