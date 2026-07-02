#!/usr/bin/env python3
"""
Inference visualization for exp67(MLP) / exp71(Transformer) / exp72(cx-Geom)

선택된 대표 프레임에 대해 각 모델의 추론 결과를 오버레이한 이미지를 생성.
오버레이 내용:
  - PG448 bbox 사각형 + cx/cy 십자선
  - 히스토리 cx 궤적 (작은 점)
  - 각 모델의 예측 액션 + softmax 바 차트 (인라인)
  - 신뢰도 수치

Usage:
  .venv/bin/python3 scripts/visualize_inference_exp67_71_72.py [--out-dir docs/v5/inference_viz]
"""
import sys, json, argparse, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image, ImageDraw, ImageFont

# ── 경로 상수 ──────────────────────────────────────────────
VLM_PATH   = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_PT  = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
ANN_PATH   = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json"
EXP67_CKPT = ROOT / "runs/v5_nav/mlp/exp67/action_mlp.pt"
EXP71_CKPT = ROOT / "runs/v5_nav/mlp/exp71/action_transformer.pt"
EXP72_CKPT = ROOT / "runs/v5_nav/mlp/exp72/action_cxgeom.pt"

ACTION_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
ACTION_COLORS = [
    (120,120,120),(72,199,116),(99,179,237),(251,146,60),
    (167,243,208),(252,211,77),(248,113,113),(196,181,253)
]
NUM_CLASSES = 8; WINDOW = 8; PROJ_DIM = 256; VIS_DIM = 1024

# ── 대표 프레임 (cls, ep_path, frame_idx) ──────────────────
SAMPLE_FRAMES = [
    (1, "episode_260408_123008_target_center_straight_path__core__fixed_center.h5", 10),
    (4, "episode_260408_174944_target_center_left_path__core__fixed_center.h5",      1),
    (5, "episode_260408_174944_target_center_left_path__core__fixed_center.h5",     12),
    (6, "episode_260409_165506_target_right_straight_path__core__fixed_center.h5",   0),
    (7, "episode_260409_121940_target_left_straight_path__core__fixed_center.h5",    0),
    (2, "episode_260408_175651_target_center_left_path__core__fixed_center.h5",      0),
    (3, "episode_260408_194433_target_center_right_path__core__fixed_center.h5",     0),
]
DATA_DIR = ROOT / "ROS_action/mobile_vla_dataset_v5"


# ── 모델 정의 ───────────────────────────────────────────────
class FrozenCLIPV2(nn.Module):
    def __init__(self, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ckpt = torch.load(str(STAGE1_PT), map_location=device, weights_only=False)
        self.processor = AutoProcessor.from_pretrained(str(VLM_PATH))
        base = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16)
        self.vm = base.vision_model.to(device).eval()
        self.proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.proj.load_state_dict(ckpt["image_proj"]); self.proj.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, pil_img):
        inp = self.processor(images=[pil_img], return_tensors="pt")
        pv  = inp["pixel_values"].to(self.device, dtype=torch.float16)
        feat = self.vm(pixel_values=pv).last_hidden_state.mean(1).float()
        return self.proj(feat).squeeze(0)  # (256,)


class ActionMLP(nn.Module):
    def __init__(self, d_in=288):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES))
    def forward(self, x): return self.net(x)


class TransformerActionHead(nn.Module):
    FRAME_DIM = 4 + PROJ_DIM  # 260
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.FRAME_DIM))
        self.pos_emb   = nn.Embedding(WINDOW + 1, self.FRAME_DIM)
        el = nn.TransformerEncoderLayer(
            d_model=self.FRAME_DIM, nhead=4, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(self.FRAME_DIM),
            nn.Linear(self.FRAME_DIM, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, NUM_CLASSES))
    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_emb(pos)
        x = self.encoder(x)
        return self.head(x[:, 0])


class CxGeomMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Sequential(
            nn.Linear(288, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1))
        self.branch_b = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.merge = nn.Sequential(
            nn.Linear(160, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES))
    def forward(self, hist, geom):
        return self.merge(torch.cat([self.branch_a(hist), self.branch_b(geom)], -1))


# ── 폰트 로드 (시스템 fallback) ─────────────────────────────
def _font(size=14):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try: return ImageFont.truetype(path, size)
        except: pass
    return ImageFont.load_default()


# ── 오버레이 드로잉 ─────────────────────────────────────────
def draw_frame_overlay(img_pil, ann_frame, history_anns, model_results):
    """
    img_pil     : PIL.Image (원본 프레임)
    ann_frame   : dict (현재 프레임 annotation)
    history_anns: list[dict] (WINDOW개 이전 프레임 annotation, 오래된→최신)
    model_results: list[(model_name, pred_cls, probs)]
    Returns composite PIL.Image
    """
    W, H = img_pil.size
    SCALE = 3  # 업스케일
    PAD_RIGHT = 320  # 모델 결과 패널 너비

    # ── 업스케일 베이스 ─────────────────────────────────────
    base = img_pil.resize((W * SCALE, H * SCALE), Image.NEAREST)
    canvas_w = W * SCALE + PAD_RIGHT
    canvas_h = H * SCALE
    canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 23, 42))
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # ── PG448 bbox 계산 ─────────────────────────────────────
    has_bbox = ann_frame.get("has_bbox", False)
    if has_bbox:
        cx  = ann_frame.get("cx_det", 0.5)
        cy  = ann_frame.get("cy_det", 0.5)
        area = ann_frame.get("area_det", 0.05)
        side = (area ** 0.5)  # sqrt(area)로 변 추정
        x1 = int((cx - side/2) * W * SCALE)
        y1 = int((cy - side/2) * H * SCALE)
        x2 = int((cx + side/2) * W * SCALE)
        y2 = int((cy + side/2) * H * SCALE)
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(W*SCALE-1,x2), min(H*SCALE-1,y2)

        # bbox 사각형 (초록)
        for off in range(3):
            draw.rectangle([x1-off, y1-off, x2+off, y2+off],
                           outline=(52, 211, 153), fill=None)

        # cx 수직선 (노랑)
        cx_px = int(cx * W * SCALE)
        cy_px = int(cy * H * SCALE)
        draw.line([(cx_px, 0), (cx_px, H*SCALE-1)], fill=(251,191,36,150), width=2)
        draw.line([(0, cy_px), (W*SCALE-1, cy_px)], fill=(251,191,36,150), width=1)

        # cx/cy 교차점 마커
        r = 6
        draw.ellipse([cx_px-r, cy_px-r, cx_px+r, cy_px+r],
                     fill=(251,191,36), outline=(15,23,42), width=2)

        # bbox 레이블
        f_small = _font(18)
        draw.text((x1+3, y1+3), f"cx={cx:.2f}  cy={cy:.2f}  A={area:.3f}",
                  fill=(52,211,153), font=f_small)

    # ── 히스토리 cx 궤적 (작은 원, 파란 계열) ─────────────
    for i, h in enumerate(history_anns):
        if not h.get("has_bbox"): continue
        hcx = h.get("cx_det", 0.5)
        hcy = h.get("cy_det", 0.5)
        alpha = int(60 + 130 * (i / max(1, len(history_anns)-1)))
        hcx_px = int(hcx * W * SCALE)
        hcy_px = int(hcy * H * SCALE)
        r2 = 4
        draw.ellipse([hcx_px-r2, hcy_px-r2, hcx_px+r2, hcy_px+r2],
                     fill=(96, 165, 250))

    # ── GT 액션 레이블 (좌하단) ─────────────────────────────
    gt_cls  = ann_frame.get("gt_class", -1)
    gt_name = ACTION_NAMES[gt_cls] if 0 <= gt_cls < NUM_CLASSES else "?"
    gt_color = ACTION_COLORS[gt_cls] if 0 <= gt_cls < NUM_CLASSES else (200,200,200)
    f_gt = _font(22)
    label = f"GT: {gt_name}"
    tw = draw.textlength(label, font=f_gt)
    draw.rectangle([4, H*SCALE-36, 10+tw+6, H*SCALE-4],
                   fill=(0,0,0,180))
    draw.text((8, H*SCALE-34), label, fill=gt_color, font=f_gt)

    # ── 우측 패널: 모델 결과 ─────────────────────────────────
    px = W * SCALE + 10
    py = 10
    f_title = _font(20)
    f_act   = _font(17)
    f_small2 = _font(13)

    MODEL_COLORS = {
        "exp67 MLP":    (167, 243, 208),
        "exp71 Trans":  (52,  211, 153),
        "exp72 Geom":   (110, 231, 183),
    }

    for mname, pred_cls, probs in model_results:
        col = MODEL_COLORS.get(mname, (200,200,200))
        pred_name = ACTION_NAMES[pred_cls] if 0 <= pred_cls < NUM_CLASSES else "?"
        pred_color = ACTION_COLORS[pred_cls] if 0 <= pred_cls < NUM_CLASSES else (200,200,200)
        correct = (pred_cls == gt_cls)

        # 모델 이름 배지
        tw2 = draw.textlength(mname, font=f_title)
        draw.rectangle([px-2, py-2, px+tw2+6, py+24], fill=(30,41,59))
        draw.text((px+2, py), mname, fill=col, font=f_title)
        py += 28

        # 예측 액션
        ck = "✓" if correct else "✗"
        ck_col = (74,222,128) if correct else (248,113,113)
        pred_txt = f"  {ck} {pred_name}  ({probs[pred_cls]*100:.1f}%)"
        draw.text((px, py), pred_txt, fill=ck_col if correct else pred_color, font=f_act)
        py += 22

        # 소프트맥스 바 차트
        bar_max_w = PAD_RIGHT - 20
        for ci, (aname, prob) in enumerate(zip(ACTION_NAMES, probs)):
            bar_w = int(bar_max_w * prob)
            bcol = ACTION_COLORS[ci]
            if ci == pred_cls:
                draw.rectangle([px, py, px+bar_w, py+13], fill=bcol)
            else:
                draw.rectangle([px, py, px+bar_w, py+13],
                               fill=(bcol[0]//3, bcol[1]//3, bcol[2]//3))
            # 레이블 (10% 이상만)
            if prob > 0.08:
                draw.text((px+bar_w+2, py), f"{aname} {prob*100:.0f}%",
                          fill=(148,163,184), font=f_small2)
            py += 15

        py += 10
        # 구분선
        draw.line([(px, py), (canvas_w-4, py)], fill=(30,41,59), width=1)
        py += 8

    return canvas


# ── 메인 ────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="docs/v5/inference_viz")
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # ── 어노테이션 로드 ─────────────────────────────────────
    with open(ANN_PATH) as f:
        ann_all = json.load(f)
    # ep stem → episode dict
    ep_map = {Path(ep["episode"]).stem: ep for ep in ann_all}

    # ── 인코더 로드 ─────────────────────────────────────────
    print("[LOAD] FrozenCLIPV2 인코더...")
    enc = FrozenCLIPV2(device).eval()

    # ── 모델 로드 ────────────────────────────────────────────
    print("[LOAD] exp67 MLP...")
    ckpt67 = torch.load(str(EXP67_CKPT), map_location=device, weights_only=False)
    d_in = ckpt67.get("d_in", 288)
    mlp67 = ActionMLP(d_in).to(device)
    mlp67.load_state_dict(ckpt67["mlp"]); mlp67.eval()

    print("[LOAD] exp71 Transformer...")
    ckpt71 = torch.load(str(EXP71_CKPT), map_location=device, weights_only=False)
    tfm71 = TransformerActionHead().to(device)
    tfm71.load_state_dict(ckpt71["model"]); tfm71.eval()

    print("[LOAD] exp72 cx-Geom...")
    ckpt72 = torch.load(str(EXP72_CKPT), map_location=device, weights_only=False)
    geom72 = CxGeomMLP().to(device)
    geom72.load_state_dict(ckpt72["model"]); geom72.eval()

    # ── 프레임별 처리 ────────────────────────────────────────
    for gt_cls, h5_name, fidx in SAMPLE_FRAMES:
        ep_stem = h5_name.replace(".h5", "")
        h5_path = DATA_DIR / h5_name
        if not h5_path.exists():
            print(f"  [SKIP] {h5_name} not found"); continue

        ep_data = ep_map.get(ep_stem)
        if ep_data is None:
            print(f"  [SKIP] annotation not found for {ep_stem}"); continue

        # 현재 프레임 annotation
        frames = ep_data["frames"]
        fr_map = {fr["frame_idx"]: fr for fr in frames}
        if fidx not in fr_map:
            fidx = frames[0]["frame_idx"] if frames else 0
        ann_frame = fr_map.get(fidx, {})

        # 이미지 로드
        with h5py.File(str(h5_path), "r") as f:
            imgs = f["observations"]["images"][:]
        img_np = imgs[fidx].astype("uint8")
        img_pil = Image.fromarray(img_np)

        # 히스토리 annotations (현재 포함 WINDOW개)
        hist_indices = [max(0, fidx - (WINDOW-1-k)) for k in range(WINDOW)]
        history_anns = [fr_map.get(i, {}) for i in hist_indices]

        # ── 피처 추출 ─────────────────────────────────────────
        print(f"[ENCODE] {h5_name}  fidx={fidx}  gt={ACTION_NAMES[gt_cls]}")
        # 히스토리 이미지 인코딩
        hist_feats = []
        for hi in hist_indices:
            h_img = Image.fromarray(imgs[hi].astype("uint8"))
            hist_feats.append(enc.encode(h_img).cpu())

        # exp67 input: flat concat (cx,cy,area,has_bbox)×WINDOW + vis_feat
        hist_flat = []
        for k, (hi, hfeat) in enumerate(zip(hist_indices, hist_feats)):
            h_ann = fr_map.get(hi, {})
            hist_flat.extend([
                h_ann.get("cx_det", 0.5), h_ann.get("cy_det", 0.5),
                h_ann.get("area_det", 0.05), float(h_ann.get("has_bbox", False))
            ])
        hist_flat.extend(hist_feats[-1].tolist())  # 현재 프레임 vis feat
        x67 = torch.tensor(hist_flat, dtype=torch.float32).unsqueeze(0).to(device)

        # exp71 input: (WINDOW, 260) 시퀀스
        seq71 = []
        for hi, hfeat in zip(hist_indices, hist_feats):
            h_ann = fr_map.get(hi, {})
            bbox = [h_ann.get("cx_det",0.5), h_ann.get("cy_det",0.5),
                    h_ann.get("area_det",0.05), float(h_ann.get("has_bbox",False))]
            seq71.append(bbox + hfeat.tolist())
        x71 = torch.tensor(seq71, dtype=torch.float32).unsqueeze(0).to(device)

        # exp72 input: hist (288) + geom (4)
        geom_vec = [
            ann_frame.get("cx_det",0.5), ann_frame.get("cy_det",0.5),
            ann_frame.get("area_det",0.05), float(ann_frame.get("has_bbox",False))
        ]
        xh72 = x67.clone()
        xg72 = torch.tensor(geom_vec, dtype=torch.float32).unsqueeze(0).to(device)

        # ── 추론 ──────────────────────────────────────────────
        with torch.no_grad():
            probs67 = F.softmax(mlp67(x67), dim=-1).squeeze(0).cpu().numpy()
            probs71 = F.softmax(tfm71(x71), dim=-1).squeeze(0).cpu().numpy()
            probs72 = F.softmax(geom72(xh72, xg72), dim=-1).squeeze(0).cpu().numpy()

        model_results = [
            ("exp67 MLP",    int(probs67.argmax()), probs67),
            ("exp71 Trans",  int(probs71.argmax()), probs71),
            ("exp72 Geom",   int(probs72.argmax()), probs72),
        ]

        # ── 시각화 ────────────────────────────────────────────
        canvas = draw_frame_overlay(img_pil, ann_frame, history_anns[:-1], model_results)

        # 파일명
        out_name = f"viz_{ACTION_NAMES[gt_cls].lower().replace('+','_')}_fidx{fidx}.png"
        out_path = out_dir / out_name
        canvas.save(str(out_path))
        print(f"  → saved: {out_path}")

    print(f"\n[DONE] 이미지 저장 위치: {out_dir}/")


if __name__ == "__main__":
    main()
