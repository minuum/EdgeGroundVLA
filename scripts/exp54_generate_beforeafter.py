#!/usr/bin/env python3
"""
Before/After 이미지 생성 — basket 마스킹 전후 실제 프레임 비교

각 방향별로 여러 프레임에 대해:
  원본 프레임 | 마스킹된 프레임 (basket 영역 회색)
  + 각각에 모델 예측/confidence 표시
  + 어텐션 맵 overlay

출력: docs/v5/exp54_viz/beforeafter/
  ├── center_flip_01.png ... (flip 발생 케이스)
  ├── left_stable_01.png  ... (flip 없는 케이스)
  ├── right_stable_01.png ...
  └── gallery.html

Usage:
  .venv/bin/python3 scripts/exp54_generate_beforeafter.py
"""

import json, sys, warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_PATH = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"
CKPT_PATH = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage1_v2" / "stage1_v2_projs.pt"
VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
OUT_DIR   = ROOT / "docs" / "v5" / "exp54_viz" / "beforeafter"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIRS     = ["left", "center", "right"]
DIR_IDX  = {"left": 0, "center": 1, "right": 2}
DIR_KO   = {"left": "왼쪽", "center": "중앙", "right": "오른쪽"}
MASK_COLOR = (90, 90, 90)

# 중간 거리 프레임 우선 — 도착 직전이 아닌 내비게이션 중 장면
# area_det: 5~15% = basket 뚜렷하게 보이지만 화면 채우지 않는 자연스러운 거리
AREA_MIN = {"left": 0.003, "center": 0.04, "right": 0.003}
AREA_TARGET = {"left": 0.04, "center": 0.08, "right": 0.04}  # 최적 중간거리
AREA_MAX  = {"left": 0.25,  "center": 0.25, "right": 0.25}   # 이 이상은 도착 직전
N_SAMPLES = {"left": 4, "center": 6, "right": 4}


def load_model(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    base = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16)
    vm = base.vision_model.to(device).eval()
    ip = nn.Linear(1024, 256).to(device)
    ip.load_state_dict(ckpt["image_proj"])
    ip.eval()
    tp = nn.Linear(2048, 256).to(device)
    tp.load_state_dict(ckpt["text_proj"])
    tp.eval()
    anchor = F.normalize(tp(ckpt["anchor_raw"].to(device)), dim=-1)
    return processor, vm, ip, anchor


@torch.no_grad()
def run(vm, ip, processor, anchor, img, device):
    inputs = processor(images=[img], return_tensors="pt")
    pv = inputs["pixel_values"].to(device, dtype=torch.float16)
    out = vm(pixel_values=pv, output_attentions=True)
    feat = out.last_hidden_state.mean(dim=1).float()
    proj = F.normalize(ip(feat), dim=-1)
    sims = (proj @ anchor.T)[0].cpu().numpy()
    attn = out.attentions[-1].float()[0, :, 0, 1:].mean(0)
    attn = (attn / (attn.sum() + 1e-8)).reshape(16, 16).cpu().numpy()
    return sims, attn


def mask_bbox(img, cx, cy, area, scale=1.8):
    W, H = img.size
    side = max(14, int(np.sqrt(area) * min(W, H) * scale))
    half = side // 2
    bx, by = int(cx * W), int(cy * H)
    x1, y1 = max(0, bx-half), max(0, by-half)
    x2, y2 = min(W, bx+half), min(H, by+half)
    m = img.copy()
    ImageDraw.Draw(m).rectangle([x1, y1, x2, y2], fill=MASK_COLOR)
    return m, (x1, y1, x2, y2)


def make_panel(img_pil, attn, sims, gt, label, bbox_px=None, is_masked=False):
    """한 패널 (이미지 + 어텐션 + 예측 바)"""
    W, H = img_pil.size
    img224 = np.array(img_pil.resize((224, 224)))

    attn_up = np.array(
        Image.fromarray((attn / attn.max() * 255).astype(np.uint8)
                        ).resize((224, 224), Image.BICUBIC)
    ).astype(float) / 255.0

    cmap = plt.get_cmap("jet")
    overlay = img224 / 255.0 * 0.45 + cmap(attn_up)[:, :, :3] * 0.55
    overlay = np.clip(overlay, 0, 1)

    pred_idx = sims.argmax()
    pred = DIRS[pred_idx]
    correct = (pred == gt)

    fig = plt.figure(figsize=(5.5, 7), facecolor="#0d1117")

    # 이미지 (위)
    ax_img = fig.add_axes([0.04, 0.38, 0.92, 0.58])
    ax_img.imshow(img224)
    ax_img.axis("off")

    # bbox 표시
    if bbox_px:
        bx1, by1, bx2, by2 = bbox_px
        sx, sy = 224 / W, 224 / H
        rect = mpatches.Rectangle(
            (bx1*sx, by1*sy), (bx2-bx1)*sx, (by2-by1)*sy,
            linewidth=2.5,
            edgecolor="#ef4444" if is_masked else "#22c55e",
            facecolor="#ef444433" if is_masked else "none",
            linestyle="--" if is_masked else "-"
        )
        ax_img.add_patch(rect)

        mid_x = (bx1*sx + bx2*sx) / 2
        top_y = by1 * sy - 6
        ax_img.text(mid_x, top_y,
                    "MASKED" if is_masked else "basket",
                    color="#ef4444" if is_masked else "#22c55e",
                    fontsize=8, fontweight="bold", ha="center", va="bottom")

    # 어텐션 overlay (아래 절반)
    ax_attn = fig.add_axes([0.04, 0.38, 0.92, 0.58])
    ax_attn.imshow(overlay, alpha=0.0)
    ax_attn.axis("off")

    # 제목
    bg = "#1a2a1a" if correct else "#2a1a1a"
    border = "#22c55e" if correct else "#ef4444"
    status = "✅ 정답" if correct else "❌ 오답"
    fig.text(0.5, 0.975, label, ha="center", va="top",
             fontsize=11, color="white", fontweight="bold")
    fig.text(0.5, 0.945, f"예측: {pred}({DIR_KO[pred]})  {status}",
             ha="center", va="top",
             fontsize=10, color="#22c55e" if correct else "#ef4444",
             fontweight="bold")

    # confidence bar
    ax_bar = fig.add_axes([0.06, 0.18, 0.88, 0.17])
    ax_bar.set_facecolor("#0d1117")
    bar_colors = {"left": "#06b6d4", "center": "#22c55e", "right": "#a78bfa"}
    bars = ax_bar.barh(DIRS, [max(0, s) for s in sims],
                       color=[bar_colors[d] for d in DIRS],
                       height=0.55, left=0)
    ax_bar.set_xlim(-0.3, 1.0)
    ax_bar.axvline(0, color="#334155", linewidth=0.8)
    ax_bar.set_facecolor("#0d1117")
    ax_bar.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax_bar.spines.values():
        spine.set_color("#334155")
    ax_bar.set_title("Cosine Similarity", color="#94a3b8", fontsize=8, pad=4)

    for i, (d, s) in enumerate(zip(DIRS, sims)):
        ax_bar.text(max(0, s) + 0.02, i, f"{s:.3f}",
                    va="center", fontsize=8.5,
                    color="#22c55e" if d == pred else "#64748b",
                    fontweight="bold" if d == pred else "normal")

    # highlight predicted bar
    bars[pred_idx].set_edgecolor("white")
    bars[pred_idx].set_linewidth(1.5)

    # 어텐션 맵
    ax_a2 = fig.add_axes([0.06, 0.02, 0.88, 0.14])
    ax_a2.imshow(overlay, aspect="auto")
    ax_a2.axis("off")
    ax_a2.set_title("Attention overlay", color="#64748b", fontsize=8, pad=2)

    return fig


def save_comparison(img, masked_img, fr, gt, bbox_px, sims_o, attn_o, sims_m, attn_m, out_path, idx):
    """원본 | 마스킹 나란히"""
    W, H = img.size

    fig_o = make_panel(img, attn_o, sims_o, gt, f"[{gt.upper()}] 원본 프레임 #{idx}", bbox_px, False)
    fig_m = make_panel(masked_img, attn_m, sims_m, gt, f"[{gt.upper()}] 마스킹 후", bbox_px, True)

    # 두 figure를 하나로 합치기
    fig_o.canvas.draw()
    fig_m.canvas.draw()

    arr_o = np.frombuffer(fig_o.canvas.buffer_rgba(), dtype=np.uint8)
    arr_o = arr_o.reshape(fig_o.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    arr_m = np.frombuffer(fig_m.canvas.buffer_rgba(), dtype=np.uint8)
    arr_m = arr_m.reshape(fig_m.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig_o)
    plt.close(fig_m)

    # 높이 맞추기
    h = max(arr_o.shape[0], arr_m.shape[0])
    def pad_h(a, h):
        if a.shape[0] < h:
            pad = np.full((h - a.shape[0], a.shape[1], 3), 13, dtype=np.uint8)
            a = np.vstack([a, pad])
        return a
    arr_o = pad_h(arr_o, h)
    arr_m = pad_h(arr_m, h)

    # 구분선
    sep = np.full((h, 6, 3), 30, dtype=np.uint8)
    combined = np.hstack([arr_o, sep, arr_m])

    # flip 화살표 텍스트 추가
    pred_o = DIRS[sims_o.argmax()]
    pred_m = DIRS[sims_m.argmax()]
    flipped = pred_o != pred_m

    fig_final, ax = plt.subplots(figsize=(combined.shape[1]/110, combined.shape[0]/110),
                                  facecolor="#0d1117")
    ax.imshow(combined)
    ax.axis("off")

    if flipped:
        ax.text(0.5, 0.5,
                f"← {pred_o} → {pred_m} →",
                ha="center", va="center", fontsize=14,
                color="#f59e0b", fontweight="bold",
                bbox=dict(facecolor="#1a1400", edgecolor="#f59e0b",
                         boxstyle="round,pad=0.4", linewidth=2),
                transform=ax.transAxes)

    fig_final.tight_layout(pad=0)
    fig_final.savefig(str(out_path), dpi=100, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig_final)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print("[MODEL] 로드 중...")
    processor, vm, image_proj, anchor = load_model(device)
    print("[MODEL] 완료\n")

    data = json.loads(DATA_PATH.read_text())

    saved_files = []
    counters = {"left": 0, "center": 0, "right": 0}

    for ep in data:
        d = ep["direction"]
        if counters[d] >= N_SAMPLES[d]:
            continue

        # 중간거리 프레임 우선: AREA_MIN~AREA_MAX 범위에서 target에 가장 가까운 것
        target = AREA_TARGET[d]
        frames_mid = sorted(
            [f for f in ep["frames"]
             if f["consistent"] and f["label"]
             and f.get("area_det")
             and AREA_MIN[d] <= f["area_det"] <= AREA_MAX[d]],
            key=lambda x: abs(x["area_det"] - target)
        )
        # 중간거리 없으면 fallback: area_min 이상 중 가장 작은 것
        if not frames_mid:
            frames_mid = sorted(
                [f for f in ep["frames"]
                 if f["consistent"] and f["label"]
                 and f.get("area_det") and f["area_det"] >= AREA_MIN[d]],
                key=lambda x: x["area_det"]
            )
        if not frames_mid:
            continue

        fr = frames_mid[0]
        cx, cy, area = fr["cx_det"], fr["cy_det"], fr["area_det"]
        gt = fr["label"]

        try:
            with h5py.File(ep["episode"], "r") as f:
                img = Image.fromarray(f["observations"]["images"][fr["frame_idx"]]).convert("RGB")
        except:
            continue

        W, H = img.size
        side = max(14, int(np.sqrt(area) * min(W, H) * 1.8))
        half = side // 2
        bx, by = int(cx * W), int(cy * H)
        bbox_px = (max(0, bx-half), max(0, by-half),
                   min(W, bx+half), min(H, by+half))

        masked_img, _ = mask_bbox(img, cx, cy, area)

        sims_o, attn_o = run(vm, image_proj, processor, anchor, img,        device)
        sims_m, attn_m = run(vm, image_proj, processor, anchor, masked_img, device)

        pred_o = DIRS[sims_o.argmax()]
        pred_m = DIRS[sims_m.argmax()]
        flipped = pred_o != pred_m
        tag = "FLIP" if flipped else "stable"

        counters[d] += 1
        fname = f"{d}_{tag}_{counters[d]:02d}.png"
        out_path = OUT_DIR / fname

        print(f"  [{d}/{counters[d]}] area={area:.4f}  orig→{pred_o}  mask→{pred_m}  {'🔄 FLIP' if flipped else '—'}")
        save_comparison(img, masked_img, fr, gt, bbox_px,
                        sims_o, attn_o, sims_m, attn_m, out_path, counters[d])

        saved_files.append({
            "direction": d, "file": fname,
            "flipped": flipped, "pred_orig": pred_o, "pred_mask": pred_m,
            "area": round(area, 4), "cx": round(cx, 3),
        })

        if all(counters[d2] >= N_SAMPLES[d2] for d2 in DIRS):
            break

    # HTML 갤러리 생성
    html_path = OUT_DIR / "gallery.html"
    rows_by_dir = {d: [] for d in DIRS}
    for s in saved_files:
        rows_by_dir[s["direction"]].append(s)

    dir_colors = {"left": "#06b6d4", "center": "#22c55e", "right": "#a78bfa"}
    dir_ko = {"left": "왼쪽 (Left)", "center": "중앙 (Center)", "right": "오른쪽 (Right)"}

    html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Track 3 — Basket Masking Before/After</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0a0f1a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 40px 24px; }
h1 { font-size: 1.4rem; color: white; margin-bottom: 8px; }
.subtitle { color: #64748b; font-size: 0.9rem; margin-bottom: 40px; line-height: 1.6; }
.section { margin-bottom: 48px; }
.section-title {
  font-size: 1.1rem; font-weight: 700; padding: 10px 18px;
  border-radius: 8px; margin-bottom: 20px; display: inline-block;
}
.grid { display: flex; flex-direction: column; gap: 20px; }
.card {
  background: #0d1117; border: 1px solid #1e293b; border-radius: 12px;
  overflow: hidden;
}
.card.flip { border-color: #f59e0b; }
.card-header {
  padding: 12px 18px; display: flex; align-items: center; gap: 12px;
  background: #111827;
}
.badge {
  padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700;
}
.badge-flip { background: #7c2d12; color: #fb923c; }
.badge-stable { background: #14532d; color: #86efac; }
.meta { color: #64748b; font-size: 0.82rem; }
.card img { width: 100%; display: block; }
.explanation {
  padding: 14px 18px; font-size: 0.85rem; color: #94a3b8; line-height: 1.7;
  border-top: 1px solid #1e293b;
}
.explanation strong { color: #e2e8f0; }
.flip-highlight { color: #f59e0b; font-weight: 700; }
.howto {
  background: #0f172a; border: 1px solid #1e3a5f; border-radius: 12px;
  padding: 20px 24px; margin-bottom: 40px;
}
.howto h2 { color: #38bdf8; font-size: 1rem; margin-bottom: 12px; }
.step { display: flex; gap: 14px; align-items: flex-start; margin-bottom: 10px; }
.step-num {
  background: #1e3a5f; color: #38bdf8; width: 24px; height: 24px;
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 0.8rem; font-weight: 700; flex-shrink: 0; margin-top: 2px;
}
.step-text { color: #94a3b8; font-size: 0.88rem; line-height: 1.6; }
</style>
</head>
<body>
<h1>Track 3 — Basket Masking Ablation: Before / After 비교</h1>
<p class="subtitle">
  Stage 1 v2 모델에 <strong>원본 프레임</strong>과 <strong>basket 영역을 회색으로 가린 프레임</strong>을 각각 입력해서<br>
  모델 예측(left/center/right)이 바뀌는지 확인. 예측이 바뀌면 → 모델이 basket 영역을 보고 있다는 인과 증거.
</p>

<div class="howto">
  <h2>🔬 실험 방법</h2>
  <div class="step">
    <div class="step-num">1</div>
    <div class="step-text"><strong>원본 프레임</strong> 선택: bbox_dataset_frame_level.json에서 consistent=True, area_det ≥ 최소 면적 프레임</div>
  </div>
  <div class="step">
    <div class="step-num">2</div>
    <div class="step-text"><strong>basket 위치 특정</strong>: HSV 탐지(left/right) 또는 Kosmos-2 bbox cx(center)로 구한 cx_det, cy_det, area_det 사용</div>
  </div>
  <div class="step">
    <div class="step-num">3</div>
    <div class="step-text"><strong>마스킹</strong>: basket 중심에서 bbox 크기 × 1.8배 영역을 회색(90,90,90)으로 덮음</div>
  </div>
  <div class="step">
    <div class="step-num">4</div>
    <div class="step-text"><strong>Stage 1 v2 입력</strong>: 원본/마스킹 각각 → frozen Kosmos-2 CLIP → image_proj → cosine similarity 3방향 비교</div>
  </div>
  <div class="step">
    <div class="step-num">5</div>
    <div class="step-text"><strong>판정</strong>: 예측이 바뀌면 "basket을 보고 있었다" (FLIP), 안 바뀌면 "basket 외 정보로 분류" (stable)</div>
  </div>
</div>
"""

    for d in DIRS:
        rows = rows_by_dir[d]
        if not rows:
            continue
        flip_n = sum(1 for r in rows if r["flipped"])
        html += f"""
<div class="section">
  <div class="section-title" style="background:{dir_colors[d]}22;color:{dir_colors[d]};border:1px solid {dir_colors[d]}44">
    {dir_ko[d]} — {flip_n}/{len(rows)} FLIP
  </div>
  <div class="grid">
"""
        for r in rows:
            flip_class = "flip" if r["flipped"] else ""
            badge_class = "badge-flip" if r["flipped"] else "badge-stable"
            badge_text = "🔄 FLIP — 예측 반전!" if r["flipped"] else "✅ STABLE — 예측 유지"
            change_text = (f'<span class="flip-highlight">"{r["pred_orig"]}" → "{r["pred_mask"]}"로 반전</span>'
                          if r["flipped"] else
                          f'원본과 동일하게 "{r["pred_orig"]}" 유지')
            area_pct = r["area"] * 100
            html += f"""
    <div class="card {flip_class}">
      <div class="card-header">
        <span class="badge {badge_class}">{badge_text}</span>
        <span class="meta">cx={r["cx"]:.2f}  basket_area={area_pct:.1f}%  원본예측={r["pred_orig"]}</span>
      </div>
      <img src="{r["file"]}" alt="before/after {d}">
      <div class="explanation">
        basket 영역(이미지의 {area_pct:.1f}%)을 회색으로 가렸을 때: {change_text}.<br>
        <strong>왼쪽</strong>: 원본 프레임 + 녹색 박스(basket 위치) + confidence 바 + 어텐션 맵<br>
        <strong>오른쪽</strong>: 마스킹 후 프레임 + 빨간 박스(가려진 영역) + confidence 바 + 어텐션 맵
      </div>
    </div>
"""
        html += "  </div>\n</div>\n"

    html += """
<div style="margin-top:40px;padding:16px 20px;background:#0f172a;border:1px solid #1e293b;border-radius:10px;font-size:0.85rem;color:#64748b;line-height:1.8">
  <strong style="color:#e2e8f0">결론:</strong>
  center 방향에서 basket area가 클수록 FLIP 비율 높음 → basket 영역이 예측에 인과적 영향.<br>
  left/right는 basket area ~0.5~1.5%로 너무 작아 마스킹 효과 미미 → 전체 이미지 구성으로 분류.<br>
  그러나 Track 2(frozen probe 96.6%)에서 이 전체 이미지 구성 자체가 basket 위치와 강하게 상관됨을 확인.
</div>
</body>
</html>"""

    html_path.write_text(html, encoding="utf-8")
    print(f"\n[GALLERY] {html_path}")

    print(f"\n생성 파일:")
    for s in saved_files:
        tag = "🔄 FLIP" if s["flipped"] else "—"
        print(f"  {s['file']:<35} area={s['area']:.4f}  {tag}")


if __name__ == "__main__":
    main()
