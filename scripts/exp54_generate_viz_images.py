#!/usr/bin/env python3
"""
교수님 보고용 시각화 이미지 생성

1. masking_comparison.png  — 원본 vs 마스킹 before/after (방향별)
2. linear_probe_results.png — zero-shot probe confusion matrix + accuracy bar
3. track_summary.png       — 5-track 증거 요약 인포그래픽

Usage:
  .venv/bin/python3 scripts/exp54_generate_viz_images.py
"""

import json, sys, warnings
from pathlib import Path
from collections import defaultdict

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
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

DATA_PATH = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"
CKPT_PATH = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage1_v2" / "stage1_v2_projs.pt"
VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
OUT_DIR   = ROOT / "docs" / "v5" / "exp54_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIRS    = ["left", "center", "right"]
DIR_IDX = {"left": 0, "center": 1, "right": 2}
MASK_COLOR = (100, 100, 100)


# ─── 모델 로드 ────────────────────────────────────────────

def load_model(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    base = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16)
    vm = base.vision_model.to(device).eval()
    image_proj = nn.Linear(1024, 256).to(device)
    image_proj.load_state_dict(ckpt["image_proj"])
    image_proj.eval()
    text_proj = nn.Linear(2048, 256).to(device)
    text_proj.load_state_dict(ckpt["text_proj"])
    text_proj.eval()
    anchor = F.normalize(text_proj(ckpt["anchor_raw"].to(device)), dim=-1)
    return processor, vm, image_proj, anchor, ckpt["val_acc"]


@torch.no_grad()
def predict(vm, proj, processor, anchor, img, device):
    inputs = processor(images=[img], return_tensors="pt")
    pv = inputs["pixel_values"].to(device, dtype=torch.float16)
    out = vm(pixel_values=pv, output_attentions=True)
    feat = out.last_hidden_state.mean(dim=1).float()
    p = F.normalize(proj(feat), dim=-1)
    sims = (p @ anchor.T)[0].cpu().numpy()
    attn = out.attentions[-1].float()
    cls_a = attn[0, :, 0, 1:].mean(0)
    cls_a = (cls_a / (cls_a.sum() + 1e-8)).reshape(16, 16).cpu().numpy()
    return sims, cls_a


def mask_basket(img, cx, cy, area, scale=1.8):
    W, H = img.size
    side = max(20, int(np.sqrt(area) * min(W, H) * scale))
    half = side // 2
    bx, by = int(cx * W), int(cy * H)
    x1, y1 = max(0, bx-half), max(0, by-half)
    x2, y2 = min(W, bx+half), min(H, by+half)
    m = img.copy()
    ImageDraw.Draw(m).rectangle([x1, y1, x2, y2], fill=MASK_COLOR)
    return m, (x1, y1, x2, y2)


# ─── Figure 1: Masking Comparison ─────────────────────────

def make_masking_figure(device, processor, vm, image_proj, anchor):
    data = json.loads(DATA_PATH.read_text())

    # 중간거리 basket 케이스 — 내비게이션 중 자연스러운 거리
    AREA_TARGET_VIZ = {"left": 0.04, "center": 0.08, "right": 0.04}
    AREA_MAX_VIZ    = {"left": 0.25, "center": 0.25, "right": 0.25}
    samples = {}
    for ep in data:
        d = ep["direction"]
        if d in samples:
            continue
        target = AREA_TARGET_VIZ[d]
        area_min = 0.005 if d != "center" else 0.03
        frames = sorted(
            [f for f in ep["frames"]
             if f["consistent"] and f["label"]
             and f.get("area_det")
             and area_min <= f["area_det"] <= AREA_MAX_VIZ[d]],
            key=lambda x: abs(x["area_det"] - target)
        )
        if not frames:
            frames = sorted(
                [f for f in ep["frames"]
                 if f["consistent"] and f["label"] and f.get("area_det")
                 and f["area_det"] >= area_min],
                key=lambda x: x["area_det"]
            )
        if not frames:
            continue
        fr = frames[0]
        try:
            with h5py.File(ep["episode"], "r") as f:
                img = Image.fromarray(f["observations"]["images"][fr["frame_idx"]]).convert("RGB")
            samples[d] = (img, fr)
        except:
            pass
        if len(samples) == 3:
            break

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.patch.set_facecolor("#0a0f1a")

    col_titles = ["Original", "Basket Masked", "Attention (orig)", "Attention (masked)"]
    for j, t in enumerate(col_titles):
        axes[0][j].set_title(t, color="white", fontsize=11, fontweight="bold", pad=8)

    cmap_jet = plt.get_cmap("jet")

    for i, d in enumerate(DIRS):
        for j in range(4):
            axes[i][j].set_facecolor("#0a0f1a")
            axes[i][j].axis("off")

        if d not in samples:
            axes[i][0].text(0.5, 0.5, f"{d}\n(sample 없음)", color="#64748b",
                           ha="center", va="center", transform=axes[i][0].transAxes)
            continue

        img, fr = samples[d]
        cx, cy, area = fr["cx_det"], fr["cy_det"], fr["area_det"]
        gt_idx = DIR_IDX[fr["label"]]

        masked_img, bbox_px = mask_basket(img, cx, cy, area)

        sims_orig,  attn_orig  = predict(vm, image_proj, processor, anchor, img,        device)
        sims_mask,  attn_mask  = predict(vm, image_proj, processor, anchor, masked_img, device)

        pred_orig = DIRS[sims_orig.argmax()]
        pred_mask = DIRS[sims_mask.argmax()]
        conf_orig = sims_orig[gt_idx]
        conf_mask = sims_mask[gt_idx]

        W, H = img.size
        img_224  = np.array(img.resize((224, 224)))
        mask_224 = np.array(masked_img.resize((224, 224)))

        def upsample_attn(a):
            return np.array(
                Image.fromarray((a / a.max() * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)
            ).astype(float) / 255.0

        attn_up_o = upsample_attn(attn_orig)
        attn_up_m = upsample_attn(attn_mask)

        # col 0: original + bbox
        axes[i][0].imshow(img_224)
        bx1, by1, bx2, by2 = bbox_px
        scale_x, scale_y = 224 / W, 224 / H
        rect = mpatches.Rectangle(
            (bx1*scale_x, by1*scale_y), (bx2-bx1)*scale_x, (by2-by1)*scale_y,
            linewidth=2, edgecolor="#22c55e", facecolor="none"
        )
        axes[i][0].add_patch(rect)
        axes[i][0].set_xlabel(
            f"gt: {d}  pred: {pred_orig}  conf: {conf_orig:.3f}",
            color="#22c55e" if pred_orig == d else "#f87171", fontsize=9
        )

        # col 1: masked
        axes[i][1].imshow(mask_224)
        flipped = pred_mask != pred_orig
        axes[i][1].set_xlabel(
            f"pred: {pred_mask}  conf: {conf_mask:.3f}" +
            ("  ← FLIP!" if flipped else ""),
            color="#f87171" if flipped else "#94a3b8", fontsize=9
        )

        # col 2: attention overlay (original)
        ov_o = img_224 / 255.0 * 0.45 + cmap_jet(attn_up_o)[:, :, :3] * 0.55
        axes[i][2].imshow(np.clip(ov_o, 0, 1))
        axes[i][2].set_xlabel(f"basket bbox↑ = basket 위치", color="#94a3b8", fontsize=9)

        # col 3: attention overlay (masked)
        ov_m = mask_224 / 255.0 * 0.45 + cmap_jet(attn_up_m)[:, :, :3] * 0.55
        axes[i][3].imshow(np.clip(ov_m, 0, 1))
        axes[i][3].set_xlabel(
            "마스킹 후 어텐션 재분산" if flipped else "어텐션 유지",
            color="#f59e0b" if flipped else "#64748b", fontsize=9
        )

        # row label
        axes[i][0].set_ylabel(f"{d.upper()}\narea={area:.3f}", color="#e2e8f0",
                               fontsize=10, fontweight="bold", rotation=0,
                               labelpad=60, va="center")

    plt.suptitle("Track 3: Basket Masking Ablation — 원본 vs 마스킹 예측 변화",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "masking_comparison.png"
    fig.savefig(str(out), dpi=110, bbox_inches="tight", facecolor="#0a0f1a")
    plt.close(fig)
    print(f"[SAVED] {out}")
    return out


# ─── Figure 2: Linear Probe Results ───────────────────────

def make_probe_figure():
    fig = plt.figure(figsize=(14, 5), facecolor="#0a0f1a")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])  # bar chart
    ax2 = fig.add_subplot(gs[1])  # confusion matrix

    # ── Bar chart
    methods = ["Random\nbaseline", "Frozen CLIP\n(zero-shot)", "Stage 1 v2\n(trained)"]
    accs    = [33.3, 96.6, 98.1]
    colors  = ["#475569", "#06b6d4", "#22c55e"]

    bars = ax1.bar(methods, accs, color=colors, width=0.5, zorder=3, edgecolor="#1e293b", linewidth=1.5)
    ax1.set_facecolor("#0a0f1a")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Accuracy (%)", color="#e2e8f0", fontsize=11)
    ax1.tick_params(colors="#94a3b8", labelsize=10)
    ax1.spines["bottom"].set_color("#334155")
    ax1.spines["left"].set_color("#334155")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.label.set_color("#e2e8f0")
    ax1.set_title("Basket 위치 분류 정확도", color="white", fontsize=12, fontweight="bold", pad=12)

    for bar, acc, c in zip(bars, accs, colors):
        ax1.text(bar.get_x() + bar.get_width()/2, acc + 1.5,
                 f"{acc:.1f}%", ha="center", va="bottom", color=c,
                 fontsize=12, fontweight="bold")

    ax1.axhline(33.3, color="#475569", linestyle="--", linewidth=1, alpha=0.6, zorder=2)
    ax1.text(2.4, 35, "random", color="#475569", fontsize=8)

    # ── Confusion matrix (zero-shot probe)
    cm = np.array([
        [291, 27, 1],
        [35, 740, 0],
        [0, 0, 750],
    ])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cmap_custom = LinearSegmentedColormap.from_list("dark_green",
        ["#0a0f1a", "#052e16", "#166534", "#22c55e"], N=256)
    im = ax2.imshow(cm_norm, cmap=cmap_custom, vmin=0, vmax=1, aspect="auto")

    for i in range(3):
        for j in range(3):
            val = cm_norm[i][j]
            n   = cm[i][j]
            color = "white" if val > 0.5 else "#94a3b8"
            ax2.text(j, i, f"{val:.1%}\n({n})", ha="center", va="center",
                     color=color, fontsize=11, fontweight="bold" if i==j else "normal")

    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(DIRS, color="#94a3b8", fontsize=11)
    ax2.set_yticklabels(DIRS, color="#94a3b8", fontsize=11)
    ax2.set_xlabel("예측", color="#e2e8f0", fontsize=11)
    ax2.set_ylabel("실제", color="#e2e8f0", fontsize=11)
    ax2.set_title("Frozen CLIP 혼동 행렬 (Zero-shot, 96.6%)",
                  color="white", fontsize=12, fontweight="bold", pad=12)
    ax2.set_facecolor("#0a0f1a")
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(colors="#94a3b8")

    plt.suptitle("Track 2: Zero-shot Linear Probe — 학습 없이 frozen CLIP만으로",
                 color="white", fontsize=13, fontweight="bold", y=1.03)

    out = OUT_DIR / "linear_probe_results.png"
    fig.savefig(str(out), dpi=110, bbox_inches="tight", facecolor="#0a0f1a")
    plt.close(fig)
    print(f"[SAVED] {out}")
    return out


# ─── Figure 3: 5-Track Summary Infographic ────────────────

def make_summary_figure():
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0a0f1a")
    ax.set_facecolor("#0a0f1a")
    ax.axis("off")

    tracks = [
        {"label": "Track 1\nKosmos-2 Caption",
         "value": '"trash can"\n"air conditioner"',
         "sub": "basket 객체 인식\n(vocabulary 불일치)",
         "color": "#06b6d4", "icon": "🔍", "strength": 0.4},
        {"label": "Track 2\nZero-shot Probe",
         "value": "96.6%",
         "sub": "frozen CLIP 학습 없이\n이미 위치 인코딩",
         "color": "#22c55e", "icon": "🧪", "strength": 1.0},
        {"label": "Track 3\nMasking Ablation",
         "value": "6/6 flip",
         "sub": "center 대형 basket\n마스킹 → 100% 반전",
         "color": "#f59e0b", "icon": "🎭", "strength": 0.9},
        {"label": "Exp A\nEarly→Late 격차",
         "value": "+8%p (left)\n+5%p (center)",
         "sub": "basket 가까울수록\n정확도 상승",
         "color": "#a78bfa", "icon": "📈", "strength": 0.75},
        {"label": "Exp B\nAttention Map",
         "value": "4.4×",
         "sub": "center late 기준\nbasket 영역 집중",
         "color": "#f472b6", "icon": "👁", "strength": 0.85},
    ]

    n = len(tracks)
    xs = np.linspace(0.1, 0.9, n)
    y_card = 0.72
    card_w, card_h = 0.14, 0.48

    for i, t in enumerate(tracks):
        x = xs[i]
        # card
        rect = mpatches.FancyBboxPatch(
            (x - card_w/2, y_card - card_h), card_w, card_h,
            boxstyle="round,pad=0.01", linewidth=2,
            edgecolor=t["color"], facecolor="#0d1b2a",
            transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(rect)

        ax.text(x, y_card - 0.03, t["icon"], ha="center", va="top",
                fontsize=20, transform=ax.transAxes)
        ax.text(x, y_card - 0.13, t["label"], ha="center", va="top",
                fontsize=8.5, color="#94a3b8", transform=ax.transAxes,
                fontweight="bold", linespacing=1.4)
        ax.text(x, y_card - 0.26, t["value"], ha="center", va="top",
                fontsize=13, color=t["color"], transform=ax.transAxes,
                fontweight="900", linespacing=1.3)
        ax.text(x, y_card - 0.42, t["sub"], ha="center", va="top",
                fontsize=8, color="#64748b", transform=ax.transAxes,
                linespacing=1.4)

        # strength bar
        bar_y = 0.12
        bar_h_max = 0.16
        bh = t["strength"] * bar_h_max
        bar_rect = mpatches.FancyBboxPatch(
            (x - 0.04, bar_y), 0.08, bh,
            boxstyle="round,pad=0.005", linewidth=0,
            facecolor=t["color"], alpha=0.85,
            transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(bar_rect)
        ax.text(x, bar_y + bh + 0.01, f"{int(t['strength']*100)}%",
                ha="center", va="bottom", fontsize=9, color=t["color"],
                fontweight="bold", transform=ax.transAxes)

    # bottom label
    ax.text(0.5, 0.03, "증거 강도", ha="center", va="bottom",
            fontsize=9, color="#475569", transform=ax.transAxes)

    # arrow → conclusion
    ax.annotate("", xy=(0.5, -0.08), xytext=(0.5, 0.0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#334155", lw=2))

    ax.text(0.5, -0.13,
            '"Frozen CLIP이 이미 96.6% — basket을 처음부터 보고 있었다. Stage 1은 이것을 텍스트와 연결한다."',
            ha="center", va="top", fontsize=11, color="#e2e8f0",
            transform=ax.transAxes, style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a",
                      edgecolor="#334155", linewidth=1.5))

    ax.set_title("\"박스를 본 건가, 복도를 외운 건가?\" — 5-Track 검증 결과",
                 color="white", fontsize=14, fontweight="bold", pad=20)

    out = OUT_DIR / "track_summary.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight", facecolor="#0a0f1a")
    plt.close(fig)
    print(f"[SAVED] {out}")
    return out


def main():
    print("[VIZ] 시각화 이미지 생성 시작\n")

    # Figure 3: summary (모델 불필요 — 먼저)
    print("[1/3] Track Summary 인포그래픽...")
    make_summary_figure()

    # Figure 2: linear probe
    print("[2/3] Linear Probe 결과...")
    make_probe_figure()

    # Figure 1: masking (모델 필요)
    print("[3/3] Masking Comparison (모델 로드 중)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, vm, image_proj, anchor, val_acc = load_model(device)
    make_masking_figure(device, processor, vm, image_proj, anchor)

    print(f"\n[DONE] 이미지 저장 위치: {OUT_DIR}/")
    print("  - track_summary.png")
    print("  - linear_probe_results.png")
    print("  - masking_comparison.png")


if __name__ == "__main__":
    main()
