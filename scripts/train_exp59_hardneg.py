#!/usr/bin/env python3
"""
Exp59: PaliGemma2 Gray Basket 단일 타겟 + Hard Negative LoRA

목적: "detect gray basket"만 정확하게 → 다른 쿼리는 <eos>
V4(brown pot) 데이터 불필요 — 목표는 gray basket 특정이지 brown pot 학습이 아님

학습 전략:
  Positive : V5 → "detect gray basket" → bbox        (basket 있음)
  Negative1: V5 → "detect brown pot"   → <eos>      (같은 이미지, basket은 pot이 아님)
  Negative2: V5 → "detect red ball"    → <eos>      (완전히 다른 물체)
  Negative3: V5 → "detect person"      → <eos>      (완전히 다른 물체)

구조 변경 (vs Exp58):
  레이어: 전체 53층 → 고수준 17층 (18~26)
  modules: q+v → q+k+v
  r: 8 → 16 / alpha: 16 → 32
  데이터: V4 전혀 사용 안 함 / V5 단일 / neg 3 phrases

성공 기준:
  "detect gray basket" on V5: >95% hit rate
  "detect brown pot"   on V5: <10% FP rate   (같은 이미지, 없는 물체)
  분리도 gap: >80%p

Usage:
    python3 scripts/train_exp59_hardneg.py --skip-annotate --epochs 20
    python3 scripts/train_exp59_hardneg.py --eval-only
"""
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
from PIL import Image

PALIGEMMA2_PATH = Path.home() / ".cache/huggingface/hub" \
    / "models--google--paligemma2-3b-mix-224" \
    / "snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"

FRAME_LEVEL_JSON = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_frame_level.json"
V4_DIR           = ROOT / "ROS_action/basket_dataset_v2"
V4_ANNOTATION    = ROOT / "docs/v5/bbox_frame_level/v4_brownpot_pseudolabels.json"
OUT_DIR          = ROOT / "runs/v5_nav/grounding/exp59"

_V5_CANDIDATES = [
    Path("/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5"),
    ROOT / "ROS_action/mobile_vla_dataset_v5",
]

LOC_RE = re.compile(r"<loc(\d{4})>")


def resolve_v5_dir():
    for c in _V5_CANDIDATES:
        if c.exists() and any(c.glob("episode_*.h5")):
            return c
    return _V5_CANDIDATES[-1]


# ─── 이미지 로딩 ─────────────────────────────────────────────────────────────

def load_v5_frame(episode: str, frame_idx: int, data_dir: Path) -> np.ndarray | None:
    ep = Path(episode)
    path = ep if ep.exists() else next(iter(data_dir.glob(f"{ep.stem}.h5")), None)
    if path is None:
        return None
    try:
        with h5py.File(path, "r") as f:
            if "observations" in f and "images" in f["observations"]:
                return f["observations"]["images"][frame_idx].astype(np.uint8)
            return f["images"][frame_idx].astype(np.uint8)
    except Exception:
        return None


def load_v4_frames(h5_path: Path, n: int) -> list[np.ndarray]:
    try:
        with h5py.File(h5_path, "r") as f:
            imgs = f["images"][:]
        total = len(imgs)
        if total == 0:
            return []
        idxs = [round(i * (total - 1) / max(n - 1, 1)) for i in range(min(n, total))]
        return [imgs[i].astype(np.uint8) for i in idxs]
    except Exception:
        return []


# ─── bbox 변환 ────────────────────────────────────────────────────────────────

def cx_cy_area_to_loc(cx: float, cy: float, area: float, label: str) -> str:
    side = math.sqrt(max(area, 1e-4))
    x1 = max(0.0, cx - side / 2)
    y1 = max(0.0, cy - side / 2)
    x2 = min(1.0, cx + side / 2)
    y2 = min(1.0, cy + side / 2)
    loc_y1, loc_x1 = int(y1 * 1023), int(x1 * 1023)
    loc_y2, loc_x2 = int(y2 * 1023), int(x2 * 1023)
    return f"<loc{loc_y1:04d}><loc{loc_x1:04d}><loc{loc_y2:04d}><loc{loc_x2:04d}> {label}"


def raw_output_to_cx(output: str) -> float | None:
    vals = [int(v) / 1023.0 for v in LOC_RE.findall(output)]
    if len(vals) >= 4:
        return (vals[1] + vals[3]) / 2  # (x1 + x2) / 2
    return None


# ─── V4 자동 주석 ─────────────────────────────────────────────────────────────

@torch.no_grad()
def annotate_v4(model, processor, device, frames_per_ep: int = 6) -> list[dict]:
    """V4 brown pot 에피소드에서 PaliGemma2 zero-shot pseudo-label 생성"""
    h5_files = sorted(V4_DIR.glob("episode_*.h5"))
    print(f"\n[V4 자동 주석] {len(h5_files)} 에피소드 처리 중...")
    prompt = "<image> detect brown pot"
    pv_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    annotations = []
    for ep_idx, h5_path in enumerate(h5_files):
        frames = load_v4_frames(h5_path, frames_per_ep)
        ep_hits = []
        for fi, frame in enumerate(frames):
            pil = Image.fromarray(frame).convert("RGB")
            inp = processor(text=prompt, images=pil, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            inp["pixel_values"] = inp["pixel_values"].to(pv_dtype)
            gen = model.generate(**inp, max_new_tokens=30, do_sample=False)
            prefix_len = inp["input_ids"].shape[1]
            raw = processor.decode(gen[0][prefix_len:], skip_special_tokens=False)

            vals = [int(v) / 1023.0 for v in LOC_RE.findall(raw)]
            if len(vals) >= 4:
                y1, x1, y2, x2 = vals[:4]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                ep_hits.append({
                    "frame_idx": fi,
                    "cx": cx, "cy": cy, "area": area,
                    "raw_output": raw,
                })

        if ep_hits:
            annotations.append({
                "episode": str(h5_path),
                "frames": ep_hits,
            })

        if (ep_idx + 1) % 10 == 0:
            print(f"  [{ep_idx+1}/{len(h5_files)}] hit_eps={len(annotations)}")

    hit_rate = len(annotations) / max(len(h5_files), 1)
    print(f"  V4 주석 완료: {len(annotations)}/{len(h5_files)} eps hit ({hit_rate*100:.1f}%)")
    return annotations


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_v5_samples(data_dir: Path, frames_per_ep: int, augment: bool) -> list[dict]:
    with open(FRAME_LEVEL_JSON) as f:
        raw = json.load(f)

    samples = []
    skipped = 0
    for ep_data in raw:
        det = [fr for fr in ep_data["frames"]
               if fr.get("detected") and fr.get("cx_det") is not None]
        if not det:
            skipped += 1
            continue
        chosen = det if len(det) <= frames_per_ep else [
            det[round(i * (len(det) - 1) / (frames_per_ep - 1))]
            for i in range(frames_per_ep)
        ]
        chosen = list({fr["frame_idx"]: fr for fr in chosen}.values())

        for fr in chosen:
            img = load_v5_frame(ep_data["episode"], fr["frame_idx"], data_dir)
            if img is None:
                skipped += 1
                continue
            cx, cy, area = fr["cx_det"], fr["cy_det"], fr["area_det"]
            samples.append({
                "image": img,
                "target": cx_cy_area_to_loc(cx, cy, area, "gray basket"),
                "prompt": "<image> detect gray basket",
                "cx": cx, "label": "gray basket",
                "episode": ep_data["episode"],
            })
            if augment:
                samples.append({
                    "image": np.fliplr(img).copy(),
                    "target": cx_cy_area_to_loc(1.0 - cx, cy, area, "gray basket"),
                    "prompt": "<image> detect gray basket",
                    "cx": 1.0 - cx, "label": "gray basket",
                    "episode": ep_data["episode"],
                })
    print(f"  [V5] {len(samples)} 샘플 로드 (skipped {skipped})")
    return samples


def load_v4_samples(annotations: list[dict], frames_per_ep: int, augment: bool) -> list[dict]:
    samples = []
    for ep_data in annotations:
        h5_path = Path(ep_data["episode"])
        frames_meta = ep_data["frames"]
        chosen = frames_meta[:frames_per_ep]
        frames = load_v4_frames(h5_path, max(f["frame_idx"] for f in chosen) + 1)

        for meta in chosen:
            fi = meta["frame_idx"]
            if fi >= len(frames):
                continue
            img = frames[fi]
            cx, cy, area = meta["cx"], meta["cy"], meta["area"]
            samples.append({
                "image": img,
                "target": cx_cy_area_to_loc(cx, cy, area, "brown pot"),
                "prompt": "<image> detect brown pot",
                "cx": cx, "label": "brown pot",
                "episode": str(h5_path),
            })
            if augment:
                samples.append({
                    "image": np.fliplr(img).copy(),
                    "target": cx_cy_area_to_loc(1.0 - cx, cy, area, "brown pot"),
                    "prompt": "<image> detect brown pot",
                    "cx": 1.0 - cx, "label": "brown pot",
                    "episode": str(h5_path),
                })
    print(f"  [V4] {len(samples)} 샘플 로드")
    return samples


# ─── Hard Negative 로더 ──────────────────────────────────────────────────────

def load_hard_negatives(
    v5_samples: list[dict],
    neg_ratio: float = 1.0,
    augment: bool = True,
) -> list[dict]:
    """
    Hard Negative 샘플 생성 — V5 이미지만 사용, 3가지 없는 물체 쿼리.

    같은 이미지에 gray basket이 있지만 다른 물체를 물어보면 <eos>를 출력해야 함.
    → "detect gray basket"과 다른 텍스트를 구별하는 능력 학습

    neg_phrases: basket과 혼동 가능한 것(brown pot) + 완전히 다른 것(red ball, person)
    neg_ratio: positive 대비 negative 비율 per phrase (0.3 = 각 phrase당 30%)
    """
    NEG_PHRASES = ["brown pot", "red ball", "person"]

    # positive 중 augmented 아닌 원본만 사용 (augmented는 flip이라 이미지 중복)
    v5_base = [s for s in v5_samples if "fliplr" not in s.get("episode", "")]
    # 실제로 augmented 구분은 episode 같고 cx가 반전된 것 → 그냥 전체 사용
    v5_base = v5_samples  # augmented 포함해서 이미지 다양성 확보

    n_per_phrase = int(len(v5_base) * neg_ratio)
    random.shuffle(v5_base)

    negs = []
    for phrase in NEG_PHRASES:
        chosen = v5_base[:n_per_phrase]
        for s in chosen:
            negs.append({
                "image": s["image"],
                "target": "<eos>",
                "prompt": f"<image> detect {phrase}",
                "cx": -1.0,
                "label": f"neg_{phrase.replace(' ', '_')}",
                "episode": s["episode"],
            })
            if augment:
                negs.append({
                    "image": np.fliplr(s["image"]).copy(),
                    "target": "<eos>",
                    "prompt": f"<image> detect {phrase}",
                    "cx": -1.0,
                    "label": f"neg_{phrase.replace(' ', '_')}",
                    "episode": s["episode"],
                })

    by_phrase = {p: sum(1 for n in negs if n["label"] == f"neg_{p.replace(' ','_')}") for p in NEG_PHRASES}
    for p, cnt in by_phrase.items():
        print(f"  [NEG] V5→'{p}': {cnt}개")
    print(f"  [NEG] 총 {len(negs)}개")
    return negs


# ─── 평가 (분리도 포함) ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, processor, samples: list[dict], device, max_eval: int = 80) -> dict:
    """
    분리도(separation) 포함 평가:
      positive: gray basket / brown pot hit rate
      negative: neg_brown_pot_on_v5 / neg_gray_basket_on_v4 → FP rate (낮을수록 좋음)
      separation: TP rate - FP rate (높을수록 좋음)
    """
    model.eval()
    pv_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # positive / negative 분리 샘플링
    pos = [s for s in samples if not s["label"].startswith("neg_")]
    neg = [s for s in samples if s["label"].startswith("neg_")]
    pos_sub = random.sample(pos, min(max_eval // 2, len(pos)))
    neg_sub = random.sample(neg, min(max_eval // 2, len(neg)))
    subset = pos_sub + neg_sub

    by_label: dict[str, dict] = {}
    for s in subset:
        lbl = s["label"]
        if lbl not in by_label:
            by_label[lbl] = {"hits": 0, "cx_errs": [], "n": 0}

        pil = Image.fromarray(s["image"]).convert("RGB")
        inp = processor(text=s["prompt"], images=pil, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        inp["pixel_values"] = inp["pixel_values"].to(pv_dtype)
        gen = model.generate(**inp, max_new_tokens=30, do_sample=False)
        prefix_len = inp["input_ids"].shape[1]
        raw = processor.decode(gen[0][prefix_len:], skip_special_tokens=False)

        vals = [int(v) / 1023.0 for v in LOC_RE.findall(raw)]
        by_label[lbl]["n"] += 1
        if len(vals) >= 4:
            by_label[lbl]["hits"] += 1
            if s["cx"] >= 0:
                pred_cx = (vals[1] + vals[3]) / 2
                by_label[lbl]["cx_errs"].append(abs(pred_cx - s["cx"]))

    result = {}
    for lbl, d in by_label.items():
        n = max(d["n"], 1)
        result[lbl] = {
            "hit_rate": d["hits"] / n,
            "cx_err": sum(d["cx_errs"]) / max(len(d["cx_errs"]), 1),
            "n": d["n"],
        }

    # TP: gray basket (positive class)
    # FP labels: neg_* (negative — 검출 안 해야 하는 것들)
    tp_labels = ["gray basket"]
    fp_labels = [l for l in by_label if l.startswith("neg_")]

    tp_n    = sum(result[l]["n"] for l in tp_labels if l in result)
    fp_n    = sum(result[l]["n"] for l in fp_labels if l in result)

    tp_rate = (sum(result[l]["hit_rate"] * result[l]["n"]
                   for l in tp_labels if l in result) / max(tp_n, 1))
    fp_rate = (sum(result[l]["hit_rate"] * result[l]["n"]
                   for l in fp_labels if l in result) / max(fp_n, 1))

    result["overall"]    = {"hit_rate": tp_rate, "n": tp_n}
    result["separation"] = {"tp_rate": tp_rate, "fp_rate": fp_rate,
                             "gap": tp_rate - fp_rate, "n_tp": tp_n, "n_fp": fp_n}
    model.train()
    return result


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-per-ep",  type=int,   default=5)
    parser.add_argument("--augment",        action="store_true", default=True)
    parser.add_argument("--no-augment",     dest="augment", action="store_false")
    parser.add_argument("--epochs",         type=int,   default=25)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--batch",          type=int,   default=4)
    parser.add_argument("--lora-r",         type=int,   default=16)   # 고수준 레이어 집중 → rank 상향
    parser.add_argument("--lora-alpha",     type=int,   default=32)
    parser.add_argument("--val-ratio",      type=float, default=0.15)
    parser.add_argument("--annotate-only",  action="store_true",
                        help="V4 auto-annotation만 실행 후 종료")
    parser.add_argument("--skip-annotate",  action="store_true",
                        help="기존 annotation JSON 재사용")
    parser.add_argument("--eval-only",      action="store_true")
    parser.add_argument("--neg-ratio",      type=float, default=1.0,
                        help="hard negative 비율 (positive 대비, default 1.0)")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(42)
    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Exp59: PaliGemma2 Hard Negative Goal-Conditioned Grounding")
    print(f"  backbone : paligemma2-3b-mix-224")
    print(f"  classes  : gray basket (V5 only) — 단일 타겟")
    print(f"  negatives: V5→brown_pot + red_ball + person → <eos>")
    print(f"  neg_ratio: {args.neg_ratio}")
    print(f"  epochs   : {args.epochs}  lr={args.lr}  batch={args.batch}")
    print("=" * 60)

    # ── 모델 로드 ──────────────────────────────────────────────────────────
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    print(f"\n[LOAD] {PALIGEMMA2_PATH}")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = PaliGemmaProcessor.from_pretrained(str(PALIGEMMA2_PATH))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PALIGEMMA2_PATH), torch_dtype=dtype
    ).to(device)

    # ── 데이터 로드 — V5만 (gray basket 단일 타겟) ──────────────────────────
    v5_dir = resolve_v5_dir()
    v5_samples = load_v5_samples(v5_dir, args.frames_per_ep, args.augment)

    # ── Hard Negative — V5 이미지에 없는 물체 쿼리 3종 ──────────────────────
    hard_negs = load_hard_negatives(v5_samples,
                                    neg_ratio=args.neg_ratio,
                                    augment=args.augment)

    all_samples = v5_samples + hard_negs
    print(f"  총 샘플: {len(all_samples)}")
    print(f"    V5 pos={len(v5_samples)}, NEG={len(hard_negs)} (brown pot / red ball / person)")

    # 에피소드 단위 train/val 분할
    eps = list({s["episode"] for s in all_samples})
    random.shuffle(eps)
    n_val = max(1, int(len(eps) * args.val_ratio))
    val_eps = set(eps[:n_val])
    train_raw = [s for s in all_samples if s["episode"] not in val_eps]
    val_raw   = [s for s in all_samples if s["episode"] in val_eps]
    print(f"  Train={len(train_raw)}  Val={len(val_raw)}")

    # ── eval-only 모드 ─────────────────────────────────────────────────────
    if args.eval_only:
        if not (OUT_DIR / "adapter_config.json").exists():
            print("ERROR: No adapter at", OUT_DIR)
            sys.exit(1)
        model = PeftModel.from_pretrained(model, str(OUT_DIR))
        model.eval()
        r = evaluate(model, processor, val_raw, device)
        for lbl, d in r.items():
            print(f"  {lbl}: hit={d['hit_rate']*100:.1f}%  n={d['n']}")
        return

    # ── LoRA 적용 — 고수준 레이어(18~26)만 타겟 ────────────────────────────
    # Vision 18~26 (시멘틱 객체 인식) + LM 18~25 (언어-시각 바인딩, loc 토큰 생성)
    # 전체 53레이어 → 17레이어 / r 8→16 (표현력 보강) / k_proj 추가 (어텐션 완전 제어)
    high_layers = list(range(18, 27))  # 18,19,...,26 → vision 18~26 + LM 18~25
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj"],
        layers_to_transform=high_layers,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    for p in model.base_model.model.vision_tower.parameters():
        p.requires_grad = False
    for p in model.base_model.model.multi_modal_projector.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    # ── 학습 샘플 텐서 준비 ────────────────────────────────────────────────
    print("\n학습 샘플 준비 중...")
    pv_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    prepared = []
    for i, s in enumerate(train_raw):
        pil = Image.fromarray(s["image"]).convert("RGB")
        enc = processor(
            text=s["prompt"],
            images=pil,
            suffix=s["target"],
            return_tensors="pt",
            padding="max_length",
            max_length=300,
            truncation=True,
        )
        prepared.append({
            "input_ids":      enc["input_ids"][0].to(device),
            "attention_mask": enc["attention_mask"][0].to(device),
            "pixel_values":   enc["pixel_values"][0].to(pv_dtype).to(device),
            "labels":         enc["labels"][0].to(device),
            "token_type_ids": enc.get("token_type_ids",
                                      torch.zeros_like(enc["input_ids"]))[0].to(device),
        })
        if (i + 1) % 50 == 0 or (i + 1) == len(train_raw):
            print(f"  {i+1}/{len(train_raw)}", end="\r", flush=True)
    print(f"\n  {len(prepared)} 샘플 준비 완료")

    # ── 학습 루프 ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps  = args.epochs * math.ceil(len(prepared) / args.batch)
    warmup_steps = max(1, int(total_steps * 0.1))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n학습 시작: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    model.train()
    best_hit = 0.0

    for epoch in range(1, args.epochs + 1):
        random.shuffle(prepared)
        ep_loss, n_batch = 0.0, 0

        for i in range(0, len(prepared), args.batch):
            batch = prepared[i: i + args.batch]
            max_len = max(b["input_ids"].shape[0] for b in batch)

            input_ids  = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            attn_mask  = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            labels     = torch.full((len(batch), max_len), -100, dtype=torch.long, device=device)
            tok_types  = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            pv         = torch.stack([b["pixel_values"] for b in batch])

            for j, b in enumerate(batch):
                L = b["input_ids"].shape[0]
                input_ids[j, :L] = b["input_ids"]
                attn_mask[j, :L] = b["attention_mask"]
                labels[j, :L]    = b["labels"]
                tok_types[j, :L] = b["token_type_ids"]

            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                pixel_values=pv,
                token_type_ids=tok_types,
                labels=labels,
            )
            loss = out.loss
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            ep_loss += loss.item()
            n_batch += 1

        avg_loss = ep_loss / max(n_batch, 1)
        lr_now   = scheduler.get_last_lr()[0]

        # 5 epoch마다 평가
        if epoch % 2 == 0 or epoch == args.epochs:
            r = evaluate(model, processor, val_raw, device, max_eval=80)
            overall_hit = r["overall"]["hit_rate"]
            sep = r.get("separation", {})
            tp  = sep.get("tp_rate", 0) * 100
            fp  = sep.get("fp_rate", 0) * 100
            gap = sep.get("gap", 0) * 100
            pos_parts = "  ".join(
                f"{lbl}={d['hit_rate']*100:.0f}%"
                for lbl, d in r.items()
                if lbl not in ("overall", "separation") and not lbl.startswith("neg_")
            )
            neg_parts = "  ".join(
                f"{lbl.replace('neg_','')}={d['hit_rate']*100:.0f}%FP"
                for lbl, d in r.items()
                if lbl.startswith("neg_")
            )
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}"
                  f"  TP={tp:.0f}% FP={fp:.0f}% gap={gap:+.0f}%p  lr={lr_now:.2e}")
            print(f"    pos: [{pos_parts}]")
            if neg_parts:
                print(f"    neg: [{neg_parts}]")
            # best 기준: separation gap (TP-FP)
            if gap >= best_hit:
                best_hit = gap
                model.save_pretrained(str(OUT_DIR))
                print(f"    [BEST] gap={best_hit:+.0f}%p → saved")
        else:
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  lr={lr_now:.2e}")

    # ── 최종 평가 ─────────────────────────────────────────────────────────
    model = PeftModel.from_pretrained(
        PaliGemmaForConditionalGeneration.from_pretrained(
            str(PALIGEMMA2_PATH), torch_dtype=dtype
        ).to(device),
        str(OUT_DIR),
    )
    model.eval()
    r = evaluate(model, processor, val_raw, device, max_eval=len(val_raw))
    print("\n최종 평가 (전체 val set):")
    for lbl, d in r.items():
        if isinstance(d, dict) and "hit_rate" in d:
            print(f"  {lbl}: hit={d['hit_rate']*100:.1f}%  n={d['n']}")
        elif lbl == "separation":
            print(f"  separation: TP={d.get('tp_rate',0)*100:.0f}%  FP={d.get('fp_rate',0)*100:.0f}%  gap={d.get('gap',0)*100:+.0f}%p")

    result = {
        "exp": "exp59",
        "backbone": "paligemma2-3b-mix-224",
        "classes": ["gray basket"],
        "train_n": len(train_raw),
        "val_n": len(val_raw),
        "best_overall_hit": best_hit,
        "final": r,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Exp58 완료  best_hit={best_hit*100:.1f}%")
    print(f"adapter → {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
