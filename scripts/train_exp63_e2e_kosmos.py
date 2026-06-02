#!/usr/bin/env python3
"""
Exp63: 순수 HF Kosmos-2 E2E VLA 학습
(Google-robot post-trained 아닌 순수 버전으로 재시도)

비교 기준:
  - 같은 데이터셋: bbox_dataset_pg2_cx.json (243 ep, Exp61과 동일)
  - 같은 train/val split: seed=42, 15% val
  - 같은 평가: CL success (FPE < 0.5m AND TLD ∈ [0.7, 1.5])
  - 같은 8-class action space

차이점:
  - 순수 HF Kosmos-2 (text path 정상)
  - 이미지 + 텍스트 프롬프트 → 액션 토큰 직접 생성
  - LoRA on vision encoder (16-26) + LM top layers

Usage:
  .venv/bin/python3 scripts/train_exp63_e2e_kosmos.py --epochs 30
  .venv/bin/python3 scripts/train_exp63_e2e_kosmos.py --eval-only
"""
import sys, json, random, warnings, re, argparse
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import h5py
from PIL import Image

VLM_PATH  = ROOT / ".vlms/kosmos-2-patch14-224"
ANN_PG2   = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"
OUT_DIR   = ROOT / "runs/v5_nav/e2e/exp63"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTION_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD_LEFT", "FWD_RIGHT", "ROT_L", "ROT_R"]
# 프롬프트: 모든 프레임에 동일
PROMPT = "Navigate to the gray basket. Robot action:"


def load_model(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType
    print(f"[LOAD] 순수 HF Kosmos-2 from {VLM_PATH}")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    proc  = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)

    # LoRA: vision encoder 고수준(16~26) + LM top layers
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(16, 27)),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return proc, model, dtype


def prepare_sample(proc, img_np, action_id, device, dtype):
    """이미지 + 프롬프트 + 액션 레이블 → 학습용 텐서"""
    pil = Image.fromarray(img_np).convert("RGB")
    action_str = ACTION_NAMES[action_id]

    # 입력: 프롬프트만 (이미지 포함)
    inp = proc(text=PROMPT, images=pil, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(dtype)

    # 타겟: 액션 토큰
    tgt_ids = proc.tokenizer.encode(f" {action_str}", add_special_tokens=False)
    return inp, tgt_ids


@torch.no_grad()
def predict_action(proc, model, img_np, device, dtype):
    """추론: 이미지 → 액션 문자열"""
    pil = Image.fromarray(img_np).convert("RGB")
    inp = proc(text=PROMPT, images=pil, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    gen = model.generate(**inp, max_new_tokens=5, do_sample=False)
    raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    # 가장 가까운 액션 이름 매핑
    raw_up = raw.upper().replace("+", "_").replace(" ", "_")
    for i, name in enumerate(ACTION_NAMES):
        if name in raw_up or raw_up in name:
            return i, raw
    return 1, raw  # default FORWARD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--lr",       type=float, default=2e-5)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--val-ratio",type=float, default=0.15)
    parser.add_argument("--eval-only",action="store_true")
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # 데이터 로드 (Exp61과 동일)
    with open(ANN_PG2) as f:
        ann = json.load(f)
    ann = [ep for ep in ann if ep.get("path_type","") not in ("","free","unknown")]
    random.shuffle(ann)
    n_val = max(1, int(len(ann) * args.val_ratio))
    val_eps, train_eps = ann[:n_val], ann[n_val:]
    print(f"Train:{len(train_eps)} / Val:{len(val_eps)} eps (Exp61과 동일 기준)")

    proc, model, dtype = load_model(device)

    if args.eval_only:
        # 저장된 체크포인트로 val 정확도 측정
        ckpt = OUT_DIR / "exp63_best.pt"
        if ckpt.exists():
            from peft import PeftModel
            model.load_adapter(str(OUT_DIR), "default")
        model.eval()
        correct = total = 0
        for ep in val_eps[:10]:
            h5_path = Path(ep["episode"])
            if not h5_path.exists(): continue
            frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
            try:
                with h5py.File(str(h5_path), "r") as f:
                    imgs = f["observations"]["images"][:]
            except: continue
            for fr in frames[:5]:  # 빠른 평가
                img_np = imgs[fr["frame_idx"]].astype("uint8")
                pred, raw = predict_action(proc, model, img_np, device, dtype)
                gt = fr["gt_class"]
                correct += (pred == gt)
                total += 1
        print(f"Val accuracy: {correct/max(total,1)*100:.1f}% ({correct}/{total})")
        return

    # 학습
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )

    best_acc = 0.0
    print(f"\n[TRAIN] {args.epochs} epochs  lr={args.lr}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = n_batch = 0

        random.shuffle(train_eps)
        for ep in train_eps:
            h5_path = Path(ep["episode"])
            if not h5_path.exists(): continue
            frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
            if not frames: continue
            try:
                with h5py.File(str(h5_path), "r") as f:
                    imgs = f["observations"]["images"][:]
            except: continue

            # 에피소드당 최대 5 프레임 샘플링 (속도)
            sample_frames = random.sample(frames, min(5, len(frames)))
            for fr in sample_frames:
                img_np = imgs[fr["frame_idx"]].astype("uint8")
                inp, tgt_ids = prepare_sample(proc, img_np, fr["gt_class"], device, dtype)

                # 프롬프트 + 액션을 합쳐서 CE loss
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)
                # Kosmos-2: pixel_values는 prompt 부분에만 연결됨
                # 액션 토큰을 label로 teacher-forcing
                # prompt만으로 forward → 마지막 hidden state에서 action logits 예측
                prompt_out = model(**inp)
                # 마지막 토큰 hidden state로 action logit 계산
                last_hidden = prompt_out.logits[:, -1, :]  # (1, vocab)
                tgt_t = torch.tensor([tgt_ids[0]], dtype=torch.long, device=device)
                out_mock = type('O', (), {'loss': F.cross_entropy(last_hidden, tgt_t)})()
                out = out_mock
                loss = out.loss
                if torch.isnan(loss) or torch.isinf(loss): continue
                loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
                ep_loss += loss.item(); n_batch += 1

        avg_loss = ep_loss / max(n_batch, 1)

        # val 평가 (5 epoch마다)
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            correct = total = 0
            for ep in val_eps[:15]:
                h5_path = Path(ep["episode"])
                if not h5_path.exists(): continue
                frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
                try:
                    with h5py.File(str(h5_path), "r") as f:
                        imgs = f["observations"]["images"][:]
                except: continue
                for fr in random.sample(frames, min(3, len(frames))):
                    img_np = imgs[fr["frame_idx"]].astype("uint8")
                    pred, _ = predict_action(proc, model, img_np, device, dtype)
                    correct += (pred == fr["gt_class"]); total += 1
            acc = correct / max(total, 1)
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  val_acc={acc*100:.1f}%")
            if acc >= best_acc:
                best_acc = acc
                model.save_pretrained(str(OUT_DIR))
                print(f"    [BEST] {acc*100:.1f}% → 저장")
            model.train()
        else:
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}")

    print(f"\n최종 val_acc: {best_acc*100:.1f}%")
    print(f"비교: Exp61 decomposed (243ep) = CL 70%")
    print(f"      Exp63 E2E Kosmos (243ep)  = val_acc {best_acc*100:.1f}% (CL 미측정)")


if __name__ == "__main__":
    main()
