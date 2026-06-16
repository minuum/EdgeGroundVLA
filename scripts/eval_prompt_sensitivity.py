#!/usr/bin/env python3
"""
Prompt sensitivity 평가 — 동일 이미지에 left/right/forward 프롬프트를 주고
액션 변화가 발생하는지 측정.

Phase A 검증: text 무감각 (action diff < 1e-3) → text 잠금 (diff >= 1e-2) 전환 확인용.

사용:
    python3 scripts/eval_prompt_sensitivity.py \
        --ckpt runs/v5_nav/kosmos/mobile_vla_v5_exp41c/.../epoch_*.ckpt \
        --config configs/mobile_vla_v5_exp41c_scratch_pta.json \
        --n-frames 30 \
        --output_json docs/v5/exp41_prompt_lockin/exp41c_sensitivity.json
"""
import os, sys, json, argparse, gc
from pathlib import Path

import lightning.fabric.plugins.environments.mpi as _mpi_env_mod
_mpi_env_mod._MPI4PY_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "RoboVLMs"))

import numpy as np
import torch

import robovlms.model.backbone as backbone_mod
import robovlms.model.policy_head as policy_head_mod
import robovlms.train.base_trainer as base_trainer_mod
import robovlms.train as train_mod

from robovlms.model.backbone.robokosmos import RoboKosMos
from robovlm_nav.models.nav_robokosmos import NavRoboKosMos
setattr(backbone_mod, "RoboKosMos", NavRoboKosMos)
setattr(backbone_mod, "RoboVLM-Nav", NavRoboKosMos)

from robovlm_nav.models.policy_head.nav_policy_impl import (
    MobileVLAClassificationDecoder,
    MobileVLALSTMDecoder,
)
from robovlm_nav.models.policy_head.hybrid_action_head import HybridActionHead
setattr(policy_head_mod, "MobileVLAClassificationDecoder", MobileVLAClassificationDecoder)
setattr(policy_head_mod, "MobileVLALSTMDecoder", MobileVLALSTMDecoder)
setattr(policy_head_mod, "NavPolicy", MobileVLAClassificationDecoder)
setattr(policy_head_mod, "NavPolicyRegression", MobileVLALSTMDecoder)
setattr(policy_head_mod, "HybridActionHead", HybridActionHead)

from robovlm_nav.trainer.nav_trainer import NavTrainer
base_trainer_mod.BaseTrainer = NavTrainer
setattr(train_mod, "NavTrainer", NavTrainer)
setattr(train_mod, "BaseTrainer", NavTrainer)

import main as main_mod
main_mod.BaseTrainer = NavTrainer

PROMPTS = {
    "left":    "<grounding>Instruction: Navigate to the left toward the gray basket. Action:",
    "right":   "<grounding>Instruction: Navigate to the right toward the gray basket. Action:",
    "forward": "<grounding>Instruction: Navigate straight forward to the gray basket. Action:",
}

CLASS_NAMES_8 = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "TURN_L", "TURN_R"]


def _resolve_data_dir(p):
    candidates = [
        Path(p),
        Path("/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5"),
        Path("/home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v5"),
        ROOT / "ROS_action" / "mobile_vla_dataset_v5",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(p)


def load_model(ckpt_path: str, config_path: str):
    os.chdir(str(ROOT))
    from main import load_config
    configs = load_config(config_path)

    vlm_path = str(ROOT / ".vlms" / "kosmos-2-patch14-224")

    def _fix_paths(d):
        for k, v in d.items():
            if isinstance(v, str) and "kosmos-2-patch14-224" in v:
                d[k] = vlm_path
            elif k == "data_dir" and isinstance(v, str):
                d[k] = _resolve_data_dir(v)
            elif isinstance(v, dict):
                _fix_paths(v)
    _fix_paths(configs)
    if isinstance(configs.get("vlm"), dict):
        configs["vlm"]["pretrained_model_name_or_path"] = vlm_path
    if isinstance(configs.get("tokenizer"), dict):
        configs["tokenizer"]["pretrained_model_name_or_path"] = vlm_path

    from robovlms.train.mobile_vla_trainer import MobileVLATrainer
    print("🔧 Building model...")
    model_wrapper = MobileVLATrainer(configs)

    print(f"📦 Loading checkpoint: {Path(ckpt_path).name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full_sd = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
    filtered = {}
    for k, v in full_sd.items():
        if any(x in k for x in ["image_to_text_projection", "act_head",
                                 "policy_head", "resampler", "action_token", "lora"]):
            new_k = k.replace("model.", "", 1) if k.startswith("model.") and not hasattr(model_wrapper, "model") else k
            filtered[new_k] = v
    missing, unexpected = model_wrapper.load_state_dict(filtered, strict=False)
    print(f"✅ Loaded {len(filtered)} weights | missing={len(missing)} unexpected={len(unexpected)}")
    del full_sd, ckpt
    gc.collect()
    model_wrapper.to("cuda").eval().half()
    return model_wrapper, configs


def build_dataset(configs, n_frames: int):
    from robovlm_nav.datasets.nav_dataset import NavDataset
    val_cfg = configs.get("val_dataset", {})
    data_dir = _resolve_data_dir(val_cfg.get("data_dir") or configs.get("train_dataset", {}).get("data_dir", ""))
    ds = NavDataset(
        data_dir=data_dir,
        episode_pattern="episode_*.h5",
        model_name="kosmos",
        window_size=val_cfg.get("window_size", 8),
        fwd_pred_next_n=val_cfg.get("fwd_pred_next_n", 1),
        discrete_action=True,
        num_classes=val_cfg.get("num_classes", 8),
        instruction_preset="default",   # 우리가 직접 prompt 교체할 거라 default OK
        grounding_prefix=True,
        is_validation=True,
        train_split=val_cfg.get("train_split", 0.9),
        stratified_split=True,
        min_episode_frames=8,
    )
    return ds


def replace_text_in_batch(batch, tokenizer, new_prompt: str):
    """기존 batch의 text/text_mask를 새 프롬프트로 교체."""
    orig = batch["text"]
    orig_shape = orig.shape  # (1, L) or (1, ws, L)
    max_len = orig_shape[-1]
    enc = tokenizer(
        new_prompt, return_tensors="pt",
        padding="max_length", max_length=max_len, truncation=True,
    )
    new_ids = enc["input_ids"]
    new_mask = enc["attention_mask"]
    if len(orig_shape) == 3:
        # (1, ws, L) — 모든 window position에 같은 텍스트
        ws = orig_shape[1]
        new_ids = new_ids.unsqueeze(1).expand(-1, ws, -1).contiguous()
        new_mask = new_mask.unsqueeze(1).expand(-1, ws, -1).contiguous()
    batch = dict(batch)
    batch["text"] = new_ids
    batch["text_mask"] = new_mask
    return batch


def run_inference(model, batch_gpu):
    out = model.forward_action(
        vision_x=batch_gpu["rgb"],
        lang_x=batch_gpu["text"],
        attention_mask=batch_gpu["text_mask"].bool(),
        text_embedding=batch_gpu.get("text_embedding"),
        vision_gripper=batch_gpu.get("hand_rgb"),
        instr_and_action_ids=batch_gpu.get("instr_and_action_ids"),
        instr_and_action_labels=batch_gpu.get("instr_and_action_labels"),
        instr_and_action_mask=batch_gpu.get("instr_and_action_mask"),
        mode="test",
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    arr = out.detach().cpu().float().numpy()
    if arr.ndim == 4:   logits = arr[0, 0, 0, :]
    elif arr.ndim == 3: logits = arr[0, 0, :]
    elif arr.ndim == 2: logits = arr[0, :]
    else:               logits = arr
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    model_wrapper, configs = load_model(args.ckpt, args.config)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer if hasattr(model_wrapper, "tokenizer") else None
    if tokenizer is None:
        # fallback: NavDataset에서 가져옴
        from transformers import AutoProcessor
        vlm_path = str(ROOT / ".vlms" / "kosmos-2-patch14-224")
        tokenizer = AutoProcessor.from_pretrained(vlm_path).tokenizer

    ds = build_dataset(configs, args.n_frames)
    n_eval = min(len(ds), args.n_frames)
    print(f"📊 Eval frames: {n_eval} / dataset size: {len(ds)}")

    per_frame = []
    diffs_l1 = []
    diffs_left_vs_right = []
    pred_change_count = 0

    with torch.no_grad():
        for i in range(n_eval):
            sample = ds[i]
            batch = ds.collater([sample])
            preds = {}
            logits_per_prompt = {}
            for key, prompt in PROMPTS.items():
                b = replace_text_in_batch(batch, tokenizer, prompt)
                gpu = {k: v.cuda().half() if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
                       else (v.cuda() if isinstance(v, torch.Tensor) else v)
                       for k, v in b.items()}
                logits = run_inference(model, gpu)
                logits_per_prompt[key] = logits
                preds[key] = int(np.argmax(logits))

            # softmax probability L1 distance — 더 안정적
            def _softmax(x):
                z = x - np.max(x)
                e = np.exp(z); return e / e.sum()
            p_left = _softmax(logits_per_prompt["left"])
            p_right = _softmax(logits_per_prompt["right"])
            p_fwd  = _softmax(logits_per_prompt["forward"])

            l1_lr = float(np.abs(p_left - p_right).sum())
            l1_lf = float(np.abs(p_left - p_fwd).sum())
            l1_rf = float(np.abs(p_right - p_fwd).sum())
            mean_l1 = (l1_lr + l1_lf + l1_rf) / 3.0

            diffs_l1.append(mean_l1)
            diffs_left_vs_right.append(l1_lr)
            if len(set(preds.values())) > 1:
                pred_change_count += 1

            per_frame.append({
                "frame_idx": i,
                "preds": {k: CLASS_NAMES_8[v] for k, v in preds.items()},
                "pred_idx": preds,
                "softmax_l1": {"left_vs_right": l1_lr, "left_vs_forward": l1_lf, "right_vs_forward": l1_rf},
                "raw_text": sample.get("raw_text", sample.get("lang", "")),
            })
            if i < 5:
                print(f"  [{i}] preds={preds} L1(L↔R)={l1_lr:.4f}")

    summary = {
        "ckpt": args.ckpt,
        "config": args.config,
        "n_frames": n_eval,
        "mean_softmax_l1": float(np.mean(diffs_l1)),
        "mean_l1_left_vs_right": float(np.mean(diffs_left_vs_right)),
        "max_l1_left_vs_right": float(np.max(diffs_left_vs_right)) if diffs_left_vs_right else 0.0,
        "frames_with_pred_change": pred_change_count,
        "pred_change_rate": pred_change_count / n_eval if n_eval else 0.0,
        "thresholds": {
            "text_locked_min_l1": 0.05,
            "text_locked_min_pred_change_rate": 0.20,
        },
        "verdict": None,
        "frames": per_frame,
    }
    locked = (summary["mean_l1_left_vs_right"] >= 0.05
              and summary["pred_change_rate"] >= 0.20)
    summary["verdict"] = "LOCKED" if locked else "TEXT_INSENSITIVE"

    print()
    print(f"📈 mean softmax L1 (avg of 3 pairs): {summary['mean_softmax_l1']:.5f}")
    print(f"📈 mean L1 (left vs right):          {summary['mean_l1_left_vs_right']:.5f}")
    print(f"📈 frames with pred change:          {pred_change_count}/{n_eval} ({summary['pred_change_rate']*100:.1f}%)")
    print(f"🏁 verdict: {summary['verdict']}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"💾 saved → {out}")


if __name__ == "__main__":
    main()
