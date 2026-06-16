#!/usr/bin/env python3
"""
Kosmos-2 prompt format 진단 스크립트.

동일 이미지에 6가지 프롬프트 포맷 × 3방향(left/right/forward)을 넣어
어떤 포맷이 text sensitivity를 만드는지 측정한다.

사용:
    python3 scripts/test_prompt_formats.py \
        --ckpt runs/v5_nav/kosmos/mobile_vla_v5_exp41c/.../last.ckpt \
        --config configs/mobile_vla_v5_exp41c_scratch_pta.json \
        --n-frames 20
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

CLASS_NAMES_8 = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "TURN_L", "TURN_R"]

# ── 6가지 포맷 ×  3방향 ─────────────────────────────────────────────────────
# 핵심 가설:
#  A. <grounding> 제거 → LM 경로 활성화?
#  B. 방향 단어를 Action: 직전에 → readout 토큰 attention 유리?
#  C. Kosmos-2 <phrase> 태그 → grounding-salient 처리?
#  D. 짧은 명령 → 단순 토큰 → 구분 쉬워짐?
#  E. Q&A 스타일 → 완성 전 방향 단어 출현?
PROMPT_SETS = {
    "A_baseline": {
        "left":    "<grounding>Instruction: Navigate to the left toward the gray basket. Action:",
        "right":   "<grounding>Instruction: Navigate to the right toward the gray basket. Action:",
        "forward": "<grounding>Instruction: Navigate straight forward to the gray basket. Action:",
    },
    "B_no_grounding": {
        "left":    "Navigate to the left toward the gray basket. Action:",
        "right":   "Navigate to the right toward the gray basket. Action:",
        "forward": "Navigate straight forward to the gray basket. Action:",
    },
    "C_dir_at_end": {
        "left":    "<grounding>Navigate to the gray basket. Direction: LEFT. Action:",
        "right":   "<grounding>Navigate to the gray basket. Direction: RIGHT. Action:",
        "forward": "<grounding>Navigate to the gray basket. Direction: STRAIGHT. Action:",
    },
    "D_phrase_tag": {
        "left":    "<grounding>Navigate <phrase>left</phrase> to the gray basket.",
        "right":   "<grounding>Navigate <phrase>right</phrase> to the gray basket.",
        "forward": "<grounding>Navigate <phrase>straight</phrase> to the gray basket.",
    },
    "E_short_no_grounding": {
        "left":    "Turn left.",
        "right":   "Turn right.",
        "forward": "Go straight.",
    },
    "F_qa_style": {
        "left":    "Should the robot go left, right, or straight? The robot should go left. Action:",
        "right":   "Should the robot go left, right, or straight? The robot should go right. Action:",
        "forward": "Should the robot go left, right, or straight? The robot should go straight. Action:",
    },
}


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


def build_dataset(configs):
    from robovlm_nav.datasets.nav_dataset import NavDataset
    val_cfg = configs.get("val_dataset", {})
    data_dir = _resolve_data_dir(val_cfg.get("data_dir") or configs.get("train_dataset", {}).get("data_dir", ""))
    return NavDataset(
        data_dir=data_dir,
        episode_pattern="episode_*.h5",
        model_name="kosmos",
        window_size=val_cfg.get("window_size", 8),
        fwd_pred_next_n=val_cfg.get("fwd_pred_next_n", 1),
        discrete_action=True,
        num_classes=val_cfg.get("num_classes", 8),
        instruction_preset="default",
        grounding_prefix=True,
        is_validation=True,
        train_split=val_cfg.get("train_split", 0.9),
        stratified_split=True,
        min_episode_frames=8,
    )


def replace_text(batch, tokenizer, prompt: str):
    orig = batch["text"]
    max_len = orig.shape[-1]
    enc = tokenizer(prompt, return_tensors="pt",
                    padding="max_length", max_length=max_len, truncation=True)
    new_ids  = enc["input_ids"]
    new_mask = enc["attention_mask"]
    if orig.ndim == 3:
        ws = orig.shape[1]
        new_ids  = new_ids.unsqueeze(1).expand(-1, ws, -1).contiguous()
        new_mask = new_mask.unsqueeze(1).expand(-1, ws, -1).contiguous()
    b = dict(batch)
    b["text"] = new_ids
    b["text_mask"] = new_mask
    return b


def run_inference(model_wrapper, batch_gpu):
    model = model_wrapper.model
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
    if arr.ndim == 4:   return arr[0, 0, 0, :]
    elif arr.ndim == 3: return arr[0, 0, :]
    elif arr.ndim == 2: return arr[0, :]
    return arr


def softmax(x):
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    model_wrapper, configs = load_model(args.ckpt, args.config)
    tokenizer = getattr(model_wrapper, "tokenizer", None)
    if tokenizer is None:
        from transformers import AutoProcessor
        tokenizer = AutoProcessor.from_pretrained(
            str(ROOT / ".vlms" / "kosmos-2-patch14-224")
        ).tokenizer

    ds = build_dataset(configs)
    n_eval = min(len(ds), args.n_frames)
    print(f"\n📊 Evaluating {n_eval} frames × {len(PROMPT_SETS)} formats × 3 directions\n")

    # format_name → list of per-frame L1(left vs right)
    format_l1s = {name: [] for name in PROMPT_SETS}
    format_pred_changes = {name: 0 for name in PROMPT_SETS}
    format_sample_preds = {name: [] for name in PROMPT_SETS}   # first 3 frames

    with torch.no_grad():
        for i in range(n_eval):
            sample = ds[i]
            batch  = ds.collater([sample])

            for fmt_name, prompts in PROMPT_SETS.items():
                logits_per_dir = {}
                preds = {}
                for direction, prompt_text in prompts.items():
                    b = replace_text(batch, tokenizer, prompt_text)
                    gpu = {
                        k: (v.cuda().half() if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
                            else (v.cuda() if isinstance(v, torch.Tensor) else v))
                        for k, v in b.items()
                    }
                    logits = run_inference(model_wrapper, gpu)
                    logits_per_dir[direction] = logits
                    preds[direction] = int(np.argmax(logits))

                p_left  = softmax(logits_per_dir["left"])
                p_right = softmax(logits_per_dir["right"])
                l1_lr   = float(np.abs(p_left - p_right).sum())

                format_l1s[fmt_name].append(l1_lr)
                if len(set(preds.values())) > 1:
                    format_pred_changes[fmt_name] += 1
                if i < 3:
                    format_sample_preds[fmt_name].append(
                        {k: CLASS_NAMES_8[v] for k, v in preds.items()}
                    )

    # ── 결과 출력 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Format':<28} {'mean L1(L↔R)':>13} {'max L1':>8} {'pred_change':>12}")
    print("-" * 72)

    results = {}
    for fmt_name in PROMPT_SETS:
        l1s = format_l1s[fmt_name]
        mean_l1 = float(np.mean(l1s))
        max_l1  = float(np.max(l1s))
        changes = format_pred_changes[fmt_name]
        change_rate = changes / n_eval
        flag = " ← SENSITIVE" if mean_l1 >= 0.05 else ("")
        print(f"  {fmt_name:<26} {mean_l1:>13.5f} {max_l1:>8.5f} {changes:>5}/{n_eval}{flag}")
        results[fmt_name] = {
            "mean_l1_lr": mean_l1,
            "max_l1_lr": max_l1,
            "pred_changes": changes,
            "pred_change_rate": change_rate,
            "sample_preds": format_sample_preds[fmt_name],
        }

    print("=" * 72)

    best = max(results, key=lambda k: results[k]["mean_l1_lr"])
    print(f"\n🏆 Best format: [{best}]  mean L1={results[best]['mean_l1_lr']:.5f}")

    # sample preds for best
    print(f"\n   Sample predictions (first 3 frames) for [{best}]:")
    for j, p in enumerate(results[best]["sample_preds"]):
        print(f"     frame {j}: {p}")

    print()
    print("Prompt texts for best format:")
    for d, t in PROMPT_SETS[best].items():
        print(f"  {d:8s}: {t}")

    if args.output_json:
        out = {
            "ckpt": args.ckpt,
            "n_frames": n_eval,
            "prompt_sets": {k: list(v.keys()) for k, v in PROMPT_SETS.items()},
            "results": results,
            "best_format": best,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n💾 Saved: {args.output_json}")


if __name__ == "__main__":
    main()
