#!/usr/bin/env python3
"""
V5 PM/DM 오프라인 평가 스크립트

- Dataset : V5 H5 (observations/images, 3D action)
- Model   : V5 best ckpt (epoch=05, val_loss=2.270)
- Classes : 6 (STOP / FORWARD / LEFT / RIGHT / FWD+LEFT / FWD+RIGHT)
- Metrics : PM (Perfect Match), per-class accuracy, confusion matrix
"""
import os, sys

# ── MPI hang 방지
import lightning.fabric.plugins.environments.mpi as _mpi_env_mod
_mpi_env_mod._MPI4PY_AVAILABLE = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "RoboVLMs"))

import json, gc
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ── robovlms 네임스페이스 주입 (train.py와 동일)
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

# ── CLI 인자 파싱 (하드코딩 대신)
import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument("--ckpt",    default=None)
_ap.add_argument("--config",  default=None)
_ap.add_argument("--data",    default=None)
_ap.add_argument("--num_classes", type=int, default=None)
_ap.add_argument("--train_split", type=float, default=None)
_ap.add_argument("--window_size", type=int, default=None)
_ap.add_argument("--instruction_preset", default=None)
_ap.add_argument("--exclude_path_types", default=None)  # 예: "center_straight"
_ap.add_argument("--include_path_families", default=None)  # 예: "left_straight,left_left,left_right"
_ap.add_argument("--eval_t", type=int, default=0)      # 0: 첫 프레임, -1: 마지막(히스토리 풀 활용)
_ap.add_argument("--eval_split", choices=["train", "val", "all"], default="val")
_ap.add_argument("--max_samples", type=int, default=None)
_ap.add_argument("--output_json", default=None)
_cli, _ = _ap.parse_known_args()

# ── 설정 (CLI 우선, fallback으로 기본값)
CKPT = _cli.ckpt or os.path.join(ROOT, "runs/v5_nav/kosmos/mobile_vla_v5_exp01/2026-04-10/v5-exp01-discrete",
                    "epoch_epoch=epoch=05-val_loss=val_loss=2.270.ckpt")
CONFIG = _cli.config or os.path.join(ROOT, "configs/mobile_vla_v5_exp01_discrete.json")
V5_DATA = _cli.data or os.path.join(ROOT, "ROS_action/v5_data_bak/mobile_vla_dataset_v5")

NUM_CLASSES = _cli.num_classes or 6
_VAL_TRAIN_SPLIT = _cli.train_split or 0.85
_VAL_WINDOW_SIZE = _cli.window_size or 6
_VAL_INSTRUCTION_PRESET = _cli.instruction_preset or "default"
_VAL_INCLUDE_PATH_FAMILIES = (
    [x.strip() for x in _cli.include_path_families.split(",") if x.strip()]
    if _cli.include_path_families
    else None
)

# 두 서버 호환: billy ↔ minum 데이터셋 경로 자동 resolve
_DATA_PATH_ALIASES = [
    "/home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v5",
    "/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5",
]

def _resolve_data_dir(p):
    if not isinstance(p, str) or os.path.exists(p):
        return p
    for src in _DATA_PATH_ALIASES:
        if src in p:
            for dst in _DATA_PATH_ALIASES:
                if dst == src:
                    continue
                cand = p.replace(src, dst)
                if os.path.exists(cand):
                    return cand
    return p


CLASS_NAMES_6 = {0:"STOP", 1:"FORWARD", 2:"LEFT", 3:"RIGHT", 4:"FWD+L", 5:"FWD+R"}
CLASS_NAMES_8 = {0:"STOP", 1:"FORWARD", 2:"LEFT", 3:"RIGHT", 4:"FWD+L", 5:"FWD+R", 6:"TURN_L", 7:"TURN_R"}
CLASS_NAMES = CLASS_NAMES_8 if NUM_CLASSES == 8 else CLASS_NAMES_6
ACTION_VEL_8 = {
    0: [0.0, 0.0, 0.0],
    1: [1.15, 0.0, 0.0],
    2: [0.0, 1.15, 0.0],
    3: [0.0, -1.15, 0.0],
    4: [1.15, 1.15, 0.0],
    5: [1.15, -1.15, 0.0],
    6: [0.0, 0.0, 0.25],
    7: [0.0, 0.0, -0.25],
}
ACTION_VEL_6 = {k: ACTION_VEL_8[k] for k in range(6)}


def _load_eval_settings_from_config():
    global V5_DATA, NUM_CLASSES, _VAL_TRAIN_SPLIT, _VAL_WINDOW_SIZE, CLASS_NAMES
    global _VAL_INCLUDE_PATH_FAMILIES
    try:
        # parent inheritance까지 머지된 config 사용
        try:
            from main import load_config
            cfg = load_config(CONFIG)
        except Exception:
            with open(CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
    except Exception:
        return

    val_cfg = cfg.get("val_dataset", {})
    train_cfg = cfg.get("train_dataset", {})
    data_cfg = val_cfg or train_cfg

    if _cli.data is None and data_cfg.get("data_dir"):
        V5_DATA = _resolve_data_dir(data_cfg["data_dir"])
    if _cli.num_classes is None and data_cfg.get("num_classes"):
        NUM_CLASSES = int(data_cfg["num_classes"])
    if _cli.train_split is None and data_cfg.get("train_split"):
        _VAL_TRAIN_SPLIT = float(data_cfg["train_split"])
    if _cli.window_size is None and data_cfg.get("window_size"):
        _VAL_WINDOW_SIZE = int(data_cfg["window_size"])
    if _cli.include_path_families is None and data_cfg.get("include_path_families") is not None:
        _VAL_INCLUDE_PATH_FAMILIES = list(data_cfg["include_path_families"] or [])

    CLASS_NAMES = CLASS_NAMES_8 if NUM_CLASSES == 8 else CLASS_NAMES_6


_load_eval_settings_from_config()

# ── 모델 로드 (inference_server와 동일한 방식)
def load_model():
    os.chdir(ROOT)
    from main import load_config, update_configs
    configs = load_config(CONFIG)

    # 경로 패치 (VLM은 ROOT 기반, 데이터는 billy↔minum 자동 resolve)
    vlm_path = os.path.join(ROOT, ".vlms", "kosmos-2-patch14-224")
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

    # 필요한 가중치만 로드 (inference_server 방식)
    print(f"📦 Loading checkpoint: {Path(CKPT).name}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    full_sd = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))

    filtered = {}
    for k, v in full_sd.items():
        if any(x in k for x in ["image_to_text_projection","act_head","policy_head","resampler","action_token","lora"]):
            new_k = k.replace("model.", "", 1) if k.startswith("model.") and not hasattr(model_wrapper, "model") else k
            filtered[new_k] = v

    missing, unexpected = model_wrapper.load_state_dict(filtered, strict=False)
    print(f"✅ Loaded {len(filtered)} weights | missing={len(missing)} unexpected={len(unexpected)}")
    del full_sd, ckpt; gc.collect()

    model_wrapper.to("cuda").eval().half()
    return model_wrapper

# ── logit 파싱 (V4 스크립트와 동일 로직)
def parse_logits(outputs, t=0):
    """
    t=0  : 첫 번째 window 위치 (parse_gt의 ac[0,0,0]과 정렬)
    t=-1 : 마지막 window 위치 (inference 시점과 정렬, ROT는 여기선 절대 안 나옴)
    """
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    if outputs is None or not isinstance(outputs, torch.Tensor):
        return None
    arr = outputs.detach().cpu().float().numpy()
    if arr.ndim == 4:   logits = arr[0, t, 0, :]
    elif arr.ndim == 3: logits = arr[0, t, :]
    elif arr.ndim == 2: logits = arr[0, :]
    else:               logits = arr
    return int(np.argmax(logits)), logits

def parse_gt(batch, t=0):
    if "action_chunck" in batch:
        ac = batch["action_chunck"].cpu().numpy()
        # ac shape: [batch, window_size, fwd_pred_next_n]
        return int(ac[0, t, 0])
    if "action" in batch:
        return int(batch["action"].cpu().numpy()[0, -1])
    return None


def softmax_np(logits):
    arr = np.asarray(logits, dtype=np.float64)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    denom = np.sum(exp)
    if denom <= 0:
        return np.zeros_like(arr, dtype=np.float64)
    return exp / denom


def class_action_3d(cls_idx):
    mapping = ACTION_VEL_8 if NUM_CLASSES == 8 else ACTION_VEL_6
    return [float(x) for x in mapping.get(int(cls_idx), [0.0, 0.0, 0.0])]

# ── 데이터셋 로드
def _build_dataset(is_validation):
    from robovlm_nav.datasets.nav_dataset import NavDataset
    return NavDataset(
        data_dir=V5_DATA,
        episode_pattern="episode_*.h5",
        model_name="kosmos",
        window_size=_VAL_WINDOW_SIZE,
        fwd_pred_next_n=3,
        discrete_action=True,
        num_classes=NUM_CLASSES,
        instruction_preset=_VAL_INSTRUCTION_PRESET,
        grounding_prefix=True,
        is_validation=is_validation,
        train_split=_VAL_TRAIN_SPLIT,
        stratified_split=True,
        exclude_path_types=_cli.exclude_path_types.split(",") if getattr(_cli, "exclude_path_types", None) else [],
        include_path_families=_VAL_INCLUDE_PATH_FAMILIES,
        min_episode_frames=8,
    )

def load_eval_dataset():
    if _cli.eval_split == "val":
        return _build_dataset(is_validation=True)
    if _cli.eval_split == "train":
        return _build_dataset(is_validation=False)

    train_ds = _build_dataset(is_validation=False)
    val_ds = _build_dataset(is_validation=True)

    class _ConcatWithCollater:
        def __init__(self, datasets):
            self.datasets = datasets
            self.offsets = []
            total = 0
            for ds in datasets:
                self.offsets.append(total)
                total += len(ds)
            self.total = total

        def __len__(self):
            return self.total

        def __getitem__(self, idx):
            for offset, ds in zip(reversed(self.offsets), reversed(self.datasets)):
                if idx >= offset:
                    return ds[idx - offset]
            raise IndexError(idx)

        def collater(self, samples):
            return self.datasets[0].collater(samples)

        def metadata(self, idx):
            for split_name, offset, ds in zip(["train", "val"], self.offsets, self.datasets):
                if idx < offset:
                    continue
                local_idx = idx - offset
                if local_idx < len(ds):
                    return sample_metadata(ds, local_idx, split_name)
            raise IndexError(idx)

    return _ConcatWithCollater([train_ds, val_ds])


def sample_metadata(ds, idx, split_name=None):
    if hasattr(ds, "metadata"):
        return ds.metadata(idx)
    ep_idx, start_frame = ds.frame_indices[idx]
    episode_file = Path(ds.episode_files[ep_idx])
    eval_offset = _cli.eval_t if _cli.eval_t >= 0 else _VAL_WINDOW_SIZE + _cli.eval_t
    family = None
    if hasattr(ds, "_extract_path_family"):
        try:
            family = ds._extract_path_family(episode_file.stem)
        except Exception:
            family = None
    return {
        "split": split_name or ("val" if getattr(ds, "is_validation", False) else "train"),
        "local_idx": int(idx),
        "episode": episode_file.name,
        "episode_stem": episode_file.stem,
        "episode_path": str(episode_file),
        "start_frame": int(start_frame),
        "eval_frame": int(start_frame + eval_offset),
        "path_family": family,
    }

# ── 메인 평가
def evaluate():
    print("=" * 65)
    print("  V5 PM/DM Offline Evaluation")
    print(f"  Ckpt : {Path(CKPT).name}")
    print(f"  Data : {V5_DATA}")
    print("=" * 65)

    model_wrapper = load_model()
    model = model_wrapper.model  # RoboKosMos backbone

    ds = load_eval_dataset()
    print(f"📊 Eval split: {_cli.eval_split} | dataset: {len(ds)} sequences\n")

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    pm, total, errors = 0, 0, 0
    rows = []
    pred_counts = np.zeros(NUM_CLASSES, dtype=int)
    n_eval = len(ds) if _cli.max_samples is None else min(len(ds), _cli.max_samples)

    with torch.no_grad():
        for i in tqdm(range(n_eval), desc="Eval"):
            try:
                sample = ds[i]
                batch = ds.collater([sample])
                gpu = {k: v.cuda().half() if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
                       else (v.cuda() if isinstance(v, torch.Tensor) else v)
                       for k, v in batch.items()}

                gt = parse_gt(gpu, t=_cli.eval_t)
                if gt is None: continue
                gt = min(gt, NUM_CLASSES - 1)

                outputs = model.forward_action(
                    vision_x=gpu["rgb"],
                    lang_x=gpu["text"],
                    attention_mask=gpu["text_mask"].bool(),
                    text_embedding=gpu.get("text_embedding"),
                    vision_gripper=gpu.get("hand_rgb"),
                    instr_and_action_ids=gpu.get("instr_and_action_ids"),
                    instr_and_action_labels=gpu.get("instr_and_action_labels"),
                    instr_and_action_mask=gpu.get("instr_and_action_mask"),
                    mode="test",
                )

                result = parse_logits(outputs, t=_cli.eval_t)  # parse_gt와 동일 t
                if result is None: errors += 1; continue

                pred, logits = result
                pred = min(pred, NUM_CLASSES - 1)

                confusion[gt, pred] += 1
                pred_counts[pred] += 1
                total += 1
                if pred == gt: pm += 1
                probs = softmax_np(logits)
                order = np.argsort(probs)[::-1][: min(5, len(probs))]
                metadata = sample_metadata(ds, i)
                rows.append(
                    {
                        "idx": i,
                        **metadata,
                        "raw_text": sample.get("raw_text", sample.get("lang", "")),
                        "gt": gt,
                        "gt_name": CLASS_NAMES.get(gt, "?"),
                        "gt_action_3d": class_action_3d(gt),
                        "pred": pred,
                        "pred_name": CLASS_NAMES.get(pred, "?"),
                        "pred_action_3d": class_action_3d(pred),
                        "correct": bool(pred == gt),
                        "logits": [float(x) for x in logits],
                        "softmax": [float(x) for x in probs],
                        "top_classes": [
                            {
                                "class_idx": int(j),
                                "class_name": CLASS_NAMES.get(int(j), "?"),
                                "logit": float(logits[j]),
                                "prob": float(probs[j]),
                                "action_3d": class_action_3d(int(j)),
                            }
                            for j in order
                        ],
                    }
                )

                if i < 15:
                    gt_n, pred_n = CLASS_NAMES.get(gt,"?"), CLASS_NAMES.get(pred,"?")
                    mark = "✅" if pred == gt else "❌"
                    print(f"  [{i:3d}] GT={gt_n:<8} PRED={pred_n:<8} logits={np.round(logits,2)} {mark}")

            except Exception as e:
                errors += 1
                if errors <= 3:
                    import traceback; traceback.print_exc()

    # ── 결과 출력
    print(f"\n{'='*65}")
    print(f"  Result Summary")
    print(f"{'='*65}")
    print(f"  Total   : {total}  |  Errors: {errors}")
    if total == 0:
        print("❌ No valid samples!"); return

    pm_rate = pm / total * 100
    print(f"  PM (Perfect Match) : {pm_rate:.2f}%  ({pm}/{total})")

    print(f"\n  {'Class':<10} {'GT_cnt':>7} {'Correct':>8} {'Acc':>8}")
    print(f"  {'-'*40}")
    for c in range(NUM_CLASSES):
        gt_tot = int(confusion[c].sum())
        correct = int(confusion[c, c])
        acc = correct / gt_tot * 100 if gt_tot > 0 else 0.0
        print(f"  {CLASS_NAMES[c]:<10} {gt_tot:>7} {correct:>8} {acc:>7.1f}%")

    print(f"\n  Confusion Matrix (row=GT, col=PRED)")
    header = "  GT\\PRED  " + "".join(f"{CLASS_NAMES[c]:>8}" for c in range(NUM_CLASSES))
    print(header)
    for r in range(NUM_CLASSES):
        row = f"  {CLASS_NAMES[r]:<9} " + "".join(f"{confusion[r,c]:>8}" for c in range(NUM_CLASSES))
        print(row)
    print()

    print("  Prediction distribution")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:<10} {int(pred_counts[c]):>7}")

    if _cli.output_json:
        payload = {
            "ckpt": CKPT,
            "config": CONFIG,
            "data": V5_DATA,
            "eval_split": _cli.eval_split,
            "eval_t": _cli.eval_t,
            "num_classes": NUM_CLASSES,
            "include_path_families": _VAL_INCLUDE_PATH_FAMILIES,
            "total": total,
            "errors": errors,
            "pm": pm,
            "pm_rate": pm / total if total else None,
            "class_names": CLASS_NAMES,
            "confusion": confusion.tolist(),
            "pred_counts": pred_counts.tolist(),
            "rows": rows,
        }
        out = Path(_cli.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"  Wrote JSON: {out}")

if __name__ == "__main__":
    os.chdir(ROOT)
    evaluate()
