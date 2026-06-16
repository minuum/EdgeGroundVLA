#!/usr/bin/env python3
"""
Measure attention weights of the action token on image vs text tokens.

Goal: causal evidence for "VLM ignores text instruction" (TEXT_IGNORE_ROOTCAUSE §6 반문 2).
Hypothesis: action token attends mostly to image(64) tokens, barely to text tokens.

Does NOT modify third_party/RoboVLMs/. Uses monkey-patch on the Kosmos text
transformer's forward to inject output_attentions=True and capture the result.
"""

import gc
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

import lightning.fabric.plugins.environments.mpi as _mpi_env_mod

_mpi_env_mod._MPI4PY_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "RoboVLMs"))

import robovlms.model.backbone as backbone_mod
import robovlms.model.policy_head as policy_head_mod
import robovlms.train as train_mod
import robovlms.train.base_trainer as base_trainer_mod
from robovlm_nav.datasets.nav_dataset import NavDataset
from robovlm_nav.models.nav_robokosmos import NavRoboKosMos
from robovlm_nav.models.policy_head.hybrid_action_head import HybridActionHead
from robovlm_nav.models.policy_head.nav_policy_impl import (
    MobileVLAClassificationDecoder,
    MobileVLALSTMDecoder,
)
from robovlm_nav.trainer.nav_trainer import NavTrainer

setattr(backbone_mod, "RoboKosMos", NavRoboKosMos)
setattr(backbone_mod, "RoboVLM-Nav", NavRoboKosMos)
setattr(policy_head_mod, "MobileVLAClassificationDecoder", MobileVLAClassificationDecoder)
setattr(policy_head_mod, "MobileVLALSTMDecoder", MobileVLALSTMDecoder)
setattr(policy_head_mod, "NavPolicy", MobileVLAClassificationDecoder)
setattr(policy_head_mod, "NavPolicyRegression", MobileVLALSTMDecoder)
setattr(policy_head_mod, "HybridActionHead", HybridActionHead)
base_trainer_mod.BaseTrainer = NavTrainer
setattr(train_mod, "NavTrainer", NavTrainer)
setattr(train_mod, "BaseTrainer", NavTrainer)

import main as main_mod

main_mod.BaseTrainer = NavTrainer
from main import load_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VLM_PATH = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_DIR = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OUT_DIR = ROOT / "docs" / "v5" / "attention_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = OUT_DIR / "summary.json"
HTML_FILE = OUT_DIR / "index.html"

MODELS = {
    "exp11": {
        "ckpt": "runs/v5_nav/kosmos/mobile_vla_v5_exp11/2026-04-16/v5-exp11-google-robot-8cls/epoch_epoch=epoch=14-val_loss=val_loss=1.010.ckpt",
        "config": "configs/mobile_vla_v5_exp11_google_robot_8cls.json",
        "window_size": 8,
        "fwd_pred_next_n": 5,
        "num_classes": 8,
    },
    "exp13": {
        "ckpt": "runs/v5_nav/kosmos/mobile_vla_v5_exp13/2026-04-17/v5-exp13-instr-cond/last-v1.ckpt",
        "config": "configs/mobile_vla_v5_exp13_instr_cond.json",
        "window_size": 8,
        "fwd_pred_next_n": 5,
        "num_classes": 8,
    },
    "exp15_head_only": {
        "ckpt": "runs/v5_nav/kosmos/mobile_vla_v5_exp15/2026-04-18/v5-exp15-head-only/epoch_epoch=epoch=14-val_loss=val_loss=1.553.ckpt",
        "config": "configs/mobile_vla_v5_exp15_head_only.json",
        "window_size": 8,
        "fwd_pred_next_n": 5,
        "num_classes": 8,
    },
    "exp21_pure_hf_head_only": {
        "ckpt": "/tmp/monavla_resume_runs/kosmos/mobile_vla_v5_exp21/2026-04-21/v5-exp21-pure-hf-head-only/epoch_epoch=epoch=14-val_loss=val_loss=2.009.ckpt",
        "config": "configs/mobile_vla_v5_exp21_pure_hf_head_only.json",
        "window_size": 8,
        "fwd_pred_next_n": 5,
        "num_classes": 8,
    },
    "exp22_pure_hf_lora": {
        "exp_dir": "runs/v5_nav/kosmos/mobile_vla_v5_exp22",
        "config": "configs/mobile_vla_v5_exp22_pure_hf_lora.json",
        "window_size": 8,
        "fwd_pred_next_n": 5,
        "num_classes": 8,
    },
    "exp23_pure_hf_both": {
        "exp_dir": "runs/v5_nav/kosmos/mobile_vla_v5_exp23",
        "config": "configs/mobile_vla_v5_exp23_pure_hf_both.json",
        "window_size": 8,
        "fwd_pred_next_n": 5,
        "num_classes": 8,
    },
    "exp41b_resume_exp40_pta": {
        "exp_dir": "runs/v5_nav/kosmos/mobile_vla_v5_exp41b",
        "config": "configs/mobile_vla_v5_exp41b_resume_exp40_pta.json",
        "window_size": 8,
        "fwd_pred_next_n": 1,
        "num_classes": 8,
    },
    "exp41c_scratch_pta": {
        "exp_dir": "runs/v5_nav/kosmos/mobile_vla_v5_exp41c",
        "config": "configs/mobile_vla_v5_exp41c_scratch_pta.json",
        "window_size": 8,
        "fwd_pred_next_n": 5,
        "num_classes": 8,
    },
    "exp42_counterfactual_pta": {
        "exp_dir": "runs/v5_nav/kosmos/mobile_vla_v5_exp42",
        "config": "configs/mobile_vla_v5_exp42_counterfactual_pta.json",
        "window_size": 8,
        "fwd_pred_next_n": 3,
        "num_classes": 8,
    },
    "exp43_cross_attn_text": {
        "exp_dir": "runs/v5_nav/kosmos/mobile_vla_v5_exp43",
        "config": "configs/mobile_vla_v5_exp43_cross_attn_text.json",
        "window_size": 8,
        "fwd_pred_next_n": 3,
        "num_classes": 8,
    },
    # Exp45: vision-only LoRA (text decoder LoRA 제거). 가설: text attention 살아남는지 확인.
    "exp45_vision_lora_only": {
        "exp_dir": "runs/v5_nav/kosmos/mobile_vla_v5_exp45",
        "config": "configs/mobile_vla_v5_exp45_vision_lora_only.json",
        "window_size": 8,
        "fwd_pred_next_n": 5,
        "num_classes": 8,
    },
    # Exp48: synthetic instruction LoRA. val_loss가 Exp45와 소수점 3자리까지 동일 → text=0% 재확인용.
    "exp48_synthetic_instr": {
        "exp_dir": "runs/v5_nav/kosmos/mobile_vla_v5_exp48",
        "config": "configs/mobile_vla_v5_exp48_synthetic_instr.json",
        "window_size": 8,
        "fwd_pred_next_n": 5,
        "num_classes": 8,
    },
}

INSTRUCTIONS = {
    "left":    "Navigate to the left toward the gray basket",
    "right":   "Navigate to the right toward the gray basket",
    "forward": "Navigate straight forward to the gray basket",
}

NUM_IMAGE_TOKENS = 64


def resolve_ckpt_path(cfg):
    ckpt_rel = cfg.get("ckpt")
    if ckpt_rel:
        ckpt_full = ROOT / ckpt_rel
        return ckpt_rel if ckpt_full.exists() else None

    exp_dir = cfg.get("exp_dir")
    if not exp_dir:
        return None
    exp_root = ROOT / exp_dir
    if not exp_root.exists():
        return None

    candidates = sorted(exp_root.glob("**/epoch*.ckpt"))
    if not candidates:
        candidates = sorted(exp_root.glob("**/last*.ckpt"))
    if not candidates:
        return None
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(best.relative_to(ROOT))


def load_model(ckpt_rel, config_rel):
    configs = load_config(str(ROOT / config_rel))
    vlm_path = VLM_PATH

    def fix_paths(d):
        for k, v in list(d.items()):
            if isinstance(v, str) and "kosmos-2-patch14-224" in v:
                d[k] = str(vlm_path)
            elif isinstance(v, dict):
                fix_paths(v)

    fix_paths(configs)
    if isinstance(configs.get("vlm"), dict):
        configs["vlm"]["pretrained_model_name_or_path"] = str(vlm_path)
    if isinstance(configs.get("tokenizer"), dict):
        configs["tokenizer"]["pretrained_model_name_or_path"] = str(vlm_path)

    from robovlms.train.mobile_vla_trainer import MobileVLATrainer

    model_wrapper = MobileVLATrainer(configs)
    ckpt = torch.load(str(ROOT / ckpt_rel), map_location="cpu", weights_only=False)
    full_sd = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
    filtered = {}
    for k, v in full_sd.items():
        if any(x in k for x in ["image_to_text_projection", "act_head", "policy_head",
                                 "resampler", "action_token", "lora", "instr_proj"]):
            new_k = k.replace("model.", "", 1) if k.startswith("model.") and not hasattr(model_wrapper, "model") else k
            filtered[new_k] = v
    model_wrapper.load_state_dict(filtered, strict=False)
    del ckpt, full_sd, filtered
    gc.collect()
    model_wrapper.to(DEVICE).eval()
    if DEVICE.type == "cuda":
        model_wrapper.half()
    return model_wrapper, configs


def find_text_transformer(model):
    target_names = {"Kosmos2TextTransformer", "Kosmos2TextModel", "Kosmos2TextForCausalLM"}
    found = []

    def walk(m, path="root"):
        cn = m.__class__.__name__
        if cn in target_names:
            found.append((path, m, cn))
        for name, child in m.named_children():
            walk(child, f"{path}.{name}")

    walk(model)
    return found


def setup_attention_capture(text_transformer):
    cache = {
        "attentions": None, "cross_attentions": None,
        "last_seq_len": None, "input_shape": None, "input_ids": None,
    }
    orig_forward = text_transformer.forward

    def new_forward(*args, **kwargs):
        kwargs["output_attentions"] = True
        ids = kwargs.get("input_ids", None)
        inp_embeds = kwargs.get("inputs_embeds", None)
        if ids is not None:
            cache["input_ids"] = ids.detach().cpu()
            cache["input_shape"] = tuple(ids.shape)
        elif inp_embeds is not None:
            cache["input_ids"] = None
            cache["input_shape"] = tuple(inp_embeds.shape[:2])
        out = orig_forward(*args, **kwargs)
        attns = getattr(out, "attentions", None)
        if attns is not None:
            cache["attentions"] = tuple(a.detach().float().cpu() for a in attns)
            cache["last_seq_len"] = cache["attentions"][-1].shape[-1]
        x_attns = getattr(out, "cross_attentions", None)
        if x_attns is not None:
            cache["cross_attentions"] = tuple(a.detach().float().cpu() for a in x_attns)
        return out

    text_transformer.forward = new_forward
    return cache


def build_one_batch(cfg, target_path_type="left_left"):
    eps = sorted(DATA_DIR.glob(f"episode_*{target_path_type}*.h5"))
    if not eps:
        raise RuntimeError(f"No episode found for {target_path_type}")
    temp_root = Path(tempfile.mkdtemp(prefix="attn_measure_"))
    for ep in eps[:3]:
        os.symlink(ep, temp_root / ep.name)
    ds = NavDataset(
        data_dir=str(temp_root),
        episode_pattern="episode_*.h5",
        model_name="kosmos",
        window_size=cfg["window_size"],
        fwd_pred_next_n=cfg["fwd_pred_next_n"],
        discrete_action=True,
        num_classes=cfg["num_classes"],
        instruction_preset="default",
        grounding_prefix=True,
        is_validation=True,
        train_split=0.0,
        stratified_split=False,
        exclude_path_types=[],
        min_episode_frames=8,
    )
    sample = ds[0]
    batch = ds.collater([sample])
    # keep temp_root alive until the caller uses the batch then cleans up itself
    return batch, ds.tokenizer, temp_root


def rebuild_text_in_batch(batch, tokenizer, new_instruction):
    orig = batch["text"]
    orig_shape = orig.shape  # e.g. (bs=1, L) or (bs=1, ws, L)
    max_len = orig_shape[-1]
    prompt = f"<grounding>{new_instruction}"
    enc = tokenizer(prompt, return_tensors="pt", padding="max_length",
                     max_length=max_len, truncation=True)
    new_text = enc["input_ids"]
    new_mask = enc["attention_mask"]
    while new_text.ndim < len(orig_shape):
        new_text = new_text.unsqueeze(1)
        new_mask = new_mask.unsqueeze(1)
    new_text = new_text.expand(orig_shape).contiguous()
    new_mask = new_mask.expand(orig_shape).contiguous()
    batch = dict(batch)
    batch["text"] = new_text
    batch["text_mask"] = new_mask
    return batch


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            t = v.to(device)
            if device.type == "cuda" and t.dtype.is_floating_point:
                t = t.half()
            out[k] = t
        else:
            out[k] = v
    return out


def forward_and_capture(model_wrapper, b, cache):
    model = model_wrapper.model
    cache["attentions"] = None
    with torch.no_grad():
        _ = model.forward_action(
            vision_x=b["rgb"],
            lang_x=b["text"],
            attention_mask=b["text_mask"].bool(),
            vision_gripper=b.get("hand_rgb"),
            instr_and_action_ids=b.get("instr_and_action_ids"),
            instr_and_action_labels=b.get("instr_and_action_labels"),
            instr_and_action_mask=b.get("instr_and_action_mask"),
            mode="test",
        )
    return cache["attentions"], cache["last_seq_len"]


def analyze_attention(attn_tuple, text_len, num_image=NUM_IMAGE_TOKENS):
    """attn_tuple: tuple of (B, heads, S, S). Action token = last position."""
    layers = []
    for li, a in enumerate(attn_tuple):
        act_row = a[0, :, -1, :]           # (heads, S)
        seq_len = act_row.shape[-1]
        img_end = min(num_image, seq_len)
        img_part = act_row[:, :img_end].sum(dim=-1)
        text_end = min(num_image + text_len, seq_len)
        text_part = act_row[:, num_image:text_end].sum(dim=-1)
        pad_end = seq_len - 1
        pad_part = act_row[:, text_end:pad_end].sum(dim=-1) if pad_end > text_end else act_row[:, :0].sum(dim=-1)
        action_part = act_row[:, -1:].sum(dim=-1)
        total = act_row.sum(dim=-1)

        mean_over_heads = act_row.mean(dim=0)           # (S,)
        topk = torch.topk(mean_over_heads, k=min(10, seq_len))
        top_positions = [(int(p), float(v)) for p, v in zip(topk.indices.tolist(), topk.values.tolist())]
        # Per-position attention in the text region (num_image : num_image + text_len)
        text_region_per_pos = [float(x) for x in mean_over_heads[num_image:text_end].tolist()]

        layers.append({
            "layer": li,
            "seq_len": int(seq_len),
            "image_sum_mean": float(img_part.mean()),
            "text_sum_mean": float(text_part.mean()),
            "pad_sum_mean": float(pad_part.mean()),
            "self_sum_mean": float(action_part.mean()),
            "total_mean": float(total.mean()),
            "image_ratio_mean": float((img_part / (total + 1e-9)).mean()),
            "text_ratio_mean": float((text_part / (total + 1e-9)).mean()),
            "pad_ratio_mean": float((pad_part / (total + 1e-9)).mean()),
            "image_per_head": [float(x) for x in img_part.tolist()],
            "text_per_head": [float(x) for x in text_part.tolist()],
            "top_positions": top_positions,
            "text_region_per_pos": text_region_per_pos,
        })
    return layers


def format_pct(v):
    return f"{v * 100:.1f}%"


def build_html(summary):
    rows = []
    for model_name, model_res in summary.items():
        for instr, data in model_res.items():
            s = data.get("summary", {})
            rows.append(
                f"<tr><td>{model_name}</td><td>{instr}</td>"
                f"<td>{data.get('text_len')}</td>"
                f"<td>{format_pct(s.get('last_layer_image_ratio', 0))}</td>"
                f"<td>{format_pct(s.get('last_layer_text_ratio', 0))}</td>"
                f"<td>{format_pct(s.get('mean_layers_image_ratio', 0))}</td>"
                f"<td>{format_pct(s.get('mean_layers_text_ratio', 0))}</td>"
                f"</tr>"
            )
    layer_rows = []
    for model_name, model_res in summary.items():
        for instr, data in model_res.items():
            per_layer = data.get("per_layer", [])
            for l in per_layer:
                layer_rows.append(
                    f"<tr><td>{model_name}</td><td>{instr}</td>"
                    f"<td>{l['layer']}</td><td>{format_pct(l['image_ratio_mean'])}</td>"
                    f"<td>{format_pct(l['text_ratio_mean'])}</td></tr>"
                )

    html = f"""<!DOCTYPE html>
<html lang=\"ko\">
<head>
<meta charset=\"UTF-8\">
<title>Action-Token Attention Analysis</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:24px; }}
 h1,h2 {{ color:#fff; }}
 .back {{ color:#60a5fa; text-decoration:none; display:inline-block; margin-bottom:16px; }}
 .note {{ background:#172554; border-left:4px solid #60a5fa; padding:14px 18px; border-radius:6px; margin-bottom:20px; line-height:1.6; }}
 table {{ width:100%; border-collapse:collapse; background:#1e293b; border-radius:8px; overflow:hidden; margin-bottom:24px; }}
 th, td {{ padding:8px 12px; border-bottom:1px solid #334155; text-align:left; font-size:0.92rem; }}
 th {{ background:#0b1220; }}
 .good {{ color:#22c55e; }} .warn {{ color:#fbbf24; }} .bad {{ color:#f87171; }}
</style>
</head>
<body>
  <a class=\"back\" href=\"../../index.html\">← Back to MoNaVLA</a>
  <h1>Action-Token Attention Analysis</h1>
  <div class=\"note\">
    Action token이 image(64) 영역과 text 영역 중 어디에 attend하는지 측정합니다.<br>
    각 모델에 동일 이미지(한 개)를 주고 좌/우/전진 instruction을 바꿔가며 forward 후, 마지막 layer의 action row에서 image/text region attention 합계를 비교합니다.<br>
    가설: <b>image_ratio가 text_ratio를 크게 상회하면 instruction 무시의 구조적 원인이 확인됨</b>.
  </div>

  <h2>Summary (Last Layer / Mean over Layers)</h2>
  <table>
    <thead><tr>
      <th>Model</th><th>Instruction</th><th>text_len</th>
      <th>Last Img%</th><th>Last Text%</th>
      <th>Mean Img%</th><th>Mean Text%</th>
    </tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>

  <h2>Per Layer</h2>
  <table>
    <thead><tr>
      <th>Model</th><th>Instruction</th><th>Layer</th>
      <th>Image%</th><th>Text%</th>
    </tr></thead>
    <tbody>
      {''.join(layer_rows)}
    </tbody>
  </table>
</body>
</html>"""
    HTML_FILE.write_text(html)


def main():
    print(f"Device: {DEVICE}")
    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    tokenizer = processor.tokenizer
    all_results = {}

    for name, cfg in MODELS.items():
        print(f"\n=== {name} ===")
        ckpt_rel = resolve_ckpt_path(cfg)
        if not ckpt_rel:
            print("  ckpt missing: no resolved checkpoint, skip")
            continue
        ckpt_full = ROOT / ckpt_rel

        model_wrapper, _ = load_model(ckpt_rel, cfg["config"])
        found = find_text_transformer(model_wrapper.model)
        print(f"  text-transformer candidates: {[(p, cn) for p, _, cn in found]}")
        if not found:
            print("  no text transformer found. skipping.")
            del model_wrapper
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            continue
        _, text_tx, _ = found[-1]
        cache = setup_attention_capture(text_tx)

        batch, ds_tok, tmp_dir = build_one_batch(cfg, target_path_type="left_left")
        tok = ds_tok if ds_tok is not None else tokenizer
        try:
            model_results = {}
            for instr_key, instr in INSTRUCTIONS.items():
                b = rebuild_text_in_batch(batch, tok, instr)
                b = to_device(b, DEVICE)
                if b["text_mask"].ndim >= 3:
                    text_len = int(b["text_mask"][0, 0, :].sum().item())
                else:
                    text_len = int(b["text_mask"][0, :].sum().item())
                attn, seq_len = forward_and_capture(model_wrapper, b, cache)
                if attn is None:
                    print(f"  {instr_key}: NO ATTENTIONS captured")
                    continue
                if instr_key == "left":
                    print(f"  [debug] batch keys: {list(b.keys())}")
                    print(f"  [debug] batch.rgb {tuple(b['rgb'].shape)} text {tuple(b['text'].shape)} mask {tuple(b['text_mask'].shape)}")
                    print(f"  [debug] tx_input_shape {cache['input_shape']} seq_len {seq_len}")
                    text_ids0 = b["text"][0].flatten().tolist()
                    try:
                        dec_full = tok.decode(text_ids0, skip_special_tokens=False)
                        print(f"  [debug] text_ids[:40]: {text_ids0[:40]}")
                        print(f"  [debug] text_ids[-20:]: {text_ids0[-20:]}")
                        print(f"  [debug] decoded[:400]: {dec_full[:400]!r}")
                    except Exception as e:
                        print(f"  [debug] decode error: {e}")
                    # Locate <image> token boundary
                    vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
                    img_ids = {vocab[k] for k in vocab if "image" in k.lower() or "patch_index" in k.lower() or k in ("<image>", "</image>")}
                    img_positions = [i for i, t in enumerate(text_ids0) if t in img_ids]
                    print(f"  [debug] #image-like positions: {len(img_positions)}, first {img_positions[:5]}, last {img_positions[-5:] if img_positions else []}")
                    mask0 = b["text_mask"][0].flatten().tolist()
                    print(f"  [debug] attention_mask_sum {sum(mask0)}")
                layers = analyze_attention(attn, text_len)
                last_layer = layers[-1]
                mean_image = float(np.mean([l["image_ratio_mean"] for l in layers]))
                mean_text = float(np.mean([l["text_ratio_mean"] for l in layers]))
                mean_pad = float(np.mean([l["pad_ratio_mean"] for l in layers]))
                print(f"  {instr_key:8s}  text_len={text_len:3d}  seq={seq_len}  "
                       f"last[img={last_layer['image_ratio_mean']:.3f} text={last_layer['text_ratio_mean']:.3f} pad={last_layer['pad_ratio_mean']:.3f}]  "
                       f"mean[img={mean_image:.3f} text={mean_text:.3f} pad={mean_pad:.3f}]")
                print(f"  [debug] {instr_key:8s} top5: {[(p, round(v,4)) for p,v in last_layer['top_positions'][:5]]}")
                print(f"  [debug] {instr_key:8s} text-region (pos 64~{64+text_len}) attn: {[round(x,5) for x in last_layer['text_region_per_pos']]}")
                if instr_key == "left":
                    print(f"  [debug] cross_attentions present: {cache.get('cross_attentions') is not None}")
                model_results[instr_key] = {
                    "text_len": text_len,
                    "seq_len": seq_len,
                    "per_layer": layers,
                    "summary": {
                        "last_layer_image_ratio": last_layer["image_ratio_mean"],
                        "last_layer_text_ratio": last_layer["text_ratio_mean"],
                        "last_layer_pad_ratio": last_layer["pad_ratio_mean"],
                        "mean_layers_image_ratio": mean_image,
                        "mean_layers_text_ratio": mean_text,
                        "mean_layers_pad_ratio": mean_pad,
                    },
                }
            all_results[name] = model_results
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        del model_wrapper
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    SUMMARY_FILE.write_text(json.dumps(all_results, indent=2))
    build_html(all_results)
    print(f"\nWrote: {SUMMARY_FILE}")
    print(f"Wrote: {HTML_FILE}")


if __name__ == "__main__":
    main()
