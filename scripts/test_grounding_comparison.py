#!/usr/bin/env python3
"""
Grounding model comparison: Kosmos-2 vs PaliGemma vs PaliGemma2 vs Moondream2

Usage:
  # sequential (default)
  python3 scripts/test_grounding_comparison.py --mode sequential

  # parallel — spawns one subprocess per model, runs concurrently
  python3 scripts/test_grounding_comparison.py --mode parallel
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/minum/26CS/MoNaVLA")
_PYTHON = ROOT / ".venv" / "bin" / "python3"
_SELF   = ROOT / "scripts" / "test_grounding_comparison.py"

_DATA_CANDIDATES = [
    Path("/home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v5"),
    Path("/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5"),
    ROOT / "ROS_action" / "mobile_vla_dataset_v5",
]

_KOSMOS_PATH = ROOT / ".vlms" / "kosmos-2-patch14-224"

_MODEL_IDS: dict[str, str] = {
    "kosmos":          str(_KOSMOS_PATH),
    "paligemma-mix":   "google/paligemma-3b-mix-224",
    "paligemma2-mix":  "google/paligemma2-3b-mix-224",
    "moondream":       "vikhyatk/moondream2",
}
_MOONDREAM_REVISION = "2025-01-09"

LABEL_KEYWORDS = ("basket", "box", "container", "bin", "crate")


# ── data helpers ──────────────────────────────────────────────────────────────

def resolve_data_dir(override: Optional[str] = None) -> Path:
    if override:
        return Path(override)
    env = os.getenv("VLA_DATA_DIR")
    if env:
        return Path(env)
    for c in _DATA_CANDIDATES:
        if c.exists():
            return c
    return _DATA_CANDIDATES[-1]


def load_image(data_dir: Path, episode: str, frame_idx: int) -> np.ndarray:
    h5_path = data_dir / f"{episode}.h5"
    with h5py.File(h5_path, "r") as f:
        img = f["observations"]["images"][frame_idx]
    return np.array(img, dtype=np.uint8)


def sample_frames(dataset_path: Path, n: int, seed: int = 42) -> list[dict]:
    data: list[dict] = json.loads(dataset_path.read_text())
    candidates: list[dict] = []
    for ep in data:
        for fr in ep["frames"]:
            if fr["has_bbox"]:
                candidates.append({
                    "episode":    ep["episode"],
                    "path_type":  ep.get("path_type", "unknown"),
                    "frame_idx":  fr["frame_idx"],
                    "kosmos_cx":  fr["cx"],
                    "kosmos_cy":  fr["cy"],
                    "gt_class":   fr["gt_class"],
                })
    random.seed(seed)
    random.shuffle(candidates)
    return candidates[:n]


def _direction(cx: float) -> str:
    if cx < 0.4:
        return "left"
    if cx > 0.6:
        return "right"
    return "center"


# ── Kosmos-2 ──────────────────────────────────────────────────────────────────

def run_kosmos(frames: list[dict], data_dir: Path) -> dict:
    from transformers import AutoProcessor, AutoModelForVision2Seq

    PROMPT = "<grounding>The gray basket is at"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(_MODEL_IDS["kosmos"])
    model = AutoModelForVision2Seq.from_pretrained(
        _MODEL_IDS["kosmos"], torch_dtype=dtype
    ).to(device).eval()
    print(f"[kosmos] model loaded on {device}", flush=True)

    rows: list[dict] = []
    for i, fr in enumerate(frames):
        img_np  = load_image(data_dir, fr["episode"], fr["frame_idx"])
        pil_img = Image.fromarray(img_np).convert("RGB")
        inputs  = processor(text=PROMPT, images=pil_img, return_tensors="pt")
        inputs  = {k: v.to(device) for k, v in inputs.items()}
        pv      = inputs["pixel_values"].to(dtype)

        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(
                pixel_values=pv,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs.get("image_embeds_position_mask"),
                use_cache=True,
                max_new_tokens=64,
            )
        latency = (time.time() - t0) * 1000

        new_ids = gen[:, inputs["input_ids"].shape[1]:]
        raw     = processor.batch_decode(new_ids, skip_special_tokens=False)[0]
        caption, entities = processor.post_process_generation(raw)

        bbox: Optional[dict] = None
        for ename, _span, boxes in entities:
            if boxes:
                x1, y1, x2, y2 = [float(v) for v in boxes[0]]
                if max(x1, y1, x2, y2) > 1.5:
                    x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
                area = (x2 - x1) * (y2 - y1)
                if area < 0.95:
                    bbox = {"entity": ename, "cx": (x1+x2)/2, "cy": (y1+y2)/2, "area": area}
                    break

        label_hit = any(k in caption.lower() for k in LABEL_KEYWORDS)
        rows.append({
            "caption":    caption,
            "detected":   bbox is not None,
            "label_hit":  label_hit,
            "cx":         bbox["cx"] if bbox else None,
            "cy":         bbox["cy"] if bbox else None,
            "latency_ms": latency,
            "kosmos_cx":  fr["kosmos_cx"],
        })
        if (i + 1) % 10 == 0:
            print(f"[kosmos] {i+1}/{len(frames)}", flush=True)

    return _summarize(rows)


# ── PaliGemma (shared for v1 + v2) ───────────────────────────────────────────

def _parse_paligemma_locs(text: str) -> Optional[dict]:
    # PaliGemma outputs <loc_XXXX> tokens in y1,x1,y2,x2 order (0-1023)
    locs = re.findall(r"<loc(\d{4})>", text)
    if len(locs) < 4:
        return None
    y1, x1, y2, x2 = [int(v) / 1023.0 for v in locs[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "cx":   (x1 + x2) / 2,
        "cy":   (y1 + y2) / 2,
        "area": (x2 - x1) * (y2 - y1),
    }


def _run_paligemma_variant(model_key: str, frames: list[dict], data_dir: Path) -> dict:
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    model_id = _MODEL_IDS[model_key]
    PROMPT   = "detect gray basket\n"
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype    = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device).eval()
    print(f"[{model_key}] model loaded on {device}", flush=True)

    rows: list[dict] = []
    any_loc_found = False

    for i, fr in enumerate(frames):
        img_np  = load_image(data_dir, fr["episode"], fr["frame_idx"])
        pil_img = Image.fromarray(img_np).convert("RGB")
        inputs  = processor(text=PROMPT, images=pil_img, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=100)
        latency = (time.time() - t0) * 1000

        full_decoded = processor.decode(gen[0], skip_special_tokens=False)
        # strip everything up to and including the prompt
        prompt_end = full_decoded.find(PROMPT.strip())
        decoded = full_decoded[prompt_end + len(PROMPT):] if prompt_end >= 0 else full_decoded

        bbox = _parse_paligemma_locs(decoded)
        if bbox:
            any_loc_found = True

        label_hit = any(k in decoded.lower() for k in LABEL_KEYWORDS)
        rows.append({
            "caption":    decoded.strip()[:120],
            "detected":   bbox is not None,
            "label_hit":  label_hit,
            "cx":         bbox["cx"] if bbox else None,
            "cy":         bbox["cy"] if bbox else None,
            "latency_ms": latency,
            "kosmos_cx":  fr["kosmos_cx"],
        })
        if (i + 1) % 10 == 0:
            print(f"[{model_key}] {i+1}/{len(frames)}", flush=True)

    summary = _summarize(rows)
    summary["detect_supported"] = any_loc_found
    return summary


def run_paligemma_mix(frames: list[dict], data_dir: Path) -> dict:
    return _run_paligemma_variant("paligemma-mix", frames, data_dir)


def run_paligemma2_mix(frames: list[dict], data_dir: Path) -> dict:
    return _run_paligemma_variant("paligemma2-mix", frames, data_dir)


# ── Moondream2 ────────────────────────────────────────────────────────────────

def run_moondream(frames: list[dict], data_dir: Path) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_IDS["moondream"],
        trust_remote_code=True,
        revision=_MOONDREAM_REVISION,
        torch_dtype=dtype,
    ).to(device).eval()
    print(f"[moondream] model loaded on {device}", flush=True)

    rows: list[dict] = []
    for i, fr in enumerate(frames):
        img_np  = load_image(data_dir, fr["episode"], fr["frame_idx"])
        pil_img = Image.fromarray(img_np).convert("RGB")

        t0 = time.time()
        try:
            with torch.no_grad():
                raw = model.detect(pil_img, "gray basket")
            objects = raw.get("objects", raw) if isinstance(raw, dict) else raw
        except Exception as e:
            print(f"[moondream] detect error frame {i}: {e}", flush=True)
            objects = []
        latency = (time.time() - t0) * 1000

        bbox: Optional[dict] = None
        if objects:
            obj  = objects[0]
            x1   = float(obj.get("x_min", obj.get("xmin", 0)))
            y1   = float(obj.get("y_min", obj.get("ymin", 0)))
            x2   = float(obj.get("x_max", obj.get("xmax", 1)))
            y2   = float(obj.get("y_max", obj.get("ymax", 1)))
            area = abs(x2 - x1) * abs(y2 - y1)
            if area < 0.95:
                bbox = {"cx": (x1+x2)/2, "cy": (y1+y2)/2, "area": area}

        rows.append({
            "caption":    f"{len(objects)} object(s) detected",
            "detected":   bbox is not None,
            "label_hit":  bbox is not None,
            "cx":         bbox["cx"] if bbox else None,
            "cy":         bbox["cy"] if bbox else None,
            "latency_ms": latency,
            "kosmos_cx":  fr["kosmos_cx"],
        })
        if (i + 1) % 10 == 0:
            print(f"[moondream] {i+1}/{len(frames)}", flush=True)

    return _summarize(rows)


# ── metrics ───────────────────────────────────────────────────────────────────

def _summarize(rows: list[dict]) -> dict:
    n        = len(rows)
    detected = [r for r in rows if r["detected"]]
    cxs      = [r["cx"] for r in detected]
    lats     = [r["latency_ms"] for r in rows]

    dir_agree = sum(
        1 for r in detected
        if r["kosmos_cx"] is not None
        and _direction(r["cx"]) == _direction(r["kosmos_cx"])
    )

    return {
        "n_frames":            n,
        "detection_rate":      len(detected) / n if n else 0.0,
        "label_acc":           sum(r["label_hit"] for r in rows) / n if n else 0.0,
        "direction_agreement": dir_agree / len(detected) if detected else 0.0,
        "mean_cx":             float(np.mean(cxs)) if cxs else None,
        "std_cx":              float(np.std(cxs))  if cxs else None,
        "mean_latency_ms":     float(np.mean(lats)),
        "samples": [
            {"caption": r["caption"], "cx": r["cx"], "detected": r["detected"]}
            for r in rows[:5]
        ],
    }


# ── output table ──────────────────────────────────────────────────────────────

def print_table(all_results: dict[str, dict]) -> None:
    print("\n" + "=" * 84)
    print(f"{'Model':<22} {'Detect':>7} {'Label':>7} {'DirAgr':>7} {'MeanCx':>7} {'Latency':>9}")
    print("-" * 84)
    for name, r in all_results.items():
        if "error" in r:
            print(f"{name:<22}  ERROR: {r['error']}")
            continue
        tag = "" if r.get("detect_supported", True) else " (vqa)"
        label = (name + tag)[:22]
        cx_str = f"{r['mean_cx']:.3f}" if r["mean_cx"] is not None else "  N/A"
        print(
            f"{label:<22}"
            f" {r['detection_rate']:>7.1%}"
            f" {r['label_acc']:>7.1%}"
            f" {r['direction_agreement']:>7.1%}"
            f" {cx_str:>7}"
            f" {r['mean_latency_ms']:>8.0f}ms"
        )
    print("=" * 84)
    print("\n[Sample captions — first 5 frames per model]")
    for name, r in all_results.items():
        if "error" in r:
            continue
        print(f"\n  [{name}]")
        for s in r.get("samples", []):
            cx_tag = f"cx={s['cx']:.2f}" if s["cx"] is not None else "no-bbox"
            print(f"    {cx_tag}  {s['caption'][:80]}")


# ── parallel / sequential runners ────────────────────────────────────────────

RUNNERS: dict[str, Any] = {
    "kosmos":         run_kosmos,
    "paligemma-mix":  run_paligemma_mix,
    "paligemma2-mix": run_paligemma2_mix,
    "moondream":      run_moondream,
}


def _run_sequential(models: list[str], frames: list[dict], data_dir: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for m in models:
        print(f"\n{'='*40}\n[{m}] starting...\n{'='*40}", flush=True)
        t0 = time.time()
        results[m] = RUNNERS[m](frames, data_dir)
        print(f"[{m}] done in {time.time()-t0:.1f}s", flush=True)
    return results


def _run_parallel(
    models: list[str],
    frames_file: Path,
    data_dir: Path,
    out_dir: Path,
) -> dict[str, dict]:
    procs: dict[str, tuple] = {}
    out_files: dict[str, Path] = {}

    for m in models:
        out_file  = out_dir / f"result_{m}.json"
        log_file  = out_dir / f"log_{m}.txt"
        out_files[m] = out_file
        cmd = [
            str(_PYTHON), str(_SELF),
            "--worker", m,
            "--frames-file", str(frames_file),
            "--data-dir",    str(data_dir),
            "--worker-out",  str(out_file),
        ]
        env = os.environ.copy()
        env["PYTHONSTARTUP"] = ""        # antigravity startup script 차단
        env["VIRTUAL_ENV"]   = str(ROOT / ".venv")
        procs[m] = (
            subprocess.Popen(
                cmd,
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(ROOT),
            ),
            log_file,
        )
        print(f"[parallel] spawned {m}  →  log: {log_file}", flush=True)

    results: dict[str, dict] = {}
    for m, (proc, log_file) in procs.items():
        ret = proc.wait()
        if ret != 0:
            print(f"[parallel] WARNING: {m} exited {ret}. see {log_file}", flush=True)
            results[m] = {"error": f"subprocess exit {ret}"}
        else:
            results[m] = json.loads(out_files[m].read_text())
            print(f"[parallel] {m} OK", flush=True)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grounding model comparison")
    p.add_argument("--dataset",    default="docs/v5/bbox_nav_step1/bbox_dataset.json")
    p.add_argument("--data-dir",   default=None)
    p.add_argument("--n-frames",   type=int, default=50)
    p.add_argument("--models",     nargs="+", default=list(RUNNERS.keys()),
                   choices=list(RUNNERS.keys()), metavar="MODEL")
    p.add_argument("--output",     default="docs/v5/grounding_comparison/results.json")
    p.add_argument("--mode",       choices=["sequential", "parallel"], default="sequential")
    p.add_argument("--seed",       type=int, default=42)
    # worker-only (used internally by parallel mode)
    p.add_argument("--worker",       default=None, help=argparse.SUPPRESS)
    p.add_argument("--frames-file",  default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-out",   default=None, help=argparse.SUPPRESS)
    return p


def main() -> None:
    args    = build_parser().parse_args()
    data_dir = resolve_data_dir(args.data_dir)

    # ── worker mode (spawned by parallel orchestrator) ───────────────────────
    if args.worker:
        frames = json.loads(Path(args.frames_file).read_text())
        result = RUNNERS[args.worker](frames, data_dir)
        Path(args.worker_out).write_text(json.dumps(result, indent=2))
        return

    # ── orchestrator ─────────────────────────────────────────────────────────
    dataset_path = ROOT / args.dataset
    out_path     = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = sample_frames(dataset_path, args.n_frames, args.seed)
    print(f"Sampled {len(frames)} frames  |  data_dir: {data_dir}  |  mode: {args.mode}")

    if args.mode == "sequential":
        all_results = _run_sequential(args.models, frames, data_dir)
    else:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=out_path.parent
        ) as tmp:
            json.dump(frames, tmp)
            frames_file = Path(tmp.name)
        try:
            all_results = _run_parallel(args.models, frames_file, data_dir, out_path.parent)
        finally:
            frames_file.unlink(missing_ok=True)

    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {out_path}")
    print_table(all_results)


if __name__ == "__main__":
    main()
