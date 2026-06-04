#!/usr/bin/env python3
"""
Kosmos-2 / PaliGemma 실시간 그라운딩 데모 — 로봇서버 배포용

카메라/H5 프레임 → VLM Grounding → bbox 오버레이 저장
로컬 HTTP 서버(7860)로 결과 실시간 확인 가능.

Usage:
  # H5 에피소드 처리 (Kosmos-2 기본)
  python3 run_grounding_realtime.py --source /path/to/episode.h5

  # Exp57 PaliGemma LoRA 사용 (백본 자동 감지)
  python3 run_grounding_realtime.py --source /path/to/episode.h5 \
      --adapter exp57

  # 여러 phrase 비교 (R2-3 데모)
  python3 run_grounding_realtime.py --source /path/to/episode.h5 \
      --adapter exp57 --phrases "gray basket" "red ball" "person"

  # Exp56 Kosmos-2 adapter 사용
  python3 run_grounding_realtime.py --source /path/to/episode.h5 \
      --adapter exp56

  # 결과 HTTP 서버
  python3 run_grounding_realtime.py --source /path/to/episode.h5 --serve
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from io import BytesIO
import base64
import threading

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ─── 설정 ──────────────────────────────────────────────────────────────────

DEFAULT_VLM         = ROOT / ".vlms" / "kosmos-2-patch14-224"
PALIGEMMA_VLM       = Path("/home/minum/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c")
DEFAULT_ADAPTERS = {
    "old":   ROOT / "docs" / "v5" / "bbox_nav_step1" / "grounding_lora",
    "exp56": ROOT / "runs" / "v5_nav" / "grounding" / "exp56",
    "exp57": ROOT / "runs" / "v5_nav" / "grounding" / "exp57",   # PaliGemma LoRA
}
# exp57는 PaliGemma backbone 필요
PALIGEMMA_ADAPTERS = {"exp57"}
OUT_DIR = ROOT / "docs" / "v5" / "grounding_demo" / "realtime_test"

BASKET_KW = ("basket", "gray box", "container", "bin", "laundry")
COLOR_TABLE = {
    "gray basket": (0,   220,  80),   # green
    "red ball":    (220,  50,  50),   # red
    "person":      (50,  150, 255),   # blue
    "white wall":  (200, 200, 200),   # gray
}


def load_frames_from_h5(h5_path: Path) -> list[np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        if "observations" in f and "images" in f["observations"]:
            imgs = f["observations"]["images"][:]
        else:
            imgs = f["images"][:]
    return [imgs[i] for i in range(len(imgs))]


def load_model(vlm_path: Path, adapter_path: Path | None, device: torch.device):
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(str(vlm_path))

    # Jetson(aarch64) bitsandbytes 4bit는 uint8 텐서 지원 안 함 → fp16으로 로드
    import platform as _platform
    _is_arm = _platform.machine().startswith("aarch")
    if device.type == "cuda" and not _is_arm:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"[LOAD] Kosmos-2 from {vlm_path} (4bit quantized)")
        model = AutoModelForVision2Seq.from_pretrained(
            str(vlm_path),
            quantization_config=bnb_config,
            device_map={"": device}
        )
    elif device.type == "cuda":
        print(f"[LOAD] Kosmos-2 from {vlm_path} (fp16, Jetson ARM)")
        model = AutoModelForVision2Seq.from_pretrained(
            str(vlm_path), torch_dtype=torch.float16
        ).to(device)
    else:
        print(f"[LOAD] Kosmos-2 from {vlm_path} (fp32, CPU)")
        model = AutoModelForVision2Seq.from_pretrained(
            str(vlm_path), torch_dtype=torch.float32
        ).to(device)

    if adapter_path is not None and adapter_path.exists():
        print(f"[LOAD] Adapter from {adapter_path}")
        try:
            model = PeftModel.from_pretrained(model, str(adapter_path))
        except TypeError as e:
            # PEFT 버전 불일치: adapter_config.json에 알 수 없는 필드 있을 때 임시 제거 후 재시도
            import json, shutil, tempfile
            cfg_path = adapter_path / "adapter_config.json"
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text())
                unknown_key = str(e).split("'")[1] if "'" in str(e) else None
                if unknown_key and unknown_key in cfg:
                    print(f"[LOAD] PEFT 버전 불일치 — '{unknown_key}' 필드 무시하고 재시도")
                    with tempfile.TemporaryDirectory() as tmp:
                        shutil.copytree(str(adapter_path), tmp + "/adapter", dirs_exist_ok=True)
                        tmp_cfg = json.loads((Path(tmp) / "adapter" / "adapter_config.json").read_text())
                        tmp_cfg.pop(unknown_key, None)
                        (Path(tmp) / "adapter" / "adapter_config.json").write_text(
                            json.dumps(tmp_cfg, indent=2)
                        )
                        model = PeftModel.from_pretrained(model, tmp + "/adapter")
                else:
                    raise
            else:
                raise
    else:
        print("[LOAD] No adapter — pure Kosmos-2")

    model.eval()
    return model, processor


def load_paligemma_model(vlm_path: Path, adapter_path: Path | None, device: torch.device):
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    from peft import PeftModel

    print(f"[LOAD] PaliGemma from {vlm_path}")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = PaliGemmaProcessor.from_pretrained(str(vlm_path))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(vlm_path), torch_dtype=dtype
    ).to(device)

    if adapter_path is not None and adapter_path.exists():
        print(f"[LOAD] PaliGemma Adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    else:
        print("[LOAD] No adapter — pure PaliGemma")

    model.eval()
    return model, processor


@torch.no_grad()
def ground(model, processor, image: np.ndarray, phrase: str, device) -> dict:
    """이미지 + 문자열 → bbox + 캡션 반환"""
    pil = Image.fromarray(image).convert("RGB")
    prompt = f"<grounding><phrase>{phrase}</phrase>"
    inp = processor(text=prompt, images=pil, return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    inp["pixel_values"] = inp["pixel_values"].to(
        torch.float16 if device.type == "cuda" else torch.float32
    )
    gen = model.generate(
        pixel_values=inp["pixel_values"],
        input_ids=inp["input_ids"],
        attention_mask=inp["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inp.get("image_embeds_position_mask"),
        use_cache=True,
        max_new_tokens=64,
    )
    new_ids = gen[:, inp["input_ids"].shape[1]:]
    raw = processor.batch_decode(new_ids, skip_special_tokens=False)[0]
    caption, entities = processor.post_process_generation(raw)

    boxes = []
    for ent_tuple in entities:
        # Kosmos-2 포맷: (entity_text, token_span, [(x1,y1,x2,y2), ...])
        # 또는 일부 버전: (entity_text, [(x1,y1,x2,y2), ...], token_span)
        ent_name = ent_tuple[0]
        ent_boxes = []
        for item in ent_tuple[1:]:
            if not item:
                continue
            if isinstance(item, (list, tuple)) and isinstance(item[0], (list, tuple)):
                # [(x1,y1,x2,y2), ...] 형태
                ent_boxes = list(item)
                break
            if isinstance(item, (list, tuple)) and len(item) == 4:
                first = item[0]
                if isinstance(first, (int, float)):
                    # 단일 flat box (x1,y1,x2,y2)
                    ent_boxes = [item]
                    break
        for box in ent_boxes:
            boxes.append({
                "entity": ent_name,
                "bbox_xyxy": [float(v) for v in box],
            })
    return {"caption": caption, "entities": entities, "boxes": boxes}


@torch.no_grad()
def ground_paligemma(model, processor, image: np.ndarray, phrase: str, device) -> dict:
    """PaliGemma detect 추론: detect <phrase> → <loc####>×4 or <eos>"""
    import re
    pil = Image.fromarray(image).convert("RGB")
    prompt = f"detect {phrase}"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    inp = processor(text=prompt, images=pil, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    gen = model.generate(**inp, max_new_tokens=64, do_sample=False)
    new_ids = gen[:, inp["input_ids"].shape[1]:]
    raw = processor.batch_decode(new_ids, skip_special_tokens=False)[0]

    loc_tokens = re.findall(r"<loc(\d{4})>", raw)
    boxes = []
    if len(loc_tokens) >= 4:
        for i in range(0, len(loc_tokens) - 3, 5):
            y1, x1, y2, x2 = [int(loc_tokens[i+j]) / 1023.0 for j in range(4)]
            boxes.append({
                "entity": phrase,
                "bbox_xyxy": [x1, y1, x2, y2],  # convert to xyxy normalized
            })
    return {"caption": raw.strip(), "entities": [], "boxes": boxes}


def draw_overlay(image: np.ndarray, phrase: str, result: dict) -> Image.Image:
    """bbox 오버레이를 그린 PIL Image 반환"""
    pil = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil)
    W, H = pil.size
    color = COLOR_TABLE.get(phrase.lower(), (255, 200, 0))

    for box_info in result.get("boxes", []):
        x1, y1, x2, y2 = box_info["bbox_xyxy"]
        px1 = int(x1 * W); py1 = int(y1 * H)
        px2 = int(x2 * W); py2 = int(y2 * H)
        draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
        label = f"{phrase}: {box_info['entity']}"
        draw.rectangle([px1, py1 - 18, px1 + len(label) * 7, py1], fill=color)
        draw.text((px1 + 2, py1 - 17), label, fill=(0, 0, 0))

    if not result.get("boxes"):
        draw.text((10, 10), f"[NO BBOX] {phrase}", fill=color)

    # 캡션 요약 하단
    cap = result.get("caption", "")[:80]
    draw.rectangle([0, H - 22, W, H], fill=(0, 0, 0))
    draw.text((5, H - 20), cap, fill=(255, 255, 255))
    return pil


def check_hit(result: dict, phrase: str) -> bool:
    """basket 관련 phrase면 entity match 확인, 아니면 bbox 존재 여부"""
    if "basket" in phrase.lower():
        return any(
            any(k in ent.lower() for k in BASKET_KW)
            for ent, _, _ in result.get("entities", [])
        )
    return len(result.get("boxes", [])) > 0


def make_html_page(out_dir: Path, latest_paths: dict[str, Path]) -> str:
    items = ""
    for phrase, img_path in latest_paths.items():
        rel = img_path.relative_to(out_dir.parent)
        items += f"""
        <div style="display:inline-block;margin:8px;text-align:center">
          <p style="color:#0f0;font-size:12px">{phrase}</p>
          <img src="{rel}" style="width:320px;border:2px solid #333">
        </div>"""
    return f"""<!DOCTYPE html>
<html>
<head>
  <title>Grounding Demo — Live</title>
  <meta http-equiv="refresh" content="2">
  <style>body{{background:#111;font-family:monospace;color:#eee}}</style>
</head>
<body>
  <h2 style="color:#0f0">Kosmos-2 Grounding — Live Feed</h2>
  <p>Auto-refresh every 2s</p>
  {items}
</body>
</html>"""


def start_http_server(out_dir: Path, port: int = 7860):
    import http.server, socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(out_dir.parent), **kwargs)
        def log_message(self, *args): pass  # suppress logs

    def run():
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"[SERVER] http://0.0.0.0:{port}/realtime_test/live.html")
            httpd.serve_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()


def process_episode(
    frames: list[np.ndarray],
    phrases: list[str],
    model,
    processor,
    device,
    out_dir: Path,
    fps_limit: float = 2.0,
    max_frames: int = 0,
    ground_fn=None,
) -> dict:
    if ground_fn is None:
        ground_fn = ground
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {p: {"hits": 0, "total": 0} for p in phrases}
    latest_paths = {}

    total = len(frames) if max_frames == 0 else min(max_frames, len(frames))
    interval = 1.0 / fps_limit

    print(f"\n{'='*60}")
    print(f"Processing {total} frames / {len(phrases)} phrase(s)")
    print(f"{'='*60}")

    for idx in range(total):
        frame = frames[idx]
        t0 = time.time()

        row_imgs = []
        for phrase in phrases:
            result = ground_fn(model, processor, frame, phrase, device)
            hit = check_hit(result, phrase)
            stats[phrase]["total"] += 1
            if hit:
                stats[phrase]["hits"] += 1

            overlay = draw_overlay(frame, phrase, result)
            row_imgs.append(overlay)
            latest_paths[phrase] = out_dir / f"latest_{phrase.replace(' ','_')}.jpg"
            overlay.save(str(latest_paths[phrase]))

        # 여러 phrase 나란히
        combined = Image.new("RGB", (320 * len(phrases), 240))
        for i, img in enumerate(row_imgs):
            combined.paste(img.resize((320, 240)), (i * 320, 0))

        frame_path = out_dir / f"frame_{idx:04d}.jpg"
        combined.save(str(frame_path))

        elapsed = time.time() - t0
        hit_str = " | ".join(
            f"{p}={'✅' if check_hit(ground_fn(model, processor, frame, p, device), p) else '❌'}"
            for p in phrases
        ) if len(phrases) == 1 else " | ".join(
            f"{p.split()[0]}={'HIT' if stats[p]['hits'] > 0 and stats[p]['total'] > 0 else 'MISS'}"
            for p in phrases
        )
        print(f"  [{idx+1:3d}/{total}] {elapsed:.1f}s  {hit_str}")

        # 라이브 HTML 업데이트
        html = make_html_page(out_dir, latest_paths)
        with open(out_dir / "live.html", "w") as f:
            f.write(html)

        # FPS 제한
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Kosmos-2 Real-time Grounding Demo")
    parser.add_argument("--source",    required=True,
                        help="H5 파일 경로 (episode_xxx.h5)")
    parser.add_argument("--phrases",   nargs="+",
                        default=["gray basket"],
                        help="테스트할 phrase 목록. 기본: 'gray basket'")
    parser.add_argument("--adapter",   default="",
                        help="LoRA adapter 경로. 'old'=72frame / 'exp56'=Exp56(Kosmos-2) / 'exp57'=Exp57(PaliGemma) / 경로 직접")
    parser.add_argument("--vlm-path",  default=str(DEFAULT_VLM))
    parser.add_argument("--fps",       type=float, default=1.0,
                        help="목표 FPS (1.0 권장 — Kosmos-2 추론 시간 고려)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="최대 처리 프레임 수 (0=전체)")
    parser.add_argument("--serve",     action="store_true",
                        help="HTTP 서버 시작 (port 7860)")
    parser.add_argument("--out-dir",   default=str(OUT_DIR))
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device    = torch.device(args.device)
    out_dir   = Path(args.out_dir)
    h5_path   = Path(args.source)
    vlm_path  = Path(args.vlm_path)

    # adapter 경로 해석 + 백본 자동감지
    adapter_path = None
    use_paligemma = False
    if args.adapter:
        if args.adapter in DEFAULT_ADAPTERS:
            adapter_path = DEFAULT_ADAPTERS[args.adapter]
            use_paligemma = args.adapter in PALIGEMMA_ADAPTERS
        else:
            adapter_path = Path(args.adapter)
            use_paligemma = (adapter_path / "adapter_config.json").exists() and \
                "paligemma" in open(adapter_path / "adapter_config.json").read().lower()

    backbone_name = "PaliGemma" if use_paligemma else "Kosmos-2"
    if use_paligemma:
        vlm_path = PALIGEMMA_VLM

    print("=" * 60)
    print(f"{backbone_name} Grounding Demo")
    print(f"  Source  : {h5_path.name}")
    print(f"  Phrases : {args.phrases}")
    print(f"  Adapter : {adapter_path or f'pure {backbone_name}'}")
    print(f"  Device  : {device}")
    print(f"  Out dir : {out_dir}")
    print("=" * 60)

    # 프레임 로딩
    print(f"\nLoading frames from {h5_path} ...")
    frames = load_frames_from_h5(h5_path)
    print(f"  Loaded {len(frames)} frames  ({frames[0].shape})")

    # 모델 로딩
    if use_paligemma:
        model, processor = load_paligemma_model(vlm_path, adapter_path, device)
        _ground_fn = lambda m, p, img, phrase, dev: ground_paligemma(m, p, img, phrase, dev)
    else:
        model, processor = load_model(vlm_path, adapter_path, device)
        _ground_fn = lambda m, p, img, phrase, dev: ground(m, p, img, phrase, dev)

    # HTTP 서버
    if args.serve:
        start_http_server(out_dir)

    # 처리
    stats = process_episode(
        frames, args.phrases, model, processor, device,
        out_dir, fps_limit=args.fps, max_frames=args.max_frames,
        ground_fn=_ground_fn,
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    for phrase, s in stats.items():
        hit_rate = s["hits"] / max(s["total"], 1)
        print(f"  '{phrase}': {s['hits']}/{s['total']} = {hit_rate*100:.1f}%")

    result_path = out_dir / "results.json"
    with open(result_path, "w") as f:
        json.dump({
            "source": str(h5_path),
            "phrases": args.phrases,
            "adapter": str(adapter_path),
            "stats": stats,
        }, f, indent=2)
    print(f"\nResults → {result_path}")
    print(f"Frames  → {out_dir}/frame_*.jpg")
    if args.serve:
        print(f"Live    → http://0.0.0.0:7860/realtime_test/live.html")


if __name__ == "__main__":
    main()
