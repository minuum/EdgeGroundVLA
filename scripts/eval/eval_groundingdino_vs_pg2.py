# -*- coding: utf-8 -*-
"""
GroundingDINO(tiny/base) zero-shot vs base PaliGemma2 zero-shot 그라운딩 비교.

목적: 둘 다 fine-tune 없이(frozen zero-shot) 동일 세트에서 비교 — "모델 교체
효과"만 분리해서 본다. (LoRA가 PG2 grounding을 깨뜨린다는 grounding_hub.html
결론과 같은 철학 — detector는 건드리지 않고 그대로 쓴다.)

세트 (전부 기존 자산 재사용, 신규 라벨링 없음):
  1. 표준 시점 49프레임 (build_basket_sample) — hit/cx_MAE/cx_std/full-frame
  2. OOD 의자 11장 — 오탐률
  3. 측면 4경로(left_left/right_right/left_right/right_left) — 꺾임 구간 추적
  4. S6 dead-zone 3프레임(56/70/85) + S7 near-miss 2프레임(39/41)
     — PG2 실패가 이미 확인된 케이스. 자동 GT 비교 없이 raw bbox만 기록
       (grounding-session-pipeline 스킬 규칙: 반드시 사람이 이미지로 스폿체크)

산출: docs/v5/grounding_hub/gdino_vs_pg2.json
      docs/v5/grounding_hub/grid_gdino_tiny.png, grid_gdino_base.png

Usage:
  .venv/bin/python3 scripts/eval/eval_groundingdino_vs_pg2.py
  .venv/bin/python3 scripts/eval/eval_groundingdino_vs_pg2.py --no-base
"""
import sys, re, json, time, argparse, gc, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
import h5py
from PIL import Image, ImageDraw, ImageFont

# GB10(Grace-Blackwell)의 nvrtc가 일부 reduction(.prod()) JIT 커널의
# --gpu-architecture를 인식 못 해 RuntimeError를 낸다. spatial_shapes.prod()처럼
# 텐서가 작을 때만 CPU로 내려 계산 — GroundingDINO 인코더가 이 경로를 씀.
_orig_prod = torch.Tensor.prod
def _safe_prod(self, *a, **kw):
    if self.is_cuda and self.numel() < 4096:
        return _orig_prod(self.cpu(), *a, **kw).to(self.device)
    return _orig_prod(self, *a, **kw)
torch.Tensor.prod = _safe_prod

from eval_exp64_grounding import build_basket_sample, CHAIRDIR  # noqa: E402

HUB_CACHE = Path.home() / ".cache/huggingface/hub"
PG2 = HUB_CACHE / "models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
LOC = re.compile(r"<loc(\d{4})>")
DSET = ROOT / "ROS_action/mobile_vla_dataset_v5"
S6S7 = ROOT / "docs/v5/grounding_frames"
OUT = ROOT / "docs/v5/grounding_hub"

S6_FAILCASE = [56, 70, 85]   # dead-zone — 정지장면 풀프레임 환각
S7_FAILCASE = [39, 41]       # near-miss — 빈 바닥 박싱
PATH_TYPES = ["left_left", "right_right", "left_right", "right_left"]


# ── 검출기 래퍼 ──────────────────────────────────────────────────────────
def make_pg_detect(model, proc):
    @torch.no_grad()
    def detect(img, phrase="gray basket"):
        inp = proc(text=f"<image>detect {phrase}", images=img, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        t0 = time.time()
        gen = model.generate(**inp, max_new_tokens=48, do_sample=False)
        lat_ms = (time.time() - t0) * 1000
        raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        locs = [int(v) / 1023 for v in LOC.findall(raw)]
        if len(locs) >= 4:
            y1, x1, y2, x2 = locs[:4]
            return {"cx": (x1 + x2) / 2, "area": (x2 - x1) * (y2 - y1),
                    "box": [x1, y1, x2, y2], "score": None, "latency_ms": lat_ms}
        return {"box": None, "latency_ms": lat_ms}
    return detect


def make_gdino_detect(model, proc, box_thr=0.35, text_thr=0.25):
    @torch.no_grad()
    def detect(img, phrase="gray basket"):
        text = phrase.strip().lower()
        if not text.endswith("."):
            text += "."
        inp = proc(images=img, text=text, return_tensors="pt").to("cuda")
        t0 = time.time()
        out = model(**inp)
        lat_ms = (time.time() - t0) * 1000
        w, h = img.size
        res = proc.post_process_grounded_object_detection(
            out, inp["input_ids"], threshold=box_thr, text_threshold=text_thr,
            target_sizes=[(h, w)],
        )[0]
        if len(res["boxes"]) == 0:
            return {"box": None, "latency_ms": lat_ms}
        i = int(res["scores"].argmax())
        x1, y1, x2, y2 = [float(v) for v in res["boxes"][i]]
        x1n, y1n, x2n, y2n = x1 / w, y1 / h, x2 / w, y2 / h
        return {"cx": (x1n + x2n) / 2, "area": (x2n - x1n) * (y2n - y1n),
                "box": [x1n, y1n, x2n, y2n], "score": float(res["scores"][i]), "latency_ms": lat_ms}
    return detect


# ── 평가 ────────────────────────────────────────────────────────────────
def eval_standard(detect, basket, chairs):
    cxs, errs, areas, hit, full, lats, boxes = [], [], [], 0, 0, [], []
    for s in basket:
        try:
            d = detect(s["img"], "gray basket")
        except Exception as e:
            d = {"box": None, "latency_ms": 0, "_err": str(e)[:120]}
        lats.append(d.get("latency_ms", 0))
        if d.get("box") is None:
            boxes.append(None)
            continue
        hit += 1
        cxs.append(d["cx"]); areas.append(d["area"]); errs.append(abs(d["cx"] - s["cx_ref"]))
        if d["area"] > 0.9:
            full += 1
        boxes.append(d["box"])
    n = len(basket)

    fp, ood_boxes = 0, []
    for _name, img in chairs:
        try:
            d = detect(img, "gray basket")
        except Exception:
            d = {"box": None}
        if d.get("box") is not None:
            fp += 1
            ood_boxes.append(d["box"])
        else:
            ood_boxes.append(None)

    return {
        "basket_hit": hit / max(n, 1), "basket_n": n,
        "cx_mae": float(np.mean(errs)) if errs else None,
        "cx_std": float(np.std(cxs)) if cxs else None,
        "fullframe_rate": full / max(n, 1),
        "ood_fp_rate": fp / max(len(chairs), 1), "ood_fp": fp, "ood_n": len(chairs),
        "latency_ms_mean": float(np.mean(lats)) if lats else None,
        "latency_ms_p50": float(np.median(lats)) if lats else None,
    }, boxes, ood_boxes


def eval_sideangle(detect, n_ep=4, n_fr=6):
    agg = {}
    for pt in PATH_TYPES:
        pattern = f"*{pt}_path__core__fixed_center.h5"
        eps = sorted(glob.glob(str(DSET / pattern)))[:n_ep]
        hit, full, total = 0, 0, 0
        for ep in eps:
            with h5py.File(ep, "r") as f:
                imgs = f["observations"]["images"][:]
            idxs = np.linspace(0, len(imgs) - 1, n_fr).astype(int)
            for idx in idxs:
                img = Image.fromarray(imgs[idx].astype("uint8")).convert("RGB")
                try:
                    d = detect(img, "gray basket")
                except Exception:
                    d = {"box": None}
                total += 1
                if d.get("box") is not None:
                    hit += 1
                    if d["area"] > 0.9:
                        full += 1
        agg[pt] = {"n_ep": len(eps), "n_frames": total,
                   "hit_rate": round(hit / max(total, 1), 3),
                   "fullframe_rate": round(full / max(total, 1), 3)}
    return agg


def eval_failcases(detect):
    """S6 dead-zone + S7 near-miss — 자동 GT 비교 없이 raw bbox만 기록 (사람 스폿체크용)."""
    out = {}
    for session, frames in (("s6", S6_FAILCASE), ("s7", S7_FAILCASE)):
        recs = []
        for fn in frames:
            fpath = S6S7 / session / f"frame_{fn:04d}.jpg"
            if not fpath.exists():
                continue
            img = Image.open(fpath).convert("RGB")
            try:
                d = detect(img, "gray basket")
            except Exception as e:
                d = {"box": None, "_err": str(e)[:120]}
            recs.append({"frame": fn, "hit": d.get("box") is not None,
                        "cx": d.get("cx"), "area": d.get("area"),
                        "score": d.get("score"), "box": d.get("box")})
        out[session] = recs
    return out


def grid(samples, boxes, path, title_each):
    cols = min(4, len(samples)); rows = (len(samples) + cols - 1) // cols
    cell, hdr = 220, 20
    cv = Image.new("RGB", (cols * cell, rows * (cell + hdr)), (15, 22, 36))
    dr = ImageDraw.Draw(cv)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    for i, s in enumerate(samples):
        r, c = divmod(i, cols)
        th = s["img"].copy(); th.thumbnail((cell - 8, cell - 8))
        ox, oy = c * cell + 4, r * (cell + hdr) + hdr
        cv.paste(th, (ox, oy))
        if boxes[i]:
            x1, y1, x2, y2 = boxes[i]
            full = (x2 - x1) * (y2 - y1) > 0.9
            col = (239, 68, 68) if full else (34, 197, 94)
            dr.rectangle([ox + x1 * th.width, oy + y1 * th.height,
                         ox + x2 * th.width, oy + y2 * th.height], outline=col, width=2)
        dr.text((ox, r * (cell + hdr) + 4), title_each(s, i)[:30], fill=(148, 163, 184), font=font)
    cv.save(path)


MODELS = [
    ("gdino_tiny", "IDEA-Research/grounding-dino-tiny"),
    ("gdino_base", "IDEA-Research/grounding-dino-base"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-vp", type=int, default=6)
    ap.add_argument("--no-base", action="store_true", help="grounding-dino-base 스킵")
    ap.add_argument("--no-pg2", action="store_true", help="base PG2 재평가 스킵(기존 JSON 보존)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    basket = build_basket_sample(args.per_vp)
    chairs = [(p.stem, Image.open(p).convert("RGB")) for p in sorted(CHAIRDIR.glob("*.jpg"))]
    print(f"[SAMPLE] basket={len(basket)} chairs={len(chairs)}\n")

    results = {}
    out_path = OUT / "gdino_vs_pg2.json"
    if out_path.exists():
        results = json.loads(out_path.read_text())

    # ── base PG2 (기준선) ───────────────────────────────────────────────
    if not args.no_pg2:
        print("[base_pg2] 로드...")
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
        proc = PaliGemmaProcessor.from_pretrained(str(PG2))
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            str(PG2), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map={"": "cuda"}).eval()
        detect = make_pg_detect(model, proc)

        stat, boxes, _ = eval_standard(detect, basket, chairs)
        stat["sideangle"] = eval_sideangle(detect)
        stat["failcases"] = eval_failcases(detect)
        results["base_pg2"] = stat
        grid(basket[:8], boxes[:8], OUT / "grid_base_pg2_gdino_compare.png",
             lambda s, i: f"{s['vp']} {s['tag']}")
        print(f"  hit={stat['basket_hit']*100:.0f}% cxMAE={stat['cx_mae']:.3f} "
              f"full-frame={stat['fullframe_rate']*100:.0f}% OOD_FP={stat['ood_fp_rate']*100:.0f}% "
              f"lat={stat['latency_ms_mean']:.0f}ms\n")
        del model, detect; gc.collect(); torch.cuda.empty_cache()

    # ── GroundingDINO tiny / base ───────────────────────────────────────
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    for key, repo in MODELS:
        if key == "gdino_base" and args.no_base:
            continue
        print(f"[{key}] 로드 ({repo})...")
        proc = AutoProcessor.from_pretrained(repo)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            repo, device_map={"": "cuda"}).eval()
        detect = make_gdino_detect(model, proc)

        stat, boxes, _ = eval_standard(detect, basket, chairs)
        stat["sideangle"] = eval_sideangle(detect)
        stat["failcases"] = eval_failcases(detect)
        results[key] = stat
        grid(basket[:8], boxes[:8], OUT / f"grid_{key}.png",
             lambda s, i: f"{s['vp']} {s['tag']}")
        print(f"  hit={stat['basket_hit']*100:.0f}% cxMAE={stat['cx_mae'] and round(stat['cx_mae'],3)} "
              f"full-frame={stat['fullframe_rate']*100:.0f}% OOD_FP={stat['ood_fp_rate']*100:.0f}% "
              f"lat={stat['latency_ms_mean']:.0f}ms\n")
        del model, detect; gc.collect(); torch.cuda.empty_cache()

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("=" * 78)
    print(f"{'model':<14}{'hit':>7}{'cxMAE':>9}{'full-frame':>12}{'OOD_FP':>9}{'lat(ms)':>10}")
    print("-" * 78)
    for key in ["base_pg2", "gdino_tiny", "gdino_base"]:
        r = results.get(key)
        if not r:
            continue
        mae = f"{r['cx_mae']:.3f}" if r.get('cx_mae') is not None else "  -"
        print(f"{key:<14}{r['basket_hit']*100:>6.0f}%{mae:>9}"
              f"{r['fullframe_rate']*100:>11.0f}%{r['ood_fp_rate']*100:>8.0f}%"
              f"{r['latency_ms_mean']:>9.0f}")
    print(f"\n[SAVE] {out_path}")
    print("\n[S6/S7 실패케이스 — 사람 스폿체크 필요]")
    for key in ["base_pg2", "gdino_tiny", "gdino_base"]:
        r = results.get(key)
        if not r:
            continue
        fc = r.get("failcases", {})
        for sess, recs in fc.items():
            parts = []
            for rec in recs:
                if rec["hit"]:
                    parts.append(f"f{rec['frame']}:HIT(a={rec['area']:.3f},cx={rec['cx']:.3f})")
                else:
                    parts.append(f"f{rec['frame']}:MISS")
            print(f"  [{key}] {sess}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
