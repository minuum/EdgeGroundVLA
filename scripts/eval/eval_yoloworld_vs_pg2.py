# -*- coding: utf-8 -*-
"""
YOLO-World zero-shot vs base PG2 / GroundingDINO — Phase B (plan §10-B).

GroundingDINO(§A 직후 phase)가 OOD 오탐(9%→91~100%)으로 production 교체 부적합
판정을 받은 뒤, "관계형 표현 가능한 다른 open-vocab detector"로 시도하는 두 번째 후보.
역시 fine-tune 없이 frozen zero-shot으로만 비교(§0 철학 유지).

세트는 eval_groundingdino_vs_pg2.py와 동일 — 표준 49프레임 + OOD 11장 +
S6 dead-zone(56/70/85) + S7 near-miss(39/41). 결과는 같은 JSON
(docs/v5/grounding_hub/gdino_vs_pg2.json)에 yoloworld_s05 키로 병합 저장해
한 표에서 base_pg2/gdino_tiny/gdino_base/yoloworld를 같이 비교한다.

conf threshold는 사전 스윕(0.03/0.05/0.1/0.2)에서 0.05를 채택 — hit 92%/OOD_FP 18%로
GroundingDINO(hit 100%/OOD_FP 91~100%)보다 OOD 억제가 훨씬 우수하면서 PG2(hit 98%/OOD_FP 9%)에
근접. latency는 ~34ms로 PG2(599ms)의 약 18배 빠름.

Usage:
  .venv/bin/python3 scripts/eval/eval_yoloworld_vs_pg2.py
  .venv/bin/python3 scripts/eval/eval_yoloworld_vs_pg2.py --conf 0.05
"""
import sys, json, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eval_exp64_grounding import build_basket_sample, CHAIRDIR  # noqa: E402

S6S7 = ROOT / "docs/v5/grounding_frames"
OUT = ROOT / "docs/v5/grounding_hub"
S6_FAILCASE = [56, 70, 85]
S7_FAILCASE = [39, 41]


def make_yoloworld_detect(model, conf):
    def detect(img, phrase="gray basket"):
        t0 = time.time()
        res = model.predict(img, conf=conf, verbose=False)
        lat_ms = (time.time() - t0) * 1000
        r = res[0]
        if len(r.boxes) == 0:
            return {"box": None, "latency_ms": lat_ms}
        i = int(r.boxes.conf.argmax())
        x1, y1, x2, y2 = [float(v) for v in r.boxes.xyxy[i]]
        w, h = img.size
        x1n, y1n, x2n, y2n = x1 / w, y1 / h, x2 / w, y2 / h
        return {"cx": (x1n + x2n) / 2, "area": (x2n - x1n) * (y2n - y1n),
                "box": [x1n, y1n, x2n, y2n], "score": float(r.boxes.conf[i]), "latency_ms": lat_ms}
    return detect


def eval_standard(detect, basket, chairs):
    cxs, errs, areas, hit, full, lats, boxes = [], [], [], 0, 0, [], []
    for s in basket:
        d = detect(s["img"], "gray basket")
        lats.append(d.get("latency_ms", 0))
        if d.get("box") is None:
            boxes.append(None); continue
        hit += 1
        cxs.append(d["cx"]); areas.append(d["area"]); errs.append(abs(d["cx"] - s["cx_ref"]))
        if d["area"] > 0.9:
            full += 1
        boxes.append(d["box"])
    n = len(basket)
    fp, ood_boxes = 0, []
    for _name, img in chairs:
        d = detect(img, "gray basket")
        if d.get("box") is not None:
            fp += 1; ood_boxes.append(d["box"])
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
    }, boxes


def eval_failcases(detect):
    out = {}
    for session, frames in (("s6", S6_FAILCASE), ("s7", S7_FAILCASE)):
        recs = []
        for fn in frames:
            fpath = S6S7 / session / f"frame_{fn:04d}.jpg"
            if not fpath.exists():
                continue
            img = Image.open(fpath).convert("RGB")
            d = detect(img, "gray basket")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-vp", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLOWorld
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(["gray basket"])
    detect = make_yoloworld_detect(model, args.conf)

    basket = build_basket_sample(args.per_vp)
    chairs = [(p.stem, Image.open(p).convert("RGB")) for p in sorted(CHAIRDIR.glob("*.jpg"))]
    print(f"[SAMPLE] basket={len(basket)} chairs={len(chairs)}")

    stat, boxes = eval_standard(detect, basket, chairs)
    stat["failcases"] = eval_failcases(detect)
    key = f"yoloworld_s_conf{args.conf}"
    print(f"  hit={stat['basket_hit']*100:.0f}% cxMAE={stat['cx_mae']:.3f} "
          f"full-frame={stat['fullframe_rate']*100:.0f}% OOD_FP={stat['ood_fp_rate']*100:.0f}% "
          f"lat={stat['latency_ms_mean']:.0f}ms")

    grid(basket[:8], boxes[:8], OUT / f"grid_{key}.png", lambda s, i: f"{s['vp']} {s['tag']}")

    out_path = OUT / "gdino_vs_pg2.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results[key] = stat
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] {out_path} (key={key})")

    print("\n[S6/S7 실패케이스]")
    for sess, recs in stat["failcases"].items():
        parts = [f"f{r['frame']}:{'HIT(a=%.3f,cx=%.3f)' % (r['area'], r['cx']) if r['hit'] else 'MISS'}" for r in recs]
        print(f"  {sess}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
