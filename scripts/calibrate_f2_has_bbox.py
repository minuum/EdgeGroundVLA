#!/usr/bin/env python3
"""EGVLA2 §2.2 백로그: Florence-2 has_bbox 면적 임계값 실측 캘리브레이션.

Florence-2 phrase grounding은 score가 없어 부재 대상도 박스를 낸다. 그래서
  정검출(pos): "gray basket" 박스 중 OWLv2(conf≥0.5)와 cx 차 <0.10 & 면적비 0.4~2.5인 것
  오검출(neg): 화면에 없는 대상 프롬프트(microwave oven 등)가 낸 박스
의 면적/종횡비/가장자리접촉 분포를 비교해, 현재 규칙(0.0008≤area≤0.55)이 둘을 얼마나 가르는지
평가하고 대안 임계값을 탐색한다. 한계: 오검출은 '부재 프롬프트 박스'라는 대리 지표이며
실제 로봇 실패 프레임(바구니가 시야 밖)이 아니다.

출력: docs/v5/detector/f2_has_bbox_calibration.json
실행: HF_HUB_OFFLINE=1 .venv_lerobot/bin/python3 scripts/calibrate_f2_has_bbox.py
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "docs/v5/bbox_nav_owl/bbox_dataset_v6_owl.json"
OUT = ROOT / "docs/v5/detector/f2_has_bbox_calibration.json"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "microsoft/Florence-2-base"
TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
POS_PHRASE = "gray basket"
NEG_PHRASES = ["microwave oven", "zebra", "surfboard", "fire hydrant", "violin"]
N_EP, PER_EP, N_OWL_MISS = 60, 6, 120
BATCH = 8


@torch.no_grad()
def ground(model, proc, pils, phrase):
    inp = proc(text=[TASK + phrase] * len(pils), images=pils, return_tensors="pt")
    out = model.generate(input_ids=inp["input_ids"].to(DEV),
                         pixel_values=inp["pixel_values"].to(DEV, torch.float16),
                         max_new_tokens=128, num_beams=3, do_sample=False)
    res = []
    for t, pil in zip(proc.batch_decode(out, skip_special_tokens=False), pils):
        p = proc.post_process_generation(t, task=TASK, image_size=(pil.width, pil.height))[TASK]
        bx = p.get("bboxes", [])
        if not bx:
            res.append(None)
            continue
        x1, y1, x2, y2 = max(bx, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        W, H = pil.width, pil.height
        res.append(dict(area=(x2 - x1) * (y2 - y1) / (W * H), cx=(x1 + x2) / 2 / W, cy=(y1 + y2) / 2 / H,
                        aspect=(x2 - x1) / max(1e-6, y2 - y1),
                        edge=int(x1 < 0.01 * W) + int(y1 < 0.01 * H) + int(x2 > 0.99 * W) + int(y2 > 0.99 * H)))
    return res


def pick_frames(ds, rng):
    sel = []
    eps = rng.choice(len(ds), N_EP, replace=False)
    for ei in eps:
        fr = ds[ei]["frames"]
        for fi in rng.choice(len(fr), min(PER_EP, len(fr)), replace=False):
            sel.append((ei, int(fi), "random"))
    miss = [(ei, i) for ei, e in enumerate(ds) for i, f in enumerate(e["frames"]) if not f["detected"]]
    for k in rng.choice(len(miss), min(N_OWL_MISS, len(miss)), replace=False):
        sel.append((*miss[k], "owl_miss"))
    return sel


def main():
    from transformers import AutoModelForCausalLM, AutoProcessor
    ds = json.loads(DATASET.read_text())
    rng = np.random.default_rng(0)
    sel = pick_frames(ds, rng)
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    by_ep = {}
    for ei, fi, kind in sel:
        by_ep.setdefault(ei, []).append((fi, kind))
    pils, metas = [], []
    for ei, items in by_ep.items():
        with h5py.File(ds[ei]["episode"], "r") as f:
            imgs = f["observations"]["images"] if "observations" in f else f["images"]
            for fi, kind in items:
                j = ds[ei]["frames"][fi]["frame_idx"]
                pils.append(Image.fromarray(np.ascontiguousarray(imgs[j][:, :, ::-1]).astype(np.uint8)).convert("RGB"))
                metas.append((ei, fi, kind))
    print(f"프레임 {len(pils)}장", flush=True)

    rows = []
    for phrase in [POS_PHRASE] + NEG_PHRASES:
        for s in range(0, len(pils), BATCH):
            for m, r in zip(metas[s:s + BATCH], ground(model, proc, pils[s:s + BATCH], phrase)):
                f = ds[m[0]]["frames"][m[1]]
                rows.append(dict(phrase=phrase, kind=m[2], box=r, owl_det=f["detected"], owl_conf=f["confidence"],
                                 owl_cx=f["cx_det"], owl_area=f["area_det"]))
        print(f"  {phrase} 완료", flush=True)

    pos, neg, pos_miss = [], [], []
    for r in rows:
        b = r["box"]
        if b is None:
            continue
        if r["phrase"] == POS_PHRASE:
            if r["owl_det"] and r["owl_conf"] >= 0.5 and abs(b["cx"] - r["owl_cx"]) < 0.10 \
                    and 0.4 <= b["area"] / max(1e-6, r["owl_area"]) <= 2.5:
                pos.append(b)
            elif r["kind"] == "owl_miss":
                pos_miss.append(b)
        else:
            neg.append(b)

    def stat(bs, k):
        a = np.array([b[k] for b in bs])
        return dict(n=len(a), mean=float(a.mean()), p5=float(np.percentile(a, 5)), p50=float(np.percentile(a, 50)),
                    p95=float(np.percentile(a, 95))) if len(a) else dict(n=0)

    def rate(bs, lo, hi):
        return float(np.mean([lo <= b["area"] <= hi for b in bs])) if bs else None

    def keep(b, lo, hi, edge_max=9, asp_max=99.0):
        return lo <= b["area"] <= hi and b["edge"] <= edge_max and b["aspect"] <= asp_max

    rules = {}
    for name, kw in {
        "현행 면적(0.0008~0.55)": dict(lo=0.0008, hi=0.55),
        "면적+가장자리접촉<=1": dict(lo=0.0008, hi=0.55, edge_max=1),
        "면적+가장자리접촉==0": dict(lo=0.0008, hi=0.55, edge_max=0),
        "면적(0.008~0.35)+접촉<=1": dict(lo=0.008, hi=0.35, edge_max=1),
        "면적(0.008~0.35)+접촉<=1+종횡비<=2.0": dict(lo=0.008, hi=0.35, edge_max=1, asp_max=2.0),
        "면적(0.008~0.25)+접촉<=1": dict(lo=0.008, hi=0.25, edge_max=1),
    }.items():
        tpr = float(np.mean([keep(b, **kw) for b in pos])); fpr = float(np.mean([keep(b, **kw) for b in neg]))
        rules[name] = dict(pos_keep=tpr, neg_keep=fpr, bal_acc=(tpr + 1 - fpr) / 2,
                           owl_miss_keep=float(np.mean([keep(b, **kw) for b in pos_miss])))
    cur = (0.0008, 0.55)
    grid = []
    for lo in [0.0008, 0.002, 0.004, 0.008, 0.015]:
        for hi in [0.15, 0.25, 0.35, 0.55, 0.8]:
            tpr, fpr = rate(pos, lo, hi), rate(neg, lo, hi)
            grid.append(dict(lo=lo, hi=hi, pos_keep=tpr, neg_keep=fpr, bal_acc=(tpr + (1 - fpr)) / 2))
    grid.sort(key=lambda g: -g["bal_acc"])
    out = dict(
        n_frames=len(pils), definitions="pos=OWLv2 일치 gray basket 박스, neg=부재 프롬프트 박스(대리 지표)",
        counts=dict(pos=len(pos), neg=len(neg), f2_box_on_owl_miss=len(pos_miss)),
        area={"pos": stat(pos, "area"), "neg": stat(neg, "area"), "owl_miss_f2": stat(pos_miss, "area")},
        aspect={"pos": stat(pos, "aspect"), "neg": stat(neg, "aspect")},
        edge_touch_ge2={"pos": float(np.mean([b["edge"] >= 2 for b in pos])) if pos else None,
                        "neg": float(np.mean([b["edge"] >= 2 for b in neg])) if neg else None},
        current_rule=dict(lo=cur[0], hi=cur[1], pos_keep=rate(pos, *cur), neg_keep=rate(neg, *cur),
                          owl_miss_keep=rate(pos_miss, *cur)),
        grid_top5=grid[:5], combined_rules=rules,
    )
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
