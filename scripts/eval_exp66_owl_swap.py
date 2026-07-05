#!/usr/bin/env python3
"""CH60-b: 운영 계보(exp66, CLIP vision + bbox w8)에서 그라운더 drop-in 교체 테스트.

exp66 학습된 헤드(ckpt 그대로, 재학습 없음)에 bbox 입력만 교체:
  - pg2_bbox : 원본 데이터셋 bbox (exp66 재현 — 하네스 검증용, 기대 SR 96.6%)
  - owl_bbox : 같은 val 프레임을 OWL-v2(th 0.25)로 재-그라운딩한 bbox

"재학습 없이 그라운더만 바꿔도 운영 성능(96.6%/0.10m)이 유지되는가"를 판정.

Usage: .venv/bin/python3 scripts/eval_exp66_owl_swap.py
출력: docs/v5/closed_loop_eval/exp66_owl_swap.json
"""
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("suite", ROOT / "scripts" / "eval_ablation_suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)

OUT = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp66_owl_swap.json"
OWL_CACHE = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp66_val_owl_bbox.json"
OWL_THRESH = 0.25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def owl_ground_val(val_eps):
    if OWL_CACHE.exists():
        print("OWL bbox 캐시 재사용")
        return json.loads(OWL_CACHE.read_text())
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    print("[OWL-v2] 로딩...")
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(DEVICE).eval()
    out = {}
    t0 = time.time()
    n = 0
    for ep in val_eps:
        h5p = Path(ep["episode"])
        if not h5p.exists():
            continue
        with h5py.File(str(h5p)) as f:
            imgs = f["observations"]["images"][:]
        for fr in ep["frames"]:
            key = f"{h5p.stem}_f{fr['frame_idx']}"
            img = Image.fromarray(imgs[fr["frame_idx"]].astype("uint8")).convert("RGB")
            W, H = img.width, img.height
            inp = proc(text=[["gray laundry basket"]], images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                o = model(**inp)
            res = proc.post_process_object_detection(o, threshold=OWL_THRESH, target_sizes=[(H, W)])[0]
            if len(res["boxes"]) == 0:
                out[key] = {"cx": 0.5, "cy": 0.5, "area": 0.05, "has_bbox": False}
            else:
                best = int(res["scores"].argmax())
                x1, y1, x2, y2 = res["boxes"][best].cpu().tolist()
                x1, x2, y1, y2 = x1 / W, x2 / W, y1 / H, y2 / H
                out[key] = {"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                            "area": (x2 - x1) * (y2 - y1), "has_bbox": True}
            n += 1
            if n % 100 == 0:
                print(f"  {n} frames ({time.time()-t0:.0f}s)")
    del model
    torch.cuda.empty_cache()
    OWL_CACHE.write_text(json.dumps(out, indent=2))
    return out


def main():
    data = json.loads(suite.EVAL_DATA.read_text())
    val_eps = suite.get_val_eps(data)
    print(f"val: {len(val_eps)} episodes")

    owl_bbox = owl_ground_val(val_eps)

    print("Stage1(FrozenCLIPV2) 로드...")
    enc = suite.FrozenCLIPV2(DEVICE)
    head, w, val_acc = suite.load_head(ROOT / "runs/v5_nav/mlp/exp66/action_mlp.pt", "mlp", 8, DEVICE)
    print(f"exp66 head: window={w} val_acc={val_acc:.3f}")

    results = {}
    for variant in ["pg2_bbox", "owl_bbox"]:
        ms = []
        for ep in val_eps:
            e = ep
            if variant == "owl_bbox":
                e = copy.deepcopy(ep)
                stem = Path(ep["episode"]).stem
                for fr in e["frames"]:
                    ob = owl_bbox.get(f"{stem}_f{fr['frame_idx']}")
                    if ob is None:
                        continue
                    fr["cx"], fr["cy"], fr["area"], fr["has_bbox"] = ob["cx"], ob["cy"], ob["area"], ob["has_bbox"]
            m = suite.rollout(e, enc, head, "mlp", w, DEVICE)
            if m is not None:
                ms.append(m)
        sr = sum(m["success"] for m in ms) / len(ms)
        fpe = float(np.mean([m["fpe"] for m in ms]))
        tld = float(np.mean([m["tld"] for m in ms]))
        results[variant] = {"sr": sr, "fpe": fpe, "tld": tld, "n": len(ms)}
        print(f"[{variant}] SR {100*sr:.1f}%  FPE {fpe:.2f}m  TLD {tld:.2f}  (n={len(ms)})")

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
