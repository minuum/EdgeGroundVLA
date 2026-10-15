#!/usr/bin/env python3
"""EGVLA2 서빙 정책 오프라인 재생(plan_20261016 §3): 캐시된 (bbox, 특징) 또는 이미지에서 정책을 돌려 정답 exec와 비교한다.

  --ckpt runs/egvla2_fm/C_pi0_adaln.pt --episodes 이름1,이름2  (기본: 세트의 마지막 5개)
  --equiv N   이미지에서 Florence-2를 직접 돌려 캐시 값과 같은지(bbox·특징 차이)도 확인(앞 N프레임)
주의: 전체 학습 체크포인트로 학습 에피소드를 재생하면 낙관적이다(동작 확인용). hold-out 성능은 S1의 fold 결과를 볼 것.
출력: stdout 요약 (+ --out JSON)
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy  # noqa: E402

DATA = ROOT / "ROS_action/mobile_vla_dataset_egvla2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--episodes", default="")
    ap.add_argument("--equiv", type=int, default=0); ap.add_argument("--out", default="")
    a = ap.parse_args()
    meta = json.loads((ROOT / "docs/v5/egvla2_fm/meta/egvla2_v3_main.json").read_text())
    names = a.episodes.split(",") if a.episodes else [d["name"] for d in meta["files"]][-5:]
    cache = np.load(ROOT / "docs/v5/egvla2_fm/v3_florence2_cache.npz")
    pol = EgVla2FMPolicy(a.ckpt, speed_source="shadow", s_max=1.0)
    rows, ang, go_tp, go_fp, go_fn = [], [], 0, 0, 0
    for n in names:
        n = n if n.endswith(".h5") else n + ".h5"
        with h5py.File(DATA / n, "r") as f:
            ex = np.array(f["actions_exec"], np.float32)
        bb, ft = cache[f"{n}|bbox"], cache[f"{n}|feat"]
        pol.reset()
        for t in range(len(ex)):
            r = pol.step(bb[t], ft[t].astype(np.float32))
            y = ex[t, :2]; p = np.array(r["first"][:2]); ym, pm = np.hypot(*y), np.hypot(*p)
            if ym > 0.2 and pm > 0.2:
                ang.append(float(np.degrees(np.arccos(np.clip(y @ p / (ym * pm), -1, 1)))))
            go_tp += int(ym > .2 and pm > .2); go_fp += int(ym <= .2 and pm > .2); go_fn += int(ym > .2 and pm <= .2)
            rows.append(dict(ep=n, t=t, pred=r["first"], true=ex[t].tolist(), shadow_s=r["action_cont"]["model_speed_scale"], s=r["action_cont"]["speed_scale"]))
    out = dict(ckpt=a.ckpt, arch=pol.meta.get("arch"), episodes=len(names), frames=len(rows), angle_deg=float(np.mean(ang)),
               go_f1=2 * go_tp / max(2 * go_tp + go_fp + go_fn, 1), mean_applied_s=float(np.mean([r["s"] for r in rows if r["s"] > 0])))
    if a.equiv:
        from robovlm_nav.serve.egvla2_fm_policy import FusedF2Extractor
        ex_ = FusedF2Extractor(); n = names[0] if names[0].endswith(".h5") else names[0] + ".h5"
        with h5py.File(DATA / n, "r") as f:
            imgs = f["images"][:a.equiv]
        db, df = [], []
        for t, im in enumerate(imgs):
            b, x = ex_(np.ascontiguousarray(im[:, :, ::-1]))
            db.append(np.abs(b - cache[f"{n}|bbox"][t]).max()); df.append(float(np.abs(x - cache[f"{n}|feat"][t].astype(np.float32)).max()))
        out["equiv"] = dict(frames=len(imgs), bbox_maxdiff=float(max(db)), feat_maxdiff=float(max(df)))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(summary=out, rows=rows[:2000]), ensure_ascii=False))


if __name__ == "__main__":
    main()
