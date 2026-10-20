#!/usr/bin/env python3
"""에피소드 끝의 정지·동일 프레임을 잘라 낸 사본을 만든다(원본 H5는 건드리지 않음). 기록: docs/v5/egvla2_fm/meta/egvla2_v3_trim.json
실행: .venv/bin/python3 -I scripts/trim_episode_frames.py <episode.h5> [--apply]   (기본: 미리보기)
규칙: 끝에서부터 '직전 프레임과 픽셀 차<0.4 이고 exec 이동·회전이 모두 0'인 프레임을 제거.
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ROS_action/mobile_vla_dataset_egvla2"
OUT = DATA / "trimmed"
REC = ROOT / "docs/v5/egvla2_fm/meta/egvla2_v3_trim.json"
KEYS = ["actions_raw", "actions_exec", "actions_scaled", "action_classes", "action_event_types", "frame_timestamps", "images"]

name = sys.argv[1]; apply = "--apply" in sys.argv
src = DATA / name
with h5py.File(src, "r") as h:
    im = h["images"]; ex = np.array(h["actions_exec"]); n = len(ex)
    keep = n
    while keep > 1:
        d = float(np.abs(im[keep - 1][::16, ::16].astype(np.int16) - im[keep - 2][::16, ::16].astype(np.int16)).mean())
        if d < 0.4 and not np.any(ex[keep - 1]):
            keep -= 1
        else:
            break
    print(f"{name}: {n}프레임 → {keep}프레임 (끝 {n - keep}프레임 제거: #{keep}~#{n - 1})")
    if apply and keep < n:
        OUT.mkdir(exist_ok=True)
        with h5py.File(OUT / name, "w") as o:
            for k in KEYS:
                o.create_dataset(k, data=h[k][:keep], compression="gzip" if k == "images" else None)
            for k, v in h.attrs.items():
                o.attrs[k] = v
            o.attrs["num_frames"] = keep
            o.attrs["trimmed_from"] = name; o.attrs["trimmed_dropped_tail_frames"] = n - keep
        rec = json.loads(REC.read_text()) if REC.exists() else {}
        rec[name] = dict(original_frames=n, kept_frames=keep, dropped=list(range(keep, n)), rule="끝의 동일 프레임(픽셀 차<0.4)이며 exec 0(STOP)",
                         copy=f"ROS_action/mobile_vla_dataset_egvla2/trimmed/{name}", original_untouched=True, date=time.strftime("%Y-%m-%d"))
        REC.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        print("저장:", OUT / name, "기록:", REC)
