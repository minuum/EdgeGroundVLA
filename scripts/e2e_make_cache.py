#!/usr/bin/env python3
"""E2E 일반 비교용 캐시: 300 에피소드에서 6프레임 간격 프레임(약 4,860장) + 실기 라이브 프레임(416장)의 Florence-2 DaViT stage-2 출력(2304토큰×512, fp16)을 저장.
뒤쪽 stage 3(마지막 단계)만 LoRA로 학습하기 위한 캐시다(전체 E2E는 1.5장/초라 학습 불가). 실행: HF_HUB_OFFLINE=1 .venv/bin/python3 scripts/e2e_make_cache.py
출력: runs/e2e_cache/{frames.json, stage2.npy, live_stage2.npy, live_meta.json}
"""
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import train_egvla2_s1 as S1  # noqa: E402
from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: E402
T = S1.T
OUT = ROOT / "runs/e2e_cache"; OUT.mkdir(parents=True, exist_ok=True)
STRIDE = 6
EXCLUDE = {"20261022_193630", "20261022_194447", "20261022_201157"}


def main():
    dev = torch.device("cuda")
    proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float32).to(dev).eval()
    vt = model.vision_tower
    eps = S1.load_main(); frames = [(i, t) for i, e in enumerate(eps) for t in range(0, len(e["bbox"]), STRIDE)]
    (OUT / "frames.json").write_text(json.dumps(dict(stride=STRIDE, frames=frames, names=[e["name"] for e in eps], cells=[e["cell"] for e in eps])))
    print("프레임", len(frames), flush=True)

    @torch.no_grad()
    def stage2(rgb_list):
        px = torch.cat([proc(images=Image.fromarray(np.ascontiguousarray(a)).convert("RGB"), return_tensors="pt")["pixel_values"] for a in rgb_list]).to(dev)
        h, size = px, (px.size(2), px.size(3))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(3):
                h, size = vt.convs[i](h, size); h, size = vt.blocks[i](h, size)
        return h.half().cpu().numpy()

    mm = np.lib.format.open_memmap(OUT / "stage2.npy", mode="w+", dtype=np.float16, shape=(len(frames), 2304, 512))
    BS = 8; k = 0; cur = None
    buf, idx = [], []
    handles = {}
    for n, (ei, t) in enumerate(frames):
        name = eps[ei]["name"]
        if name not in handles:
            for h_ in handles.values():
                h_.close()
            handles = {name: h5py.File(T.DATA / name, "r")}
        rgb = np.ascontiguousarray(np.asarray(handles[name]["images"][t])[:, :, ::-1])
        buf.append(rgb); idx.append(n)
        if len(buf) == BS or n == len(frames) - 1:
            mm[idx[0]:idx[-1] + 1] = stage2(buf); buf, idx = [], []
        if n % 400 == 0:
            print(n, "/", len(frames), flush=True)
    mm.flush()
    # 라이브(SYNC EGVLA2 세션)
    live_rgb, live_meta = [], []
    for rp in sorted(glob.glob(str(ROOT / "ROS_action/dryrun_closedloop_20261022/reports/*.json"))):
        d = json.load(open(rp)); sid = d["session_id"]
        hp = ROOT / f"ROS_action/dryrun_closedloop_20261022/h5/session_{sid}.h5"
        if sid in EXCLUDE or not hp.exists() or "egvla2" not in d["model_name"]:
            continue
        with h5py.File(hp, "r") as f:
            if f["observations/images"].shape[0] < 3:
                continue
            for t in range(f["observations/images"].shape[0]):
                live_rgb.append(np.asarray(f["observations/images"][t])); live_meta.append(f"{sid}|{t}")
    lm = np.lib.format.open_memmap(OUT / "live_stage2.npy", mode="w+", dtype=np.float16, shape=(len(live_rgb), 2304, 512))
    for s in range(0, len(live_rgb), BS):
        lm[s:s + BS] = stage2(live_rgb[s:s + BS])
    lm.flush(); (OUT / "live_meta.json").write_text(json.dumps(live_meta)); print("완료 train", len(frames), "live", len(live_rgb))


if __name__ == "__main__":
    main()
