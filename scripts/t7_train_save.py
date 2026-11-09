#!/usr/bin/env python3
"""T7 (2): V0(이미지 전용)·V4(확장 어휘)를 300에피소드 전체로 학습해 체크포인트 저장. 학습 방식은 eval_text_cond_t4.fit과 동일(120ep, stride 9, seed 0).
출력: runs/egvla2_fm/textcond/EG4-00_V0_full.pt, EG4-04_V4_full.pt"""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_text_cond_t4 as E

out = ROOT / "runs/egvla2_fm/textcond"; out.mkdir(parents=True, exist_ok=True)
eps, D = E.load(); tr = np.arange(len(D["ep"])); print("train frames", len(tr), "eps", len(eps), flush=True)
import os
SD = int(os.environ.get("SEED", 0)); sfx = "" if SD == 0 else f"_s{SD}"
for pn, v in [("EG4-00_V0_full", "V0_img_only"), ("EG4-04_V4_full", "V4_big_vocab")]:
    pn += sfx; m = E.fit(D, tr, v, SD)
    torch.save(dict(state=m.state_dict(), variant=v, bdim=12, fdim=768 if v == "V0_img_only" else 1536, epochs=E.EPOCHS, stride=E.STRIDE, seed=SD,
                    n_frames=int(len(tr)), n_episodes=len(eps), recipe="eval_text_cond_t4.fit"), out / f"{pn}.pt"); print("saved", pn, flush=True)
