#!/usr/bin/env python3
"""서빙 설계 그대로(gray basket 고정 박스 + 지시어 문구 특징)의 V4·V4R 10시드 열린고리 재생. 키 접미사 _gb. 기존 t13_obj_replay.json에 합친다."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import robovlm_nav.models.egvla2_fm_heads as HD
DEV = AB.DEV
C = np.load(ROOT / "runs/egvla2_fm/t13_obj_served_cache.npz"); names = sorted({k.split("|")[0] for k in C.files if k.endswith("|vis")})
RP = json.load(open(ROOT / "runs/egvla2_fm/t13_obj_replay.json"))
@torch.no_grad()
def step1(m, bb, ft): torch.manual_seed(0); return HD.sample_chunk(m, bb, ft, 8).mean(1)[:, 0, :2].cpu().numpy()
for v, base in {"V4": "EG4-04_V4_full", "V4R": "EG4-07_V4R_full"}.items():
    for sd in range(10):
        sfx = "" if sd == 0 else f"_s{sd}"; ck = torch.load(ROOT / f"runs/egvla2_fm/textcond/{base}{sfx}.pt", map_location=DEV, weights_only=False)
        m = HD.PiZeroFM(12, ck["fdim"]).to(DEV); m.load_state_dict(ck["state"]); m.eval()
        for n in names:
            raw = C[f"{n}|bbox"].astype(np.float32); bbw = AB.window(raw, 3); bbw[raw[:, 3] < 0.5] = 0; bt = torch.tensor(bbw, device=DEV); vis = torch.tensor(C[f"{n}|vis"], device=DEV)
            for tag in ("ins", "gray"):
                f = torch.cat([vis, torch.tensor(C[f"{n}|txt_{tag}"], device=DEV)], -1)
                RP[n].setdefault(f"{v}_{tag}_gb", []).append(step1(m, bt, f).round(3).tolist())
            RP[n]["gb_has"] = [round(float(x), 0) for x in raw[:, 3]]; RP[n]["gb_bbox"] = raw.round(3).tolist()
    print(v, flush=True)
json.dump(RP, open(ROOT / "runs/egvla2_fm/t13_obj_replay.json", "w")); print("saved")
