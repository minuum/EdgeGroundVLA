#!/usr/bin/env python3
"""S3: A3 레시피(300에피소드 전체, λ=1.0 서보 라벨, 서브골 드롭아웃 30%, 120 epoch) + 이미지 특징 처리 변형.
plan_20261115_a3_improvement.md S3. 근거(S2-b): 의자·가운데 이미지 특징이 들어가면 박스와 무관하게 명령 0 → 헤드가 이미지 특징 지름길로 정지.
변형: D00/D10/D20 = 이미지 특징은 그대로, 박스 가림 확률만 0/0.1/0.2(A3 기본 0.3) / F30/F50 = 확률 p로 이미지 특징을 0으로 / S30 = 확률 0.3로 같은 배치의 다른 프레임 특징으로 교체(장면 정보를 행동과 분리).
사용: VARIANT=F30|F50|S30 SEED=0 .venv/bin/python3 scripts/a3c_train.py   출력 runs/egvla2_fm/a3c/A3C_{VARIANT}_s{SEED}.pt (배포 서버 형식, 배포 폴더는 건드리지 않음)"""
import os, sys, time, json
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB, eval_target_cond as TC, train_egvla2_s1 as S1
import robovlm_nav.models.egvla2_fm_heads as HD
T = S1.T; DEV = AB.DEV; BS = AB.BS
V = os.environ["VARIANT"]; SD = int(os.environ.get("SEED", 0)); EP = int(os.environ.get("EPOCHS", TC.EPOCHS))
CFG = {"F30": ("zero", .3), "F50": ("zero", .5), "S30": ("swap", .3), "D00": ("none", 0.), "D10": ("none", 0.), "D20": ("none", 0.), "D30": ("none", 0.)}[V]
DROP = {"D00": 0.0, "D10": 0.1, "D20": 0.2, "D30": 0.3}.get(V, 0.3)   # 박스(서브골) 가림 확률, A3 기본 0.3
dst = ROOT / f"runs/egvla2_fm/a3c/A3C_{V}_s{SD}.pt"
if dst.exists(): print("exists", dst); sys.exit(0)
eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); tr = T.to_t(E, range(len(E)))
torch.manual_seed(SD); g = torch.Generator(device=DEV); g.manual_seed(SD)
m = HD.PiZeroFM(tr[0].shape[1], tr[1].shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); n = len(tr[0]); t0 = time.time()
for ep in range(EP):
    m.train(); perm = torch.randperm(n, device=DEV)
    for s in range(0, n, BS):
        i = perm[s:s + BS]; B = len(i); bb = tr[0][i]; ft = tr[1][i]; a = TC.servo_label(tr[2][i], bb[:, 8], 1.0)
        bb = bb * (torch.rand(B, 1, device=DEV) > DROP).float()
        u = torch.rand(B, 1, device=DEV, generator=g) < CFG[1]
        if CFG[0] == "zero": ft = torch.where(u, torch.zeros_like(ft), ft)
        elif CFG[0] == "swap": ft = torch.where(u, ft[torch.randperm(B, device=DEV)], ft)
        x0 = torch.randn_like(a); t = torch.rand(B, device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
        l = torch.nn.functional.mse_loss(m(bb, ft, xt, t), a - x0); opt.zero_grad(); l.backward(); opt.step()
    if ep % 10 == 0 or ep == EP - 1: print(f"{V} s{SD} ep{ep+1}/{EP} loss {l.item():.4f} {time.time()-t0:.0f}s", flush=True)
ck = dict(arch="C_pi0_adaln", state=m.state_dict(), bdim=tr[0].shape[1], fdim=tr[1].shape[1], window=T.WINDOW, H=T.H, epochs=EP, n_episodes=len(eps), frames=int(n), azmask=False,
          subgoal_dropout=DROP, seed=SD, servo_label_lambda=1.0, variant="A3C_" + V, note=f"A3 recipe + image-feature {CFG[0]} p={CFG[1]} subgoal_dropout={DROP} (plan_20261115_a3_improvement S3)")
torch.save(ck, dst); print("saved", dst.name, flush=True)
