#!/usr/bin/env python3
"""타깃 위치에 반응하는 모델 만들기 시험(오프라인): 반사실 민감도 S = 타깃 cx를 ±0.2 옮겼을 때 예측 방향각 변화의 교정 방향 평균(도, 양수=타깃 쪽으로 교정, 현재 배포 모델 약 -0.7°)
과 사람 방향 오차를 함께 본다. 변형: base / featdrop50(학습 중 50% 확률로 이미지 특징을 0으로 → bbox가 필요해짐) / servo_label0.5·1.0(방향 라벨을 타깃 서보 방향과 λ만큼 혼합).
분할: fold0·fold1·lopo_center·lopo_strong_right, seed 0, 서브골 드롭아웃 30%, 120 epoch. 출력: docs/v5/egvla2_fm/target_cond.json (분할 단위 저장·재개)
실행: .venv_lerobot/bin/python3 scripts/eval_target_cond.py  (SMOKE=1: epoch 2·분할 1개)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_runtime_ablation as RA  # noqa: E402
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
import robovlm_nav.models.egvla2_fm_heads as HD  # noqa: E402
T = S1.T
DEV, BS = AB.DEV, AB.BS
SMOKE = bool(os.getenv("SMOKE"))
EPOCHS = 2 if SMOKE else 120
OUT = ROOT / "docs/v5/egvla2_fm" / ("target_cond_smoke.json" if SMOKE else "target_cond.json")
VARIANTS = {"base": {}, "featdrop50": dict(fdrop=0.5), "servo_label0.5": dict(lam=0.5), "servo_label1.0": dict(lam=1.0)}
SK, SM = 2.0, 0.7


def servo_label(a, cx, lam):
    """(lx,ly) 방향을 타깃 서보 방향(ly = clip(2(cx-0.5), ±0.7))과 lam만큼 혼합. 크기 보존, az 불변."""
    xy = a[..., :2]; mag = xy.norm(dim=-1, keepdim=True); u = xy / mag.clamp_min(1e-6)
    sy = (SK * (cx - 0.5)).clamp(-SM, SM)[:, None, None]; s = torch.cat([(1 - sy ** 2).sqrt(), sy], -1)
    b = (1 - lam) * u + lam * s; b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    out = torch.where(mag > 0.2, b * mag, xy)
    return torch.cat([out, a[..., 2:]], -1)


def fit(tr, fdrop=0.0, lam=0.0, seed=0, drop=0.3):
    torch.manual_seed(seed)
    m = HD.PiZeroFM(tr[0].shape[1], tr[1].shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); n = len(tr[0])
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; bb = tr[0][i]; ft = tr[1][i]; a = tr[2][i]
            if lam > 0:
                a = servo_label(a, bb[:, 8], lam)
            bb = bb * (torch.rand(len(i), 1, device=DEV) > drop).float()
            if fdrop > 0:
                ft = ft * (torch.rand(len(i), 1, device=DEV) > fdrop).float()
            x0 = torch.randn_like(a); t = torch.rand(len(i), device=DEV)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            l = torch.nn.functional.mse_loss(m(bb, ft, xt, t), a - x0); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


def angle_of(m, bb, ft):
    S = RA.draw(m, bb, ft, 8, 10).mean(1)[:, 0]
    return np.degrees(np.arctan2(S[:, 1], S[:, 0])), np.linalg.norm(S, axis=1)


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    splits = {"fold0": folds[0], "fold1": folds[1],
              "lopo_center": [i for i, e in enumerate(eps) if e["cell"].startswith("center|")], "lopo_strong_right": [i for i, e in enumerate(eps) if e["cell"].startswith("strong_right|")]}
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, te_idx in splits.items():
        res.setdefault(name, {}); tr_idx = [i for i in range(len(E)) if i not in te_idx]
        tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx)
        gt = te[2][..., :2].cpu().numpy(); g_first = gt[:, 0]
        mv = np.linalg.norm(g_first, axis=1) > 0.2
        for vn, kw in VARIANTS.items():
            if vn in res[name]:
                continue
            m = fit(tr, **kw)
            a0, mag0 = angle_of(m, te[0], te[1])
            def shift(dx):
                b = te[0].clone()
                for k in (0, 4, 8):
                    b[:, k] = (b[:, k] + dx).clamp(0, 1)
                return b
            ap, _ = angle_of(m, shift(0.2), te[1]); am, _ = angle_of(m, shift(-0.2), te[1])
            S = float(np.mean(((ap - am) / 2)[mv]))                    # 양수 = 타깃 쪽 교정(+0.2 → 각도 증가)
            S = float(np.clip(S, -180, 180))
            res[name][vn] = dict(first=RA.angle(np.stack([np.cos(np.radians(a0)), np.sin(np.radians(a0))], 1) * mag0[:, None], g_first), sens=S)
            OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            print(name, vn, {k: round(v, 2) for k, v in res[name][vn].items()}, flush=True)
        if SMOKE:
            break


if __name__ == "__main__":
    main()
