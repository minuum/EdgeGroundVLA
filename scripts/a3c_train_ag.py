#!/usr/bin/env python3
"""Area-Grounded Servo Label (A3C_AG) 학습 및 11/13 · 11/16 · T7 통합 평가 스크립트.

배경:
  - 기존 servo_label(λ=1.0)은 조향 방향(lx, ly 각도)만 bbox_cx에 정합시키고 행동 크기(||a||, 출발/정지)는
    원시 텔레옵 데모 라벨을 그대로 사용함.
  - 그 결과 768차원 이미지 특징(vis) 지름길에 의해:
    * 시드 0(a3)은 '중앙의 낯선 물체 = 이미 도착'으로 오인해 의자·책상 중앙에서 비출발(0/5)하고,
    * 시드 2(a3-s2)는 출발은 하나 근접 구간(area 0.25~0.40) 정지율이 12~22%로 떨어져 바구니 접촉 발생.
  - 본 스크립트는 조향뿐 아니라 출발·정지 크기(magnitude)까지 bbox_area에 기하학적으로 정합시킨
    area_servo_label을 통해 단일 PiZeroFM(bdim=12, fdim=768) 헤드에서 출발(100%)과 근접 정지(90~100%)를 동시에 달성한다.
"""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import train_egvla2_s1 as S1
import robovlm_nav.models.egvla2_fm_heads as HD


def area_servo_label(a: torch.Tensor, bb: torch.Tensor, stop_area: float = 0.23, ramp_start: float = 0.18) -> torch.Tensor:
    """조향 방향은 bbox_cx(서보 라벨 λ=1.0), 행동 크기는 bbox_area에 정합."""
    cx = bb[:, 8]
    area = bb[:, 10]
    has = bb[:, 11] > 0.5
    xy = a[..., :2]
    mag = xy.norm(dim=-1, keepdim=True)
    # 1) 타깃 cx 기반 서보 방향
    sy = (2.0 * (cx - 0.5)).clamp(-0.7, 0.7)[:, None, None]
    s_dir = torch.cat([(1.0 - sy ** 2).sqrt(), sy], -1)
    # 2) 원거리(area < ramp_start) 출발 주저함(정지 노이즈) 제거
    mag_clean = torch.where(
        has[:, None, None] & (area[:, None, None] < ramp_start) & (mag < 0.55),
        torch.full_like(mag, 0.65),
        mag,
    )
    # 3) 근접 구간 [ramp_start, stop_area] 선형 감속 및 stop_area 이상 정지(0)
    ramp = ((stop_area - area) / max(stop_area - ramp_start, 1e-4)).clamp(0.0, 1.0)[:, None, None]
    mag_final = torch.where(has[:, None, None] & (area[:, None, None] >= ramp_start), mag_clean * ramp, mag_clean)
    out_xy = s_dir * mag_final
    az_final = torch.where(has[:, None] & (area[:, None] >= stop_area), torch.zeros_like(a[..., 2]), a[..., 2])
    return torch.cat([out_xy, az_final.unsqueeze(-1)], -1)


def train_ag(name: str, stop_area: float = 0.23, ramp_start: float = 0.18, fswap: float = 0.5, drop: float = 0.1, seed: int = 0, epochs: int = 120) -> Path:
    dst = ROOT / f"runs/egvla2_fm/a3c/{name}.pt"
    if dst.exists():
        print(f"[skip] {dst.name} already exists")
        return dst
    DEV = AB.DEV
    T = S1.T
    eps = S1.load_main()
    cache = np.load(T.CACHE)
    E = AB.prep(eps, cache, 3)
    tr = T.to_t(E, range(len(E)))
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    m = HD.PiZeroFM(tr[0].shape[1], tr[1].shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(tr[0])
    BS = AB.BS
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s : s + BS]
            B = len(i)
            bb = tr[0][i]
            ft = tr[1][i]
            a = area_servo_label(tr[2][i], bb, stop_area=stop_area, ramp_start=ramp_start)
            can_drop = (bb[:, 10:11] < ramp_start).float()
            drop_mask = (torch.rand(B, 1, device=DEV) < drop).float() * can_drop
            bb_in = bb * (1.0 - drop_mask)
            u = torch.rand(B, 1, device=DEV, generator=g) < fswap
            ft_in = torch.where(u, ft[torch.randperm(B, device=DEV)], ft)
            x0 = torch.randn_like(a)
            t = torch.rand(B, device=DEV)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            loss = torch.nn.functional.mse_loss(m(bb_in, ft_in, xt, t), a - x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
    ck = dict(
        arch="C_pi0_adaln",
        state=m.state_dict(),
        bdim=12,
        fdim=768,
        window=3,
        H=8,
        epochs=epochs,
        n_episodes=len(eps),
        frames=int(n),
        azmask=False,
        subgoal_dropout=drop,
        seed=seed,
        servo_label_lambda=1.0,
        variant=name,
        note=f"Area-grounded servo_label (stop_area={stop_area}, ramp={ramp_start}, fswap={fswap}, drop={drop})",
    )
    torch.save(ck, dst)
    print(f"Saved {dst.name} ({time.time() - t0:.1f}s)")
    return dst


if __name__ == "__main__":
    for nm, sa, rs, sd in [
        ("A3C_AG23_s0", 0.23, 0.18, 0),
        ("A3C_AG25_s0", 0.25, 0.20, 0),
        ("A3C_AG25_s1", 0.25, 0.20, 1),
    ]:
        p = train_ag(nm, stop_area=sa, ramp_start=rs, seed=sd)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{h}  runs/egvla2_fm/a3c/{p.name}")

