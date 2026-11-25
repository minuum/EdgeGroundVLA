#!/usr/bin/env python3
"""Step B (바닥 접지선 y2 정지 정합: A3C_Y2_s0) 및 Step C (텍스트 조건부 거리·안전 통합: EG4-08_V4R_Y2_s0) 학습·평가 스크립트.

핵심 아이디어:
  - 11/25 실기에서 동일 area(~0.29)로 정지했을 때 바구니는 y2=1.000까지 가까이 접근했으나,
    의자는 덩치가 커서 y2=0.819(아직 먼 거리)에서 멈추는 문제가 확인됨.
  - Step B (A3C_Y2_s0.pt, bdim=12, fdim=768):
      2D 면적(area) 대신 바닥 접지선 y2 = clamp(cy + 0.5*sqrt(area), 0, 1)을 정지·감속 기준으로 정합
      (y2_ramp=0.86, y2_stop=0.95) + 50% Feature Swap.
  - Step C (EG4-08_V4R_Y2_s0.pt, bdim=12, fdim=1536):
      V4R(텍스트 조건부 비목표 차단 + 박스 훼손 증강) + Step B(y2 바닥 접지선 정지) + 50% 시각 특징 교체 통합.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import eval_bbox_robustness as RB
import eval_subgoal_ablation as AB
import eval_text_cond_t4 as E4
import eval_text_cond_t5 as E5
import train_egvla2_s1 as S1
import robovlm_nav.models.egvla2_fm_heads as HD
from audit_overfitting import predict_head

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def y2_servo_label(a: torch.Tensor, bb: torch.Tensor, y2_stop: float = 0.95, y2_ramp: float = 0.86) -> torch.Tensor:
    """조향은 bbox_cx(서보 λ=1.0), 감속·정지는 바닥 접지선 y2 = clamp(cy + 0.5*sqrt(area), 0, 1)에 정합."""
    cx = bb[:, 8]
    cy = bb[:, 9]
    area = bb[:, 10].clamp(min=0.0)
    has = bb[:, 11] > 0.5
    y2 = (cy + 0.5 * torch.sqrt(area + 1e-6)).clamp(0.0, 1.0)

    xy = a[..., :2]
    mag = xy.norm(dim=-1, keepdim=True)
    sy = (2.0 * (cx - 0.5)).clamp(-0.7, 0.7)[:, None, None]
    s_dir = torch.cat([(1.0 - sy ** 2).sqrt(), sy], -1)

    # 원거리(y2 < y2_ramp) 출발 주저함 제거
    mag_clean = torch.where(
        has[:, None, None] & (y2[:, None, None] < y2_ramp) & (mag < 0.55),
        torch.full_like(mag, 0.65),
        mag,
    )
    # 근접 [y2_ramp, y2_stop] 선형 감속 및 y2 >= y2_stop 완전 정지
    ramp = ((y2_stop - y2) / max(y2_stop - y2_ramp, 1e-4)).clamp(0.0, 1.0)[:, None, None]
    mag_final = torch.where(has[:, None, None] & (y2[:, None, None] >= y2_ramp), mag_clean * ramp, mag_clean)
    out_xy = s_dir * mag_final
    az_final = torch.where(has[:, None] & (y2[:, None] >= y2_stop), torch.zeros_like(a[..., 2]), a[..., 2])
    return torch.cat([out_xy, az_final.unsqueeze(-1)], -1)


def train_step_b(y2_stop: float = 0.95, y2_ramp: float = 0.86, epochs: int = 120, seed: int = 0) -> Path:
    dst = ROOT / "runs/egvla2_fm/a3c/A3C_Y2_s0.pt"
    if dst.exists():
        print(f"[skip] {dst.name} already exists", flush=True)
        return dst
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
    n, BS = len(tr[0]), AB.BS
    t0 = time.time()
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s : s + BS]
            B = len(i)
            bb, ft = tr[0][i], tr[1][i]
            a = y2_servo_label(tr[2][i], bb, y2_stop=y2_stop, y2_ramp=y2_ramp)
            y2 = (bb[:, 9:10] + 0.5 * torch.sqrt(bb[:, 10:11].clamp(min=0.0) + 1e-6)).clamp(0.0, 1.0)
            drop_mask = (torch.rand(B, 1, device=DEV) < 0.1).float() * (y2 < y2_ramp).float()
            bb_in = bb * (1.0 - drop_mask)
            u = torch.rand(B, 1, device=DEV, generator=g) < 0.5
            ft_in = torch.where(u, ft[torch.randperm(B, device=DEV)], ft)
            x0 = torch.randn_like(a)
            t = torch.rand(B, device=DEV)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            loss = torch.nn.functional.mse_loss(m(bb_in, ft_in, xt, t), a - x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save(
        dict(
            arch="C_pi0_adaln",
            state=m.state_dict(),
            bdim=12,
            fdim=768,
            window=3,
            H=8,
            epochs=epochs,
            n_episodes=len(eps),
            frames=int(n),
            seed=seed,
            variant="A3C_Y2_s0",
            note=f"Step B: Ground-line y2 servo_label (y2_stop={y2_stop}, y2_ramp={y2_ramp}, fswap=0.5)",
        ),
        dst,
    )
    print(f"[Step B] Saved {dst.name} in {time.time() - t0:.1f}s", flush=True)
    return dst


def train_step_c(y2_stop: float = 0.95, y2_ramp: float = 0.86, seed: int = 0) -> Path:
    dst = ROOT / "runs/egvla2_fm/textcond/EG4-08_V4R_Y2_s0.pt"
    if dst.exists():
        print(f"[skip] {dst.name} already exists", flush=True)
        return dst
    eps, D = E5.load()
    tr = np.arange(len(D["ep"]))
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    m = HD.PiZeroFM(12, 1536).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(tr)
    trt = torch.tensor(tr, device=DEV)
    pos_set = torch.tensor(list(range(E4.NP)), device=DEV)
    neg_set = torch.tensor([E4.NP + j for j in range(E4.NN)], device=DEV)
    t0 = time.time()
    for _ in range(E4.EPOCHS):
        m.train()
        perm = trt[torch.randperm(n, device=DEV)]
        for s in range(0, n, E4.BS):
            i = perm[s : s + E4.BS]
            B = len(i)
            bb0 = D["bb"][i]
            a = y2_servo_label(D["ch"][i], bb0, y2_stop=y2_stop, y2_ramp=y2_ramp)
            # V4R 박스 훼손 증강
            u = torch.rand(B, 1, device=DEV, generator=g)
            bb = torch.where(
                u < 0.3,
                torch.zeros_like(bb0),
                torch.where(
                    (u >= 0.3) & (u < 0.5),
                    RB.corrupt(bb0, "jitter", g),
                    torch.where((u >= 0.5) & (u < 0.6), RB.corrupt(bb0, "wrong", g), bb0),
                ),
            )
            k = pos_set[torch.randint(0, len(pos_set), (B,), device=DEV)]
            neg = torch.rand(B, device=DEV) < E4.P_NEG
            k = torch.where(neg, neg_set[torch.randint(0, len(neg_set), (B,), device=DEV)], k)
            a = torch.where(neg[:, None, None], torch.zeros_like(a), a)
            keep = (torch.rand(B, 1, device=DEV) < E4.KEEP).float()
            bb = torch.where(neg[:, None], bb0 * keep, bb)
            # 시각 배경 50% 교체로 외형 과적합 차단
            vis_i = D["vis"][i]
            sw = torch.rand(B, 1, device=DEV, generator=g) < 0.5
            vis_in = torch.where(sw, vis_i[torch.randperm(B, device=DEV)], vis_i)
            ft = torch.cat([vis_in, E4.txt(D, i, k)], -1)
            x0 = torch.randn_like(a)
            t = torch.rand(B, device=DEV)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            loss = torch.nn.functional.mse_loss(m(bb, ft, xt, t), a - x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            arch="PiZeroFM_textcond",
            state=m.state_dict(),
            variant="V4R_Y2_DistCond",
            fdim=1536,
            bdim=12,
            epochs=E4.EPOCHS,
            stride=E4.STRIDE,
            seed=seed,
            n_frames=int(n),
            n_episodes=len(eps),
            note=f"Step C: V4R + y2 ground-line stop (y2_stop={y2_stop}, y2_ramp={y2_ramp}) + 50% vis swap",
        ),
        dst,
    )
    print(f"[Step C] Saved {dst.name} in {time.time() - t0:.1f}s", flush=True)
    return dst


def evaluate_all(p_b: Path, p_c: Path) -> None:
    C13 = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz")
    C16 = np.load(ROOT / "runs/egvla2_fm/t16_cache.npz")
    D13 = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
    S16 = json.load(open(ROOT / "docs/v5/egvla2_fm/a3c/real_1116_sessions_bbox_area.json"))
    OB16 = {"go to the gray basket": "바구니", "go to the brown wooden chair": "의자", "go to the small desk": "책상"}

    bw13, vis13, meta13 = [], [], []
    idx = 0
    for x in D13:
        sid = x["sid"]
        vis = C13[f"{sid}|vis"].astype(np.float32)
        raw = C13[f"{sid}|bbox"][: len(vis)].astype(np.float32)
        L = len(vis)
        bw = AB.window(raw, 3); bw[raw[:, 3] < 0.5] = 0
        bw13.append(bw); vis13.append(vis); meta13.append((sid, x["obj"], x["pos"], idx, idx + L, raw)); idx += L
    BW13, VIS13 = np.concatenate(bw13, 0), np.concatenate(vis13, 0)

    bw16, vis16, meta16 = [], [], []
    idx = 0
    for sid, s in S16.items():
        vis = C16[f"{sid}|vis"].astype(np.float32)
        raw = C16[f"{sid}|bbox"][: len(vis)].astype(np.float32)
        L = len(vis)
        bw = AB.window(raw, 3); bw[raw[:, 3] < 0.5] = 0
        bw16.append(bw); vis16.append(vis); meta16.append((sid, OB16[s["instruction"]], idx, idx + L, raw)); idx += L
    BW16, VIS16 = np.concatenate(bw16, 0), np.concatenate(vis16, 0)
    raw16 = np.concatenate([raw for (_, _, _, _, raw) in meta16], 0)
    y2_16 = np.clip(raw16[:, 1] + 0.5 * np.sqrt(np.maximum(0.0, raw16[:, 2])), 0.0, 1.0)

    # 11/25 의자 도착 프레임(y2~0.819) vs 바구니 도착 프레임(y2~1.000)에서 모델이 계속 가는지/멈추는지 확인
    bw25_b, bw25_c = [], []
    for hp in sorted((ROOT / "_received_1125_sessions").glob("session_20261125_*.h5")):
        with h5py.File(hp, "r") as f:
            bb = f["grounding"]["bbox"][:].astype(np.float32)
            bw = AB.window(bb, 3)
            if "basket" in str(f.attrs.get("instruction", "")):
                bw25_b.append(bw[-1:])
            elif "chair" in str(f.attrs.get("instruction", "")):
                bw25_c.append(bw[-1:])
    BW25_B = np.concatenate(bw25_b, 0)
    BW25_C = np.concatenate(bw25_c, 0)
    mean_vis_b = np.repeat(VIS13[:1], len(BW25_B), axis=0)
    mean_vis_c = np.repeat(VIS13[:1], len(BW25_C), axis=0)

    perm13 = np.random.default_rng(42).permutation(len(VIS13))
    results = {}

    # Step B 평가 (768-d) vs AG23 대조
    for name, ck_path in [("AG23 (기존 area=0.23)", ROOT / "runs/egvla2_fm/a3c/A3C_AG23_s0.pt"), ("Step B (A3C_Y2_s0)", p_b)]:
        ck = torch.load(ck_path, map_location=DEV, weights_only=False)
        mod = HD.PiZeroFM(int(ck.get("bdim", 12)), int(ck.get("fdim", 768))).to(DEV)
        mod.load_state_dict(ck["state"]); mod.eval()
        c13 = predict_head(mod, BW13, VIS13)
        c16 = predict_head(mod, BW16, VIS16)
        c13_sw = predict_head(mod, BW13, VIS13[perm13])
        n13, n16 = np.linalg.norm(c13, axis=1), np.linalg.norm(c16, axis=1)
        ch_go = float(np.mean([(n13[s:e] > 0.2).mean() for (_, o, p, s, e, _) in meta13 if "가운데" in p and o == "의자"])) * 100
        bk_go = float(np.mean([(n13[s:e] > 0.2).mean() for (_, o, p, s, e, _) in meta13 if "가운데" in p and o == "바구니"])) * 100
        flip = float(((n13 < 0.2) != (np.linalg.norm(c13_sw, axis=1) < 0.2)).mean() * 100)
        st_y2_low = float((n16[(y2_16 >= 0.78) & (y2_16 < 0.86)] < 0.2).mean() * 100)
        st_y2_hi = float((n16[y2_16 >= 0.95] < 0.2).mean() * 100)
        # 11/25 도착 프레임(바구니 y2=1.00, 의자 y2=0.819) 정지율
        st25_b = float((np.linalg.norm(predict_head(mod, BW25_B, mean_vis_b), axis=1) < 0.2).mean() * 100)
        st25_c = float((np.linalg.norm(predict_head(mod, BW25_C, mean_vis_c), axis=1) < 0.2).mean() * 100)
        results[name] = dict(
            basket_center_go=round(bk_go, 1),
            chair_center_go=round(ch_go, 1),
            vis_shuffle_flip_pct=round(flip, 1),
            stop_y2_078_086_pct=round(st_y2_low, 1),
            stop_y2_ge_095_pct=round(st_y2_hi, 1),
            stop_1125_basket_arrival_pct=round(st25_b, 1),
            stop_1125_chair_arrival_pct=round(st25_c, 1),
        )
        print(f"★ {name:24s} | 출발(바:{bk_go:5.1f}% 의:{ch_go:5.1f}%) | 셔플뒤집힘:{flip:4.1f}% | 11/25도착정지(바구니:{st25_b:5.1f}% 의자:{st25_c:5.1f}%)", flush=True)

    # Step C 평가 (1536-d: positive text vs negative text)
    _, D = E5.load()
    z_i = torch.zeros(1, dtype=torch.long, device=DEV)
    txt_pos = E4.txt(D, z_i, torch.tensor([0], dtype=torch.long, device=DEV)).cpu().numpy()
    txt_neg = E4.txt(D, z_i, torch.tensor([E4.NP], dtype=torch.long, device=DEV)).cpu().numpy()
    ck_c = torch.load(p_c, map_location=DEV, weights_only=False)
    mod_c = HD.PiZeroFM(12, 1536).to(DEV)
    mod_c.load_state_dict(ck_c["state"]); mod_c.eval()
    f13_pos = np.concatenate([VIS13, np.repeat(txt_pos, len(VIS13), axis=0)], axis=1)
    f13_neg = np.concatenate([VIS13, np.repeat(txt_neg, len(VIS13), axis=0)], axis=1)
    c13_pos = predict_head(mod_c, BW13, f13_pos)
    c13_neg = predict_head(mod_c, BW13, f13_neg)
    neg_stop = float((np.linalg.norm(c13_neg, axis=1) < 0.2).mean() * 100)
    bk_go_c = float(np.mean([(np.linalg.norm(c13_pos[s:e], axis=1) > 0.2).mean() for (_, o, p, s, e, _) in meta13 if "가운데" in p and o == "바구니"])) * 100
    f25_b = np.concatenate([mean_vis_b, np.repeat(txt_pos, len(BW25_B), axis=0)], axis=1)
    f25_c = np.concatenate([mean_vis_c, np.repeat(txt_pos, len(BW25_C), axis=0)], axis=1)
    st25_bc = float((np.linalg.norm(predict_head(mod_c, BW25_B, f25_b), axis=1) < 0.2).mean() * 100)
    st25_cc = float((np.linalg.norm(predict_head(mod_c, BW25_C, f25_c), axis=1) < 0.2).mean() * 100)
    results["Step C (EG4-08_V4R_Y2_s0)"] = dict(
        basket_center_go=round(bk_go_c, 1),
        neg_instruction_stop_pct=round(neg_stop, 1),
        stop_1125_basket_arrival_pct=round(st25_bc, 1),
        stop_1125_chair_arrival_pct=round(st25_cc, 1),
    )
    print(f"★ Step C (V4R_Y2_s0)      | 바구니출발:{bk_go_c:5.1f}% | 비목표정지:{neg_stop:5.1f}% | 11/25도착정지(바구니:{st25_bc:5.1f}% 의자:{st25_cc:5.1f}%)", flush=True)

    out_p = ROOT / "docs/v5/egvla2_fm/step_bc_results.json"
    out_p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out_p}", flush=True)


def main() -> None:
    p_b = train_step_b(y2_stop=0.95, y2_ramp=0.86, epochs=120, seed=0)
    p_c = train_step_c(y2_stop=0.95, y2_ramp=0.86, seed=0)
    evaluate_all(p_b, p_c)


if __name__ == "__main__":
    main()
