#!/usr/bin/env python3
"""Phase 3-1: VLM-E2E-SoftLoc 고도화 (EG3-10, VLM-E2E-SoftLoc+)

기존 Phase 2(VLM-E2E-SoftLoc, 6.00°)의 원인을 분해·개선한다:
  1) 에피소드 경계를 지키는 실제 3프레임 시계열 Soft-Loc 윈도우 구축 (단일 프레임 반복 제거)
  2) Decoder Hidden State 80% swap + CosineAnnealingLR (200ep)으로 기하(Soft-Loc) 정합 극대화
  3) 대조군: 기존 A3C_AG23_s0 헤드에 Soft-Loc 좌표만 직접 연결(Zero-Shot Soft-Loc)했을 때의 오차 상한 측정
이미 캐시된 runs/e2e_cache/dec1pass_softloc_cache.npz를 재사용해 ~20초 내 완료된다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import eval_subgoal_ablation as AB
import train_egvla2_s1 as S1
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
from scripts.a3c_train_ag import area_servo_label

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = S1.T


@torch.no_grad()
def batch_predict_head(model, bw_np: np.ndarray, feat_np: np.ndarray, steps: int = 10, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(bw_np), bs):
        b = torch.tensor(bw_np[s : s + bs], dtype=torch.float32, device=DEV)
        f = torch.tensor(feat_np[s : s + bs], dtype=torch.float32, device=DEV)
        x = torch.zeros(len(b), 8, 3, device=DEV)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((len(b),), k * dt, device=DEV)
            x = x + dt * model(b, f, x, t)
        out.append(x[:, 0, :2].cpu().numpy())
    return np.concatenate(out, 0)


def build_ep_windows(frames: list[list[int]], sb_arr: np.ndarray) -> np.ndarray:
    """학습 프레임 목록(frames=[(ei, t), ...])에서 에피소드 경계를 유지하며 3프레임 Soft-Loc 윈도우 생성."""
    out = np.zeros((len(sb_arr), 12), dtype=np.float32)
    ep_groups: dict[int, list[int]] = {}
    for idx, (ei, _t) in enumerate(frames):
        ep_groups.setdefault(ei, []).append(idx)
    for idxs in ep_groups.values():
        sub = sb_arr[idxs].astype(np.float32)
        w = AB.window(sub, 3)
        w[sub[:, 3] < 0.5] = 0
        for local_i, global_i in enumerate(idxs):
            out[global_i] = w[local_i]
    return out


def main() -> None:
    t0 = time.time()
    Z = np.load(ROOT / "runs/e2e_cache/dec1pass_softloc_cache.npz")
    eps = S1.load_main()
    cache_c = np.load(T.CACHE)
    E = AB.prep(eps, cache_c, 3)
    frames = json.load(open(ROOT / "runs/e2e_cache/frames.json"))["frames"]

    C13 = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz")
    C16 = np.load(ROOT / "runs/egvla2_fm/t16_cache.npz")
    D13 = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
    S16 = json.load(open(ROOT / "docs/v5/egvla2_fm/a3c/real_1116_sessions_bbox_area.json"))
    OB16 = {"go to the gray basket": "바구니", "go to the brown wooden chair": "의자", "go to the small desk": "책상"}

    bw13_gt, vis13_gt, bw13_sl, meta13 = [], [], [], []
    idx = 0
    for x in D13:
        sid = x["sid"]
        vis = C13[f"{sid}|vis"].astype(np.float32)
        raw = C13[f"{sid}|bbox"][: len(vis)].astype(np.float32)
        L = len(vis)
        bw = AB.window(raw, 3)
        bw[raw[:, 3] < 0.5] = 0
        sl_raw = Z["r13_sb"][idx : idx + L]
        sl_bw = AB.window(sl_raw, 3)
        sl_bw[sl_raw[:, 3] < 0.5] = 0
        bw13_gt.append(bw)
        vis13_gt.append(vis)
        bw13_sl.append(sl_bw)
        meta13.append((sid, x["obj"], x["pos"], x["res"], idx, idx + L, raw))
        idx += L
    BW13_GT = np.concatenate(bw13_gt, 0)
    VIS13_GT = np.concatenate(vis13_gt, 0)
    BW13_SL = np.concatenate(bw13_sl, 0)

    bw16_gt, vis16_gt, bw16_sl, meta16 = [], [], [], []
    idx = 0
    for sid, s in S16.items():
        vis = C16[f"{sid}|vis"].astype(np.float32)
        raw = C16[f"{sid}|bbox"][: len(vis)].astype(np.float32)
        L = len(vis)
        bw = AB.window(raw, 3)
        bw[raw[:, 3] < 0.5] = 0
        sl_raw = Z["r16_sb"][idx : idx + L]
        sl_bw = AB.window(sl_raw, 3)
        sl_bw[sl_raw[:, 3] < 0.5] = 0
        bw16_gt.append(bw)
        vis16_gt.append(vis)
        bw16_sl.append(sl_bw)
        meta16.append((sid, OB16[s["instruction"]], idx, idx + L, raw))
        idx += L
    BW16_GT = np.concatenate(bw16_gt, 0)
    VIS16_GT = np.concatenate(vis16_gt, 0)
    BW16_SL = np.concatenate(bw16_sl, 0)

    p_a3 = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt").model.to(DEV)
    p_ag = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/a3c/A3C_AG23_s0.pt").model.to(DEV)
    ref_a3_13 = batch_predict_head(p_a3, BW13_GT, VIS13_GT)
    ref_a3_16 = batch_predict_head(p_a3, BW16_GT, VIS16_GT)

    bins = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 1.0]

    def eval_pred(name: str, c13: np.ndarray, c16: np.ndarray) -> dict:
        n13 = np.linalg.norm(c13, axis=1)
        n16 = np.linalg.norm(c16, axis=1)
        c_all = np.concatenate([c13, c16], 0)
        r_all = np.concatenate([ref_a3_13, ref_a3_16], 0)
        n_all = np.linalg.norm(c_all, axis=1)
        rn_all = np.linalg.norm(r_all, axis=1)
        mv = (n_all > 0.2) & (rn_all > 0.2)
        cos = np.clip((c_all[mv] * r_all[mv]).sum(1) / (n_all[mv] * rn_all[mv]), -1, 1)
        ang_med = round(float(np.median(np.degrees(np.arccos(cos)))), 2)
        lat_agree = round(float((np.sign(c_all[mv, 1]) == np.sign(r_all[mv, 1])).mean()), 3)

        def g13(pos_sub: str, obj_name: str) -> float:
            return round(
                float(np.mean([(n13[s:e] > 0.2).mean() for (_, o, p, _, s, e, _) in meta13 if pos_sub in p and o == obj_name])),
                3,
            )

        a16 = np.concatenate([raw[:, 2] for (_, _, _, _, raw) in meta16])
        b = np.digitize(a16, bins[1:-1])
        sr_bins = [round(float((n16[b == k] < 0.2).mean()), 3) for k in range(7)]
        obj_ge25 = {
            o_: round(
                float(np.concatenate([(n16[s:e][raw[:, 2] >= 0.25] < 0.2) for (_, o, s, e, raw) in meta16 if o == o_]).mean()),
                3,
            )
            for o_ in ("바구니", "의자")
        }
        res = dict(
            name=name,
            ang_med_vs_a3=ang_med,
            lat_agree_vs_a3=lat_agree,
            chair_center_13=g13("가운데", "의자"),
            desk_center_13=g13("가운데", "작은 책상"),
            pot_center_13=g13("가운데", "화분"),
            basket_center_13=g13("가운데", "바구니"),
            stop_lt10_16=sr_bins[0],
            stop_25_30_16=sr_bins[4],
            stop_30_40_16=sr_bins[5],
            basket_ge25_stop_16=obj_ge25["바구니"],
            chair_ge25_stop_16=obj_ge25["의자"],
        )
        print(
            f"★ {name:34s} | 각도차:{ang_med:>5.2f}°(좌우:{lat_agree:.3f}) | 11/13 의자:{res['chair_center_13']:.2f} 책상:{res['desk_center_13']:.2f} 화분:{res['pot_center_13']:.2f} | 11/16 정지(<.10:{res['stop_lt10_16']:.2f} 25-30:{res['stop_25_30_16']:.2f} 30-40:{res['stop_30_40_16']:.2f} 바구니≥.25:{res['basket_ge25_stop_16']:.2f} 의자≥.25:{res['chair_ge25_stop_16']:.2f})",
            flush=True,
        )
        return res

    results = {}

    # 1) 대조군: 기존 A3C_AG23_s0 헤드에 Soft-Loc 좌표만 Zero-Shot 직접 연결
    c13_zs = batch_predict_head(p_ag, BW13_SL, VIS13_GT)
    c16_zs = batch_predict_head(p_ag, BW16_SL, VIS16_GT)
    results["VLM-E2E-SoftLoc (Zero-Shot A3C_AG23)"] = eval_pred("VLM-E2E-SoftLoc (Zero-Shot AG23)", c13_zs, c16_zs)

    # 2) EG3-10: VLM-E2E-SoftLoc+ (시계열 3프레임 Soft-Loc 윈도우 + 80% Hidden Swap + Cosine LR 200ep)
    BB_TR_GT = torch.from_numpy(np.stack([E[ei]["bbox"][t] for ei, t in frames])).float().to(DEV)
    BB_TR_SL_WIN = torch.from_numpy(build_ep_windows(frames, Z["tr_sb"])).float().to(DEV)
    CH_TR = torch.from_numpy(np.stack([E[ei]["chunk"][t] for ei, t in frames])).float().to(DEV)
    CH_AG_TR = area_servo_label(CH_TR, BB_TR_GT, stop_area=0.23, ramp_start=0.18)
    HLOC_TR = torch.from_numpy(Z["tr_hloc"]).float().to(DEV)
    N_TR = len(frames)

    torch.manual_seed(0)
    g = torch.Generator(device=DEV)
    g.manual_seed(0)
    m_v2 = HD.PiZeroFM(12, 768).to(DEV)
    opt = torch.optim.AdamW(m_v2.parameters(), lr=1.2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-5)

    for ep in range(200):
        m_v2.train()
        perm = torch.randperm(N_TR, device=DEV)
        for s in range(0, N_TR, 64):
            i = perm[s : s + 64]
            B = len(i)
            a = CH_AG_TR[i]
            # 70% 실제 3프레임 Soft-Loc 시계열 윈도우 + 30% GT 윈도우 (Soft-Loc 고유 편향 보정)
            use_sl = (torch.rand(B, 1, device=DEV, generator=g) < 0.7).float()
            bb = use_sl * BB_TR_SL_WIN[i] + (1.0 - use_sl) * BB_TR_GT[i]
            ft = HLOC_TR[i]
            # 80% Decoder Hidden Swap: 조향·정지가 768-d 외형 지름길 대신 미분가능 Soft-Loc 기하에 정합되도록 유도
            u = torch.rand(B, 1, device=DEV, generator=g) < 0.8
            ft_in = torch.where(u, ft[torch.randperm(B, device=DEV)], ft)
            x0 = torch.randn_like(a)
            t = torch.rand(B, device=DEV)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            loss = F.mse_loss(m_v2(bb, ft_in, xt, t), a - x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    c13_v2 = batch_predict_head(m_v2, BW13_SL, Z["r13_hloc"])
    c16_v2 = batch_predict_head(m_v2, BW16_SL, Z["r16_hloc"])
    results["VLM-E2E-SoftLoc+ (EG3-10, 3F-Win+Swap80)"] = eval_pred("VLM-E2E-SoftLoc+ (EG3-10)", c13_v2, c16_v2)

    torch.save(
        dict(arch="C_pi0_adaln", state=m_v2.state_dict(), bdim=12, fdim=768, variant="VLM_E2E_SoftLoc_v2"),
        ROOT / "runs/egvla2_e2e_lora/EG3_10_SoftLoc_v2.pt",
    )

    # track_a_dec1pass_results.json에 병합 저장
    out_path = ROOT / "docs/v5/egvla2_fm/track_a_dec1pass_results.json"
    prev = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    prev.update(results)
    out_path.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
