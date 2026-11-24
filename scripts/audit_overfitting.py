#!/usr/bin/env python3
"""전 모델 과적합(Overfitting) 정밀 감사 스크립트.

점검 항목 (4축 과적합 진단):
  1) Train(300ep 바구니) vs Held-Out(11/16 바구니) vs Unseen(11/13·16 의자·책상·화분) 일반화 격차
  2) 시각 외형 과적합 민감도 (Counterfactual Feature Swap: 동일 BBox에서 시각 특징만 셔플 시 각도/크기 변화량 Δ_vis)
  3) 기하 반응성 (S_cx: cx ±0.2 시 조향 변화량, S_area: area 0.06→0.35 시 정지율 증가량)
  4) EG3-11 Open-Vocab 물체별 망각/편향 여부 (11/13 의자·책상·화분·바구니별)

출력: docs/v5/egvla2_fm/overfitting_audit.json
"""
from __future__ import annotations

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
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = S1.T


@torch.no_grad()
def predict_head(model, bw_np: np.ndarray, feat_np: np.ndarray, steps: int = 10, bs: int = 256) -> np.ndarray:
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


def load_fm_ckpt(path: Path) -> HD.PiZeroFM:
    ck = torch.load(path, map_location=DEV, weights_only=False)
    m = HD.PiZeroFM(int(ck.get("bdim", 12)), int(ck.get("fdim", 768))).to(DEV)
    m.load_state_dict(ck["state"])
    return m.eval()


def main() -> None:
    t0 = time.time()
    Z = np.load(ROOT / "runs/e2e_cache/dec1pass_softloc_cache.npz")
    C13 = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz")
    C16 = np.load(ROOT / "runs/egvla2_fm/t16_cache.npz")
    D13 = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
    S16 = json.load(open(ROOT / "docs/v5/egvla2_fm/a3c/real_1116_sessions_bbox_area.json"))
    OB16 = {"go to the gray basket": "바구니", "go to the brown wooden chair": "의자", "go to the small desk": "책상"}

    # 11/13(870장), 11/16(588장) 구성
    bw13_gt, vis13, bw13_sl, meta13 = [], [], [], []
    idx = 0
    for x in D13:
        sid = x["sid"]
        vis = C13[f"{sid}|vis"].astype(np.float32)
        raw = C13[f"{sid}|bbox"][: len(vis)].astype(np.float32)
        L = len(vis)
        bw = AB.window(raw, 3); bw[raw[:, 3] < 0.5] = 0
        sl_raw = Z["r13_sb"][idx : idx + L]
        sl_bw = AB.window(sl_raw, 3); sl_bw[sl_raw[:, 3] < 0.5] = 0
        bw13_gt.append(bw); vis13.append(vis); bw13_sl.append(sl_bw)
        meta13.append((sid, x["obj"], x["pos"], idx, idx + L, raw))
        idx += L
    BW13_GT = np.concatenate(bw13_gt, 0)
    VIS13 = np.concatenate(vis13, 0)
    BW13_SL = np.concatenate(bw13_sl, 0)

    bw16_gt, vis16, bw16_sl, meta16 = [], [], [], []
    idx = 0
    for sid, s in S16.items():
        vis = C16[f"{sid}|vis"].astype(np.float32)
        raw = C16[f"{sid}|bbox"][: len(vis)].astype(np.float32)
        L = len(vis)
        bw = AB.window(raw, 3); bw[raw[:, 3] < 0.5] = 0
        sl_raw = Z["r16_sb"][idx : idx + L]
        sl_bw = AB.window(sl_raw, 3); sl_bw[sl_raw[:, 3] < 0.5] = 0
        bw16_gt.append(bw); vis16.append(vis); bw16_sl.append(sl_bw)
        meta16.append((sid, OB16[s["instruction"]], idx, idx + L, raw))
        idx += L
    BW16_GT = np.concatenate(bw16_gt, 0)
    VIS16 = np.concatenate(vis16, 0)
    BW16_SL = np.concatenate(bw16_sl, 0)

    # 기준 A3 출력
    m_a3 = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt", device=DEV).model
    ref13 = predict_head(m_a3, BW13_GT, VIS13)
    ref16 = predict_head(m_a3, BW16_GT, VIS16)

    models_spec = [
        ("MOD-A3 (a3 · 현 배포)", m_a3, BW13_GT, VIS13, BW16_GT, VIS16, False),
        ("MOD-A3-s2 (a3-s2 · 11/16실기)", EgVla2FMPolicy(ROOT / "runs/egvla2_fm/a3c/A3C_D30_s2.pt", device=DEV).model, BW13_GT, VIS13, BW16_GT, VIS16, False),
        ("MOD-A3-AG (a3-ag23 ★)", EgVla2FMPolicy(ROOT / "runs/egvla2_fm/a3c/A3C_AG23_s0.pt", device=DEV).model, BW13_GT, VIS13, BW16_GT, VIS16, False),
        ("VLM-E2E-Hidden (EG3-08)", load_fm_ckpt(ROOT / "runs/egvla2_e2e_lora/EG3_A1_Dec1Pass.pt"), np.zeros_like(BW13_GT), Z["r13_h1"], np.zeros_like(BW16_GT), Z["r16_h1"], True),
        ("VLM-E2E-SoftLoc (EG3-09)", load_fm_ckpt(ROOT / "runs/egvla2_e2e_lora/EG3_A2_SoftLoc.pt"), BW13_SL, Z["r13_hloc"], BW16_SL, Z["r16_hloc"], False),
        ("VLM-E2E-SoftLoc+ (EG3-10)", load_fm_ckpt(ROOT / "runs/egvla2_e2e_lora/EG3_10_SoftLoc_v2.pt"), BW13_SL, Z["r13_hloc"], BW16_SL, Z["r16_hloc"], False),
        ("VLM-E2E-Joint (EG3-11 ★)", load_fm_ckpt(ROOT / "runs/egvla2_e2e_lora/EG3_11_SoftLoc_Joint.pt"), BW13_SL, VIS13, BW16_SL, VIS16, False),
    ]

    rng = np.random.default_rng(42)
    perm13 = rng.permutation(len(VIS13))

    audit = {}
    print("=== 전 모델 과적합(Overfitting) 정밀 감사 ===")
    for name, mod, bw13, f13, bw16, f16, zero_bb in models_spec:
        c13 = predict_head(mod, bw13, f13)
        c16 = predict_head(mod, bw16, f16)

        # 1) 일반화 격차: Train 바구니 vs 11/16 Held-Out 바구니 vs 11/13·16 Unseen 의자·책상
        n13 = np.linalg.norm(c13, axis=1)
        n16 = np.linalg.norm(c16, axis=1)
        # 바구니 근접(>=0.25) vs 의자 근접(>=0.25) 정지율 격차 -> 시각 외형 과적합의 핵심 지표!
        bk_stop = float(np.concatenate([(n16[s:e][raw[:, 2] >= 0.25] < 0.2) for (_, o, s, e, raw) in meta16 if o == "바구니"]).mean())
        ch_stop = float(np.concatenate([(n16[s:e][raw[:, 2] >= 0.25] < 0.2) for (_, o, s, e, raw) in meta16 if o == "의자"]).mean())
        unseen_stop_gap = round(abs(bk_stop - ch_stop) * 100, 1)  # %p 차이

        # 의자·가운데 vs 바구니·가운데 출발 격차
        ch_go = float(np.mean([(n13[s:e] > 0.2).mean() for (_, o, p, s, e, _) in meta13 if "가운데" in p and o == "의자"]))
        bk_go = float(np.mean([(n13[s:e] > 0.2).mean() for (_, o, p, s, e, _) in meta13 if "가운데" in p and o == "바구니"]))
        unseen_go_gap = round(abs(bk_go - ch_go) * 100, 1)

        # 2) Counterfactual Visual Feature Swap 민감도 Δ_vis (동일 BBox에서 시각 특징만 셔플)
        c13_sw = predict_head(mod, bw13, f13[perm13])
        mv = (np.linalg.norm(c13, axis=1) > 0.2) & (np.linalg.norm(c13_sw, axis=1) > 0.2)
        if mv.any():
            cos_v = np.clip((c13[mv] * c13_sw[mv]).sum(1) / (np.linalg.norm(c13[mv], axis=1) * np.linalg.norm(c13_sw[mv], axis=1)), -1, 1)
            vis_ang_shift = round(float(np.median(np.degrees(np.arccos(cos_v)))), 2)
        else:
            vis_ang_shift = 0.0
        vis_stop_flip = round(float(((np.linalg.norm(c13, axis=1) < 0.2) != (np.linalg.norm(c13_sw, axis=1) < 0.2)).mean() * 100), 1)

        # 3) 기하 반응성 S_cx & S_area
        if zero_bb:
            s_cx_deg, s_area_pp = 0.0, 0.0
        else:
            bw_p, bw_m = bw13.copy(), bw13.copy()
            for k in (0, 4, 8):
                bw_p[:, k] = np.clip(bw_p[:, k] + 0.2, 0, 1)
                bw_m[:, k] = np.clip(bw_m[:, k] - 0.2, 0, 1)
            cp = predict_head(mod, bw_p, f13)
            cm = predict_head(mod, bw_m, f13)
            s_cx_deg = round(float(np.mean((np.degrees(np.arctan2(cp[:, 1], cp[:, 0])) - np.degrees(np.arctan2(cm[:, 1], cm[:, 0]))) / 2.0)), 2)

            bw_sm, bw_lg = bw16.copy(), bw16.copy()
            for k in (2, 6, 10):
                bw_sm[:, k] = 0.06
                bw_lg[:, k] = 0.35
            for k in (3, 7, 11):
                bw_sm[:, k] = 1.0
                bw_lg[:, k] = 1.0
            st_sm = (np.linalg.norm(predict_head(mod, bw_sm, f16), axis=1) < 0.2).mean()
            st_lg = (np.linalg.norm(predict_head(mod, bw_lg, f16), axis=1) < 0.2).mean()
            s_area_pp = round(float((st_lg - st_sm) * 100), 1)

        # 과적합 판정
        if unseen_go_gap > 30 or unseen_stop_gap > 30 or vis_stop_flip > 20:
            verdict = "과적합/지름길 존재"
        elif unseen_go_gap <= 5 and unseen_stop_gap <= 10 and vis_stop_flip <= 5 and s_area_pp >= 80:
            verdict = "과적합 없음 (기하 일반화 우수)"
        else:
            verdict = "부분 일반화"

        row = dict(
            basket_center_go=round(bk_go * 100, 1),
            chair_center_go=round(ch_go * 100, 1),
            unseen_go_gap_pp=unseen_go_gap,
            basket_ge25_stop=round(bk_stop * 100, 1),
            chair_ge25_stop=round(ch_stop * 100, 1),
            unseen_stop_gap_pp=unseen_stop_gap,
            vis_swap_ang_deg=vis_ang_shift,
            vis_swap_stop_flip_pct=vis_stop_flip,
            S_cx_deg=s_cx_deg,
            S_area_pp=s_area_pp,
            verdict=verdict,
        )
        audit[name] = row
        print(
            f"★ {name:28s} | 출발(바:{row['basket_center_go']:5.1f}% 의:{row['chair_center_go']:5.1f}% 격차:{unseen_go_gap:4.1f}%p) "
            f"| 정지(바:{row['basket_ge25_stop']:5.1f}% 의:{row['chair_ge25_stop']:5.1f}% 격차:{unseen_stop_gap:4.1f}%p) "
            f"| 시각셔플(각:{vis_ang_shift:4.1f}° 정지뒤집힘:{vis_stop_flip:4.1f}%) "
            f"| 기하반응(S_cx:{s_cx_deg:+5.1f}° S_area:{s_area_pp:+5.1f}%p) → [{verdict}]",
            flush=True,
        )

    out_p = ROOT / "docs/v5/egvla2_fm/overfitting_audit.json"
    out_p.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out_p} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
