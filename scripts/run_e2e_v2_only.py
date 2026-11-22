#!/usr/bin/env python3
"""이미 추출된 real_1113_1116_stage2.npz 캐시를 재사용해 고도화 E2E-VLA v2 2종(EG3-06, EG3-07)만
빠르게(batch=128, 15ep, 에폭별 진행 로그) 학습하고 11/13 · 11/16 실기 프레임에서 검증한다.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import train_egvla2_s1 as S1
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
from scripts.a3c_train_ag import area_servo_label
from scripts.eval_e2e_on_real_a3 import (
    PretrainedAlignedProj,
    SpatialGroundingE2E,
    batch_predict_head,
    stage3_tokens,
)

DEV = torch.device("cuda")
T = S1.T
E2E_CACHE = ROOT / "runs/e2e_cache"
OUT_DIR = ROOT / "runs/egvla2_e2e_lora"


def main():
    t0 = time.time()
    C13 = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz")
    C16 = np.load(ROOT / "runs/egvla2_fm/t16_cache.npz")
    D13 = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
    S16 = json.load(open(ROOT / "docs/v5/egvla2_fm/a3c/real_1116_sessions_bbox_area.json"))
    OB16 = {"go to the gray basket": "바구니", "go to the brown wooden chair": "의자", "go to the small desk": "책상"}

    bw13_all, vis13_all, meta13 = [], [], []
    idx = 0
    for x in D13:
        sid = x["sid"]; vis = C13[f"{sid}|vis"].astype(np.float32); raw = C13[f"{sid}|bbox"][:len(vis)].astype(np.float32)
        bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0
        bw13_all.append(bw); vis13_all.append(vis)
        meta13.append((sid, x["obj"], x["pos"], x["res"], idx, idx + len(vis), raw)); idx += len(vis)
    BW13 = np.concatenate(bw13_all, 0); VIS13 = np.concatenate(vis13_all, 0)

    bw16_all, vis16_all, meta16 = [], [], []
    idx = 0
    for sid, s in S16.items():
        vis = C16[f"{sid}|vis"].astype(np.float32); raw = C16[f"{sid}|bbox"][:len(vis)].astype(np.float32)
        bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0
        bw16_all.append(bw); vis16_all.append(vis)
        meta16.append((sid, OB16[s["instruction"]], idx, idx + len(vis), raw)); idx += len(vis)
    BW16 = np.concatenate(bw16_all, 0); VIS16 = np.concatenate(vis16_all, 0)

    p_a3 = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt").model
    ref_a3_13 = batch_predict_head(p_a3, BW13, torch.tensor(VIS13))
    ref_a3_16 = batch_predict_head(p_a3, BW16, torch.tensor(VIS16))

    z = np.load(E2E_CACHE / "real_1113_1116_stage2.npz")
    S2_13 = torch.from_numpy(z["s2_13"]).to(DEV, dtype=torch.float32)
    S2_16 = torch.from_numpy(z["s2_16"]).to(DEV, dtype=torch.float32)

    eps = S1.load_main(); cache_c = np.load(T.CACHE); E = AB.prep(eps, cache_c, 3)
    fr = json.load(open(E2E_CACHE / "frames.json")); frames = fr["frames"]
    S2_TR = torch.from_numpy(np.array(np.load(E2E_CACHE / "stage2.npy", mmap_mode="r"))).to(DEV, dtype=torch.bfloat16)
    BB_TR = torch.from_numpy(np.stack([E[ei]["bbox"][t] for ei, t in frames])).float().to(DEV)
    CH_TR = torch.from_numpy(np.stack([E[ei]["chunk"][t] for ei, t in frames])).float().to(DEV)
    CH_AG_TR = area_servo_label(CH_TR, BB_TR, stop_area=0.23, ramp_start=0.18)
    N_TR = len(frames)

    fl_base = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float32).eval()
    bins = [0, .1, .15, .2, .25, .3, .4, 1.0]

    def eval_predictions(name, c13, c16, feat13_t, head_mod):
        n13 = np.linalg.norm(c13, axis=1); n16 = np.linalg.norm(c16, axis=1)
        c_all = np.concatenate([c13, c16], 0); r_all = np.concatenate([ref_a3_13, ref_a3_16], 0)
        n_all = np.linalg.norm(c_all, axis=1); rn_all = np.linalg.norm(r_all, axis=1)
        mv = (n_all > 0.2) & (rn_all > 0.2)
        cos = np.clip((c_all[mv] * r_all[mv]).sum(1) / (n_all[mv] * rn_all[mv]), -1, 1)
        ang_med = round(float(np.median(np.degrees(np.arccos(cos)))), 2)
        lat_agree = round(float((np.sign(c_all[mv, 1]) == np.sign(r_all[mv, 1])).mean()), 3)

        bw_p, bw_m = BW13.copy(), BW13.copy()
        for k in (0, 4, 8):
            bw_p[:, k] = np.clip(bw_p[:, k] + 0.2, 0, 1); bw_m[:, k] = np.clip(bw_m[:, k] - 0.2, 0, 1)
        cp = batch_predict_head(head_mod, bw_p, feat13_t); cm = batch_predict_head(head_mod, bw_m, feat13_t)
        S_deg = round(float(np.mean((np.degrees(np.arctan2(cp[:, 1], cp[:, 0])) - np.degrees(np.arctan2(cm[:, 1], cm[:, 0]))) / 2.0)), 2)

        def g13(pos_sub, obj_name):
            return round(float(np.mean([(n13[s:e] > 0.2).mean() for (_, o, p, _, s, e, _) in meta13 if pos_sub in p and o == obj_name])), 3)
        a16 = np.concatenate([raw[:, 2] for (_, _, _, _, raw) in meta16]); b = np.digitize(a16, bins[1:-1])
        sr_bins = [round(float((n16[b == k] < 0.2).mean()), 3) for k in range(7)]
        obj_ge25 = {o_: round(float(np.concatenate([(n16[s:e][raw[:, 2] >= 0.25] < 0.2) for (_, o, s, e, raw) in meta16 if o == o_]).mean()), 3) for o_ in ("바구니", "의자")}

        res = dict(name=name, ang_med_vs_a3=ang_med, lat_agree_vs_a3=lat_agree, S_cx_sens_deg=S_deg,
                   chair_center_13=g13("가운데", "의자"), desk_center_13=g13("가운데", "작은 책상"),
                   basket_center_13=g13("가운데", "바구니"), side_13=round(float(np.mean([(n13[s:e] > 0.2).mean() for (_, _, p, _, s, e, _) in meta13 if "가운데" not in p])), 3),
                   stop_lt10_16=sr_bins[0], stop_25_30_16=sr_bins[4], stop_30_40_16=sr_bins[5],
                   basket_ge25_stop_16=obj_ge25["바구니"], chair_ge25_stop_16=obj_ge25["의자"])
        print(f"★ {name:28s} | 각도차:{ang_med:>5.2f}°(좌우:{lat_agree:.3f}) S:{S_deg:>6.2f}° | 11/13 의자중:{res['chair_center_13']:.2f} 책상중:{res['desk_center_13']:.2f} 바구니중:{res['basket_center_13']:.2f} 옆:{res['side_13']:.2f} | 11/16 정지(<.10:{res['stop_lt10_16']:.2f} 25-30:{res['stop_25_30_16']:.2f} 30-40:{res['stop_30_40_16']:.2f} 바구니≥.25:{res['basket_ge25_stop_16']:.2f} 의자≥.25:{res['chair_ge25_stop_16']:.2f})", flush=True)
        return res

    results = {}
    EP = 15
    BS = 128

    # 1. EG3-06: E2E-LoRA-AG
    print("\n[1/2] EG3-06 (E2E-LoRA-AG, 15ep, batch=128) 학습 시작...", flush=True)
    torch.manual_seed(0); g_gen = torch.Generator(device=DEV); g_gen.manual_seed(0)
    lcfg_ag = LoraConfig(r=8, lora_alpha=16, target_modules=r".*blocks\.3\..*(qkv|proj)", lora_dropout=0.05, bias="none")
    vt_ag = get_peft_model(AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float32).vision_tower, lcfg_ag).to(DEV)
    b3_ag = vt_ag.get_base_model()
    proj_ag = PretrainedAlignedProj(fl_base).to(DEV)
    head_ag = HD.PiZeroFM(12, 768).to(DEV)
    opt_ag = torch.optim.AdamW([
        dict(params=[p for p in vt_ag.parameters() if p.requires_grad], lr=1e-4),
        dict(params=proj_ag.parameters(), lr=3e-4),
        dict(params=head_ag.parameters(), lr=1e-3),
    ], weight_decay=1e-2)
    t_ag = time.time()
    for ep in range(EP):
        vt_ag.train(); proj_ag.train(); head_ag.train()
        perm = torch.randperm(N_TR, device=DEV); tot = 0.0; nb = 0
        for s in range(0, N_TR, BS):
            i = perm[s:s+BS]; B = len(i)
            bb = BB_TR[i]; a = CH_AG_TR[i]
            can_drop = (bb[:, 10:11] < 0.18).float()
            bb_in = bb * (1.0 - (torch.rand(B, 1, device=DEV) < 0.10).float() * can_drop)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ft = proj_ag(stage3_tokens(b3_ag, S2_TR[i]).mean(1).float()).float()
            u = torch.rand(B, 1, device=DEV, generator=g_gen) < 0.5
            ft_in = torch.where(u, ft[torch.randperm(B, device=DEV)], ft)
            x0 = torch.randn_like(a); t = torch.rand(B, device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            l = F.mse_loss(head_ag(bb_in, ft_in, xt, t), a - x0); opt_ag.zero_grad(); l.backward(); opt_ag.step()
            tot += float(l); nb += 1
        print(f"  [EG3-06] ep {ep+1:2d}/{EP} loss={tot/nb:.4f} ({time.time()-t_ag:.0f}s)", flush=True)
    vt_ag.eval(); proj_ag.eval(); head_ag.eval()
    with torch.no_grad():
        def eval_ag(s2_t):
            out = []
            for s in range(0, len(s2_t), 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(proj_ag(stage3_tokens(b3_ag, s2_t[s:s+64]).mean(1).float()).float())
            return torch.cat(out, 0)
        fag_13, fag_16 = eval_ag(S2_13), eval_ag(S2_16)
    results["EG3-06 (E2E-LoRA-AG)"] = eval_predictions("EG3-06 (E2E-LoRA-AG)", batch_predict_head(head_ag, BW13, fag_13), batch_predict_head(head_ag, BW16, fag_16), fag_13, head_ag)
    torch.save(dict(proj=proj_ag.state_dict(), head=head_ag.state_dict(), lora={k: v.cpu() for k, v in vt_ag.state_dict().items() if "lora_" in k}), OUT_DIR / "EG3_06_E2E_LoRA_AG.pt")
    del vt_ag, b3_ag; torch.cuda.empty_cache()

    # 2. EG3-07: E2E-LoRA-SpatialAux
    print("\n[2/2] EG3-07 (E2E-LoRA-SpatialAux, 15ep, batch=128) 학습 시작...", flush=True)
    torch.manual_seed(0); g_gen2 = torch.Generator(device=DEV); g_gen2.manual_seed(0)
    lcfg_sp = LoraConfig(r=8, lora_alpha=16, target_modules=r".*blocks\.3\..*(qkv|proj)", lora_dropout=0.05, bias="none")
    vt_sp = get_peft_model(AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float32).vision_tower, lcfg_sp).to(DEV)
    b3_sp = vt_sp.get_base_model()
    sp_mod = SpatialGroundingE2E(fl_base).to(DEV)
    head_sp = HD.PiZeroFM(12, 768).to(DEV)
    opt_sp = torch.optim.AdamW([
        dict(params=[p for p in vt_sp.parameters() if p.requires_grad], lr=2e-4),
        dict(params=sp_mod.parameters(), lr=5e-4),
        dict(params=head_sp.parameters(), lr=1e-3),
    ], weight_decay=1e-2)
    t_sp = time.time()
    for ep in range(EP):
        vt_sp.train(); sp_mod.train(); head_sp.train()
        perm = torch.randperm(N_TR, device=DEV); tot_f = 0.0; tot_g = 0.0; nb = 0
        for s in range(0, N_TR, BS):
            i = perm[s:s+BS]; B = len(i)
            bb = BB_TR[i]; a = CH_AG_TR[i]; gt_cur_box = bb[:, 8:12]
            can_drop = (bb[:, 10:11] < 0.18).float()
            bb_in = bb * (1.0 - (torch.rand(B, 1, device=DEV) < 0.10).float() * can_drop)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok = stage3_tokens(b3_sp, S2_TR[i]).float()
                ft, pred_box = sp_mod(tok)
            loss_gnd = F.smooth_l1_loss(pred_box, gt_cur_box)
            u = torch.rand(B, 1, device=DEV, generator=g_gen2) < 0.5
            ft_in = torch.where(u, ft.float()[torch.randperm(B, device=DEV)], ft.float())
            x0 = torch.randn_like(a); t = torch.rand(B, device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            loss_fm = F.mse_loss(head_sp(bb_in, ft_in, xt, t), a - x0)
            loss = loss_fm + 0.5 * loss_gnd
            opt_sp.zero_grad(); loss.backward(); opt_sp.step()
            tot_f += float(loss_fm); tot_g += float(loss_gnd); nb += 1
        print(f"  [EG3-07] ep {ep+1:2d}/{EP} fm={tot_f/nb:.4f} gnd={tot_g/nb:.4f} ({time.time()-t_sp:.0f}s)", flush=True)
    vt_sp.eval(); sp_mod.eval(); head_sp.eval()
    with torch.no_grad():
        def eval_sp(s2_t):
            fl, bl = [], []
            for s in range(0, len(s2_t), 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f, pb = sp_mod(stage3_tokens(b3_sp, s2_t[s:s+64]).float())
                fl.append(f.float()); bl.append(pb.float().cpu().numpy())
            return torch.cat(fl, 0), np.concatenate(bl, 0)
        fsp_13, pb_13 = eval_sp(S2_13); fsp_16, pb_16 = eval_sp(S2_16)
    results["EG3-07 (E2E-SpatialAux)"] = eval_predictions("EG3-07 (E2E-SpatialAux)", batch_predict_head(head_sp, BW13, fsp_13), batch_predict_head(head_sp, BW16, fsp_16), fsp_13, head_sp)

    bk_err_cx = [np.abs(pb_16[s:e, 0] - raw[:, 0]) for (_, o, s, e, raw) in meta16 if o == "바구니"]
    bk_err_ar = [np.abs(pb_16[s:e, 2] - raw[:, 2]) for (_, o, s, e, raw) in meta16 if o == "바구니"]
    cx_mae = round(float(np.mean(np.concatenate(bk_err_cx))), 3); ar_mae = round(float(np.mean(np.concatenate(bk_err_ar))), 3)
    print(f"  [EG3-07 내부 AuxGnd 헤드 (generate() 없이 직접 예측)] 11/16 바구니 cx MAE={cx_mae}, area MAE={ar_mae}", flush=True)

    torch.save(dict(sp_mod=sp_mod.state_dict(), head=head_sp.state_dict(), lora={k: v.cpu() for k, v in vt_sp.state_dict().items() if "lora_" in k}, aux_basket_cx_mae=cx_mae, aux_basket_area_mae=ar_mae), OUT_DIR / "EG3_07_E2E_LoRA_SpatialAux.pt")
    (ROOT / "docs/v5/egvla2_fm/e2e_v2_results.json").write_text(json.dumps(dict(models=results, aux_basket_cx_mae=cx_mae, aux_basket_area_mae=ar_mae), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
