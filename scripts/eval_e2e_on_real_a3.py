#!/usr/bin/env python3
"""실제 A3/a3-s2 실기 세션(11/13 65세션 870장 + 11/16 45세션 588장) 위에서
기존 E2E 모델들(EG3-01, EG3-03 B0, EG3-04 B1)의 프레임별 성능을 시뮬레이션 검증하고,
기존 E2E의 3대 병목(사전학습 투영 파괴, 2D 공간 좌표 소실 평균 풀링, 원시 라벨 시각 지름길 과적합)을
해결한 고도화 E2E-VLA v2(EG3-06 E2E-LoRA-AG, EG3-07 E2E-LoRA-SpatialAux)를 학습·비교한다.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import train_egvla2_s1 as S1
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
from scripts.a3c_train_ag import area_servo_label
from scripts.train_egvla2_e2e_lora import E2EVLA

DEV = torch.device("cuda")
T = S1.T
E2E_CACHE = ROOT / "runs/e2e_cache"
OUT_DIR = ROOT / "runs/egvla2_e2e_lora"
OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
X0_SINGLE = torch.randn(1, 8, 8, 3, device=DEV)


@torch.no_grad()
def batch_predict_head(head: nn.Module, bw_np: np.ndarray, feat_t: torch.Tensor) -> np.ndarray:
    head.eval()
    B = len(bw_np)
    bb = torch.tensor(bw_np, device=DEV, dtype=torch.float32).repeat_interleave(8, 0)
    ft = feat_t.to(DEV, dtype=torch.float32).repeat_interleave(8, 0)
    x = X0_SINGLE.repeat(B, 1, 1, 1).view(B * 8, 8, 3).clone()
    for i in range(10):
        t = torch.full((B * 8,), i / 10.0, device=DEV)
        x = x + head(bb, ft, x, t) / 10.0
    return x.view(B, 8, 8, 3).mean(1)[:, 0, :2].cpu().numpy()


class PretrainedAlignedProj(nn.Module):
    """Florence-2의 사전학습된 image_projection(1024->768) + image_proj_norm으로 초기화."""
    def __init__(self, fl_model):
        super().__init__()
        self.proj = nn.Linear(1024, 768, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(fl_model.image_projection.T)
        self.norm = nn.LayerNorm(768)
        with torch.no_grad():
            self.norm.weight.copy_(fl_model.image_proj_norm.weight)
            self.norm.bias.copy_(fl_model.image_proj_norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class SpatialGroundingE2E(nn.Module):
    """24x24 DaViT stage-3 공간 토큰에서 2D 좌표 임베딩 + 공간 어텐션으로
    1) 보조 서브골 BBox (cx, cy, area, has) 직접 예측 (generate() 없이 15ms 내 추론 가능)
    2) 타깃 집중 특징 + 전역 특징 결합 (768-d)
    """
    def __init__(self, fl_model):
        super().__init__()
        self.vproj = PretrainedAlignedProj(fl_model)
        # 24x24 2D normalized coordinate grid (1, 576, 2)
        gy, gx = torch.meshgrid(torch.linspace(0, 1, 24), torch.linspace(0, 1, 24), indexing="ij")
        coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)
        self.register_buffer("coords", coords)
        self.coord_emb = nn.Linear(2, 768)
        self.attn_head = nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 1))
        self.box_head = nn.Sequential(nn.Linear(768 + 2, 128), nn.GELU(), nn.Linear(128, 4))
        self.fuse = nn.Sequential(nn.LayerNorm(768 * 2), nn.Linear(768 * 2, 768), nn.GELU())

    def forward(self, tokens_1024: torch.Tensor):
        # tokens_1024: (B, 576, 1024)
        h = self.vproj(tokens_1024) + self.coord_emb(self.coords)  # (B, 576, 768)
        logits = self.attn_head(h).squeeze(-1)                     # (B, 576)
        w = torch.softmax(logits, dim=-1)                          # (B, 576)
        exp_xy = torch.einsum("bn,bnd->bd", w, self.coords.expand(len(h), -1, -1))  # (B, 2) soft-argmax (cx, cy)
        att_feat = torch.einsum("bn,bnd->bd", w, h)                # (B, 768)
        glb_feat = h.mean(dim=1)                                   # (B, 768)
        raw_box = self.box_head(torch.cat([att_feat, exp_xy], dim=-1))
        pred_cx = torch.sigmoid(raw_box[:, 0] + torch.logit(exp_xy[:, 0].clamp(0.02, 0.98)))
        pred_cy = torch.sigmoid(raw_box[:, 1] + torch.logit(exp_xy[:, 1].clamp(0.02, 0.98)))
        pred_area = torch.sigmoid(raw_box[:, 2]) * 0.65
        pred_has = torch.sigmoid(raw_box[:, 3])
        pred_bbox = torch.stack([pred_cx, pred_cy, pred_area, pred_has], dim=-1)
        feat_768 = self.fuse(torch.cat([glb_feat, att_feat], dim=-1))
        return feat_768, pred_bbox


def stage3_tokens(base, h, size=(48, 48)):
    h, size = base.convs[3](h, size)
    h, size = base.blocks[3](h, size)
    return h  # (B, 576, 1024)


def main():
    t_start = time.time()
    print("[1/5] 데이터셋 및 11/13(870장)·11/16(588장) 실기 프레임 로드 중...", flush=True)
    C13 = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz")
    C16 = np.load(ROOT / "runs/egvla2_fm/t16_cache.npz")
    D13 = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
    S16 = json.load(open(ROOT / "docs/v5/egvla2_fm/a3c/real_1116_sessions_bbox_area.json"))
    R16 = json.load(open(ROOT / "docs/v5/egvla2_fm/a3c/replay_1116.json"))
    OB16 = {"go to the gray basket": "바구니", "go to the brown wooden chair": "의자", "go to the small desk": "책상"}

    bw13_all, vis13_all, meta13, img13_paths = [], [], [], []
    idx = 0
    for x in D13:
        sid = x["sid"]
        vis = C13[f"{sid}|vis"].astype(np.float32)
        raw = C13[f"{sid}|bbox"][:len(vis)].astype(np.float32)
        bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0
        bw13_all.append(bw); vis13_all.append(vis)
        meta13.append((sid, x["obj"], x["pos"], x["res"], idx, idx + len(vis), raw))
        for fi in range(len(vis)):
            img13_paths.append(ROOT / f"docs/v5/figures/objtest/{sid}_{fi}.jpg")
        idx += len(vis)
    BW13 = np.concatenate(bw13_all, 0); VIS13 = np.concatenate(vis13_all, 0)

    bw16_all, vis16_all, meta16, img16_paths = [], [], [], []
    idx = 0
    for sid, s in S16.items():
        vis = C16[f"{sid}|vis"].astype(np.float32)
        raw = C16[f"{sid}|bbox"][:len(vis)].astype(np.float32)
        bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0
        bw16_all.append(bw); vis16_all.append(vis)
        meta16.append((sid, OB16[s["instruction"]], idx, idx + len(vis), raw))
        for fi in R16[sid]["frames"]:
            img16_paths.append(ROOT / f"docs/v5/figures/objtest_1116/{sid}_{fi}.jpg")
        idx += len(vis)
    BW16 = np.concatenate(bw16_all, 0); VIS16 = np.concatenate(vis16_all, 0)

    # Reference A3 predictions on 11/13 and 11/16
    p_a3 = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt").model
    p_s2 = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/a3c/A3C_D30_s2.pt").model
    p_ag = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/a3c/A3C_AG23_s0.pt").model
    ref_a3_13 = batch_predict_head(p_a3, BW13, torch.tensor(VIS13))
    ref_a3_16 = batch_predict_head(p_a3, BW16, torch.tensor(VIS16))
    ref_s2_13 = batch_predict_head(p_s2, BW13, torch.tensor(VIS13))
    ref_s2_16 = batch_predict_head(p_s2, BW16, torch.tensor(VIS16))

    bins = [0, .1, .15, .2, .25, .3, .4, 1.0]

    def eval_predictions(name, c13, c16, feat13_t=None, feat16_t=None, head_mod=None):
        n13 = np.linalg.norm(c13, axis=1)
        n16 = np.linalg.norm(c16, axis=1)

        # 1) Steering comparison vs A3 on moving frames (11/13 basket + 11/16 basket)
        c_all = np.concatenate([c13, c16], 0)
        r_all = np.concatenate([ref_a3_13, ref_a3_16], 0)
        n_all = np.linalg.norm(c_all, axis=1)
        rn_all = np.linalg.norm(r_all, axis=1)
        mv_mask = (n_all > 0.2) & (rn_all > 0.2)
        if mv_mask.any():
            cos = np.clip((c_all[mv_mask] * r_all[mv_mask]).sum(1) / (n_all[mv_mask] * rn_all[mv_mask]), -1, 1)
            ang_med = round(float(np.median(np.degrees(np.arccos(cos)))), 2)
            ang_mean = round(float(np.mean(np.degrees(np.arccos(cos)))), 2)
            lat_agree = round(float((np.sign(c_all[mv_mask, 1]) == np.sign(r_all[mv_mask, 1])).mean()), 3)
        else:
            ang_med, ang_mean, lat_agree = None, None, None

        # Counterfactual cx sensitivity S (shift cx by +0.2 vs -0.2 on 11/13)
        S_deg = None
        if head_mod is not None and feat13_t is not None:
            bw_p, bw_m = BW13.copy(), BW13.copy()
            for k in (0, 4, 8):
                bw_p[:, k] = np.clip(bw_p[:, k] + 0.2, 0, 1)
                bw_m[:, k] = np.clip(bw_m[:, k] - 0.2, 0, 1)
            cp = batch_predict_head(head_mod, bw_p, feat13_t)
            cm = batch_predict_head(head_mod, bw_m, feat13_t)
            ap = np.degrees(np.arctan2(cp[:, 1], cp[:, 0]))
            am = np.degrees(np.arctan2(cm[:, 1], cm[:, 0]))
            S_deg = round(float(np.mean((ap - am) / 2.0)), 2)

        # 2) 11/13 group movement rates
        def g13(pos_sub, obj_name):
            vals = [(n13[s:e] > 0.2).mean() for (sid, o, p, r, s, e, raw) in meta13 if pos_sub in p and o == obj_name]
            return round(float(np.mean(vals)), 3)

        # 3) 11/16 stop rates by area bin & object >= 0.25
        a16 = np.concatenate([raw[:, 2] for (_, _, _, _, raw) in meta16])
        b = np.digitize(a16, bins[1:-1])
        sr_bins = [round(float((n16[b == k] < 0.2).mean()), 3) for k in range(7)]
        obj_ge25 = {}
        for target_o in ("바구니", "의자"):
            m_list = [(n16[s:e][raw[:, 2] >= 0.25] < 0.2) for (_, o, s, e, raw) in meta16 if o == target_o]
            obj_ge25[target_o] = round(float(np.concatenate(m_list).mean()), 3)

        res = dict(
            name=name,
            ang_med_vs_a3=ang_med,
            ang_mean_vs_a3=ang_mean,
            lat_agree_vs_a3=lat_agree,
            S_cx_sens_deg=S_deg,
            chair_center_13=g13("가운데", "의자"),
            desk_center_13=g13("가운데", "작은 책상"),
            pot_center_13=g13("가운데", "화분"),
            basket_center_13=g13("가운데", "바구니"),
            side_13=round(float(np.mean([(n13[s:e] > 0.2).mean() for (_, o, p, r, s, e, raw) in meta13 if "가운데" not in p])), 3),
            stop_lt10_16=sr_bins[0],
            stop_25_30_16=sr_bins[4],
            stop_30_40_16=sr_bins[5],
            basket_ge25_stop_16=obj_ge25["바구니"],
            chair_ge25_stop_16=obj_ge25["의자"],
        )
        print(f"★ {name:28s} | 각도차:{str(ang_med):>5s}°(좌우:{str(lat_agree):>5s}) S:{str(S_deg):>6s}° | 11/13 의자중:{res['chair_center_13']:.2f} 책상중:{res['desk_center_13']:.2f} 바구니중:{res['basket_center_13']:.2f} 옆:{res['side_13']:.2f} | 11/16 정지(<.10:{res['stop_lt10_16']:.2f} 25-30:{res['stop_25_30_16']:.2f} 30-40:{res['stop_30_40_16']:.2f} 바구니≥.25:{res['basket_ge25_stop_16']:.2f} 의자≥.25:{res['chair_ge25_stop_16']:.2f})", flush=True)
        return res

    results = {}
    results["a3 (EG2-09 배포)"] = eval_predictions("a3 (EG2-09 배포)", ref_a3_13, ref_a3_16, torch.tensor(VIS13), torch.tensor(VIS16), p_a3)
    results["a3-s2 (EG2-14 11/16실기)"] = eval_predictions("a3-s2 (EG2-14 11/16실기)", ref_s2_13, ref_s2_16, torch.tensor(VIS13), torch.tensor(VIS16), p_s2)
    c_ag_13 = batch_predict_head(p_ag, BW13, torch.tensor(VIS13))
    c_ag_16 = batch_predict_head(p_ag, BW16, torch.tensor(VIS16))
    results["a3-ag23 (EG2-16 면적정합)"] = eval_predictions("a3-ag23 (EG2-16 면적정합)", c_ag_13, c_ag_16, torch.tensor(VIS13), torch.tensor(VIS16), p_ag)

    # 2. Extract DaViT stage2 (for B0/B1/Upgraded E2E) and evaluate EG3-01 (e2e_vla_florence2_lora_latest.pt)
    print("\n[2/5] Florence-2 로드 및 11/13·11/16 실기 프레임(1,458장) 비전 백본 통과 중...", flush=True)
    proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    fl_base = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float32).to(DEV).eval()

    # Cache pixel_values / stage2 features for 11/13 and 11/16 so we don't recompute
    real_s2_cache = ROOT / "runs/e2e_cache/real_1113_1116_stage2.npz"
    if not real_s2_cache.exists():
        vt = fl_base.vision_tower
        def extract_s2(paths):
            out_s2 = []
            BS = 32
            for s in range(0, len(paths), BS):
                ims = [Image.open(p).convert("RGB") for p in paths[s:s+BS]]
                px = proc(images=ims, return_tensors="pt")["pixel_values"].to(DEV)
                h, size = px, (px.size(2), px.size(3))
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for i in range(3):
                        h, size = vt.convs[i](h, size); h, size = vt.blocks[i](h, size)
                out_s2.append(h.half().cpu().numpy())
            return np.concatenate(out_s2, 0)
        s2_13 = extract_s2(img13_paths)
        s2_16 = extract_s2(img16_paths)
        np.savez(real_s2_cache, s2_13=s2_13, s2_16=s2_16)
        print(f"  Saved {real_s2_cache} ({time.time()-t_start:.1f}s)", flush=True)
    else:
        z = np.load(real_s2_cache)
        s2_13, s2_16 = z["s2_13"], z["s2_16"]
        print(f"  Loaded cached {real_s2_cache} ({time.time()-t_start:.1f}s)", flush=True)

    S2_13 = torch.from_numpy(s2_13).to(DEV, dtype=torch.float32)
    S2_16 = torch.from_numpy(s2_16).to(DEV, dtype=torch.float32)

    # Evaluate EG3-01 (e2e_vla_florence2_lora_latest.pt - trained with dummy bbox [0.5,0.5,0.2,0.2] on 15 episodes)
    print("\n[3/5] 기존 EG3-01 (e2e-lora-smoke, 354MB 체크포인트) 실기 프레임 시뮬레이션 평가...", flush=True)
    fl_eg301 = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float32)
    lcfg_all = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv", "proj"], lora_dropout=0.05, bias="none")
    fl_eg301.vision_tower = get_peft_model(fl_eg301.vision_tower, lcfg_all)
    head_301 = HD.PiZeroFM(12, 768)
    e2e_301 = E2EVLA(fl_eg301, head_301, feat_dim=1024, hidden_dim=128)
    ck301 = torch.load(ROOT / "runs/egvla2_e2e_lora/e2e_vla_florence2_lora_latest.pt", map_location="cpu", weights_only=False)
    e2e_301.vlm.vision_tower.load_state_dict(ck301["vision_lora"])
    e2e_301.head.load_state_dict(ck301["action_head"])
    e2e_301.feat_proj.load_state_dict(ck301["feat_proj"])
    e2e_301.to(DEV).eval()

    @torch.no_grad()
    def extract_eg301_feats(paths):
        out_f = []
        for s in range(0, len(paths), 32):
            ims = [Image.open(p).convert("RGB") for p in paths[s:s+32]]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = e2e_301.extract_vlm_features(px).float()
            out_f.append(f)
        return torch.cat(out_f, 0)

    f301_13 = extract_eg301_feats(img13_paths)
    f301_16 = extract_eg301_feats(img16_paths)
    # EG3-01 trained with dummy bbox [0.5, 0.5, 0.2, 0.2] * 3
    dummy13 = np.tile(np.array([0.5, 0.5, 0.2, 0.2] * 3, np.float32), (len(BW13), 1))
    dummy16 = np.tile(np.array([0.5, 0.5, 0.2, 0.2] * 3, np.float32), (len(BW16), 1))
    c301_13 = batch_predict_head(e2e_301.head, dummy13, f301_13)
    c301_16 = batch_predict_head(e2e_301.head, dummy16, f301_16)
    results["EG3-01 (e2e-lora-smoke 15ep)"] = eval_predictions("EG3-01 (e2e-lora-smoke 15ep)", c301_13, c301_16, f301_13, f301_16, e2e_301.head)
    del e2e_301, fl_eg301
    torch.cuda.empty_cache()

    # 3. Load 300-episode stage2 cache (4,860 frames) into memory for fast B0 / B1 / Upgraded E2E training!
    print("\n[4/5] 300에피소드 stage2 캐시(4,860장) 로드 및 기존 E2E(EG3-03 B0, EG3-04 B1) 학습·평가...", flush=True)
    eps = S1.load_main(); cache_c = np.load(T.CACHE); E = AB.prep(eps, cache_c, 3)
    fr = json.load(open(E2E_CACHE / "frames.json")); frames = fr["frames"]
    stage2_mmap = np.load(E2E_CACHE / "stage2.npy", mmap_mode="r")
    S2_TR = torch.from_numpy(np.asarray(stage2_mmap)).to(DEV, dtype=torch.bfloat16)  # (4860, 2304, 512) in GPU memory (~10.7GB)
    bbs = np.stack([E[ei]["bbox"][t] for ei, t in frames])
    chs = np.stack([E[ei]["chunk"][t] for ei, t in frames])
    BB_TR = torch.from_numpy(bbs).float().to(DEV)
    CH_TR = torch.from_numpy(chs).float().to(DEV)
    N_TR = len(frames)

    # Precompute frozen stage-3 tokens on S2_TR, S2_13, S2_16
    vt_base = fl_base.vision_tower.eval()
    @torch.no_grad()
    def get_frozen_tokens(s2_tensor):
        out = []
        for s in range(0, len(s2_tensor), 64):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok = stage3_tokens(vt_base, s2_tensor[s:s+64].float())
            out.append(tok.float())
        return torch.cat(out, 0)

    tok_tr_frozen = get_frozen_tokens(S2_TR)
    tok_13_frozen = get_frozen_tokens(S2_13)
    tok_16_frozen = get_frozen_tokens(S2_16)
    pool_tr_frozen = tok_tr_frozen.mean(1)
    pool_13_frozen = tok_13_frozen.mean(1)
    pool_16_frozen = tok_16_frozen.mean(1)

    # 3-A. EG3-03: B0_frozen_davit (120ep, original recipe: random Proj, actions_exec, drop=0.3)
    torch.manual_seed(0)
    proj_b0 = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 768), nn.GELU()).to(DEV)
    head_b0 = HD.PiZeroFM(12, 768).to(DEV)
    opt_b0 = torch.optim.AdamW(list(proj_b0.parameters()) + list(head_b0.parameters()), lr=1e-3, weight_decay=1e-2)
    for ep in range(120):
        perm = torch.randperm(N_TR, device=DEV)
        for s in range(0, N_TR, 64):
            i = perm[s:s+64]; bb = BB_TR[i] * (torch.rand(len(i), 1, device=DEV) > 0.3).float()
            ft = proj_b0(pool_tr_frozen[i]); a = CH_TR[i]
            x0 = torch.randn_like(a); t = torch.rand(len(i), device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            l = F.mse_loss(head_b0(bb, ft, xt, t), a - x0); opt_b0.zero_grad(); l.backward(); opt_b0.step()
    proj_b0.eval(); head_b0.eval()
    with torch.no_grad():
        fb0_13, fb0_16 = proj_b0(pool_13_frozen), proj_b0(pool_16_frozen)
    results["EG3-03 (B0 고정DaViT 원본)"] = eval_predictions("EG3-03 (B0 고정DaViT 원본)", batch_predict_head(head_b0, BW13, fb0_13), batch_predict_head(head_b0, BW16, fb0_16), fb0_13, fb0_16, head_b0)

    # 3-B. EG3-04: B1_lora_last_stage (20ep, original recipe: random Proj, actions_exec, drop=0.3)
    torch.manual_seed(0)
    lcfg_b1 = LoraConfig(r=8, lora_alpha=16, target_modules=r".*blocks\.3\..*(qkv|proj)", lora_dropout=0.05, bias="none")
    vt_b1 = get_peft_model(AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float32).vision_tower, lcfg_b1).to(DEV)
    b3_b1 = vt_b1.get_base_model()
    proj_b1 = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 768), nn.GELU()).to(DEV)
    head_b1 = HD.PiZeroFM(12, 768).to(DEV)
    opt_b1 = torch.optim.AdamW([
        dict(params=[p for p in vt_b1.parameters() if p.requires_grad], lr=2e-4),
        dict(params=list(proj_b1.parameters()) + list(head_b1.parameters()), lr=1e-3),
    ], weight_decay=1e-2)
    t_b1 = time.time()
    for ep in range(20):
        vt_b1.train(); proj_b1.train(); head_b1.train()
        perm = torch.randperm(N_TR, device=DEV)
        for s in range(0, N_TR, 64):
            i = perm[s:s+64]; bb = BB_TR[i] * (torch.rand(len(i), 1, device=DEV) > 0.3).float(); a = CH_TR[i]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ft = proj_b1(stage3_tokens(b3_b1, S2_TR[i]).mean(1).float())
            x0 = torch.randn_like(a); t = torch.rand(len(i), device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            l = F.mse_loss(head_b1(bb, ft.float(), xt, t), a - x0); opt_b1.zero_grad(); l.backward(); opt_b1.step()
    print(f"  Trained EG3-04 (B1 original 20ep) in {time.time()-t_b1:.1f}s", flush=True)
    vt_b1.eval(); proj_b1.eval(); head_b1.eval()
    with torch.no_grad():
        def eval_b1_feats(s2_t):
            out = []
            for s in range(0, len(s2_t), 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(proj_b1(stage3_tokens(b3_b1, s2_t[s:s+64]).mean(1).float()).float())
            return torch.cat(out, 0)
        fb1_13, fb1_16 = eval_b1_feats(S2_13), eval_b1_feats(S2_16)
    results["EG3-04 (B1 LoRA 원본 20ep)"] = eval_predictions("EG3-04 (B1 LoRA 원본 20ep)", batch_predict_head(head_b1, BW13, fb1_13), batch_predict_head(head_b1, BW16, fb1_16), fb1_13, fb1_16, head_b1)
    torch.save(dict(proj=proj_b1.state_dict(), head=head_b1.state_dict()), OUT_DIR / "EG3_04_B1_orig.pt")
    del vt_b1, b3_b1
    torch.cuda.empty_cache()

    # 4. Deeply Upgraded E2E-VLA v2 Architectures!
    print("\n[5/5] 고도화 E2E-VLA v2 학습 및 실기 시뮬레이션 검증...", flush=True)
    CH_AG_TR = area_servo_label(CH_TR, BB_TR, stop_area=0.23, ramp_start=0.18)

    # 4-A. EG3-06: E2E-LoRA-AG (Pretrained ImageProj + DaViT Stage3 LoRA + Area-Grounded Servo Label + Feature Swap 50% + Drop 10%)
    torch.manual_seed(0)
    g_gen = torch.Generator(device=DEV); g_gen.manual_seed(0)
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
    for ep in range(40):
        vt_ag.train(); proj_ag.train(); head_ag.train()
        perm = torch.randperm(N_TR, device=DEV)
        for s in range(0, N_TR, 64):
            i = perm[s:s+64]; B = len(i)
            bb = BB_TR[i]; a = CH_AG_TR[i]
            can_drop = (bb[:, 10:11] < 0.18).float()
            bb_in = bb * (1.0 - (torch.rand(B, 1, device=DEV) < 0.10).float() * can_drop)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ft = proj_ag(stage3_tokens(b3_ag, S2_TR[i]).mean(1).float()).float()
            u = torch.rand(B, 1, device=DEV, generator=g_gen) < 0.5
            ft_in = torch.where(u, ft[torch.randperm(B, device=DEV)], ft)
            x0 = torch.randn_like(a); t = torch.rand(B, device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            l = F.mse_loss(head_ag(bb_in, ft_in, xt, t), a - x0); opt_ag.zero_grad(); l.backward(); opt_ag.step()
    print(f"  Trained EG3-06 (E2E-LoRA-AG 40ep) in {time.time()-t_ag:.1f}s (loss={l.item():.4f})", flush=True)
    vt_ag.eval(); proj_ag.eval(); head_ag.eval()
    with torch.no_grad():
        def eval_ag_feats(s2_t):
            out = []
            for s in range(0, len(s2_t), 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(proj_ag(stage3_tokens(b3_ag, s2_t[s:s+64]).mean(1).float()).float())
            return torch.cat(out, 0)
        fag_13, fag_16 = eval_ag_feats(S2_13), eval_ag_feats(S2_16)
    results["★ EG3-06 (E2E-LoRA-AG 고도화)"] = eval_predictions("★ EG3-06 (E2E-LoRA-AG 고도화)", batch_predict_head(head_ag, BW13, fag_13), batch_predict_head(head_ag, BW16, fag_16), fag_13, fag_16, head_ag)
    torch.save(dict(proj=proj_ag.state_dict(), head=head_ag.state_dict(), lora={k: v.cpu() for k, v in vt_ag.state_dict().items() if "lora_" in k}), OUT_DIR / "EG3_06_E2E_LoRA_AG.pt")
    del vt_ag, b3_ag
    torch.cuda.empty_cache()

    # 4-B. EG3-07: E2E-LoRA-SpatialAux (24x24 2D Spatial Grounding Attention + Aux BBox Regression Loss + Area-Grounded FM Loss)
    torch.manual_seed(0)
    g_gen2 = torch.Generator(device=DEV); g_gen2.manual_seed(0)
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
    for ep in range(40):
        vt_sp.train(); sp_mod.train(); head_sp.train()
        perm = torch.randperm(N_TR, device=DEV)
        for s in range(0, N_TR, 64):
            i = perm[s:s+64]; B = len(i)
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
    print(f"  Trained EG3-07 (E2E-LoRA-SpatialAux 40ep) in {time.time()-t_sp:.1f}s (fm={loss_fm.item():.4f}, gnd={loss_gnd.item():.4f})", flush=True)
    vt_sp.eval(); sp_mod.eval(); head_sp.eval()
    with torch.no_grad():
        def eval_sp(s2_t):
            f_list, b_list = [], []
            for s in range(0, len(s2_t), 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tok = stage3_tokens(b3_sp, s2_t[s:s+64]).float()
                    f, pb = sp_mod(tok)
                f_list.append(f.float()); b_list.append(pb.float().cpu().numpy())
            return torch.cat(f_list, 0), np.concatenate(b_list, 0)
        fsp_13, pb_13 = eval_sp(S2_13)
        fsp_16, pb_16 = eval_sp(S2_16)

    # Mode 1: Guided by Grounder BBox + SpatialAux E2E Features
    results["★ EG3-07 (E2E-SpatialAux + 지시어BBox)"] = eval_predictions(
        "★ EG3-07 (E2E-SpatialAux + 지시어BBox)",
        batch_predict_head(head_sp, BW13, fsp_13),
        batch_predict_head(head_sp, BW16, fsp_16),
        fsp_13, fsp_16, head_sp
    )

    # Check internal Auxiliary BBox prediction accuracy on 11/16 basket sessions (generate()-free!)
    bk_err_cx, bk_err_area = [], []
    for (sid, o, s, e, raw) in meta16:
        if o == "바구니":
            bk_err_cx.append(np.abs(pb_16[s:e, 0] - raw[:, 0]))
            bk_err_area.append(np.abs(pb_16[s:e, 2] - raw[:, 2]))
    cx_mae = round(float(np.mean(np.concatenate(bk_err_cx))), 3)
    ar_mae = round(float(np.mean(np.concatenate(bk_err_area))), 3)
    print(f"  [EG3-07 내부 AuxGnd 헤드 (generate() 없이 ~15ms 직접 예측)] 11/16 바구니 cx MAE={cx_mae}, area MAE={ar_mae}", flush=True)

    torch.save(dict(sp_mod=sp_mod.state_dict(), head=head_sp.state_dict(), lora={k: v.cpu() for k, v in vt_sp.state_dict().items() if "lora_" in k}, aux_basket_cx_mae=cx_mae, aux_basket_area_mae=ar_mae), OUT_DIR / "EG3_07_E2E_LoRA_SpatialAux.pt")

    out_json = ROOT / "docs/v5/egvla2_fm/e2e_vs_a3_real_sessions.json"
    out_json.write_text(json.dumps(dict(models=results, aux_basket_cx_mae=cx_mae, aux_basket_area_mae=ar_mae), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved all results to {out_json} (total elapsed {time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
