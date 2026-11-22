#!/usr/bin/env python3
"""Phase 2 (Track A1 & Track A2 프로토타입):
1) Track A1 (Frozen Decoder 1-Pass Hidden State E2E):
   Florence-2 Encoder + 6-layer Decoder에 [</s><s> + phrase_tokens]를 1-Pass로 통과시켜 얻은
   Decoder Last Hidden State (768-d)를 Action Expert(PiZeroFM)가 직접 읽어 액션을 생성.
2) Track A2 (Differentiable Prefix-Forced Soft-Loc E2E):
   문자열 파싱(generate) 없이 Decoder의 <loc_0~999> Softmax Expectation 연속 좌표 (cx, cy, area, has) +
   Decoder Hidden State (768-d)를 Area-Grounded PiZeroFM으로 직접 전달해 11/13(다물체 65세션) · 11/16(근접 정지 45세션)에서 5축 검증!
출력: docs/v5/egvla2_fm/track_a_dec1pass_results.json, runs/egvla2_e2e_lora/EG3_A1_Dec1Pass.pt, EG3_A2_SoftLoc.pt
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import precompute_v6_florence2_fused as P
import train_egvla2_s1 as S1
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
from scripts.a3c_train_ag import area_servo_label

DEV = torch.device("cuda")
T = S1.T
LOC_START = 50269
LOC_END = 51269
LOC_GRID = torch.linspace(0.0, 1.0, 1000, device=DEV, dtype=torch.float32)

torch.manual_seed(0)
X0_SINGLE = torch.randn(1, 8, 8, 3, device=DEV)


@torch.no_grad()
def extract_dec_and_softloc(model, proc, pil, phrase):
    """1회 이미지+인코더 통과 후:
      - dec_h1 (768-d): [</s><s> + phrase] 1-Pass Decoder Last Hidden State (Track A1용)
      - soft_box (4-d): 4-step KV-cached <loc_0~999> Softmax Expectation (cx, cy, area, has) (Track A2용)
      - dec_h_loc (768-d): 4개 좌표 토큰 위치의 Decoder Hidden State 평균 (Track A2용)
    """
    inp = proc(text=[P.TASK + phrase], images=[pil], return_tensors="pt")
    pv = inp["pixel_values"].to(DEV, torch.float16)
    ids = inp["input_ids"].to(DEV)
    p_toks = proc.tokenizer.encode(phrase, add_special_tokens=False)
    dec_prefix = torch.tensor([[2, 0] + p_toks], device=DEV, dtype=torch.long)

    img = model._encode_image(pv)
    emb, att = model._merge_input_ids_with_image_features(img, model.get_input_embeddings()(ids))
    enc_out = model.language_model.model.encoder(inputs_embeds=emb, attention_mask=att)

    out = model.language_model(
        encoder_outputs=enc_out,
        attention_mask=att,
        decoder_input_ids=dec_prefix,
        use_cache=True,
        output_hidden_states=True,
    )
    dec_h1 = out.decoder_hidden_states[-1][0, -1, :].float().cpu().numpy()
    pkv = out.past_key_values
    coords = []
    loc_hs = [dec_h1]
    has_prob = 1.0
    for step in range(4):
        logits = out.logits[:, -1, :].float()
        if step == 0:
            has_prob = float(torch.softmax(logits, dim=-1)[0, LOC_START:LOC_END].sum().item())
        loc_probs = torch.softmax(logits[:, LOC_START:LOC_END], dim=-1)
        coords.append(float((loc_probs * LOC_GRID).sum(dim=-1).item()))
        if step < 3:
            next_tok = torch.argmax(logits[:, LOC_START:LOC_END], dim=-1, keepdim=True) + LOC_START
            out = model.language_model(
                encoder_outputs=enc_out,
                attention_mask=att,
                decoder_input_ids=next_tok,
                past_key_values=pkv,
                use_cache=True,
                output_hidden_states=True,
            )
            pkv = out.past_key_values
            loc_hs.append(out.decoder_hidden_states[-1][0, -1, :].float().cpu().numpy())
    x1, y1, x2, y2 = coords
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    cx = (x1 + x2) * 0.5; cy = (y1 + y2) * 0.5; area = w * h
    has = 1.0 if (has_prob > 0.35 and area > 1e-4) else 0.0
    soft_box = np.array([cx, cy, area, has], np.float32)
    dec_h_loc = np.mean(loc_hs, axis=0).astype(np.float32)
    return dec_h1, soft_box, dec_h_loc


@torch.no_grad()
def batch_predict_head(head, bw_np, feat_np):
    head.eval()
    B = len(bw_np)
    bb = torch.tensor(bw_np, device=DEV, dtype=torch.float32).repeat_interleave(8, 0)
    ft = torch.tensor(feat_np, device=DEV, dtype=torch.float32).repeat_interleave(8, 0)
    x = X0_SINGLE.repeat(B, 1, 1, 1).view(B * 8, 8, 3).clone()
    for i in range(10):
        t = torch.full((B * 8,), i / 10.0, device=DEV)
        x = x + head(bb, ft, x, t) / 10.0
    return x.view(B, 8, 8, 3).mean(1)[:, 0, :2].cpu().numpy()


def main():
    t0 = time.time()
    print("=== Phase 2: Track A1 (Decoder 1-Pass) & Track A2 (Soft-Loc E2E) ===", flush=True)
    cache_file = ROOT / "runs/e2e_cache/dec1pass_softloc_cache.npz"
    D13 = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
    S16 = json.load(open(ROOT / "docs/v5/egvla2_fm/a3c/real_1116_sessions_bbox_area.json"))
    R16 = json.load(open(ROOT / "docs/v5/egvla2_fm/a3c/replay_1116.json")); R16.pop("_validation", None)
    OB16 = {"go to the gray basket": "바구니", "go to the brown wooden chair": "의자", "go to the small desk": "책상"}

    eps = S1.load_main(); cache_c = np.load(T.CACHE); E = AB.prep(eps, cache_c, 3)
    fr = json.load(open(ROOT / "runs/e2e_cache/frames.json")); frames = fr["frames"]  # 4,860 training frames (stride 6)

    if not cache_file.exists():
        print(f"[1/3] Florence-2 fp16으로 학습(4,860장) + 11/13(870장) + 11/16(588장) Decoder 1-Pass & Soft-Loc 추출 중...", flush=True)
        proc = AutoProcessor.from_pretrained(P.MODEL, trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained(P.MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()

        tr_h1, tr_sb, tr_hloc = [], [], []
        handles = {}
        for k, (ei, t) in enumerate(frames):
            name = eps[ei]["name"]
            if name not in handles:
                for h_ in handles.values(): h_.close()
                handles = {name: h5py.File(T.DATA / name, "r")}
            rgb = np.ascontiguousarray(np.asarray(handles[name]["images"][t])[:, :, ::-1])
            h1, sb, hl = extract_dec_and_softloc(mod, proc, Image.fromarray(rgb), P.PHRASE)
            tr_h1.append(h1); tr_sb.append(sb); tr_hloc.append(hl)
            if (k + 1) % 800 == 0:
                print(f"  train {k+1}/{len(frames)} ({time.time()-t0:.0f}s)", flush=True)
        for h_ in handles.values(): h_.close()

        r13_h1, r13_sb, r13_hloc = [], [], []
        for x in D13:
            sid, ins, n = x["sid"], x["ins"], x["n"]
            for fi in range(n):
                pil = Image.open(ROOT / f"docs/v5/figures/objtest/{sid}_{fi}.jpg").convert("RGB")
                h1, sb, hl = extract_dec_and_softloc(mod, proc, pil, ins)
                r13_h1.append(h1); r13_sb.append(sb); r13_hloc.append(hl)
        print(f"  11/13 done ({len(r13_h1)} frames, {time.time()-t0:.0f}s)", flush=True)

        r16_h1, r16_sb, r16_hloc = [], [], []
        for sid, s in S16.items():
            ins = s["instruction"]
            for fn in R16[sid]["frames"]:
                pil = Image.open(ROOT / f"docs/v5/figures/objtest_1116/{sid}_{fn}.jpg").convert("RGB")
                h1, sb, hl = extract_dec_and_softloc(mod, proc, pil, ins)
                r16_h1.append(h1); r16_sb.append(sb); r16_hloc.append(hl)
        print(f"  11/16 done ({len(r16_h1)} frames, {time.time()-t0:.0f}s)", flush=True)

        np.savez(
            cache_file,
            tr_h1=np.stack(tr_h1), tr_sb=np.stack(tr_sb), tr_hloc=np.stack(tr_hloc),
            r13_h1=np.stack(r13_h1), r13_sb=np.stack(r13_sb), r13_hloc=np.stack(r13_hloc),
            r16_h1=np.stack(r16_h1), r16_sb=np.stack(r16_sb), r16_hloc=np.stack(r16_hloc),
        )
        del mod; torch.cuda.empty_cache()
    else:
        print(f"[1/3] 기존 캐시 로드: {cache_file}", flush=True)

    Z = np.load(cache_file)

    # Prepare 11/13 and 11/16 windows for Track A2 (using Soft-Loc boxes!) and ground-truth reference
    C13 = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz")
    C16 = np.load(ROOT / "runs/egvla2_fm/t16_cache.npz")

    bw13_gt, vis13_gt, bw13_sl, meta13 = [], [], [], []
    idx = 0
    for x in D13:
        sid = x["sid"]; vis = C13[f"{sid}|vis"].astype(np.float32); raw = C13[f"{sid}|bbox"][:len(vis)].astype(np.float32)
        L = len(vis)
        bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0
        sl_raw = Z["r13_sb"][idx:idx+L]
        sl_bw = AB.window(sl_raw, 3); sl_bw[sl_raw[:, 3] < .5] = 0
        bw13_gt.append(bw); vis13_gt.append(vis); bw13_sl.append(sl_bw)
        meta13.append((sid, x["obj"], x["pos"], x["res"], idx, idx + L, raw))
        idx += L
    BW13_GT = np.concatenate(bw13_gt, 0); VIS13_GT = np.concatenate(vis13_gt, 0); BW13_SL = np.concatenate(bw13_sl, 0)

    bw16_gt, vis16_gt, bw16_sl, meta16 = [], [], [], []
    idx = 0
    for sid, s in S16.items():
        vis = C16[f"{sid}|vis"].astype(np.float32); raw = C16[f"{sid}|bbox"][:len(vis)].astype(np.float32)
        L = len(vis)
        bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0
        sl_raw = Z["r16_sb"][idx:idx+L]
        sl_bw = AB.window(sl_raw, 3); sl_bw[sl_raw[:, 3] < .5] = 0
        bw16_gt.append(bw); vis16_gt.append(vis); bw16_sl.append(sl_bw)
        meta16.append((sid, OB16[s["instruction"]], idx, idx + L, raw))
        idx += L
    BW16_GT = np.concatenate(bw16_gt, 0); VIS16_GT = np.concatenate(vis16_gt, 0); BW16_SL = np.concatenate(bw16_sl, 0)

    p_a3 = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt").model
    ref_a3_13 = batch_predict_head(p_a3, BW13_GT, VIS13_GT)
    ref_a3_16 = batch_predict_head(p_a3, BW16_GT, VIS16_GT)

    # Training tensors (4,860 frames)
    BB_TR_GT = torch.from_numpy(np.stack([E[ei]["bbox"][t] for ei, t in frames])).float().to(DEV)
    CH_TR = torch.from_numpy(np.stack([E[ei]["chunk"][t] for ei, t in frames])).float().to(DEV)
    CH_AG_TR = area_servo_label(CH_TR, BB_TR_GT, stop_area=0.23, ramp_start=0.18)

    # Build 3-frame Soft-Loc window for training (approximate via repeating current soft-loc or matching stride)
    tr_sb_t = torch.from_numpy(Z["tr_sb"]).float().to(DEV)
    BB_TR_SL = tr_sb_t.repeat(1, 3)  # (N, 12)
    H1_TR = torch.from_numpy(Z["tr_h1"]).float().to(DEV)
    HLOC_TR = torch.from_numpy(Z["tr_hloc"]).float().to(DEV)
    N_TR = len(frames)

    bins = [0, .1, .15, .2, .25, .3, .4, 1.0]
    def eval_pred(name, c13, c16):
        n13 = np.linalg.norm(c13, axis=1); n16 = np.linalg.norm(c16, axis=1)
        c_all = np.concatenate([c13, c16], 0); r_all = np.concatenate([ref_a3_13, ref_a3_16], 0)
        n_all = np.linalg.norm(c_all, axis=1); rn_all = np.linalg.norm(r_all, axis=1)
        mv = (n_all > 0.2) & (rn_all > 0.2)
        cos = np.clip((c_all[mv] * r_all[mv]).sum(1) / (n_all[mv] * rn_all[mv]), -1, 1)
        ang_med = round(float(np.median(np.degrees(np.arccos(cos)))), 2)
        lat_agree = round(float((np.sign(c_all[mv, 1]) == np.sign(r_all[mv, 1])).mean()), 3)
        def g13(pos_sub, obj_name):
            return round(float(np.mean([(n13[s:e] > 0.2).mean() for (_, o, p, _, s, e, _) in meta13 if pos_sub in p and o == obj_name])), 3)
        a16 = np.concatenate([raw[:, 2] for (_, _, _, _, raw) in meta16]); b = np.digitize(a16, bins[1:-1])
        sr_bins = [round(float((n16[b == k] < 0.2).mean()), 3) for k in range(7)]
        obj_ge25 = {o_: round(float(np.concatenate([(n16[s:e][raw[:, 2] >= 0.25] < 0.2) for (_, o, s, e, raw) in meta16 if o == o_]).mean()), 3) for o_ in ("바구니", "의자")}
        res = dict(name=name, ang_med_vs_a3=ang_med, lat_agree_vs_a3=lat_agree,
                   chair_center_13=g13("가운데", "의자"), desk_center_13=g13("가운데", "작은 책상"),
                   pot_center_13=g13("가운데", "화분"), basket_center_13=g13("가운데", "바구니"),
                   stop_lt10_16=sr_bins[0], stop_25_30_16=sr_bins[4], stop_30_40_16=sr_bins[5],
                   basket_ge25_stop_16=obj_ge25["바구니"], chair_ge25_stop_16=obj_ge25["의자"])
        print(f"★ {name:30s} | 각도차:{ang_med:>5.2f}°(좌우:{lat_agree:.3f}) | 11/13 의자중:{res['chair_center_13']:.2f} 책상중:{res['desk_center_13']:.2f} 화분중:{res['pot_center_13']:.2f} 바구니중:{res['basket_center_13']:.2f} | 11/16 정지(<.10:{res['stop_lt10_16']:.2f} 25-30:{res['stop_25_30_16']:.2f} 30-40:{res['stop_30_40_16']:.2f} 바구니≥.25:{res['basket_ge25_stop_16']:.2f} 의자≥.25:{res['chair_ge25_stop_16']:.2f})", flush=True)
        return res

    results = {}

    # 2. Track A1: Pure 1-Pass Frozen Decoder Hidden State (ZERO BBox input! Pure H_dec -> PiZeroFM)
    print("\n[2/3] Track A1 (Frozen Decoder 1-Pass Hidden State -> PiZeroFM, 120ep) 학습 중...", flush=True)
    torch.manual_seed(0)
    m_a1 = HD.PiZeroFM(12, 768).to(DEV)
    opt_a1 = torch.optim.AdamW(m_a1.parameters(), lr=1e-3, weight_decay=1e-2)
    zero_bb_tr = torch.zeros(N_TR, 12, device=DEV)
    for ep in range(120):
        m_a1.train(); perm = torch.randperm(N_TR, device=DEV)
        for s in range(0, N_TR, 64):
            i = perm[s:s+64]; a = CH_AG_TR[i]
            x0 = torch.randn_like(a); t = torch.rand(len(i), device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            l = F.mse_loss(m_a1(zero_bb_tr[i], H1_TR[i], xt, t), a - x0); opt_a1.zero_grad(); l.backward(); opt_a1.step()
    c13_a1 = batch_predict_head(m_a1, np.zeros_like(BW13_GT), Z["r13_h1"])
    c16_a1 = batch_predict_head(m_a1, np.zeros_like(BW16_GT), Z["r16_h1"])
    results["Track A1 (Decoder 1-Pass 순수 E2E)"] = eval_pred("Track A1 (Decoder 1-Pass 순수 E2E)", c13_a1, c16_a1)
    torch.save(dict(arch="C_pi0_adaln", state=m_a1.state_dict(), bdim=12, fdim=768, variant="TrackA1_Dec1Pass"), ROOT / "runs/egvla2_e2e_lora/EG3_A1_Dec1Pass.pt")

    # 3. Track A2: Differentiable Prefix-Forced Soft-Loc Bottleneck + Decoder Hidden State -> Area-Grounded PiZeroFM!
    print("\n[3/3] Track A2 (Differentiable Soft-Loc + Decoder Hidden State -> PiZeroFM, 120ep) 학습 중...", flush=True)
    torch.manual_seed(0); g = torch.Generator(device=DEV); g.manual_seed(0)
    m_a2 = HD.PiZeroFM(12, 768).to(DEV)
    opt_a2 = torch.optim.AdamW(m_a2.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(120):
        m_a2.train(); perm = torch.randperm(N_TR, device=DEV)
        for s in range(0, N_TR, 64):
            i = perm[s:s+64]; B = len(i); a = CH_AG_TR[i]
            # Use GT/Soft-Loc boxes (50/50 mix during training for robustness)
            use_sl = (torch.rand(B, 1, device=DEV) < 0.5).float()
            bb = use_sl * BB_TR_SL[i] + (1.0 - use_sl) * BB_TR_GT[i]
            ft = HLOC_TR[i]
            u = torch.rand(B, 1, device=DEV, generator=g) < 0.5
            ft_in = torch.where(u, ft[torch.randperm(B, device=DEV)], ft)
            x0 = torch.randn_like(a); t = torch.rand(B, device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            l = F.mse_loss(m_a2(bb, ft_in, xt, t), a - x0); opt_a2.zero_grad(); l.backward(); opt_a2.step()
    c13_a2 = batch_predict_head(m_a2, BW13_SL, Z["r13_hloc"])
    c16_a2 = batch_predict_head(m_a2, BW16_SL, Z["r16_hloc"])
    results["Track A2 (Soft-Loc 미분가능 E2E)"] = eval_pred("Track A2 (Soft-Loc 미분가능 E2E)", c13_a2, c16_a2)
    torch.save(dict(arch="C_pi0_adaln", state=m_a2.state_dict(), bdim=12, fdim=768, variant="TrackA2_SoftLoc"), ROOT / "runs/egvla2_e2e_lora/EG3_A2_SoftLoc.pt")

    out_path = ROOT / "docs/v5/egvla2_fm/track_a_dec1pass_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path} (total elapsed {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
