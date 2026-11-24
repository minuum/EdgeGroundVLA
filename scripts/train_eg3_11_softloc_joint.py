#!/usr/bin/env python3
"""Phase 3 최종 완성 (EG3-11: VLM-E2E-SoftLoc-Joint / Distill):
액션 손실(loss_fm)이 미분 가능한 Prefix-Forced Soft-Loc(Softmax Expectation)을 통과해
Florence-2 BART Decoder LoRA(r=4)까지 실제로 역전파되면서,
동시에 사전학습 Florence-2의 <loc_0~999> 분포를 KL Distillation으로 보존하는
'100% End-to-End 학습·추론 VLA' 공동 미세조정 및 11/13·11/16 실기 프레임 전수 검증.

출력:
  - runs/egvla2_e2e_lora/EG3_11_SoftLoc_Joint.pt
  - docs/v5/egvla2_fm/track_a_dec1pass_results.json (EG3-11 결과 병합)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
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
from scripts.train_e2e_dec1pass import LOC_START, LOC_END, extract_dec_and_softloc

DEV = torch.device("cuda")
T = S1.T
LOC_GRID = torch.linspace(0.0, 1.0, 1000, device=DEV, dtype=torch.float32)


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


def main() -> None:
    t0 = time.time()
    print("=== Phase 3 최종: EG3-11 (VLM-E2E-SoftLoc-Joint · Decoder LoRA + KL Distillation) ===", flush=True)

    proc = AutoProcessor.from_pretrained(P.MODEL, trust_remote_code=True, local_files_only=True)
    fl = AutoModelForCausalLM.from_pretrained(
        P.MODEL, trust_remote_code=True, torch_dtype=torch.float16, local_files_only=True
    ).to(DEV).eval()

    # 1) 300개 학습 에피소드에서 균등 1,500프레임(에피소드당 5프레임) 샘플링 → Frozen Encoder 출력 + Teacher <loc> 분포 캐싱
    eps = S1.load_main()
    cache_c = np.load(T.CACHE)
    E = AB.prep(eps, cache_c, 3)

    phrase = "gray basket"
    p_toks = proc.tokenizer.encode(phrase, add_special_tokens=False)
    prefix_ids = [2, 0] + p_toks  # [</s>, <s>, gray, basket]
    pref_len = len(prefix_ids)

    print("[1/4] 300개 학습 에피소드에서 1,500프레임 Encoder 출력 + Teacher <loc> 분포 추출 중...", flush=True)
    enc_list, vis_list, tprob_list, tid_list, gt_bb_list, ch_ag_list = [], [], [], [], [], []
    lm = fl.language_model

    for ei, ep in enumerate(eps):
        L = len(E[ei]["bbox"])
        idxs = np.linspace(0, L - 1, min(5, L), dtype=int)
        with h5py.File(T.DATA / ep["name"], "r") as hf:
            imgs = hf["observations"]["images"] if "observations" in hf else hf["images"]
            pils = [Image.fromarray(np.ascontiguousarray(imgs[j][:, :, ::-1]).astype(np.uint8)).convert("RGB") for j in idxs]

        bb_ep = torch.tensor(E[ei]["bbox"][idxs], dtype=torch.float32, device=DEV)
        ch_ep = torch.tensor(E[ei]["chunk"][idxs], dtype=torch.float32, device=DEV)
        ch_ag_ep = area_servo_label(ch_ep, bb_ep, stop_area=0.23, ramp_start=0.18).cpu().numpy()

        with torch.no_grad():
            inp = proc(text=[P.TASK + phrase] * len(pils), images=pils, return_tensors="pt")
            pv = inp["pixel_values"].to(DEV, torch.float16)
            ids = inp["input_ids"].to(DEV)
            img_feats = fl._encode_image(pv)  # (B, 577, 768)
            vis_f = img_feats.float().mean(dim=1).cpu().numpy()  # (B, 768)
            emb, att = fl._merge_input_ids_with_image_features(img_feats, fl.get_input_embeddings()(ids))
            enc_out = lm.model.encoder(inputs_embeds=emb, attention_mask=att)
            enc_h = enc_out.last_hidden_state  # (B, S, 768) fp16

            # Frozen Teacher 4-step <loc> 분포 및 greedy token ID 추출
            B = len(pils)
            dec_in = torch.tensor([prefix_ids] * B, device=DEV, dtype=torch.long)
            out = lm(encoder_outputs=enc_out, attention_mask=att, decoder_input_ids=dec_in, use_cache=True)
            pkv = out.past_key_values
            probs_steps, ids_steps = [], []
            for st in range(4):
                z = out.logits[:, -1, LOC_START:LOC_END].float()
                probs_steps.append(F.softmax(z, dim=-1).cpu().numpy().astype(np.float16))
                nt = torch.argmax(z, dim=-1, keepdim=True) + LOC_START
                ids_steps.append(nt.squeeze(1).cpu().numpy())
                if st < 3:
                    out = lm(encoder_outputs=enc_out, attention_mask=att, decoder_input_ids=nt, past_key_values=pkv, use_cache=True)
                    pkv = out.past_key_values

        enc_list.append(enc_h.cpu().numpy())  # (B, S, 768) fp16
        vis_list.append(vis_f)
        tprob_list.append(np.stack(probs_steps, axis=1))  # (B, 4, 1000) fp16
        tid_list.append(np.stack(ids_steps, axis=1))      # (B, 4)
        gt_bb_list.append(E[ei]["bbox"][idxs])
        ch_ag_list.append(ch_ag_ep)
        if (ei + 1) % 100 == 0:
            print(f"  ep {ei+1}/300 ({time.time()-t0:.0f}s)", flush=True)

    ENC_TR = torch.from_numpy(np.concatenate(enc_list, 0))           # CPU fp16 (~1.3GB)
    VIS_TR = torch.from_numpy(np.concatenate(vis_list, 0)).float()   # (N, 768)
    TPROB_TR = torch.from_numpy(np.concatenate(tprob_list, 0))       # (N, 4, 1000) fp16
    TID_TR = torch.from_numpy(np.concatenate(tid_list, 0)).long()    # (N, 4)
    GTBB_TR = torch.from_numpy(np.concatenate(gt_bb_list, 0)).float()  # (N, 12)
    CHAG_TR = torch.from_numpy(np.concatenate(ch_ag_list, 0)).float()  # (N, 8, 3)
    N_TR = len(VIS_TR)
    print(f"  추출 완료: N={N_TR}, enc={tuple(ENC_TR.shape)} ({time.time()-t0:.1f}s)", flush=True)

    # 2) BART Decoder에만 경량 LoRA(r=4) 부착 + PiZeroFM(A3C_AG23_s0 초기화) 공동 미세조정!
    print("[2/4] Florence-2 BART Decoder LoRA(r=4) + PiZeroFM 공동 역전파 학습 시작...", flush=True)
    for p in fl.parameters():
        p.requires_grad = False

    lcfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    # lm.model.decoder에 직접 LoRA 래핑 (Encoder는 100% 동결)
    lm.model.decoder = get_peft_model(lm.model.decoder, lcfg)
    lm.model.decoder.to(DEV, dtype=torch.float32)  # 안정적 LoRA fp32 그래디언트
    dec_lora_params = [p for p in lm.model.decoder.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in dec_lora_params)
    print(f"  Decoder LoRA 학습 파라미터: {n_lora:,}개", flush=True)

    head = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/a3c/A3C_AG23_s0.pt", device=DEV).model
    for p in head.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(
        [
            dict(params=dec_lora_params, lr=2e-5, weight_decay=1e-2),
            dict(params=head.parameters(), lr=1e-4, weight_decay=1e-2),
        ]
    )
    EPOCHS = 8
    BS = 32
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=2e-6)
    action_grad_norm_step1 = None

    lm_head_w = lm.lm_head.weight.detach().float()
    lm_head_b = lm.final_logits_bias.detach().float() if hasattr(lm, "final_logits_bias") else None
    pref_t = torch.tensor(prefix_ids, device=DEV, dtype=torch.long)

    for ep in range(EPOCHS):
        lm.model.decoder.train()
        head.train()
        perm = torch.randperm(N_TR)
        s_fm = s_gnd = s_kl = 0.0
        nb = 0
        for s in range(0, N_TR, BS):
            i = perm[s : s + BS]
            B = len(i)
            enc_h = ENC_TR[i].to(DEV, dtype=torch.float32)
            vis_f = VIS_TR[i].to(DEV)
            tprob = TPROB_TR[i].to(DEV, dtype=torch.float32)
            tid = TID_TR[i].to(DEV)
            gt_bb = GTBB_TR[i].to(DEV)
            a_ag = CHAG_TR[i].to(DEV)

            # Teacher-forced decoder input: [2, 0, phrase..., loc_x1, loc_y1, loc_x2]
            # 출력 위치 [pref_len-1 .. pref_len+2]가 4개의 <loc> 토큰 로짓 예측!
            dec_in = torch.cat([pref_t.unsqueeze(0).expand(B, -1), tid[:, :3]], dim=1)
            dec_out = lm.model.decoder(input_ids=dec_in, encoder_hidden_states=enc_h, use_cache=False)
            h_4 = dec_out.last_hidden_state[:, pref_len - 1 : pref_len + 3, :]  # (B, 4, 768)
            logits_4 = F.linear(h_4, lm_head_w)
            if lm_head_b is not None:
                logits_4 = logits_4 + lm_head_b
            z_loc = logits_4[:, :, LOC_START:LOC_END]  # (B, 4, 1000)

            # 완전 미분 가능 Softmax Expectation 좌표 추출
            p_loc = F.softmax(z_loc, dim=-1)
            coords = (p_loc * LOC_GRID).sum(dim=-1)  # (B, 4): [x1, y1, x2, y2]
            x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            area = F.relu(x2 - x1) * F.relu(y2 - y1)
            has = gt_bb[:, 11]  # 존재 프레임 마스크
            soft_cur = torch.stack([cx, cy, area, has], dim=-1)  # (B, 4) — 미분 가능!
            # 이전 2프레임(gt_bb[:, :8]) + 현재 미분가능 soft_cur(4) 결합 -> (B, 12)
            bb_in = torch.cat([gt_bb[:, :8], soft_cur], dim=-1)

            # 50% visual feature swap (기하 정합 유지)
            u = torch.rand(B, 1, device=DEV) < 0.5
            vis_in = torch.where(u, vis_f[torch.randperm(B, device=DEV)], vis_f)

            x0 = torch.randn_like(a_ag)
            t = torch.rand(B, device=DEV)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a_ag
            loss_fm = F.mse_loss(head(bb_in, vis_in, xt, t), a_ag - x0)

            # Step 1에서 순수 Action Loss(loss_fm)만으로 Decoder LoRA에 그래디언트가 실제로 흐르는지 수학적 검증!
            if action_grad_norm_step1 is None:
                opt.zero_grad()
                loss_fm.backward(retain_graph=True)
                gnorm = float(sum(p.grad.detach().float().norm().item() for p in dec_lora_params if p.grad is not None))
                action_grad_norm_step1 = round(gnorm, 6)
                print(f"  [검증 PASS] 순수 Action Loss(loss_fm) → Decoder LoRA 역전파 그래디언트 norm = {action_grad_norm_step1}", flush=True)
                assert action_grad_norm_step1 > 0.0, "Action gradient did not reach Decoder LoRA!"
                opt.zero_grad()

            loss_gnd = F.smooth_l1_loss(soft_cur[:, :3], gt_bb[:, 8:11])
            loss_kl = F.kl_div(F.log_softmax(z_loc, dim=-1), tprob.clamp(min=1e-6), reduction="batchmean") / 4.0
            loss = loss_fm + 0.5 * loss_gnd + 0.3 * loss_kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec_lora_params, 1.0)
            opt.step()
            s_fm += float(loss_fm)
            s_gnd += float(loss_gnd)
            s_kl += float(loss_kl)
            nb += 1
        sched.step()
        print(f"  [EG3-11] ep {ep+1}/{EPOCHS} fm={s_fm/nb:.4f} gnd={s_gnd/nb:.4f} kl={s_kl/nb:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # 3) Held-Out 11/13(65세션, 의자·책상·화분·바구니) + 11/16(45세션) 전수 Soft-Loc 추론 및 Zero-Forgetting 검증!
    print("[3/4] 11/13(870장) + 11/16(588장) 미학습 다물체 Held-Out Soft-Loc + Joint Head 평가 중...", flush=True)
    # LoRA 가중치를 병합(merge)해 fp16 고속 추론
    lm.model.decoder = lm.model.decoder.merge_and_unload().to(DEV, dtype=torch.float16).eval()
    fl.eval()
    head.eval()

    D13 = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
    S16 = json.load(open(ROOT / "docs/v5/egvla2_fm/a3c/real_1116_sessions_bbox_area.json"))
    R16 = json.load(open(ROOT / "docs/v5/egvla2_fm/a3c/replay_1116.json"))
    R16.pop("_validation", None)
    C13 = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz")
    C16 = np.load(ROOT / "runs/egvla2_fm/t16_cache.npz")
    OB16 = {"go to the gray basket": "바구니", "go to the brown wooden chair": "의자", "go to the small desk": "책상"}

    def run_eval_split(items, cache_npz, is_13: bool):
        bw_gt_l, vis_l, bw_sl_l, sl_raw_l, meta = [], [], [], [], []
        idx = 0
        for item in items:
            if is_13:
                sid, obj_name, pos_name, ins = item["sid"], item["obj"], item["pos"], item["ins"]
            else:
                sid, s = item
                ins = s["instruction"]
                obj_name = OB16[ins]
                pos_name = ""
            ph = ins.lower().replace("go to the ", "").replace("go to ", "").strip()
            vis = cache_npz[f"{sid}|vis"].astype(np.float32)
            raw = cache_npz[f"{sid}|bbox"][: len(vis)].astype(np.float32)
            L = len(vis)
            sbs = []
            if is_13:
                for fi in range(L):
                    pil = Image.open(ROOT / f"docs/v5/figures/objtest/{sid}_{fi}.jpg").convert("RGB")
                    _, sb, _ = extract_dec_and_softloc(fl, proc, pil, ph)
                    sbs.append(sb)
            else:
                for fn in R16[sid]["frames"][:L]:
                    pil = Image.open(ROOT / f"docs/v5/figures/objtest_1116/{sid}_{fn}.jpg").convert("RGB")
                    _, sb, _ = extract_dec_and_softloc(fl, proc, pil, ph)
                    sbs.append(sb)
            sl_raw = np.stack(sbs, 0)
            bw_gt = AB.window(raw, 3); bw_gt[raw[:, 3] < 0.5] = 0
            bw_sl = AB.window(sl_raw, 3); bw_sl[sl_raw[:, 3] < 0.5] = 0
            bw_gt_l.append(bw_gt); vis_l.append(vis); bw_sl_l.append(bw_sl); sl_raw_l.append(sl_raw)
            meta.append((sid, obj_name, pos_name, idx, idx + L, raw))
            idx += L
        return np.concatenate(bw_gt_l, 0), np.concatenate(vis_l, 0), np.concatenate(bw_sl_l, 0), np.concatenate(sl_raw_l, 0), meta

    BW13_GT, VIS13, BW13_SL, SL13, meta13 = run_eval_split(D13, C13, True)
    print(f"  11/13 평가 완료 ({time.time()-t0:.0f}s)", flush=True)
    BW16_GT, VIS16, BW16_SL, SL16, meta16 = run_eval_split(list(S16.items()), C16, False)
    print(f"  11/16 평가 완료 ({time.time()-t0:.0f}s)", flush=True)

    p_a3 = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt", device=DEV).model
    ref13 = batch_predict_head(p_a3, BW13_GT, VIS13)
    ref16 = batch_predict_head(p_a3, BW16_GT, VIS16)
    c13 = batch_predict_head(head, BW13_SL, VIS13)
    c16 = batch_predict_head(head, BW16_SL, VIS16)

    # 지표 집계
    n13, n16 = np.linalg.norm(c13, axis=1), np.linalg.norm(c16, axis=1)
    c_all = np.concatenate([c13, c16], 0)
    r_all = np.concatenate([ref13, ref16], 0)
    mv = (np.linalg.norm(c_all, axis=1) > 0.2) & (np.linalg.norm(r_all, axis=1) > 0.2)
    cos = np.clip((c_all[mv] * r_all[mv]).sum(1) / (np.linalg.norm(c_all[mv], axis=1) * np.linalg.norm(r_all[mv], axis=1)), -1, 1)
    ang_med = round(float(np.median(np.degrees(np.arccos(cos)))), 2)
    lat_agree = round(float((np.sign(c_all[mv, 1]) == np.sign(r_all[mv, 1])).mean()), 3)

    def g13(pos_sub: str, obj_name: str) -> float:
        return round(float(np.mean([(n13[s:e] > 0.2).mean() for (_, o, p, s, e, _) in meta13 if pos_sub in p and o == obj_name])), 3)

    bins = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 1.0]
    a16 = np.concatenate([raw[:, 2] for (_, _, _, _, _, raw) in meta16])
    b = np.digitize(a16, bins[1:-1])
    sr_bins = [round(float((n16[b == k] < 0.2).mean()), 3) for k in range(7)]
    obj_ge25 = {
        o_: round(float(np.concatenate([(n16[s:e][raw[:, 2] >= 0.25] < 0.2) for (_, o, _, s, e, raw) in meta16 if o == o_]).mean()), 3)
        for o_ in ("바구니", "의자")
    }

    # Open-Vocab 망각 검증 (11/13 미학습 의자·책상·화분 cx/area MAE vs beam=3)
    raw13_all = np.concatenate([raw for (_, _, _, _, _, raw) in meta13], 0)
    valid13 = (raw13_all[:, 3] > 0.5) & (SL13[:, 3] > 0.5)
    ov_cx_mae = round(float(np.abs(SL13[valid13, 0] - raw13_all[valid13, 0]).mean()), 4)
    ov_ar_mae = round(float(np.abs(SL13[valid13, 2] - raw13_all[valid13, 2]).mean()), 4)
    ov_det_rate = round(float((SL13[raw13_all[:, 3] > 0.5, 3] > 0.5).mean()), 4)

    res = dict(
        name="VLM-E2E-SoftLoc-Joint (EG3-11, DecLoRA+KL)",
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
        open_vocab_1113_det_rate=ov_det_rate,
        open_vocab_1113_cx_mae=ov_cx_mae,
        open_vocab_1113_area_mae=ov_ar_mae,
        action_grad_norm_to_decoder_lora=action_grad_norm_step1,
    )
    print(
        f"\n★ EG3-11 (VLM-E2E-SoftLoc-Joint) | 각도차:{ang_med:>5.2f}°(좌우:{lat_agree:.3f}) "
        f"| 11/13 의자:{res['chair_center_13']:.2f} 책상:{res['desk_center_13']:.2f} 화분:{res['pot_center_13']:.2f} "
        f"| 11/16 정지(<.10:{res['stop_lt10_16']:.2f} 25-30:{res['stop_25_30_16']:.2f} 30-40:{res['stop_30_40_16']:.2f} "
        f"바구니≥.25:{res['basket_ge25_stop_16']:.2f} 의자≥.25:{res['chair_ge25_stop_16']:.2f}) "
        f"| 11/13 Open-Vocab 유지: det={ov_det_rate:.3f} cx_mae={ov_cx_mae:.4f} ar_mae={ov_ar_mae:.4f}",
        flush=True,
    )

    # 4) 저장
    out_ckpt = ROOT / "runs/egvla2_e2e_lora/EG3_11_SoftLoc_Joint.pt"
    torch.save(
        dict(
            arch="C_pi0_adaln",
            state=head.state_dict(),
            bdim=12,
            fdim=768,
            variant="EG3_11_SoftLoc_Joint",
            metrics=res,
        ),
        out_ckpt,
    )
    out_json = ROOT / "docs/v5/egvla2_fm/track_a_dec1pass_results.json"
    prev = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    prev["VLM-E2E-SoftLoc-Joint (EG3-11, DecLoRA+KL)"] = res
    out_json.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4/4] Saved {out_ckpt} & updated {out_json} in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
