#!/usr/bin/env python3
"""A(고정 융합 특징) 대 B(DaViT 풀링) 격차 원인 분리 (계획: docs/plans/plan_20261028_feature_gap_control.md, 승인 10/29).
R0: stage3 토큰에 Florence 투영 경로(위치 임베딩·전역 토큰·투영·LN)를 적용해 평균 → 캐시된 A 특징과 코사인 비교(학습 없음).
R1: R0 특징으로 기존 방식(GS.fit)과 같은 헤드 학습(구현 일치 점검). P1 3×3 격자 / P2 가로 8구간 / P3 어텐션 풀링: 투영된 토큰 위에서 풀링 모듈+헤드 학습.
실행: MODE=r0 → R0만(≥0.99여야 다음 단계)  MODE=all → R0 후 R1·P1·P2·P3 × 3분할   SMOKE=1: epoch 2·분할 1개
출력: docs/v5/egvla2_fm/feature_gap.json, 로그는 호출 측에서 runs/e2e_cache/logs/ 로 리다이렉트
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_e2e_general as EG  # noqa: E402  (Proj·head_loss·stage3 재사용, 기존 파일은 수정하지 않음)
import eval_subgoal_ablation as AB  # noqa: E402
import gap_suite as GS  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
import robovlm_nav.models.egvla2_fm_heads as HD  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
T = S1.T
DEV, BS = GS.DEV, GS.BS
SMOKE = bool(os.getenv("SMOKE")); MODE = os.getenv("MODE", "all")
EPOCHS = 2 if SMOKE else 120
CACHE = ROOT / "runs/e2e_cache"
OUT = ROOT / "docs/v5/egvla2_fm" / ("feature_gap_smoke.json" if SMOKE else "feature_gap.json")
G = 24  # stage3 격자 한 변(768px 입력 → 24×24=576토큰)


def encode_stage3(model, h):
    """stage3 토큰 h(B,576,1024) → Florence _encode_image와 같은 경로(위치 임베딩 + visual_temporal_embed + [전역평균;576토큰] 투영·LN). 반환 (B,577,768)."""
    b = h.shape[0]; x = h.float().view(b, G, G, -1); x = (x + model.image_pos_embed(x)).view(b, G * G, -1)
    tpe = model.visual_temporal_embed(x.view(b, 1, -1, x.shape[-1])[:, :, 0]); x = x.view(b, 1, -1, x.shape[-1]) + tpe.view(1, 1, 1, x.shape[-1]); x = x[:, 0]   # T=1
    x = torch.cat([x.mean(1, keepdim=True), x], 1)                                  # image_feature_source = [spatial_avg_pool, temporal_avg_pool]
    return model.image_proj_norm(x @ model.image_projection)


def stage3_from_cache(model, stage2, idx):
    base = model.vision_tower; h = torch.from_numpy(np.stack([np.asarray(stage2[k]) for k in idx])).to(DEV).float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, size = base.convs[3](h, (48, 48)); h, size = base.blocks[3](h, size)
    return h


def florence_tokens(model, stage2, idx, bs=32):
    """투영·LN 후 576 공간 토큰(전역 토큰 제외) (n,576,768) fp16, GPU."""
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs): out.append(encode_stage3(model, stage3_from_cache(model, stage2, idx[s:s + bs]))[:, 1:].half())
    return torch.cat(out)


class P1(nn.Module):       # 3×3 격자 평균 → 칸마다 768→128 → 결합 → 768
    def __init__(s): super().__init__(); s.r = nn.Linear(768, 128); s.o = nn.Sequential(nn.Linear(9 * 128, 768), nn.GELU())
    def forward(s, t):
        b = t.shape[0]; g = t.float().view(b, 3, G // 3, 3, G // 3, 768).mean((2, 4)).reshape(b, 9, 768)
        return s.o(s.r(g).reshape(b, -1))


class P2(nn.Module):       # 가로 8구간(세로·구간 내 3열 평균) → 구간마다 768→96 → 결합 768
    def __init__(s): super().__init__(); s.r = nn.Linear(768, 96); s.o = nn.GELU()
    def forward(s, t):
        b = t.shape[0]; g = t.float().view(b, G, 8, G // 8, 768).mean((1, 3))
        return s.o(s.r(g).reshape(b, -1))


class P3(nn.Module):       # 학습형 질의 1개 어텐션 풀링
    def __init__(s): super().__init__(); s.q = nn.Parameter(torch.randn(1, 1, 768) * 0.02); s.a = nn.MultiheadAttention(768, 8, batch_first=True); s.n = nn.LayerNorm(768)
    def forward(s, t):
        x = t.float(); o, _ = s.a(s.q.expand(len(x), -1, -1), x, x); return s.n(o[:, 0])


def train_pool(pool, head, bb, toks, ch, epochs, seed=0):
    torch.manual_seed(seed); params = list(pool.parameters()) + list(head.parameters()); opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2); n = len(bb)
    for _ in range(epochs):
        pool.train(); head.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; l = EG.head_loss(head, bb[i], pool(toks[i]), ch[i]); opt.zero_grad(); l.backward(); opt.step()
    pool.eval(); head.eval()


def apply(pool, toks, bs=128):
    with torch.no_grad(): return torch.cat([pool(toks[s:s + bs]) for s in range(0, len(toks), bs)])


def global_token_feature(model, stage2, idx, bs=32):
    """R0 본체: A와 같은 577토큰 평균 특징 (n,768) fp32."""
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs): out.append(encode_stage3(model, stage3_from_cache(model, stage2, idx[s:s + bs])).mean(1))
    return torch.cat(out)


def main():
    eps = S1.load_main(); cache_c = np.load(T.CACHE); E = AB.prep(eps, cache_c, 3); folds = S1.make_folds(eps)
    frames = json.load(open(CACHE / "frames.json"))["frames"]; stage2 = np.load(CACHE / "stage2.npy", mmap_mode="r"); live_stage2 = np.load(CACHE / "live_stage2.npy", mmap_mode="r")
    lm = json.load(open(CACHE / "live_meta.json"))
    ep_of = np.array([f[0] for f in frames]); bbs = np.stack([E[ei]["bbox"][t] for ei, t in frames]); fts = np.stack([E[ei]["feat"][t] for ei, t in frames]); chs = np.stack([E[ei]["chunk"][t] for ei, t in frames])
    BB, FT, CH = [torch.from_numpy(x).float().to(DEV) for x in (bbs, fts, chs)]
    lb, lf_fused, win = GS.live_inputs()
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float32).to(DEV).eval()
    print("image_feature_source", model.image_feature_source, flush=True)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    # ---- R0 복원 점검 ----
    rng = np.random.default_rng(0); sub = np.sort(rng.choice(len(frames), 256 if not SMOKE else 32, replace=False))
    r0 = global_token_feature(model, stage2, sub); a = FT[torch.from_numpy(sub).to(DEV)]
    cos = torch.nn.functional.cosine_similarity(r0, a, dim=1); nr = (r0.norm(dim=1) / a.norm(dim=1))
    r0_live = global_token_feature(model, live_stage2, np.arange(len(lm))); lfa = torch.from_numpy(np.asarray(lf_fused)).float().to(DEV)
    cl = torch.nn.functional.cosine_similarity(r0_live, lfa, dim=1)
    res["R0"] = dict(n_train_frames=len(sub), cos_mean=float(cos.mean()), cos_min=float(cos.min()), norm_ratio_mean=float(nr.mean()), max_abs_diff_mean=float((r0 - a).abs().max(1).values.mean()),
                     live_cos_mean=float(cl.mean()), live_cos_min=float(cl.min()), pass_099=bool(cos.mean() >= 0.99 and cl.mean() >= 0.99))
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False)); print("R0", res["R0"], flush=True)
    if MODE == "r0" or not res["R0"]["pass_099"]:
        print("중단: R0 점검만 요청됐거나 0.99 미달(계획 §3-1)", flush=True); return
    # ---- 3분할 비교 ----
    allidx = np.arange(len(frames)); R0F = global_token_feature(model, stage2, allidx); TOK = florence_tokens(model, stage2, allidx); LTOK = florence_tokens(model, live_stage2, np.arange(len(lm)))
    splits = {"fold0": folds[0], "lopo_center": [i for i, e in enumerate(eps) if e["cell"].startswith("center|")], "lopo_strong_right": [i for i, e in enumerate(eps) if e["cell"].startswith("strong_right|")]}
    live = lambda f: (lb, f.detach().float().cpu().numpy(), win)
    for name, te_eps in splits.items():
        res.setdefault(name, {}); tes = set(te_eps)
        tr = torch.from_numpy(np.array([k for k in allidx if ep_of[k] not in tes])).to(DEV); te = torch.from_numpy(np.array([k for k in allidx if ep_of[k] in tes])).to(DEV)
        print(name, "train", len(tr), "test", len(te), flush=True)
        if "R1_replica" not in res[name]:
            m = GS.fit((BB[tr], R0F[tr], CH[tr])); res[name]["R1_replica"] = GS.split_metrics(m, (BB[te], R0F[te], CH[te]), live(r0_live)); OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(name, "R1", flush=True)
        for nm, cls in (("P1_grid3x3", P1), ("P2_horiz8", P2), ("P3_attn", P3)):
            if nm in res[name]: continue
            t0 = time.time(); torch.manual_seed(0); pool, head = cls().to(DEV), HD.PiZeroFM(12, 768).to(DEV); train_pool(pool, head, BB[tr], TOK[tr], CH[tr], EPOCHS)
            res[name][nm] = GS.split_metrics(head, (BB[te], apply(pool, TOK[te]), CH[te]), live(apply(pool, LTOK))); res[name][nm]["train_seconds"] = time.time() - t0
            OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(name, nm, f"{time.time() - t0:.0f}s", flush=True)
        if SMOKE: break


if __name__ == "__main__":
    main()
