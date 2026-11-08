#!/usr/bin/env python3
"""T5 텍스트 토큰 어텐션 헤드: V4 평균 풀링(T4와 동일) vs V5 크로스 어텐션(이미지 질의 → 문구 토큰 6개). 어휘·평가·학습법은 T4와 동일.
캐시: precompute_text_vocab.py(풀링) + precompute_text_tokens.py(토큰). plan: docs/plans/plan_20261108_text_token_attention_head.md. 출력: docs/v5/egvla2_fm/text_cond_t5.json
실행: .venv_lerobot/bin/python3 scripts/eval_text_cond_t5.py (SMOKE=1)"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import eval_target_cond as TC
import train_egvla2_s1 as S1
import robovlm_nav.models.egvla2_fm_heads as HD
import text_vocab as TV
T = S1.T; DEV = AB.DEV
SMOKE = bool(os.getenv("SMOKE")); EPOCHS = 2 if SMOKE else 120
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
OUT = ROOT / "docs/v5/egvla2_fm" / ("text_cond_t5_smoke.json" if SMOKE else "text_cond_t5.json")
STRIDE, BS, GO, P_NEG, KEEP = 9, 256, 0.2, 0.2, 0.5
VARIANTS = ["V0_img_only", "V4_big_vocab", "V5_token_attn"]
OLD_POS = [TV.POS_TRAIN.index(p) for p in ["gray basket", "gray bin", "gray container", "the gray basket", "go to the gray bin", "basket"]]
OLD_NEG = [TV.NP + TV.NEG_TRAIN.index(p) for p in ["red chair", "person", "blue bottle"]]
NP, NN = TV.NP, TV.NN
IDX = {p: i for i, p in enumerate(TV.ALL)}
H_POS = [IDX[p] for p in TV.POS_HELD]; H_NEG = [IDX[p] for p in TV.NEG_HELD]; H_HARD = [IDX[p] for p in TV.HARD_NEG]; H_AMB = [IDX[p] for p in TV.AMBIG]


def load():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3)
    tc = np.load(ROOT / "runs/egvla2_fm/text_vocab_cache.npz")
    BB, CH, VIS, TXT, EPI = [], [], [], [], []
    for i, e in enumerate(E):
        n = len(e["bbox"]); sel = np.arange(0, n, STRIDE); vis = tc[f"{e['name']}|vis"].astype(np.float32); assert len(vis) == len(sel)
        BB.append(e["bbox"][sel]); CH.append(e["chunk"][sel]); VIS.append(vis)
        TXT.append(np.stack([tc[f"{e['name']}|txt|{k}"].astype(np.float32) for k in range(len(TV.ALL))], 1).astype(np.float16)); EPI.append(np.full(len(sel), i))
    tk = np.load(ROOT / "runs/egvla2_fm/text_vocab_tokens.npz"); TOK = []
    for e in E: TOK.append(np.stack([tk[f"{e['name']}|tok|{k}"] for k in range(len(TV.ALL))], 1))
    g = lambda x, dt=None: torch.tensor(np.concatenate(x)).to(DEV)
    return eps, dict(bb=g(BB), ch=g(CH), vis=g(VIS), txt=g(TXT), tok=g(TOK), ep=np.concatenate(EPI))


def txt(D, i, k):   # k: 텐서(B,) 또는 int
    t = D["txt"][i]
    return (t[:, k] if isinstance(k, int) else t[torch.arange(len(i), device=DEV), k]).float()


class AttnWrap(torch.nn.Module):
    """ft = [이미지 768 ; 문구 토큰 6×768 펼침] → 이미지가 질의인 크로스 어텐션으로 문구를 읽어 768-d로 → PiZeroFM."""
    def __init__(s):
        super().__init__(); s.q = torch.nn.Linear(768, 768); s.ln = torch.nn.LayerNorm(768)
        s.att = torch.nn.MultiheadAttention(768, 4, batch_first=True); s.fm = HD.PiZeroFM(12, 1536)
    def forward(s, bb, ft, xt, t):
        v = ft[:, :768]; tok = s.ln(ft[:, 768:].view(len(ft), 6, 768))
        o, _ = s.att(s.q(v)[:, None], tok, tok); return s.fm(bb, torch.cat([v, o[:, 0]], -1), xt, t)


def tokf(D, i, k):
    t = D["tok"][i]; t = t[:, k] if isinstance(k, int) else t[torch.arange(len(i), device=DEV), k]
    return t.float().flatten(1)


def fit(D, tr, variant, seed):
    torch.manual_seed(seed)
    m = (AttnWrap() if variant == "V5_token_attn" else HD.PiZeroFM(12, 768 if variant == "V0_img_only" else 1536)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); n = len(tr); trt = torch.tensor(tr, device=DEV)
    pos_set = torch.tensor(list(range(NP)), device=DEV)
    neg_set = torch.tensor([NP + j for j in range(NN)], device=DEV)
    for _ in range(EPOCHS):
        m.train(); perm = trt[torch.randperm(n, device=DEV)]
        for s in range(0, n, BS):
            i = perm[s:s + BS]; B = len(i); bb0 = D["bb"][i]; a = TC.servo_label(D["ch"][i], bb0[:, 8], 1.0)
            bb = bb0 * (torch.rand(B, 1, device=DEV) > 0.3).float()
            if variant == "V0_img_only": ft = D["vis"][i]
            else:
                k = pos_set[torch.randint(0, len(pos_set), (B,), device=DEV)]
                neg = torch.rand(B, device=DEV) < P_NEG
                k = torch.where(neg, neg_set[torch.randint(0, len(neg_set), (B,), device=DEV)], k)
                a = torch.where(neg[:, None, None], torch.zeros_like(a), a)
                keep = (torch.rand(B, 1, device=DEV) < KEEP).float(); bb = torch.where(neg[:, None], bb0 * keep, bb)
                ft = torch.cat([D["vis"][i], (tokf if variant == "V5_token_attn" else txt)(D, i, k)], -1)
            x0 = torch.randn_like(a); t = torch.rand(B, device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            l = torch.nn.functional.mse_loss(m(bb, ft, xt, t), a - x0); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


@torch.no_grad()
def step1(m, bb, ft):
    torch.manual_seed(0); return HD.sample_chunk(m, bb, ft, 8).mean(1)[:, 0, :2]


@torch.no_grad()
def evaluate(m, D, te, variant):
    te = torch.tensor(te, device=DEV); bb = D["bb"][te]; z = torch.zeros_like(bb); y = D["ch"][te][:, 0, :2]; ym = y.norm(dim=-1)
    F = lambda k: D["vis"][te] if variant == "V0_img_only" else torch.cat([D["vis"][te], (tokf if variant == "V5_token_attn" else txt)(D, te, k)], -1)
    f = lambda k, b: step1(m, b, F(k))
    ref = f(0, bb)
    def ang(S):
        pm = S.norm(dim=-1); both = (pm > GO) & (ym > GO); c = (S[both] * y[both]).sum(-1) / (pm[both] * ym[both])
        return torch.rad2deg(torch.acos(c.clamp(-1, 1))).mean().item()
    stop = lambda S: (S.norm(dim=-1) < GO).float().mean().item()
    per = {}
    for p, k in IDX.items():
        if k < NP + NN and variant != "V0_img_only" and False: continue
        S = f(k, bb); per[p] = dict(stop=stop(S), eff=(S - ref).norm(dim=-1).mean().item(), stop_nobbox=stop(f(k, z)))
    mean = lambda ks, key: float(np.mean([per[TV.ALL[k]][key] for k in ks]))
    res = dict(ang_ref=ang(ref), stop_ref=stop(ref),
               held_pos_stop=mean(H_POS, "stop"), held_pos_eff=mean(H_POS, "eff"), held_pos_ang=float(np.mean([ang(f(k, bb)) for k in H_POS])),
               held_neg_stop_bbox=mean(H_NEG, "stop"), held_neg_stop_nobbox=mean(H_NEG, "stop_nobbox"),
               hard_neg_stop_bbox=mean(H_HARD, "stop"), hard_neg_stop_nobbox=mean(H_HARD, "stop_nobbox"),
               ambig_stop=mean(H_AMB, "stop"),
               train_pos_stop=mean(range(NP), "stop"), train_neg_stop_nobbox=mean(range(NP, NP + NN), "stop_nobbox"), per=per)
    return res


def main():
    eps, D = load(); folds = S1.make_folds(eps)
    cells = lambda p: [i for i, e in enumerate(eps) if e["cell"].startswith(p)]
    splits = {"fold0": folds[0], "fold1": folds[1], "lopo_center": cells("center|")}
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, te_eps in splits.items():
        te = np.where(np.isin(D["ep"], te_eps))[0]; tr = np.where(~np.isin(D["ep"], te_eps))[0]
        for v in VARIANTS:
            for sd in SEEDS:
                key = f"{name}|{v}|{sd}"
                if key in res: continue
                mdl = fit(D, tr, v, sd); res[key] = evaluate(mdl, D, te, v); res[key]["t"] = round(time.time(), 1); OUT.write_text(json.dumps(res, indent=1))
                print(key, {a: round(b, 3) for a, b in res[key].items() if isinstance(b, float)}, flush=True)


if __name__ == "__main__":
    main()
