#!/usr/bin/env python3
"""T1·T2 오프라인: 헤드가 텍스트를 쓰는가. 변형 V0 이미지 전용(A3와 같은 학습법) / V1 +텍스트 특징(문구 고정) / V2 +패러프레이즈 증강 / V3 +음성 지시(정답=정지).
입력 특징 = [이미지 인코더 풀링(문구 무관) ; 융합 인코더의 텍스트 토큰 풀링(문구별)] (V0은 앞 768만). 학습법은 A3와 동일(서보 라벨 1.0, 서브골 드롭아웃 30%, 120 epoch, PiZeroFM).
캐시: scripts/precompute_text_fused.py (3프레임마다 1개). 분할: fold0, fold1, lopo_center. 출력: docs/v5/egvla2_fm/text_cond_t2.json (분할·시드 단위 저장·재개)
plan: docs/plans/plan_20261106_text_conditioned_e2e_vla.md §2. 실행: .venv_lerobot/bin/python3 scripts/eval_text_cond_t2.py (SMOKE=1)"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import eval_target_cond as TC
import train_egvla2_s1 as S1
import robovlm_nav.models.egvla2_fm_heads as HD
T = S1.T
DEV = AB.DEV
SMOKE = bool(os.getenv("SMOKE"))
EPOCHS = 2 if SMOKE else 120
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
OUT = ROOT / "docs/v5/egvla2_fm" / ("text_cond_t2_smoke.json" if SMOKE else "text_cond_t2.json")
STRIDE, BS, GO = 3, 256, 0.2
NPOS, NNEG = 6, 3
VARIANTS = ["V0_img_only", "V1_text_const", "V2_paraphrase_aug", "V3_aug_neg"]
P_NEG, P_NEG_KEEP_BBOX = 0.2, 0.5


def load():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3)
    tc = np.load(ROOT / "runs/egvla2_fm/text_fused_cache.npz")
    BB, CH, VIS, TXT, EPI = [], [], [], [], []
    for i, e in enumerate(E):
        n = len(e["bbox"]); sel = np.arange(0, n, STRIDE)
        vis = tc[f"{e['name']}|vis"].astype(np.float32); assert len(vis) == len(sel), (e["name"], len(vis), len(sel))
        BB.append(e["bbox"][sel]); CH.append(e["chunk"][sel]); VIS.append(vis)
        TXT.append(np.stack([tc[f"{e['name']}|txt|{k}"].astype(np.float32) for k in range(NPOS + NNEG)], 1)); EPI.append(np.full(len(sel), i))
    g = lambda x: torch.tensor(np.concatenate(x)).to(DEV)
    D = dict(bb=g(BB), ch=g(CH), vis=g(VIS), txt=g(TXT), ep=np.concatenate(EPI))
    hp = ROOT / "runs/egvla2_fm/text_fused_cache_heldout.npz"        # 학습에 없던 문구(미지 패러프레이즈 2 + 미지 음성 2): 3번째 프레임마다 1개
    if hp.exists():
        hc = np.load(hp); rows, H_ = [], []
        for i, e in enumerate(E):
            n = len(e["bbox"]); m3 = len(np.arange(0, n, STRIDE)); r = np.full(m3, -1)
            ht = np.stack([hc[f"{e['name']}|txt|{k}"].astype(np.float32) for k in range(4)], 1); assert len(ht) == len(range(0, m3, 3))
            r[::3] = len(H_) + np.arange(len(ht)); rows.append(r); H_.append(ht)
        D["hrow"] = torch.tensor(np.concatenate(rows)).to(DEV); D["htxt"] = g(H_)
    return eps, D


def feats(D, i, k, variant):
    return D["vis"][i] if variant == "V0_img_only" else torch.cat([D["vis"][i], D["txt"][i, k]], -1)


def fit(D, tr, variant, seed):
    torch.manual_seed(seed)
    m = HD.PiZeroFM(12, 768 if variant == "V0_img_only" else 1536).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); n = len(tr); trt = torch.tensor(tr, device=DEV)
    for _ in range(EPOCHS):
        m.train(); perm = trt[torch.randperm(n, device=DEV)]
        for s in range(0, n, BS):
            i = perm[s:s + BS]; B = len(i); bb0 = D["bb"][i]; a = TC.servo_label(D["ch"][i], bb0[:, 8], 1.0)
            bb = bb0 * (torch.rand(B, 1, device=DEV) > 0.3).float()
            k = torch.zeros(B, dtype=torch.long, device=DEV)
            if variant in ("V2_paraphrase_aug", "V3_aug_neg", "V4_neg_only"):
                k = torch.randint(0, NPOS, (B,), device=DEV)
            if variant == "V3_aug_neg":
                neg = torch.rand(B, device=DEV) < P_NEG
                k = torch.where(neg, NPOS + torch.randint(0, NNEG, (B,), device=DEV), k)
                a = torch.where(neg[:, None, None], torch.zeros_like(a), a)                     # 장면에 없는 물체 → 정지
                keep = (torch.rand(B, 1, device=DEV) < P_NEG_KEEP_BBOX).float()                  # 절반은 박스가 남아 있어도(오검출 상황) 정지해야 함
                bb = torch.where(neg[:, None], bb0 * keep, bb)
            ft = torch.cat([D["vis"][i], D["txt"][i, k]], -1) if variant != "V0_img_only" else D["vis"][i]
            x0 = torch.randn_like(a); t = torch.rand(B, device=DEV)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            l = torch.nn.functional.mse_loss(m(bb, ft, xt, t), a - x0); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


@torch.no_grad()
def first_step(m, bb, ft):
    torch.manual_seed(0)
    return HD.sample_chunk(m, bb, ft, 8).mean(1)[:, 0, :2]


@torch.no_grad()
def evaluate(m, D, te, variant):
    te = torch.tensor(te, device=DEV); bb = D["bb"][te]; y = D["ch"][te][:, 0, :2]; ym = y.norm(dim=-1); z = torch.zeros_like(bb)

    def ang(S):
        pm = S.norm(dim=-1); both = (pm > GO) & (ym > GO)
        cos = (S[both] * y[both]).sum(-1) / (pm[both] * ym[both]); return torch.rad2deg(torch.acos(cos.clamp(-1, 1))).mean().item()
    f = lambda k, b: first_step(m, b, feats(D, te, k, variant))
    S0 = f(0, bb); res = dict(ang_ref=ang(S0), ang_para=float(np.mean([ang(f(k, bb)) for k in range(1, NPOS)])))
    neg_zero = [(f(NPOS + j, z).norm(dim=-1) < GO).float().mean().item() for j in range(NNEG)]
    neg_bbox = [(f(NPOS + j, bb).norm(dim=-1) < GO).float().mean().item() for j in range(NNEG)]
    res.update(stop_neg_nobbox=float(np.mean(neg_zero)), stop_neg_bbox=float(np.mean(neg_bbox)),
               stop_pos_bbox=(S0.norm(dim=-1) < GO).float().mean().item(),                          # 정상 지시에서 정지 → 낮아야 함
               text_effect=float(np.mean([(f(NPOS + j, bb) - S0).norm(dim=-1).mean().item() for j in range(NNEG)])),   # 문구만 바꿨을 때 첫 스텝 명령 변화 크기
               para_effect=float(np.mean([(f(k, bb) - S0).norm(dim=-1).mean().item() for k in range(1, NPOS)])))
    if "htxt" in D:                                                                                   # 학습에 없던 문구
        sel = te[D["hrow"][te] >= 0]; r = D["hrow"][sel]; bb_, y_ = D["bb"][sel], D["ch"][sel][:, 0, :2]; ym_ = y_.norm(dim=-1)
        fh = lambda j, b: first_step(m, b, D["vis"][sel] if variant == "V0_img_only" else torch.cat([D["vis"][sel], D["htxt"][r, j]], -1))
        ref_ = first_step(m, bb_, feats(D, sel, 0, variant))
        def ang_(S):
            pm = S.norm(dim=-1); both = (pm > GO) & (ym_ > GO); c = (S[both] * y_[both]).sum(-1) / (pm[both] * ym_[both])
            return torch.rad2deg(torch.acos(c.clamp(-1, 1))).mean().item()
        res.update(unseen_ang_para=float(np.mean([ang_(fh(j, bb_)) for j in (0, 1)])), unseen_ang_ref=ang_(ref_), unseen_n=int(len(sel)),
                   unseen_para_effect=float(np.mean([(fh(j, bb_) - ref_).norm(dim=-1).mean().item() for j in (0, 1)])),
                   unseen_stop_pos=float(np.mean([(fh(j, bb_).norm(dim=-1) < GO).float().mean().item() for j in (0, 1)])),
                   unseen_stop_neg_nobbox=float(np.mean([(fh(j, torch.zeros_like(bb_)).norm(dim=-1) < GO).float().mean().item() for j in (2, 3)])),
                   unseen_stop_neg_bbox=float(np.mean([(fh(j, bb_).norm(dim=-1) < GO).float().mean().item() for j in (2, 3)])))
    return res


def pick_probes(eps, D, te, n=6):
    """시험 분할에서 서로 다른 셀 n곳의 프레임(미지 문구 캐시가 있는 프레임, 사람 명령이 이동 중)을 결정적으로 고른다."""
    cand = te[(D["hrow"][te] >= 0).cpu().numpy()] if "hrow" in D else te
    cand = [int(i) for i in cand if float(D["ch"][i][0, :2].norm()) > GO]
    seen, out = set(), []
    for i in cand[len(cand) // 7::max(1, len(cand) // 40)]:
        c = eps[int(D["ep"][i])]["cell"]
        if c not in seen: seen.add(c); out.append(i)
        if len(out) == n: break
    return out


def export_probes(eps, D, idx):
    """실험에 실제로 쓰는 원본 프레임(데이터셋 H5, BGR 저장 → RGB 변환)을 저장하고 메타를 기록한다."""
    import h5py
    from PIL import Image
    outd = ROOT / "docs/v5/figures/textcond"; outd.mkdir(parents=True, exist_ok=True); meta = []
    for r, i in enumerate(idx):
        e = eps[int(D["ep"][i])]; pos = int((D["ep"][:i] == D["ep"][i]).sum()); fr = pos * STRIDE
        with h5py.File(ROOT / "ROS_action/mobile_vla_dataset_egvla2" / e["name"], "r") as f:
            im = f["observations"]["images"][fr] if "observations" in f else f["images"][fr]
        Image.fromarray(np.ascontiguousarray(im[:, :, ::-1]).astype(np.uint8)).convert("RGB").resize((480, 270)).save(outd / f"probe_{r}.jpg", quality=92)
        bb = D["bb"][i].cpu().numpy()[8:12].tolist(); ch = D["ch"][i]
        meta.append(dict(i=r, ep=e["name"], frame=fr, cell=e["cell"], bbox=bb, human=ch[0, :2].tolist(),
                         servo=TC.servo_label(ch[None], D["bb"][i][8:9], 1.0)[0, 0, :2].tolist()))
    (ROOT / "docs/v5/egvla2_fm/text_cond_probes.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False)); return meta


@torch.no_grad()
def probe_preds(m, D, idx, variant):
    ix = torch.tensor(idx, device=DEV); bb = D["bb"][ix]; z = torch.zeros_like(bb); r = D["hrow"][ix]
    F = lambda k: D["vis"][ix] if variant == "V0_img_only" else torch.cat([D["vis"][ix], D["txt"][ix, k]], -1)
    H_ = lambda j: D["vis"][ix] if variant == "V0_img_only" else torch.cat([D["vis"][ix], D["htxt"][r, j]], -1)
    g = lambda ft, b: [[round(float(x), 4) for x in v] for v in first_step(m, b, ft).cpu()]
    return dict(ref=g(F(0), bb), para_seen=g(F(4), bb), para_unseen=g(H_(0), bb), neg_seen_nobbox=g(F(6), z), neg_seen_bbox=g(F(6), bb),
                neg_unseen_nobbox=g(H_(2), z), neg_unseen_bbox=g(H_(2), bb))


def main():
    eps, D = load()
    if "htxt" not in D and not SMOKE: raise SystemExit("보류 문구 캐시(text_fused_cache_heldout.npz)가 아직 없음")
    folds = S1.make_folds(eps)
    cells = lambda p: [i for i, e in enumerate(eps) if e["cell"].startswith(p)]
    splits = {"fold0": folds[0], "fold1": folds[1], "lopo_center": cells("center|")}
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    probes = None
    for name, te_eps in splits.items():
        te = np.where(np.isin(D["ep"], te_eps))[0]; tr = np.where(~np.isin(D["ep"], te_eps))[0]
        if name == "fold0" and "hrow" in D: probes = pick_probes(eps, D, te); export_probes(eps, D, probes)
        for v in VARIANTS:
            for sd in SEEDS:
                key = f"{name}|{v}|{sd}"
                if key in res: continue
                mdl = fit(D, tr, v, sd); res[key] = evaluate(mdl, D, te, v)
                if name == "fold0" and probes: res[key]["probe"] = probe_preds(mdl, D, probes, v)
                res[key]["t"] = round(time.time(), 1); OUT.write_text(json.dumps(res, indent=1))
                print(key, {a: round(b, 3) for a, b in res[key].items() if isinstance(b, float)}, flush=True)


main()
