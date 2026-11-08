#!/usr/bin/env python3
"""C 상황(인식이 흔들릴 때): 박스를 훼손해도 방향을 유지하나 + 박스 훼손 증강(V0R) 보완안 검증.
plan: docs/plans/plan_20261108_bbox_robustness_and_supplement.md. 캐시·로더는 T4(text_vocab_cache) 재사용. 출력: docs/v5/egvla2_fm/bbox_robustness.json
변형 V0(A3 방식) / V0R(박스 훼손 증강) / V4(텍스트 조건화, T4 설정) / V5(토큰 어텐션, 캐시가 있을 때). 실행: .venv_lerobot/bin/python3 scripts/eval_bbox_robustness.py (SMOKE=1)"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_text_cond_t4 as T4
HD, TC, S1, DEV = T4.HD, T4.TC, T4.S1, T4.DEV
SMOKE = bool(os.getenv("SMOKE")); EPOCHS = 2 if SMOKE else 120
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
OUT = ROOT / "docs/v5/egvla2_fm" / ("bbox_robustness_smoke.json" if SMOKE else "bbox_robustness.json")
BS, GO = 256, 0.2
VARIANTS = ["V0_img_only", "V0R_bbox_aug", "V4_big_vocab"]
CX, AR, HAS = [0, 4, 8], [2, 6, 10], [3, 7, 11]


def corrupt(bb, kind, g):
    """bb (B,12) = [cx,cy,area,has]×3. kind: drop / jitter / wrong. g: torch.Generator."""
    out = bb.clone(); B = len(bb)
    if kind == "drop": return torch.zeros_like(bb)
    has = (bb[:, HAS[-1]] > 0.5).float()[:, None]
    if kind == "jitter":
        n = torch.randn(B, 1, device=bb.device, generator=g) * 0.15; m = torch.randn(B, 1, device=bb.device, generator=g) * 0.3
        for c in CX: out[:, c] = torch.where(has[:, 0] > 0, (bb[:, c] + n[:, 0]).clamp(0, 1), bb[:, c])
        for a in AR: out[:, a] = torch.where(has[:, 0] > 0, bb[:, a] * torch.exp(m[:, 0]), bb[:, a])
    if kind == "wrong":
        for c in CX: out[:, c] = torch.where(has[:, 0] > 0, 1 - bb[:, c], bb[:, c])
    return out


def fit_robust(D, tr, seed):
    """V0R: V0 + 박스 훼손 증강(제거 30% · jitter 20% · 반전 10%), 정답은 훼손 전 박스 기준 서보 라벨."""
    torch.manual_seed(seed); g = torch.Generator(device=DEV); g.manual_seed(seed)
    m = HD.PiZeroFM(12, 768).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); n = len(tr); trt = torch.tensor(tr, device=DEV)
    for _ in range(EPOCHS):
        m.train(); perm = trt[torch.randperm(n, device=DEV)]
        for s in range(0, n, BS):
            i = perm[s:s + BS]; B = len(i); bb0 = D["bb"][i]; a = TC.servo_label(D["ch"][i], bb0[:, 8], 1.0)
            u = torch.rand(B, 1, device=DEV, generator=g)
            bb = torch.where(u < 0.3, torch.zeros_like(bb0), torch.where((u >= 0.3) & (u < 0.5), corrupt(bb0, "jitter", g), torch.where((u >= 0.5) & (u < 0.6), corrupt(bb0, "wrong", g), bb0)))
            ft = D["vis"][i]; x0 = torch.randn_like(a); t = torch.rand(B, device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            l = torch.nn.functional.mse_loss(m(bb, ft, xt, t), a - x0); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


@torch.no_grad()
def evaluate(m, D, te, variant):
    te = torch.tensor(te, device=DEV); bb = D["bb"][te]; y = D["ch"][te][:, 0, :2]; ym = y.norm(dim=-1)
    ft = D["vis"][te] if variant in ("V0_img_only", "V0R_bbox_aug") else torch.cat([D["vis"][te], T4.txt(D, te, 0)], -1)
    def ang(S):
        pm = S.norm(dim=-1); both = (pm > GO) & (ym > GO); c = (S[both] * y[both]).sum(-1) / (pm[both] * ym[both])
        return torch.rad2deg(torch.acos(c.clamp(-1, 1))).mean().item()
    g = torch.Generator(device=DEV); g.manual_seed(123); res = {}
    conds = {"clean": bb, "drop100": corrupt(bb, "drop", g), "jitter": corrupt(bb, "jitter", g), "wrong": corrupt(bb, "wrong", g)}
    half = (torch.rand(len(bb), 1, device=DEV, generator=g) < 0.5); conds["drop50"] = torch.where(half, torch.zeros_like(bb), bb)
    for k, b in conds.items():
        S = T4.step1(m, b, ft); res[k] = dict(ang=ang(S), stop=(S.norm(dim=-1) < GO).float().mean().item())
    return res


def main():
    eps, D = T4.load(); folds = S1.make_folds(eps)
    cells = lambda p: [i for i, e in enumerate(eps) if e["cell"].startswith(p)]
    splits = {"fold0": folds[0], "fold1": folds[1], "lopo_center": cells("center|")}
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, te_eps in splits.items():
        te = np.where(np.isin(D["ep"], te_eps))[0]; tr = np.where(~np.isin(D["ep"], te_eps))[0]
        for v in VARIANTS:
            for sd in SEEDS:
                key = f"{name}|{v}|{sd}"
                if key in res: continue
                T4.EPOCHS = EPOCHS
                mdl = fit_robust(D, tr, sd) if v == "V0R_bbox_aug" else T4.fit(D, tr, v, sd)
                res[key] = evaluate(mdl, D, te, v); res[key]["t"] = round(time.time(), 1); OUT.write_text(json.dumps(res, indent=1))
                print(key, {c: round(x["ang"], 1) for c, x in res[key].items() if isinstance(x, dict)}, flush=True)


if __name__ == "__main__":
    main()
