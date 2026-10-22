#!/usr/bin/env python3
"""덜 보수적인 후보 ablation(학습·데이터·목표 변형). 분할: 5-fold + 안 본 목표 위치 5종, seed 0, FM-π0(층2·폭128), 서브골 드롭아웃 30%.
변형: ep250(학습 길게) / H4·H16(청크 길이 재학습) / w_first(첫 스텝 손실 4배) / unit_target(방향만 학습, 크기 1로) / smooth_labels(사람 조작 5프레임 이동평균)
      / feat_prev3(3프레임 전 특징 추가=시간 맥락) / bbox_noise(bbox 입력 잡음) / w_curved(곡선 구간 프레임 가중 3배) + 기준(base) 대비,
      평가만 바꾸는 초기 노이즈 크기 σ(0·0.5·1·1.5). 모든 평가 정답은 원본 사람 조작(첫 스텝, 3스텝 평균).
출력: docs/v5/egvla2_fm/bold_ablation.json (분할 단위 저장·재개). 실행: .venv_lerobot/bin/python3 scripts/eval_bold_ablation.py  (SMOKE=1: epoch 2·분할 1개)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_runtime_ablation as RA  # noqa: E402
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
import robovlm_nav.models.egvla2_fm_heads as HD  # noqa: E402
T = S1.T
DEV, BS = AB.DEV, AB.BS
SMOKE = bool(os.getenv("SMOKE"))
EPOCHS = 2 if SMOKE else 120
SEED = int(os.getenv("SEED", 0))
OUT = ROOT / "docs/v5/egvla2_fm" / ("bold_ablation_smoke.json" if SMOKE else "bold_ablation.json" if SEED == 0 else f"bold_ablation_seed{SEED}.json")
H0 = HD.H
NOISE_MASK = torch.tensor([1, 1, 1, 0] * 3, dtype=torch.float32)       # bbox 창(3)×(cx,cy,area,has)


def rebuild_chunk(a, H):
    out = np.zeros((len(a), H, a.shape[1]), np.float32)
    for t in range(len(a)):
        seg = a[t:t + H]; out[t, :len(seg)] = seg
    return out


def smooth(a, w=5):
    k = np.ones(w) / w; pad = w // 2
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(ap[:, c], k, mode="valid") for c in range(a.shape[1])], 1).astype(np.float32)


def variant_data(E, kind):
    """E(기준 데이터) → 변형 데이터. 청크 길이 H 변형은 (E, H) 반환."""
    out = []
    for e in E:
        e2 = dict(e); a = e["chunk"][:, 0, :]
        if kind == "smooth_labels":
            e2["chunk"] = rebuild_chunk(smooth(a), H0)
        elif kind in ("H4", "H16"):
            e2["chunk"] = rebuild_chunk(a, int(kind[1:]))
        elif kind == "feat_prev3":
            f = e["feat"]; idx = np.maximum(np.arange(len(f)) - 3, 0); e2["feat"] = np.concatenate([f, f[idx]], 1).astype(np.float32)
        out.append(e2)
    return out


def fit(tr, kind, seed=0, drop=0.3):
    H = int(kind[1:]) if kind in ("H4", "H16") else H0
    HD.H = H
    torch.manual_seed(seed)
    m = HD.PiZeroFM(tr[0].shape[1], tr[1].shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); n = len(tr[0])
    ep = EPOCHS * 2 if kind == "ep250" and not SMOKE else EPOCHS
    sw = torch.ones(H, device=DEV)
    if kind == "w_first":
        sw[0] = 4.0
    sw = sw / sw.mean()
    fw = torch.ones(n, device=DEV)
    if kind == "w_curved":
        ang = torch.rad2deg(torch.atan2(tr[2][:, 0, 1], tr[2][:, 0, 0])).abs(); fw = torch.where(ang > 20, 3.0, 1.0).to(DEV)
    nm = NOISE_MASK.to(DEV)
    for _ in range(ep):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; bb = tr[0][i]; a = tr[2][i]
            if kind == "bbox_noise":
                bb = bb + 0.02 * torch.randn_like(bb) * nm
            bb = bb * (torch.rand(len(i), 1, device=DEV) > drop).float()
            if kind == "unit_target":
                xy = a[..., :2]; nrm = xy.norm(dim=-1, keepdim=True)
                a = torch.cat([torch.where(nrm > 0.2, xy / nrm.clamp_min(1e-6), xy), a[..., 2:]], -1)
            x0 = torch.randn_like(a); t = torch.rand(len(i), device=DEV)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            per = ((m(bb, tr[1][i], xt, t) - (a - x0)) ** 2).mean(-1)          # (B,H)
            l = ((per * sw[None]).mean(1) * fw[i]).mean() / fw[i].mean()
            opt.zero_grad(); l.backward(); opt.step()
    m.eval(); HD.H = H0
    return m, H


@torch.no_grad()
def draw_sigma(m, bb, ft, k, steps, sigma, H, seed=0):
    torch.manual_seed(seed); HD.H = H
    B = len(bb); b2, f2 = bb.repeat_interleave(k, 0), ft.repeat_interleave(k, 0)
    x = sigma * torch.randn(B * k, H, HD.ADIM, device=bb.device)
    for i in range(steps):
        t = torch.full((B * k,), i / steps, device=bb.device); x = x + m(b2, f2, x, t) / steps
    HD.H = H0
    return x.view(B, k, H, HD.ADIM)[..., :2].cpu().numpy()


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    splits = {f"fold{i}": folds[i] for i in range(len(folds))}
    for pos in ("center", "weak_left", "strong_left", "weak_right", "strong_right"):
        splits[f"lopo_{pos}"] = [i for i, e in enumerate(eps) if e["cell"].startswith(pos + "|")]
    kinds = ["base", "ep250", "H4", "H16", "w_first", "unit_target", "smooth_labels", "feat_prev3", "bbox_noise", "w_curved"]
    if os.getenv("KINDS"):
        kinds = [k for k in kinds if k in os.getenv("KINDS").split(",")]
    data = {k: variant_data(E, k) if k in ("smooth_labels", "H4", "H16", "feat_prev3") else E for k in kinds}
    order = [x for x in os.getenv("SPLITS", "").split(",") if x in splits]
    if order:
        splits = {x: splits[x] for x in order}
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, te_idx in splits.items():
        res.setdefault(name, {})
        tr_idx = [i for i in range(len(E)) if i not in te_idx]
        gt = T.to_t(E, te_idx)[2][..., :2].cpu().numpy(); g_first, g_burst = gt[:, 0], gt[:, :3].mean(1)
        for kind in kinds:
            if kind in res[name]:
                continue
            tr, te = T.to_t(data[kind], tr_idx), T.to_t(data[kind], te_idx)
            m, H = fit(tr, kind, seed=SEED)
            HD.H = H
            S = RA.draw(m, te[0], te[1], 8, 10).mean(1); HD.H = H0
            Sm = RA.draw(m, torch.zeros_like(te[0]), te[1], 8, 10).mean(1) if False else None
            r = dict(first=RA.angle(S[:, 0], g_first), n1_vs_burst=RA.angle(S[:, 0], g_burst), params=int(sum(p.numel() for p in m.parameters())))
            if kind == "base":
                for sg in (0.0, 0.5, 1.0, 1.5):
                    Ss = draw_sigma(m, te[0], te[1], 8, 10, sg, H).mean(1); r[f"sigma{sg}|first"] = RA.angle(Ss[:, 0], g_first)
            res[name][kind] = r
            OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            print(name, kind, {k: round(v, 2) for k, v in r.items() if not isinstance(v, int)}, flush=True)
        if SMOKE:
            break


if __name__ == "__main__":
    main()
