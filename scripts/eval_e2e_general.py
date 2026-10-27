#!/usr/bin/env python3
"""E2E 일반 비교: 같은 데이터·같은 분할·같은 라벨·같은 지표에서 기존 EGVLA2(고정 융합 특징)와 '비전 마지막 단계 LoRA' E2E를 비교한다.
데이터: 300 에피소드의 6프레임 간격 프레임(약 4,860장), 분할 fold0·lopo_center·lopo_strong_right, bbox 창은 실제 값(상수 아님), 라벨 actions_exec 청크(H=8), 서브골 드롭아웃 30%.
팔: A = 고정 융합 특징(기존 EGVLA2 방식, 120 epoch) / B0 = 고정 DaViT 풀링 특징 1024→768 투영+헤드(120 및 20 epoch) / B1 = DaViT 마지막 단계(conv3+block3)에 LoRA(r=8, qkv/proj)+투영+헤드를 행동 손실로 함께 학습(20 epoch).
제한: 전체 E2E(전 단계 LoRA)는 젯슨급 GPU에서 1.5장/초라 불가능해서 stage-2 출력 캐시 위에서 마지막 단계만 학습한다(런타임 실측 40장/초). 평가는 gap_suite 지표 + 실기 라이브 프레임(416장)의 규칙 전문가 일치(이미지는 같은 파이프라인으로 처리).
실행: HF_HUB_OFFLINE=1 .venv/bin/python3 scripts/eval_e2e_general.py   (SMOKE=1: epoch 2·분할 1개)   출력: docs/v5/egvla2_fm/e2e_general.json, training_status.json(대시보드)
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import gap_suite as GS  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
import training_status as TS  # noqa: E402
import robovlm_nav.models.egvla2_fm_heads as HD  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
T = S1.T
DEV, BS = GS.DEV, GS.BS
SMOKE = bool(os.getenv("SMOKE"))
E_A, E_B0, E_B0S, E_B1 = (2, 2, 2, 2) if SMOKE else (120, 120, 20, 20)
OUT = ROOT / "docs/v5/egvla2_fm" / ("e2e_general_smoke.json" if SMOKE else "e2e_general.json")
CACHE = ROOT / "runs/e2e_cache"


class Proj(nn.Module):
    def __init__(self):
        super().__init__(); self.n = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 768), nn.GELU())

    def forward(self, x):
        return self.n(x)


def stage3_features(base, h, size=(48, 48)):
    h, size = base.convs[3](h, size); h, size = base.blocks[3](h, size)
    return h.mean(1)                                                            # (B,1024) 풀링


def pooled_all(base, cache, idx, bs=64):
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(idx), bs):
            h = torch.from_numpy(np.asarray(cache[idx[s:s + bs]])).to(DEV).float(); out.append(stage3_features(base, h).float())
    return torch.cat(out)


def head_loss(head, bb, ft, a, drop=0.3):
    b = bb * (torch.rand(len(bb), 1, device=DEV) > drop).float()
    x0 = torch.randn_like(a); t = torch.rand(len(a), device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
    return F.mse_loss(head(b, ft, xt, t), a - x0)


def train_pooled(proj, head, bb, pooled, ch, epochs, seed=0):
    torch.manual_seed(seed); params = list(proj.parameters()) + list(head.parameters()); opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2); n = len(bb)
    for _ in range(epochs):
        proj.train(); head.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; l = head_loss(head, bb[i], proj(pooled[i]), ch[i]); opt.zero_grad(); l.backward(); opt.step()
    proj.eval(); head.eval()


def main():
    eps = S1.load_main(); cache_c = np.load(T.CACHE); E = AB.prep(eps, cache_c, 3); folds = S1.make_folds(eps)
    fr = json.load(open(CACHE / "frames.json")); frames = fr["frames"]; stage2 = np.load(CACHE / "stage2.npy", mmap_mode="r"); live_stage2 = np.load(CACHE / "live_stage2.npy", mmap_mode="r")
    ep_of = np.array([f[0] for f in frames]); bbs = np.stack([E[ei]["bbox"][t] for ei, t in frames]); fts = np.stack([E[ei]["feat"][t] for ei, t in frames]); chs = np.stack([E[ei]["chunk"][t] for ei, t in frames])
    BB, FT, CH = [torch.from_numpy(x).float().to(DEV) for x in (bbs, fts, chs)]
    splits = {"fold0": folds[0], "lopo_center": [i for i, e in enumerate(eps) if e["cell"].startswith("center|")], "lopo_strong_right": [i for i, e in enumerate(eps) if e["cell"].startswith("strong_right|")]}
    lb, lf_fused, win = GS.live_inputs()                                          # 라이브 bbox·고정 융합 특징·bbox 창(416장)
    lm = json.load(open(CACHE / "live_meta.json")); z = np.load(ROOT / "docs/v5/egvla2_fm/live_feats.npz"); ref = [m.split("|")[0].replace("session_", "") + "|" + m.split("|")[1] for m in z["meta"]]
    assert [m for m in lm] == ref, "라이브 프레임 순서 불일치"
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float32)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, te_eps in splits.items():
        res.setdefault(name, {}); tes = set(te_eps)
        tr_idx = np.array([k for k in range(len(frames)) if ep_of[k] not in tes]); te_idx = np.array([k for k in range(len(frames)) if ep_of[k] in tes])
        tr_t = torch.from_numpy(tr_idx).to(DEV); te_t = torch.from_numpy(te_idx).to(DEV); live = lambda feats: (lb, feats, win)
        print(name, "train", len(tr_idx), "test", len(te_idx), flush=True)
        if "A_frozen_fused" not in res[name]:
            m = GS.fit((BB[tr_t], FT[tr_t], CH[tr_t])) if not SMOKE else GS.fit((BB[tr_t], FT[tr_t], CH[tr_t]))
            res[name]["A_frozen_fused"] = GS.split_metrics(m, (BB[te_t], FT[te_t], CH[te_t]), live(lf_fused)); OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(name, "A", flush=True)
        base = model.vision_tower.to(DEV).eval()
        pooled = pooled_all(base, stage2, np.arange(len(frames))); lpooled = pooled_all(base, live_stage2, np.arange(len(lm)))
        for nm, ep in (("B0_frozen_davit_120ep", E_B0), ("B0_frozen_davit_20ep", E_B0S)):
            if nm in res[name]:
                continue
            proj, head = Proj().to(DEV), HD.PiZeroFM(12, 768).to(DEV); train_pooled(proj, head, BB[tr_t], pooled[tr_t], CH[tr_t], ep)
            with torch.no_grad():
                res[name][nm] = GS.split_metrics(head, (BB[te_t], proj(pooled[te_t]), CH[te_t]), live(proj(lpooled)))
            OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(name, nm, flush=True)
        if "B1_lora_last_stage_20ep" not in res[name]:
            torch.manual_seed(0)
            cfg = LoraConfig(r=8, lora_alpha=16, target_modules=r".*blocks\.3\..*(qkv|proj)", lora_dropout=0.05, bias="none")
            vt = get_peft_model(AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float32).vision_tower, cfg).to(DEV)
            b3 = vt.get_base_model(); proj, head = Proj().to(DEV), HD.PiZeroFM(12, 768).to(DEV)
            lora = [p for p in vt.parameters() if p.requires_grad]
            opt = torch.optim.AdamW([dict(params=lora, lr=2e-4), dict(params=list(proj.parameters()) + list(head.parameters()), lr=1e-3)], weight_decay=1e-2)
            ntr = len(tr_idx); hist = []; t0 = time.time()
            def feats(idx_np, train):
                h = torch.from_numpy(np.asarray(stage2[np.sort(idx_np)])).to(DEV).float() if False else torch.from_numpy(np.stack([np.asarray(stage2[k]) for k in idx_np])).to(DEV).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = stage3_features(b3, h)
                return proj(f.float())
            for ep in range(E_B1):
                vt.train(); proj.train(); head.train(); perm = np.random.default_rng(ep).permutation(tr_idx); tot = 0.0; nb = 0
                for s in range(0, ntr, 32):
                    ii = perm[s:s + 32]; it = torch.from_numpy(ii).to(DEV); ft = feats(ii, True)
                    l = head_loss(head, BB[it], ft, CH[it]); opt.zero_grad(); l.backward(); opt.step(); tot += float(l); nb += 1
                hist.append(dict(ep=ep + 1, loss=tot / nb, angle=None))
                TS.write(f"E2E-LoRA last-stage ({name})", "running", ep + 1, E_B1, loss=tot / nb, history=hist)
                print(name, "B1 epoch", ep + 1, round(tot / nb, 4), f"{time.time() - t0:.0f}s", flush=True)
            vt.eval(); proj.eval(); head.eval()
            def feats_eval(arr, idxs):
                out = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for s in range(0, len(idxs), 64):
                        h = torch.from_numpy(np.stack([np.asarray(arr[k]) for k in idxs[s:s + 64]])).to(DEV).float(); out.append(proj(stage3_features(b3, h).float()).float())
                return torch.cat(out)
            with torch.no_grad():
                fte = feats_eval(stage2, te_idx); fl = feats_eval(live_stage2, np.arange(len(lm)))
                res[name]["B1_lora_last_stage_20ep"] = GS.split_metrics(head, (BB[te_t], fte, CH[te_t]), live(fl)); res[name]["B1_lora_last_stage_20ep"]["train_seconds"] = time.time() - t0
            OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(name, "B1 done", flush=True)
            TS.write(f"E2E-LoRA last-stage ({name})", "idle", E_B1, E_B1, history=hist)
        if SMOKE:
            break


if __name__ == "__main__":
    main()
