#!/usr/bin/env python3
"""전 단계 LoRA E2E(해상도 384) 일반 비교. eval_e2e_general.py와 같은 데이터·분할·라벨·지표·bbox 창을 쓰고, 입력 해상도를 384로 낮춰 전 단계(모든 conv/block의 qkv·proj)에 LoRA를 건다.
같은 단위 비교를 위해 기준 팔도 384로 맞춘다: B0@384 = 고정 DaViT 풀링 특징(384) + 투영 + 헤드(120·20 epoch), B2@384 = 전 단계 LoRA(r=8)+투영+헤드를 행동 손실로 함께 학습(EPOCHS 기본 8).
입력은 runs/e2e_cache/px768.npy(정규화된 768 입력, fp16)를 학습 때 384로 보간한다. 실행: HF_HUB_OFFLINE=1 .venv/bin/python3 scripts/eval_e2e_fullstage.py (SMOKE=1: epoch 1·분할 1개, 프레임 64장)
출력: docs/v5/egvla2_fm/e2e_fullstage.json, training_status.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_e2e_general as G  # noqa: E402
import eval_subgoal_ablation as AB  # noqa: E402
import gap_suite as GS  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
import training_status as TS  # noqa: E402
import robovlm_nav.models.egvla2_fm_heads as HD  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
T = S1.T
DEV = GS.DEV
SMOKE = bool(os.getenv("SMOKE"))
RES = int(os.getenv("RES", 384)); E_B2 = 1 if SMOKE else int(os.getenv("EPOCHS", 8)); E_B0, E_B0S = (1, 1) if SMOKE else (120, 20)
OUT = ROOT / "docs/v5/egvla2_fm" / ("e2e_fullstage_smoke.json" if SMOKE else "e2e_fullstage.json")
CACHE = ROOT / "runs/e2e_cache"


def px_batch(arr, idx):
    x = torch.from_numpy(np.stack([np.asarray(arr[k]) for k in idx])).to(DEV).float()
    return F.interpolate(x, size=(RES, RES), mode="bilinear", antialias=True, align_corners=False)


def pooled_px(base, arr, idxs, bs=32):
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(idxs), bs):
            out.append(base.forward_features_unpool(px_batch(arr, idxs[s:s + bs])).float().mean(1))
    return torch.cat(out)


def main():
    eps = S1.load_main(); cache_c = np.load(T.CACHE); E = AB.prep(eps, cache_c, 3); folds = S1.make_folds(eps)
    fr = json.load(open(CACHE / "frames.json")); frames = fr["frames"]; px = np.load(CACHE / "px768.npy", mmap_mode="r"); lpx = np.load(CACHE / "live_px768.npy", mmap_mode="r")
    ep_of = np.array([f[0] for f in frames]); bbs = np.stack([E[ei]["bbox"][t] for ei, t in frames]); chs = np.stack([E[ei]["chunk"][t] for ei, t in frames])
    BB, CH = [torch.from_numpy(x).float().to(DEV) for x in (bbs, chs)]
    splits = {"fold0": folds[0], "lopo_center": [i for i, e in enumerate(eps) if e["cell"].startswith("center|")], "lopo_strong_right": [i for i, e in enumerate(eps) if e["cell"].startswith("strong_right|")]}
    lb, _, win = GS.live_inputs(); nlive = len(lpx)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, te_eps in splits.items():
        res.setdefault(name, {}); tes = set(te_eps)
        tr_idx = np.array([k for k in range(len(frames)) if ep_of[k] not in tes]); te_idx = np.array([k for k in range(len(frames)) if ep_of[k] in tes])
        if SMOKE:
            tr_idx, te_idx = tr_idx[:64], te_idx[:32]
        tr_t, te_t = torch.from_numpy(tr_idx).to(DEV), torch.from_numpy(te_idx).to(DEV); live = lambda f_: (lb, f_.detach().float().cpu().numpy() if torch.is_tensor(f_) else f_, win)
        print(name, "train", len(tr_idx), "test", len(te_idx), "res", RES, flush=True)
        base_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float32)
        base = base_model.vision_tower.to(DEV).eval()
        if any(k not in res[name] for k in (f"B0_frozen_davit{RES}_120ep", f"B0_frozen_davit{RES}_20ep")):
            pooled_tr = pooled_px(base, px, tr_idx); pooled_te = pooled_px(base, px, te_idx); pooled_live = pooled_px(base, lpx, np.arange(nlive))
            for nm, ep in ((f"B0_frozen_davit{RES}_120ep", E_B0), (f"B0_frozen_davit{RES}_20ep", E_B0S)):
                if nm in res[name]:
                    continue
                proj, head = G.Proj().to(DEV), HD.PiZeroFM(12, 768).to(DEV); G.train_pooled(proj, head, BB[tr_t], pooled_tr, CH[tr_t], ep)
                with torch.no_grad():
                    res[name][nm] = GS.split_metrics(head, (BB[te_t], proj(pooled_te), CH[te_t]), live(proj(pooled_live)))
                OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(name, nm, flush=True)
        nm2 = f"B2_lora_all_stages_{RES}_{E_B2}ep"
        if nm2 not in res[name]:
            torch.manual_seed(0)
            cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv", "proj"], lora_dropout=0.05, bias="none")
            vt = get_peft_model(AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float32).vision_tower, cfg).to(DEV)
            b3 = vt.get_base_model(); proj, head = G.Proj().to(DEV), HD.PiZeroFM(12, 768).to(DEV)
            lora = [p for p in vt.parameters() if p.requires_grad]
            opt = torch.optim.AdamW([dict(params=lora, lr=2e-4), dict(params=list(proj.parameters()) + list(head.parameters()), lr=1e-3)], weight_decay=1e-2)
            hist = []; t0 = time.time(); ntr = len(tr_idx); BSZ = 16
            for ep in range(E_B2):
                vt.train(); proj.train(); head.train(); perm = np.random.default_rng(ep).permutation(tr_idx); tot = 0.0; nb = 0
                for s in range(0, ntr, BSZ):
                    ii = perm[s:s + BSZ]; it = torch.from_numpy(ii).to(DEV)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        f = b3.forward_features_unpool(px_batch(px, ii)).float().mean(1)
                    l = G.head_loss(head, BB[it], proj(f.float()), CH[it]); opt.zero_grad(); l.backward(); opt.step(); tot += float(l); nb += 1
                    if nb % 20 == 0:
                        TS.write(f"E2E full-stage LoRA {RES}px ({name})", "running", ep + 1, E_B2, loss=tot / nb, history=hist, extra=dict(step=nb, steps=ntr // BSZ))
                hist.append(dict(ep=ep + 1, loss=tot / nb, angle=None)); TS.write(f"E2E full-stage LoRA {RES}px ({name})", "running", ep + 1, E_B2, loss=tot / nb, history=hist)
                print(name, "B2 epoch", ep + 1, round(tot / nb, 4), f"{time.time() - t0:.0f}s", flush=True)
            vt.eval(); proj.eval(); head.eval()
            def feats(arr, idxs):
                out = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for s in range(0, len(idxs), 32):
                        out.append(proj(b3.forward_features_unpool(px_batch(arr, idxs[s:s + 32])).float().mean(1).float()).float())
                return torch.cat(out)
            with torch.no_grad():
                r = GS.split_metrics(head, (BB[te_t], feats(px, te_idx), CH[te_t]), live(feats(lpx, np.arange(nlive)))); r["train_seconds"] = time.time() - t0; res[name][nm2] = r
            OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(name, nm2, "done", flush=True)
            TS.write(f"E2E full-stage LoRA {RES}px ({name})", "idle", E_B2, E_B2, history=hist)
        if SMOKE:
            break


if __name__ == "__main__":
    main()
