#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5대 목표 경로(LOPO) 전체에 대한 Base vs E1b_tuned 모델 전수 평가 스크립트"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import json
import time
import numpy as np
import torch

import train_egvla2_s1 as S1
import gap_suite as GS
import gap_e1b as E1B
import eval_subgoal_ablation as AB

T = S1.T
DEV = GS.DEV

OUT_JSON = ROOT / "docs/v5/egvla2_fm/all_lopo_eval.json"

def fmt_val(v, unit="°", fmt="+.2f"):
    if v is None:
        return "N/A"
    try:
        return f"{v:{fmt}}{unit}"
    except Exception:
        return str(v)

def print_result(tag, r):
    s_str = fmt_val(r.get('S'))
    s_img_str = fmt_val(r.get('S_img'))
    de_str = fmt_val(r.get('dir_err'), fmt=".2f")
    spin_r = r.get('spin_recall')
    spin_str = f"{spin_r*100:.1f}%" if spin_r is not None else "N/A"
    mode_acc = r.get('mode_acc')
    mode_str = f"{mode_acc*100:.1f}%" if mode_acc is not None else "N/A"
    print(f"    -> {tag}: S={s_str}, DirErr={de_str}, S_img={s_img_str}, SpinRecall={spin_str}, ModeAcc={mode_str}")

def main():
    print(f"Using device: {DEV}")
    eps = S1.load_main()
    cache = np.load(T.CACHE)
    E = AB.prep(eps, cache, 3)
    folds = S1.make_folds(eps)
    by_name = {e["name"]: i for i, e in enumerate(eps)}

    # 5대 목표 위치 정의
    pos_list = ["strong_left", "weak_left", "center", "weak_right", "strong_right"]
    
    splits = {"fold0": folds[0]}
    for p in pos_list:
        splits[f"lopo_{p}"] = [i for i, e in enumerate(eps) if e["cell"].startswith(p + "|")]

    print("=== 5대 목표 위치 분할 목록 ===")
    for sname, s_idx in splits.items():
        print(f"  {sname}: {len(s_idx)} 에피소드")

    # 증강 캐시 로드 (E1b_tuned: thresh=0.22, gain=2.6)
    print("\nBuilding counterfactual augmentation cache (thresh=0.22, gain=2.6)...")
    aug_rows = E1B.build(E, by_name, None, thresh=0.22, gain=2.6)
    live = GS.live_inputs()

    res = {}
    if OUT_JSON.exists():
        try:
            res = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:
            res = {}

    for sname, te_idx in splits.items():
        print(f"\n==================================================")
        print(f">>> Evaluating Split: {sname} ({len(te_idx)} test episodes)")
        print(f"==================================================")
        res.setdefault(sname, {})
        tr_idx = [i for i in range(len(E)) if i not in te_idx]
        tr0, te0 = T.to_t(E, tr_idx), T.to_t(E, te_idx)
        tes = set(te_idx)

        # 1. Base 모델 (C_pi0_adaln_drop30, 일반 학습)
        if "Base_C_pi0" not in res[sname]:
            print(f"  [1/2] Training Base model on {sname} (120 epochs)...")
            m_base = E1B.fit(tr0, [], 0.0, gain=1.0)
            r_base = GS.split_metrics(m_base, te0, live)
            r_base["S_img"] = E1B.s_img(m_base, [x for x in aug_rows if x["ei"] in tes])
            res[sname]["Base_C_pi0"] = r_base
            print_result("Base", r_base)
        else:
            print_result("Base (cached)", res[sname]["Base_C_pi0"])

        # 2. E1b_tuned 모델 (cx 증강 가중치 1.0, thresh=0.22, gain=2.6)
        if "E1b_tuned" not in res[sname]:
            print(f"  [2/2] Training E1b_tuned model on {sname} (120 epochs)...")
            aug = [r for r in aug_rows if r["ei"] not in tes and r["kind"] == "shift"]
            m_e1b = E1B.fit(tr0, aug, 1.0, gain=2.6)
            r_e1b = GS.split_metrics(m_e1b, te0, live)
            r_e1b["S_img"] = E1B.s_img(m_e1b, [x for x in aug_rows if x["ei"] in tes])
            res[sname]["E1b_tuned"] = r_e1b
            print_result("E1b_tuned", r_e1b)
        else:
            print_result("E1b_tuned (cached)", res[sname]["E1b_tuned"])

        OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n==================================================")
    print(">>> 5대 LOPO 경로 전수 평가 완료! 결과 요약 표:")
    print("==================================================")
    print(f"{'분할(Split)':<20} | {'Base S':<9} | {'E1b S':<9} | {'Base DirErr':<12} | {'E1b DirErr':<12} | {'Base SpinR':<10} | {'E1b SpinR':<10}")
    print("-" * 92)
    for sname in splits.keys():
        b = res[sname]["Base_C_pi0"]
        e = res[sname]["E1b_tuned"]
        b_s = fmt_val(b.get('S'))
        e_s = fmt_val(e.get('S'))
        b_sp = f"{b['spin_recall']*100:.1f}%" if b.get('spin_recall') is not None else "N/A"
        e_sp = f"{e['spin_recall']*100:.1f}%" if e.get('spin_recall') is not None else "N/A"
        b_de = fmt_val(b.get('dir_err'), fmt=".2f")
        e_de = fmt_val(e.get('dir_err'), fmt=".2f")
        print(f"{sname:<20} | {b_s:<9} | {e_s:<9} | {b_de:<12} | {e_de:<12} | {b_sp:<10} | {e_sp:<10}")

if __name__ == "__main__":
    main()
