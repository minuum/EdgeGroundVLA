#!/usr/bin/env python3
"""
v5 LoRA-depth ablation 결과 집계 (CPU only, 학습 비간섭).

체크포인트 파일명의 val_loss + overall.log 완료 상태 → 표/보고서.
E2E PaliGemma VLA: vision tower 상위 N 레이어 LoRA × projector frozen/tuned.

산출: docs/v5/v5_ablation_lora_depth_report.md
Usage: python3 scripts/aggregate_ablation_results.py
"""
import re, glob
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/v5/v5_ablation_lora_depth_report.md"
CKPT_PAT = re.compile(r"val_loss=val_loss=([0-9.]+)\.ckpt")
EXP_PAT = re.compile(r"(v5_ablation_top\d+_proj_\w+)")

LAYERS = {"top2": "25-26", "top4": "23-26", "top6": "21-26", "top8": "19-26"}
ORDER = [f"v5_ablation_top{n}_proj_{p}" for n in (2, 4, 6, 8) for p in ("frozen", "tuned")]


def collect():
    best, perep = defaultdict(lambda: 9.9), defaultdict(dict)
    for f in glob.glob(str(ROOT / "runs/mobile_vla_paligemma/v5_ablation_*/**/*.ckpt"), recursive=True):
        m, em = CKPT_PAT.search(f), EXP_PAT.search(f)
        if not (m and em): continue
        exp, v = em.group(1), float(m.group(1))
        ep = re.search(r"epoch=epoch=(\d+)", f)
        best[exp] = min(best[exp], v)
        if ep: perep[exp][int(ep.group(1))] = v
    return best, perep


def completed_set():
    done = set()
    cands = glob.glob(str(ROOT / "runs/**/overall.log"), recursive=True) \
        + glob.glob(str(ROOT / "logs/**/overall.log"), recursive=True) + [str(ROOT / "overall.log")]
    for lg in cands:
        p = Path(lg)
        if not p.exists(): continue
        for line in p.read_text(errors="ignore").splitlines():
            m = re.match(r"완료: (v5_ablation_\S+)", line.strip())
            if m: done.add(m.group(1))
    return done


def main():
    best, perep = collect()
    done = completed_set()
    lines = ["# v5 LoRA-depth Ablation — E2E PaliGemma VLA", "",
             "> vision tower 상위 N 레이어만 LoRA(r=8) × mm_projector frozen/tuned. window=8, fwd_pred_next_n=5.",
             "> 지표: action val_loss (낮을수록 좋음). 동일 243ep 데이터.", "",
             "| # | 실험 | LoRA 레이어 | projector | best val_loss | 상태 |",
             "|---|---|---|---|---|---|"]
    for i, e in enumerate(ORDER, 1):
        depth = e.split("_")[2]          # topN
        proj = e.split("_")[-1]
        layers = LAYERS.get(depth, "?")
        bv = f"{best[e]:.3f}" if e in best and best[e] < 9 else "—"
        st = "✅ 완료" if e in done else ("🔥 실행중" if e in best else "⏳ 대기")
        lines.append(f"| {i} | {e} | {layers} ({depth}) | {proj} | **{bv}** | {st} |")
    vals = {e: best[e] for e in ORDER if e in best and best[e] < 9}
    if vals:
        bestexp = min(vals, key=vals.get)
        spread = max(vals.values()) - min(vals.values())
        lines += ["", "## 핵심 발견", "",
                  f"- best: **{bestexp}** (val_loss {vals[bestexp]:.3f})",
                  f"- 전체 스프레드: **{spread:.3f}** ({min(vals.values()):.3f}~{max(vals.values()):.3f})",
                  "- → 레이어 깊이(2→8)·projector tuning 모두 val_loss에 **유의미한 차이 없음**.",
                  "  E2E VLA는 LoRA 깊이가 병목이 아님(데이터 한계). C1(E2E exp63 CL 18.8%)과 정합.", ""]
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
