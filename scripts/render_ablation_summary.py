# -*- coding: utf-8 -*-
"""Table1/Table2 JSON → docs/v5/grounding_ablation/SUMMARY.md 렌더."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GDIR = ROOT / "docs/v5/grounding_ablation"


def f(x, p=3):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"


def main():
    lines = ["# Grounding Ablation 결과", ""]
    t1p = GDIR / "table1.json"
    if t1p.exists():
        t1 = json.loads(t1p.read_text())
        lines += ["## Table 1 — Grounding 품질 (동일 val 셋)", "",
                  "| model | hit율 | cx MAE(vs HSV)↓ | cx_std(ep내)↓ | 캔박스율↓ | full-frame율↓ | 선택성 gap↑ |",
                  "|---|---|---|---|---|---|---|"]
        for tag, r in t1.items():
            lines.append(f"| {tag} | {r['hit_rate']*100:.0f}% | {f(r['cx_mae_vs_hsv'])} | "
                         f"{f(r['cx_std_in_ep'])} | {f(r['canned_rate'],2)} | {f(r['fullframe_rate'],3)} | "
                         f"{r['selectivity_gap']:.2f} |")
        lines.append("")
    else:
        lines += ["## Table 1 — (아직 생성 안 됨)", ""]

    t2p = GDIR / "table2.json"
    if t2p.exists():
        d = json.loads(t2p.read_text()); t2 = d["table"]
        controls = sorted({c for g in t2.values() for c in g})
        lines += [f"## Table 2 — 전체 파이프라인 CL (success-fpe={d.get('success_fpe')})", "",
                  "| grounding ↓ \\ control → | " + " | ".join(controls) + " |",
                  "|" + "---|" * (len(controls) + 1)]
        for g, row in t2.items():
            cells = [f"{row[c]['success_rate']*100:.1f}% (FPE {row[c]['mean_fpe']:.2f})" if c in row else "—" for c in controls]
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
        lines.append("")
    else:
        lines += ["## Table 2 — (아직 생성 안 됨)", ""]

    (GDIR / "SUMMARY.md").write_text("\n".join(lines))
    print(f"[SAVE] {GDIR}/SUMMARY.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
