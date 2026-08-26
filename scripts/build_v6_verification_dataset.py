#!/usr/bin/env python3
"""V6 검증셋 전수 자동화 — 합의추정 vs 사람검증 분리 (2026-08-24).

288개 표본 전체에 OWL(gt_cx)·Florence-2 phrase(pred_cx)를 계산해두고(아직 안
계산된 건 이 스크립트가 채움, GPU 필요), 다음 두 계층으로 나눈다:

  1. 합의추정(auto_consensus) — OWL과 Florence가 서로 근접(|Δ|<=0.05)하면
     두 독립 아키텍처가 우연히 같은 틀린 답에 동시에 도달했을 가능성은 낮다고
     보고, 중간값을 잠정 정답으로 자동 채움. **사람이 확인한 진짜 정답이 아니라
     추정치**임을 항상 명시해야 한다.
  2. 사람검증 필요(needs_human) — 둘이 다르면(|Δ|>0.05) 어느 쪽이 맞는지 자동으로
     알 수 없다 — 이 프레임만 `/verify` 도구에서 사람이 클릭하면 된다. 288개 전부가
     아니라 이 서브셋만 클릭하면 되므로 수작업이 크게 줄어든다.

주의: 파이프라인에 끼워넣지 않는다. 검증 인프라 구축 전용. 합의추정은 어디까지나
근사치이며, 최종 신뢰 기준은 여전히 사람 클릭(true_cx)이다.

출력: docs/v5/detector/v6_verification_dataset.json
      (기존 human_labels.json은 건드리지 않음 — 별도 파일로 분리)
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "label"))

HIT_TOL = 0.05
OUT = ROOT / "docs/v5/detector/v6_verification_dataset.json"


def main():
    import serve_v6_phrase_grounding_verify as srv

    print(f"표본 {len(srv.SAMPLE)}개 대상으로 gt_cx/pred_cx 전수 계산...", flush=True)
    rows = []
    for i in range(len(srv.SAMPLE)):
        data = srv.render_card(i)
        fidx, bin_, owl_ok = srv.SAMPLE[i]
        fr = srv.FRAMES[fidx]
        rows.append(dict(
            key=f"{fr['stem']}|{fr['frame_idx']}", stem=fr["stem"], frame_idx=fr["frame_idx"],
            bin=bin_, owl_ok=owl_ok, direction=fr["direction"], approach=fr["approach"],
            gt_cx=data["gt_cx"], pred_cx=data["pred_cx"],
        ))
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(srv.SAMPLE)}", flush=True)

    n_agree = n_disagree = n_flo_missing = 0
    for r in rows:
        if r["pred_cx"] is None:
            n_flo_missing += 1
            r["status"] = "florence_no_detection"
            r["auto_consensus_cx"] = None
        else:
            diff = abs(r["gt_cx"] - r["pred_cx"])
            if diff <= HIT_TOL:
                n_agree += 1
                r["status"] = "auto_consensus"
                r["auto_consensus_cx"] = (r["gt_cx"] + r["pred_cx"]) / 2
            else:
                n_disagree += 1
                r["status"] = "needs_human"
                r["auto_consensus_cx"] = None

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    print(f"\n총 {len(rows)}개")
    print(f"  합의추정(자동, |Δ|<=0.05): {n_agree}개 ({n_agree/len(rows)*100:.1f}%)")
    print(f"  사람검증 필요(불일치): {n_disagree}개 ({n_disagree/len(rows)*100:.1f}%)")
    print(f"  Florence 미검출: {n_flo_missing}개")
    print(f"\n저장 → {OUT}")
    print(f"\n→ /verify 도구에서 사람이 클릭해야 하는 건 이 {n_disagree}개뿐입니다"
          f"(전체 {len(rows)}개가 아니라).")


if __name__ == "__main__":
    main()
