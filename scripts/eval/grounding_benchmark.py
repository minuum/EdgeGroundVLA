#!/usr/bin/env python3
"""고정 그라운딩 벤치마크 — 사람 라벨 기반 3지표 채점.

데이터:
  1) docs/v5/hsv_owlv2_preview_20260704/{meta,human_labels,owlv2_scores}.json
     - 296프레임: 5모델(H1/H2/Ow/PG/Kr) 예측 + 사람 O/X + 객체없음 79
  2) docs/v5/bbox_truth_mini.json — 72프레임 사람 정답 bbox (IoU는 eval_iou_truth_mini.py가 병합)

지표:
  - acc_present : 객체있음 217프레임 정확도. 기존 5모델은 사람 O/X 그대로.
                  신규 모델은 합의 cx(사람 O 받은 모델들 cx 중앙값) 대비 |Δcx|<0.15 자동 채점
  - fp_absent   : 객체없음 79프레임에서 검출(has_bbox=True) 비율. Ow는 score th 0.25 적용 버전 별도
  - iou_*       : truth_mini 72프레임 (C 단계에서 병합)

Usage: .venv/bin/python3 scripts/eval/grounding_benchmark.py
출력: docs/v5/grounding_benchmark/results.json + 콘솔 테이블
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GAL = ROOT / "docs" / "v5" / "hsv_owlv2_preview_20260704"
OUT_DIR = ROOT / "docs" / "v5" / "grounding_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = OUT_DIR / "results.json"

MODELS = ["h1", "h2", "ow", "pg", "kr"]
OWL_THRESH = 0.25  # owlv2_threshold_roc.py 확정값
CONSENSUS_TOL = 0.15


def load():
    meta = json.loads((GAL / "meta.json").read_text())
    labels = json.loads((GAL / "human_labels.json").read_text())
    scores = json.loads((GAL / "owlv2_scores.json").read_text())
    return meta, labels, scores


def consensus_cx(m: dict, lab: dict) -> float | None:
    """사람 O 받은 모델들의 cx 중앙값 — 신규 모델 자동 채점 기준."""
    cxs = [m[mid]["cx"] for mid in MODELS if lab.get(mid) == "ok" and m.get(mid)]
    return statistics.median(cxs) if cxs else None


def benchmark_stored_models():
    """저장된 5모델 예측을 사람 라벨로 채점."""
    meta, labels, scores = load()
    results = {}
    for mid in MODELS:
        ok = ng = 0
        fp = n_absent = 0
        for m in meta:
            lab = labels.get(m["key"], {})
            if lab.get("no_target") == "yes":
                n_absent += 1
                if m.get(mid) is not None:
                    fp += 1
                continue
            v = lab.get(mid)
            if v == "ok":
                ok += 1
            elif v == "ng":
                ng += 1
        results[mid] = {
            "source": "human_ox",
            "acc_present": ok / (ok + ng) if (ok + ng) else None,
            "n_present_labeled": ok + ng,
            "fp_absent": fp / n_absent if n_absent else None,
            "n_absent": n_absent,
        }
    # OWL threshold 0.25 적용 버전 (score 기반 부재판정)
    ok = ng = fp = n_absent = 0
    for m in meta:
        lab = labels.get(m["key"], {})
        s = scores.get(m["key"], 0.0)
        detected = m.get("ow") is not None and s >= OWL_THRESH
        if lab.get("no_target") == "yes":
            n_absent += 1
            if detected:
                fp += 1
            continue
        v = lab.get("ow")
        if v is None:
            continue
        # threshold로 거부되면: 사람 O였어도 놓침(오답), X였으면 "잘 거부"로 정답 처리
        if not detected:
            ok += 1 if v == "ng" else 0
            ng += 1 if v == "ok" else 0
        else:
            ok += 1 if v == "ok" else 0
            ng += 1 if v == "ng" else 0
    results["ow_th025"] = {
        "source": "human_ox+score_threshold",
        "acc_present": ok / (ok + ng) if (ok + ng) else None,
        "n_present_labeled": ok + ng,
        "fp_absent": fp / n_absent if n_absent else None,
        "n_absent": n_absent,
    }
    return results


def benchmark_new_model(predictions: dict[str, dict | None], name: str) -> dict:
    """신규 그라운더 채점 — predictions: {key: {cx,...}|None}.
    합의 cx 기준 자동 채점 (근사 지표, 사람 O/X와 완전히 같지 않음)."""
    meta, labels, _ = load()
    ok = ng = fp = n_absent = skipped = 0
    for m in meta:
        lab = labels.get(m["key"], {})
        pred = predictions.get(m["key"])
        if lab.get("no_target") == "yes":
            n_absent += 1
            if pred is not None:
                fp += 1
            continue
        ref = consensus_cx(m, lab)
        if ref is None:
            skipped += 1
            continue
        if pred is not None and abs(pred["cx"] - ref) < CONSENSUS_TOL:
            ok += 1
        else:
            ng += 1
    return {
        "source": "consensus_cx_auto",
        "acc_present": ok / (ok + ng) if (ok + ng) else None,
        "n_present_labeled": ok + ng, "skipped": skipped,
        "fp_absent": fp / n_absent if n_absent else None, "n_absent": n_absent,
    }


def main():
    existing = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    existing.update(benchmark_stored_models())
    RESULTS.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    print(f"{'model':<10}{'acc_present':>13}{'fp_absent':>12}{'n_lab':>7}")
    for name, r in existing.items():
        if "acc_present" not in r:
            continue
        acc = f"{100*r['acc_present']:.1f}%" if r["acc_present"] is not None else "—"
        fp = f"{100*r['fp_absent']:.1f}%" if r["fp_absent"] is not None else "—"
        print(f"{name:<10}{acc:>13}{fp:>12}{r['n_present_labeled']:>7}")
    print(f"\n저장: {RESULTS}")


if __name__ == "__main__":
    main()
