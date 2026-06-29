"""
CH54 Preview 전략 오프라인 어블레이션 (수정판)

Oracle 모델링:
  - 6/26 데이터: frame 0 has_bbox=False 는 PG2 콜드스타트 아티팩트
    → 워밍업 후 frame 0 실제 결과 ≈ frame 1 결과 (로봇 미이동이므로)
    → warmup_bbox[0] = bbox[1] 로 패치

전략:
  A. Baseline      현재 상태 (워밍업 없음, frame 0 항상 실패 → STOP)
  B. Warmup-only   Stage 0만 적용, 프리뷰 없음
  C. ROT_R         has_bbox=False → ROT_R 반복 (max_retry 변화)
  D. ROT_L         has_bbox=False → ROT_L 반복
  E. Alternate     R→L→R 교대
  F. Warmup+ROT_R  Stage 0 + 그래도 실패 시 ROT_R (복합)

측정:
  - frame 0 그라운딩 성공률
  - 첫 VLA 입력까지 낭비 스텝 수 (낮을수록 좋음)
  - 첫 VLA 입력 cx 분포 (중앙에 가까울수록 좋음)
  - cx deviation from 0.5 (절대값 평균, 낮을수록 정렬 잘 됨)
"""

import h5py, numpy as np, json
from pathlib import Path

ROOT     = Path("docs/inference_sessions")
SESSIONS = sorted(ROOT.glob("session_2026062[246]*.h5"))


def load_session(sp):
    with h5py.File(sp, "r") as f:
        if "grounding/bbox" not in f:
            return None
        bbox = f["grounding/bbox"][:]
        acts = f["actions"][:]
    return {"bbox": bbox.copy(), "acts": acts, "name": sp.stem}


def warmup_patch(bbox):
    """워밍업 후 frame 0 oracle: frame 0 결과를 frame 1로 대체."""
    w = bbox.copy()
    if len(w) > 1:
        w[0] = w[1]   # frame 0 ≈ frame 1 (no movement between)
    return w


def simulate(sessions, strategy: str, max_retry: int = 5, use_warmup: bool = False):
    results = []
    for d in sessions:
        bbox_raw = d["bbox"]
        N = len(bbox_raw)

        # oracle 선택
        bbox = warmup_patch(bbox_raw) if use_warmup else bbox_raw

        # --- 전략별 시뮬레이션 ---
        if strategy == "none":
            # 프리뷰 없음: frame 0 결과 그대로, has_bbox=False면 STOP → frame 1부터 VLA
            f0_ok = bool(bbox[0, 3] > 0.5)
            rot_steps = 0
            wasted = 0 if f0_ok else 1   # frame 0 STOP = 1 낭비 스텝
            first_f = 0 if f0_ok else 1
            if first_f < N and bbox[first_f, 3] > 0.5:
                first_cx = float(bbox[first_f, 0])
            else:
                first_cx = None

        else:
            # 프리뷰 루프
            f0_ok = bool(bbox[0, 3] > 0.5)
            if f0_ok:
                rot_steps = 0
                wasted = 0
                first_f = 0
                first_cx = float(bbox[0, 0])
            else:
                rot_steps = 0
                found = False
                # ROT 방향 결정 함수
                def rot_dir(attempt):
                    if strategy == "rot_r": return "R"
                    if strategy == "rot_l": return "L"
                    if strategy == "alternate": return "R" if attempt % 2 == 0 else "L"
                    return "R"

                for attempt in range(max_retry):
                    rot_steps += 1
                    # oracle: 실제 회전 후 상태
                    # 보수적 모델: ROT 후 basket이 보일 확률 = frame(attempt+1) oracle
                    t = min(attempt + 1, N - 1)
                    if bbox[t, 3] > 0.5:
                        found = True
                        first_f = t
                        first_cx = float(bbox[t, 0])
                        break

                if not found:
                    first_f = max_retry
                    first_cx = None
                    rot_steps = max_retry

                wasted = rot_steps   # 각 ROT = 1 낭비 스텝

        results.append({
            "name": d["name"],
            "f0_ok": f0_ok,
            "rot_steps": rot_steps,
            "wasted_steps": wasted,
            "first_vla_frame": first_f,
            "first_cx": first_cx,
        })
    return results


def report(name, results):
    f0_ok    = np.mean([r["f0_ok"] for r in results])
    wasted   = np.mean([r["wasted_steps"] for r in results])
    w_std    = np.std([r["wasted_steps"] for r in results])
    fvf      = np.mean([r["first_vla_frame"] for r in results])
    cxs      = [r["first_cx"] for r in results if r["first_cx"] is not None]
    cx_mu    = np.mean(cxs) if cxs else float("nan")
    cx_dev   = np.mean([abs(c - 0.5) for c in cxs]) if cxs else float("nan")
    cx_l     = np.mean([c < 0.40 for c in cxs]) if cxs else 0
    cx_c     = np.mean([(0.40 <= c <= 0.60) for c in cxs]) if cxs else 0
    cx_r     = np.mean([c > 0.60 for c in cxs]) if cxs else 0
    print(f"  {name:<26} | f0={f0_ok:.0%}  waste={wasted:.2f}±{w_std:.2f}  "
          f"fvf={fvf:.2f}  cx={cx_mu:.3f}(dev={cx_dev:.3f}) [L{cx_l:.0%}/C{cx_c:.0%}/R{cx_r:.0%}]")
    return {
        "strategy": name, "f0_ok": f0_ok,
        "waste_mean": float(wasted), "waste_std": float(w_std),
        "first_vla_frame": float(fvf),
        "cx_mean": float(cx_mu), "cx_dev": float(cx_dev),
    }


sessions = [d for sp in SESSIONS if (d := load_session(sp)) is not None]
print(f"세션 {len(sessions)}개  ({', '.join(sorted(set(s['name'].split('_')[1] for s in sessions)))})")
print(f"\n{'전략':<26} | f0_ok  waste(±std)  fvf   cx(dev) [L/C/R]")
print("─" * 90)

all_metrics = []
print("[오라클: 현재 h5 그대로]")
all_metrics.append(report("A.baseline(no warmup)",    simulate(sessions, "none")))
all_metrics.append(report("C.rot_r(no warmup,r=5)",   simulate(sessions, "rot_r",    max_retry=5)))

print("\n[오라클: warmup 적용 후 — frame0≈frame1]")
all_metrics.append(report("B.warmup_only",             simulate(sessions, "none",     use_warmup=True)))
all_metrics.append(report("F.warmup+rot_r(r=1)",       simulate(sessions, "rot_r",    max_retry=1,  use_warmup=True)))
all_metrics.append(report("F.warmup+rot_r(r=3)",       simulate(sessions, "rot_r",    max_retry=3,  use_warmup=True)))
all_metrics.append(report("F.warmup+rot_r(r=5)",       simulate(sessions, "rot_r",    max_retry=5,  use_warmup=True)))
all_metrics.append(report("F.warmup+rot_l(r=5)",       simulate(sessions, "rot_l",    max_retry=5,  use_warmup=True)))
all_metrics.append(report("F.warmup+alternate(r=5)",   simulate(sessions, "alternate", max_retry=5, use_warmup=True)))

print("\n── max_retry sweep (warmup + ROT_R) ───────────────────────────────────────")
for r in [1, 2, 3, 5, 7, 10]:
    all_metrics.append(report(f"  warmup+rot_r(r={r:>2})",
                               simulate(sessions, "rot_r", max_retry=r, use_warmup=True)))

out = Path("docs/v5/ablate_preview_strategy.json")
with open(out, "w") as fp:
    json.dump(all_metrics, fp, indent=2)
print(f"\n결과 저장: {out}")
