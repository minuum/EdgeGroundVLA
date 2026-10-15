#!/usr/bin/env python3
"""EGVLA2 수집 배치 검증 + 기존 데이터(V6 이산 / v1 / 파일럿 v2)와의 차이 분석.

사용: python3 scripts/validate_egvla2_batch.py [--since 20261014] [--out docs/v5/egvla2_fm/validation_batch.json]
검사: (파일) 읽기·필수 데이터셋/속성·shape 정합·NaN·범위·타임스탬프·이미지 이상(검정/중복)·exec 단계 값·raw→단계 매핑 일치·방향 일치·정지 프레임·az 스파이크·드리프트 스냅
      (집합) 경로/방향 분포, 속력 단계 분포(가이드 목표 15~25/35~45/35~45), 크기 히스토그램, 진행도별 프로파일, 8-class 분포 vs V6, 속력-거리 상관(캐시 있을 때)
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "ROS_action/mobile_vla_dataset_egvla2"
META = ROOT / "docs/v5/egvla2_fm/meta/manual_dataset_sets.json"
CACHE = ROOT / "docs/v5/egvla2_fm/v3_florence2_cache.npz"
V6_JSON = ROOT / "docs/v5/bbox_nav_owl/bbox_dataset_v6_owl.json"
args = sys.argv[1:]
since = args[args.index("--since") + 1] if "--since" in args else "20261014"
out_path = Path(args[args.index("--out") + 1]) if "--out" in args else ROOT / "docs/v5/egvla2_fm/validation_batch.json"
TH_LO, TH_HI = 0.35, 0.70
LEVELS = (0.4, 0.7, 1.0)
CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
REQ = ["actions_raw", "actions_exec", "actions_scaled", "action_classes", "action_event_types", "frame_timestamps", "images"]


def level_of(mag):
    return np.where(mag < 1e-3, 0.0, np.where(mag < TH_LO, 0.4, np.where(mag < TH_HI, 0.7, 1.0)))


def check_file(path):
    r = dict(file=path.name, problems=[], warnings=[])
    try:
        with h5py.File(path, "r") as h:
            at = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h.attrs.items()}
            r["attrs"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in at.items() if k in (
                "action_space", "derive_version", "drive_mode", "cx_path", "cx_position", "scenario", "joystick_speed_setting", "flagged_model_issue", "stop_inject_n", "num_frames")}
            miss = [k for k in REQ if k not in h]
            if miss: r["problems"].append(f"필수 데이터셋 누락 {miss}"); return r
            raw, ex, sc = (np.array(h[k], dtype=float) for k in ("actions_raw", "actions_exec", "actions_scaled"))
            t = np.array(h["frame_timestamps"], dtype=float); cls = np.array(h["action_classes"]); ev = np.array(h["action_event_types"])
            im = h["images"]; n = len(raw)
            shapes = {k: h[k].shape[0] for k in REQ}
            if len(set(shapes.values())) != 1: r["problems"].append(f"길이 불일치 {shapes}")
            if at.get("action_space") != "egvla2_continuous_v3": r["problems"].append(f"action_space={at.get('action_space')}")
            if "measured_speed_cmps" not in h.attrs: r["warnings"].append("measured_speed_cmps 속성 없음")
            for nm, a in (("raw", raw), ("exec", ex), ("scaled", sc), ("t", t)):
                if not np.isfinite(a).all(): r["problems"].append(f"{nm}에 NaN/inf")
            if im.shape[1:] != (720, 1280, 3) or im.dtype != np.uint8: r["problems"].append(f"이미지 shape/dtype {im.shape} {im.dtype}")
            # 이미지: 일부 프레임 샘플 검사(밝기, 연속 중복)
            idx = np.linspace(0, n - 1, min(n, 24)).astype(int)
            fr = np.stack([im[i][::8, ::8].astype(np.float32) for i in idx]); r["img_mean_brightness"] = float(fr.mean())
            dark = int(sum(f.mean() < 12 for f in fr)); r["dark_sampled"] = dark
            if dark: r["warnings"].append(f"어두운 프레임 {dark}/{len(idx)} (샘플)")
            dup = 0; pairs = 0
            for i in range(1, n, 3):
                pairs += 1
                if np.abs(im[i][::16, ::16].astype(np.int16) - im[i - 1][::16, ::16].astype(np.int16)).mean() < 0.4: dup += 1
            r["dup_consecutive_frac"] = dup / max(1, pairs)
            if r["dup_consecutive_frac"] > 0.15: r["warnings"].append(f"연속 동일 프레임 비율 {r['dup_consecutive_frac']:.0%}")
    except Exception as e:  # 읽기 실패 = 전송 중 파일일 가능성
        r["problems"].append(f"읽기 실패: {e}"); return r
    dt = np.diff(t)
    r.update(n=n, duration=float(t[-1] - t[0]), dt_median=float(np.median(dt)), dt_p95=float(np.percentile(dt, 95)), dt_max=float(dt.max()))
    if (dt <= 0).any(): r["problems"].append("타임스탬프 비단조")
    em = np.hypot(ex[:, 0], ex[:, 1])
    idle = np.array([em[k] < 1e-3 and abs(ex[k, 2]) < 1e-3 for k in range(len(dt))])
    r["idle_gaps_gt05"] = int(np.sum((dt > 0.5) & idle)); r["active_gaps_gt05"] = int(np.sum((dt > 0.5) & ~idle))
    r["active_dt_max"] = float(dt[~idle].max()) if (~idle).any() else 0.0
    if r["active_gaps_gt05"]: r["warnings"].append(f"주행 중 프레임 간격>0.5s {r['active_gaps_gt05']}회(최대 {r['active_dt_max']:.2f}s)")
    if n < 20: r["warnings"].append(f"프레임 {n}개로 짧음")
    mag = np.hypot(raw[:, 0], raw[:, 1]); em = np.hypot(ex[:, 0], ex[:, 1]); sm = np.hypot(sc[:, 0], sc[:, 1])
    r["raw_mag_max"] = float(mag.max()); r["scaled_mag_max"] = float(sm.max())
    if sm.max() > 1.0001: r["problems"].append(f"scaled 크기 {sm.max():.3f} > 1")
    mv = em > 1e-3
    bad_level = int(np.sum(mv & ~np.isclose(em, LEVELS[0], atol=0.02) & ~np.isclose(em, LEVELS[1], atol=0.02) & ~np.isclose(em, LEVELS[2], atol=0.02)))
    if bad_level: r["problems"].append(f"exec 크기가 0.4/0.7/1.0이 아닌 프레임 {bad_level}")
    # 구동 규칙: |az|>0.1(raw)이면 회전만(이동 0, exec az=±단계), 아니면 이동 (exec az=0)
    az_raw = np.abs(raw[:, 2]); spin = az_raw > 0.1
    exp = np.where(spin, 0.0, level_of(mag)); agree = float(np.mean(np.isclose(em, exp, atol=0.02)))
    r["level_mapping_agree"] = agree
    if agree < 0.99: r["warnings"].append(f"raw 크기→단계 매핑(0.35/0.70, 회전 우선) 일치 {agree:.1%}")
    r["spin_frames"] = int(spin.sum()); r["spin_with_translation_raw"] = int(np.sum(spin & (mag > 1e-3)))
    sp_exec = np.abs(ex[:, 2])
    bad_spin = int(np.sum(spin & ~(np.isclose(sp_exec, level_of(az_raw), atol=0.02))))
    if bad_spin: r["warnings"].append(f"회전 프레임의 exec az 단계 불일치 {bad_spin}")
    if np.any((~spin) & (sp_exec > 1e-3)): r["warnings"].append("회전 입력 없는데 exec az≠0")
    both = (mag > 1e-3) & mv
    if both.any():
        a_raw = np.degrees(np.arctan2(raw[both, 1], raw[both, 0])); a_ex = np.degrees(np.arctan2(ex[both, 1], ex[both, 0]))
        d = np.abs((a_raw - a_ex + 180) % 360 - 180); r["angle_diff_p95"] = float(np.percentile(d, 95))
        if d.max() > 5: r["warnings"].append(f"raw/exec 각도 차 최대 {d.max():.1f}°")
    small_raw = int(np.sum(((np.abs(raw[:, :2]) > 0) & (np.abs(raw[:, :2]) < 0.03)).any(1)))
    small = int(np.sum(((np.abs(ex) > 0) & (np.abs(ex) < 0.03)).any(1))) + int(np.sum(((np.abs(sc[:, :2]) > 0) & (np.abs(sc[:, :2]) < 0.03)).any(1)))
    r["small_residual_raw"] = small_raw; r["small_residual_exec_scaled"] = small   # raw에는 센터링 잔류가 남는 것이 정상(스냅은 exec/scaled에 적용)
    if small: r["warnings"].append(f"exec/scaled에 0.03 미만 잔류 성분 {small} (스냅 누락 의심)")
    z = raw[:, 2]; nz = np.where(np.abs(z) > 1e-3)[0]
    iso = sum(1 for k in nz if (k == 0 or abs(z[k - 1]) < 1e-3) and (k == n - 1 or abs(z[k + 1]) < 1e-3)); r["az_nonzero"] = int(len(nz)); r["az_isolated"] = int(iso)
    tail_stop = 0
    for k in range(n - 1, -1, -1):
        if em[k] < 1e-3 and np.abs(ex[k, 2]) < 1e-3: tail_stop += 1
        else: break
    r["tail_stop"] = tail_stop
    if tail_stop == 0: r["warnings"].append("마지막 프레임이 정지가 아님")
    if (ev != b"joystick").mean() > 0.4: r["warnings"].append(f"조이스틱 외 이벤트 비율 {(ev != b'joystick').mean():.0%}")
    r["_raw"], r["_exec"], r["_cls"], r["_mag"] = raw, ex, cls, mag
    return r


def load_baselines():
    base = {}
    S = json.load(open(META))["sets"]
    names = {x if x.endswith(".h5") else x + ".h5" for x in S["pilot_v2_gauge_20261013"]}
    v1, p2 = [], []
    for f in sorted(glob.glob(str(D / "*.h5"))):
        try:
            with h5py.File(f) as h:
                sp = h.attrs.get("action_space")
                if sp == "egvla2_continuous_v1": v1.append(np.array(h["actions"], dtype=float))
                elif Path(f).name in names: p2.append((np.array(h["actions_raw"], dtype=float), np.array(h["actions_exec"], dtype=float), np.array(h["action_classes"])))
        except Exception:
            pass
    base["v1"], base["pilot_v2"] = v1, p2
    v6 = json.loads(V6_JSON.read_text()); c = Counter(fr["gt_class"] for e in v6 for fr in e["frames"])
    tot = sum(c.values()); base["v6_class"] = {CLASS_NAMES[k]: c.get(k, 0) / tot for k in range(8)} if all(isinstance(k, int) for k in c) else {str(k): v / tot for k, v in c.items()}
    # V6 원시 액션 값 종류
    vals = set()
    for e in v6[:60]:
        with h5py.File(e["episode"], "r") as h: vals.update(map(tuple, np.round(np.array(h["actions"]), 3).tolist()))
    base["v6_unique_actions"] = len(vals)
    return base


def main():
    files = sorted(p for p in D.glob("*.h5") if p.name.startswith(f"episode_{since}") or p.name >= f"episode_{since}")
    rows = [check_file(p) for p in files]
    ok = [r for r in rows if "_raw" in r]
    problems = [r for r in rows if r["problems"]]
    print(f"[배치] 파일 {len(rows)}개 중 읽힘 {len(ok)}, 문제 {len(problems)}, 경고 {sum(1 for r in rows if r['warnings'])}")
    for r in rows:
        if r["problems"] or r["warnings"]:
            print("  -", r["file"], "문제:", r["problems"], "경고:", r["warnings"])
    # ---- 집합 통계 ----
    raw = np.concatenate([r["_raw"] for r in ok]); ex = np.concatenate([r["_exec"] for r in ok]); mag = np.hypot(raw[:, 0], raw[:, 1]); em = np.hypot(ex[:, 0], ex[:, 1])
    mv = em > 1e-3
    lvl = {str(v): float(np.mean(np.isclose(em[mv], v, atol=.02))) for v in LEVELS}
    paths = Counter((r["attrs"].get("cx_position"), r["attrs"].get("cx_path")) for r in ok)
    by_dir = Counter(str(r["attrs"].get("cx_position")) for r in ok)
    mid = float(np.mean((mag[mag > 1e-3] > .15) & (mag[mag > 1e-3] < .9)))
    ax, ay = np.abs(raw[:, 0]), np.abs(raw[:, 1]); b = (ax > 1e-3) & (ay > 1e-3); corner = float(np.mean((ax[b] >= .99) & (ay[b] >= .99))) if b.any() else 0.0
    iso = sum(r["az_isolated"] for r in ok); nzaz = sum(r["az_nonzero"] for r in ok)
    spin_tot = sum(r["spin_frames"] for r in ok); spin_tr = sum(r["spin_with_translation_raw"] for r in ok)
    prof = []
    rel = np.concatenate([np.arange(len(r["_raw"])) / max(1, len(r["_raw"]) - 1) for r in ok])
    for lo, hi in [(0, .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1.01)]:
        mk = (rel >= lo) & (rel < hi) & mv; prof.append(dict(range=[lo, hi], mean_mag=float(mag[mk].mean()), mean_level=float(em[mk].mean()), n=int(mk.sum())))
    agg = dict(files=len(ok), frames=int(len(raw)), moving_frames=int(mv.sum()), paths={str(k): v for k, v in paths.items()}, level_dist=lvl, mid_ratio=mid, corner=corner,
               az_spike=[iso, nzaz], spin=dict(frames=spin_tot, with_translation=spin_tr), dt_median=float(np.median([r["dt_median"] for r in ok])), ep_len_mean=float(np.mean([r["n"] for r in ok])), profile=prof,
               mag_hist=np.histogram(mag[mag > 1e-3], bins=np.arange(0, 1.6, .1))[0].tolist(),
               gate=dict(mid_ge_50=mid >= .5, corner_le_20=corner <= .2, az_spike_le_25=(iso / nzaz if nzaz else 0) <= .25,
                         level_targets=dict(slow_15_25=.15 <= lvl["0.4"] <= .25, mid_35_45=.35 <= lvl["0.7"] <= .45, fast_35_45=.35 <= lvl["1.0"] <= .45)))
    print(f"\n[집합] 에피소드 {len(ok)} / 프레임 {len(raw)} (이동 {int(mv.sum())}) / 경로 {dict(paths)}")
    print(f"  속력 단계 분포 {lvl}  (목표 느림 15~25 / 보통 35~45 / 빠름 35~45)")
    print(f"  회전(|az|>0.1) 프레임 {spin_tot} 중 이동 입력 동시 {spin_tr} (exec는 회전만)")
    print(f"  중간값 {mid:.0%}  코너 {corner:.0%}  az 스파이크 {iso}/{nzaz}  프레임 간격 중앙값 {agg['dt_median']:.3f}s  평균 길이 {agg['ep_len_mean']:.0f}프레임")
    print("  진행도별 프로파일:", [(p['range'], round(p['mean_mag'], 2), round(p['mean_level'], 2)) for p in prof])
    # ---- 기존 데이터와 비교 ----
    base = load_baselines()
    cmp = {}
    def stat(name, r_, e_):
        m_ = np.hypot(r_[:, 0], r_[:, 1]); nz = m_ > 1e-3; e_m = np.hypot(e_[:, 0], e_[:, 1]); mvv = e_m > 1e-3
        return dict(frames=int(len(r_)), unique_raw=int(len(np.unique(np.round(r_, 3), axis=0))), mid=float(np.mean((m_[nz] > .15) & (m_[nz] < .9))) if nz.any() else None,
                    levels={str(v): float(np.mean(np.isclose(e_m[mvv], v, atol=.02))) for v in LEVELS} if mvv.any() else None)
    cmp["신규 배치"] = stat("new", raw, ex)
    if base["pilot_v2"]:
        pr = np.concatenate([x[0] for x in base["pilot_v2"]]); pe = np.concatenate([x[1] for x in base["pilot_v2"]]); cmp["파일럿 v2(15ep)"] = stat("p2", pr, pe)
    if base["v1"]:
        v1 = np.concatenate(base["v1"]); m1 = np.hypot(v1[:, 0], v1[:, 1]); nz = m1 > 1e-3
        cmp["v1 아카이브(40ep)"] = dict(frames=int(len(v1)), unique_raw=int(len(np.unique(np.round(v1, 3), axis=0))), mid=float(np.mean((m1[nz] > .15) & (m1[nz] < .9))), levels=None)
    cmp["V6(이산, 키보드 LUT)"] = dict(unique_actions_in_60ep=base["v6_unique_actions"], note="±1.15/0 값만")
    print("\n[기존 데이터와 비교]")
    for k, v in cmp.items(): print(" ", k, v)
    # 8-class 분포 비교
    cls_new = np.concatenate([r["_cls"] for r in ok]); cnt = Counter(cls_new.tolist()); tot = len(cls_new)
    cdist = {CLASS_NAMES[k]: cnt.get(k, 0) / tot for k in range(8)}
    print("\n[8-class 분포] 신규 vs V6")
    for k in CLASS_NAMES: print(f"  {k:8s} 신규 {cdist[k]:6.1%}   V6 {base['v6_class'].get(k, 0):6.1%}")
    # 속력-거리 상관 (F2 bbox 캐시가 있는 파일만)
    corr = None
    try:
        from scipy.stats import spearmanr
        C = np.load(CACHE); A, L = [], []
        for r in ok:
            if f"{r['file']}|bbox" in C:
                bb = C[f"{r['file']}|bbox"]; e = np.hypot(r["_exec"][:, 0], r["_exec"][:, 1]); m_ = e > 1e-3
                if len(bb) == len(e): A += bb[m_, 2].tolist(); L += e[m_].tolist()
        if len(A) > 100:
            rho = float(spearmanr(L, A)[0]); corr = dict(n=len(A), rho_speed_area=rho, files_with_cache=sum(1 for r in ok if f"{r['file']}|bbox" in C))
            print(f"\n[속력-거리] ρ(속력, 바구니 면적)={rho:+.2f} (n={len(A)}, 캐시 {corr['files_with_cache']}개) — 판정 기준 |ρ|≥0.3")
    except Exception as e:
        print("속력-거리 계산 생략:", e)
    out = dict(since=since, aggregate=agg, compare=cmp, class_dist_new=cdist, class_dist_v6=base["v6_class"], speed_distance=corr,
               files=[{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print("\n저장:", out_path)


if __name__ == "__main__":
    main()
