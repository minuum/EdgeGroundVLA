#!/usr/bin/env python3
"""Jetson-vs-local OWL-v2 confidence gap 정량화 (64-17 요청 1번 우회 해법).

원리: 젯슨이 threshold T에서 has_bbox=False로 판정한 프레임은 젯슨 score < T 가
확정이다. 같은 이미지를 로컬에서 재실행해 score를 뽑으면, 그 프레임에서
로컬 score >= T 인 경우가 곧 "확정 gap 사례"이고, 초과분이 gap 크기다.
대조군으로 젯슨이 검출한 프레임(젯슨 score >= T)도 함께 재실행해
confidence 분포 shift 자체를 본다.

서버(soda) 재현 조건 — stage2_v2_inference_server.py OwlV2Grounder.run() 기준:
  - google/owlv2-base-patch16-ensemble, dtype 명시 없음(fp32)
  - phrase="gray basket" (H5 attrs instruction과 동일)
  - 저장 원본 720x1280 PIL을 그대로 processor에 전달 (별도 pre-resize 없음)
  - post_process_object_detection(threshold=owl_thresh), target_sizes=(H,W)

출력: docs/v5/grounding_analysis/jetson_local_gap.json
"""
import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RECV = Path("/home/minum/MoNaVLA/inference_sessions_recv")
OUT = ROOT / "docs/v5/grounding_analysis"
OUT.mkdir(parents=True, exist_ok=True)
POS = {"trackA_strong_left": "강좌", "trackA_weak_left": "약좌",
       "trackF_center": "중앙", "trackA_weak_right": "약우",
       "trackA_strong_right": "강우"}
THRESH = 0.20          # 100개 세트에 적용된 젯슨 threshold
PHRASE = "gray basket"
CKPT = "exp73_owl_trackF_v6_mlp_holdaware_seed0.pt"


def load_set100() -> pd.DataFrame:
    """episode_log.csv에서 확정 100세트 복원 (#230=이상경로 메모, #200~209=설정리셋 제외)."""
    df = pd.read_csv(RECV / "20260731" / "episode_log.csv")
    df.columns = [c.strip() for c in df.columns]
    d = df[df["체크포인트"] == CKPT].copy()
    d["dt"] = pd.to_datetime(d["날짜"])
    w = d[(d["dt"] >= pd.Timestamp("2026-07-30 19:25"))
          & ~d["#"].between(200, 209)
          & (d["#"] != 230)
          & d["경로"].isin(POS)].copy()
    w["pos"] = w["경로"].map(POS)
    w["ok"] = w["결과"] == "성공"
    assert len(w) == 100, f"기대 100건, 실제 {len(w)}건"
    return w


def collect_frames(w: pd.DataFrame, hit_sample: int, seed: int = 0):
    """젯슨 미검출 프레임 전량 + 검출 프레임 무작위 표본을 모은다."""
    idx = {Path(f).stem.replace("session_", ""): f
           for f in glob.glob(str(RECV / "2026073*" / "h5" / "*.h5"))}
    miss, hit = [], []
    for _, r in w.iterrows():
        f = idx[str(r["session_id"])]
        with h5py.File(f, "r") as h:
            hb = h["grounding/bbox"][:, 3]
            bb = h["grounding/bbox"][:]
        for i, v in enumerate(hb):
            rec = dict(sid=str(r["session_id"]), file=f, fi=i, pos=r["pos"],
                       ok=bool(r["ok"]), jet_cx=float(bb[i, 0]),
                       jet_area=float(bb[i, 2]))
            (miss if v == 0 else hit).append(rec)
    rng = np.random.default_rng(seed)
    if hit_sample and len(hit) > hit_sample:
        hit = [hit[i] for i in rng.choice(len(hit), hit_sample, replace=False)]
    return miss, hit


def run_local(frames, proc, model, device):
    """로컬에서 OWL-v2 재실행 → max score 및 최고박스 cx 반환."""
    out = []
    cache_file, cache_imgs = None, None
    for n, r in enumerate(frames, 1):
        if r["file"] != cache_file:
            if cache_imgs is not None:
                cache_imgs.close()
            cache_imgs = h5py.File(r["file"], "r")
            cache_file = r["file"]
        img = Image.fromarray(
            np.array(cache_imgs["observations/images"][r["fi"]]).astype(np.uint8)
        ).convert("RGB")
        W, H = img.width, img.height
        inp = proc(text=[[PHRASE]], images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(**inp)
        res = proc.post_process_object_detection(
            o, threshold=0.0, target_sizes=[(H, W)])[0]
        s = res["scores"]
        if len(s):
            b = int(s.argmax())
            x1, y1, x2, y2 = [float(v) for v in res["boxes"][b]]
            loc_score, loc_cx = float(s[b]), (x1 + x2) / 2 / W
        else:
            loc_score, loc_cx = 0.0, None
        out.append({**{k: v for k, v in r.items() if k != "file"},
                    "local_score": loc_score, "local_cx": loc_cx})
        if n % 50 == 0:
            print(f"    {n}/{len(frames)}", flush=True)
    if cache_imgs is not None:
        cache_imgs.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hit-sample", type=int, default=200)
    args = ap.parse_args()

    w = load_set100()
    miss, hit = collect_frames(w, args.hit_sample)
    print(f"확정 100세트 로드 완료 | 젯슨 미검출 {len(miss)}프레임 전량, "
          f"검출 표본 {len(hit)}프레임")

    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OWL-v2] 로딩 (fp32, {device}) — 서버와 동일 조건")
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(device).eval()

    print("  [1/2] 젯슨 미검출 프레임 재실행")
    miss_r = run_local(miss, proc, model, device)
    print("  [2/2] 젯슨 검출 프레임 재실행(대조군)")
    hit_r = run_local(hit, proc, model, device)

    ms = np.array([r["local_score"] for r in miss_r])
    hs = np.array([r["local_score"] for r in hit_r])
    gap = ms >= THRESH
    res = {
        "threshold": THRESH, "phrase": PHRASE,
        "n_miss": len(ms), "n_hit": len(hs),
        "miss": {
            "local_ge_thresh": int(gap.sum()),
            "local_ge_thresh_pct": float(100 * gap.mean()),
            "score_mean": float(ms.mean()), "score_median": float(np.median(ms)),
            "score_p90": float(np.percentile(ms, 90)),
            "score_max": float(ms.max()),
            "excess_mean_when_gap": float((ms[gap] - THRESH).mean()) if gap.any() else 0.0,
        },
        "hit": {
            "score_mean": float(hs.mean()), "score_median": float(np.median(hs)),
            "below_thresh": int((hs < THRESH).sum()),
            "below_thresh_pct": float(100 * (hs < THRESH).mean()),
        },
        "by_pos": {}, "frames": {"miss": miss_r, "hit": hit_r},
    }
    for p in POS.values():
        sel = [r for r in miss_r if r["pos"] == p]
        if not sel:
            continue
        a = np.array([r["local_score"] for r in sel])
        res["by_pos"][p] = {
            "n_miss": len(a),
            "local_ge_thresh": int((a >= THRESH).sum()),
            "local_ge_thresh_pct": float(100 * (a >= THRESH).mean()),
            "score_mean": float(a.mean()),
        }

    (OUT / "jetson_local_gap.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("\n" + "=" * 66)
    print(f"젯슨 미검출({len(ms)}프레임)을 로컬 재실행한 결과")
    print("=" * 66)
    print(f"  로컬 score >= {THRESH} (확정 gap): "
          f"{gap.sum()}/{len(ms)} ({100 * gap.mean():.1f}%)")
    print(f"  로컬 score 평균 {ms.mean():.4f} / 중앙 {np.median(ms):.4f} / "
          f"p90 {np.percentile(ms, 90):.4f} / 최대 {ms.max():.4f}")
    if gap.any():
        print(f"  gap 사례의 threshold 초과분 평균 +{(ms[gap] - THRESH).mean():.4f}")
    print(f"\n  [대조군] 젯슨 검출 {len(hs)}프레임의 로컬 score 평균 {hs.mean():.4f}, "
          f"threshold 미달 {(hs < THRESH).sum()}건({100 * (hs < THRESH).mean():.1f}%)")
    print("\n  위치별 미검출→로컬재검출률:")
    for p, v in res["by_pos"].items():
        print(f"    {p:4s} {v['local_ge_thresh']:3d}/{v['n_miss']:3d} "
              f"({v['local_ge_thresh_pct']:5.1f}%)  로컬 score 평균 {v['score_mean']:.4f}")
    print(f"\n저장: {OUT / 'jetson_local_gap.json'}")


if __name__ == "__main__":
    main()
