#!/usr/bin/env python3
"""OWL-v2 / Kosmos-2 그라운딩 피처의 좌우 비대칭성 검정 — horizontal flip 대조.

질문(교수님, 2026-08-04): 학습 데이터의 좌/우 에피소드 수는 균형인데
(V6: 좌 90ep / 우 90ep, 프레임 6910/6870, 라벨 cx 평균 0.4976),
실기 결과가 좌우로 갈리는 이유가 **모델 자체가 이미지 좌/우에 비대칭**이기 때문일 수 있는가?

증명 설계 — paired flip test:
  모델이 좌우 대칭이면, 이미지를 좌우 반전했을 때
    · 검출 여부가 같아야 하고
    · confidence가 같아야 하며
    · cx_flip = 1 - cx_orig 이어야 한다.
  같은 물체·같은 장면에서 **위치만 좌↔우로 바뀐** 쌍을 비교하므로
  내용(content) 교란이 제거된다. 편차가 남으면 그것이 모델의 비대칭이다.

출력: docs/v5/grounding_analysis/lr_symmetry.json
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json"
OUT = ROOT / "docs/v5/grounding_analysis"
OUT.mkdir(parents=True, exist_ok=True)
PHRASE = "gray basket"
THRESH = 0.20
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_frames(n, seed=0):
    """V6 LIVE&detected 프레임을 cx 좌/우 균등하게 표본."""
    ann = json.loads(ANN.read_text())
    rows = [dict(ep=e["episode"], fi=f["frame_idx"], cx=f["cx_det"], area=f["area_det"])
            for e in ann for f in e["frames"]
            if (not f["grounding_cached"]) and f["detected"]]
    rng = np.random.default_rng(seed)
    left = [r for r in rows if r["cx"] < 0.5]
    right = [r for r in rows if r["cx"] >= 0.5]
    k = n // 2
    pick = ([left[i] for i in rng.choice(len(left), min(k, len(left)), replace=False)]
            + [right[i] for i in rng.choice(len(right), min(k, len(right)), replace=False)])
    return pick


def owl_run(proc, model, pil):
    """OWL-v2 1회 추론 → (max score, cx). 서버와 동일 조건(fp32)."""
    W, H = pil.width, pil.height
    inp = proc(text=[[PHRASE]], images=pil, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model(**inp)
    res = proc.post_process_object_detection(out, threshold=0.0,
                                             target_sizes=[(H, W)])[0]
    s = res["scores"]
    if not len(s):
        return 0.0, None
    b = int(s.argmax())
    x1, _, x2, _ = [float(v) for v in res["boxes"][b]]
    return float(s[b]), (x1 + x2) / 2 / W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=300)
    a = ap.parse_args()
    rows = sample_frames(a.n)
    print(f"표본 {len(rows)}프레임 (좌 {sum(r['cx']<0.5 for r in rows)} / "
          f"우 {sum(r['cx']>=0.5 for r in rows)})")

    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    print("[OWL-v2] 로딩 (fp32, 서버 동일)")
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(DEV).eval()

    recs, cur, hf = [], None, None
    for i, r in enumerate(rows, 1):
        if r["ep"] != cur:
            if hf is not None:
                hf.close()
            hf = h5py.File(r["ep"], "r"); cur = r["ep"]
        im = np.array(hf["images"][r["fi"]])[:, :, ::-1]          # BGR→RGB (주석과 동일)
        pil = Image.fromarray(im.astype("uint8"))
        flip = pil.transpose(Image.FLIP_LEFT_RIGHT)
        s_o, cx_o = owl_run(proc, model, pil)
        s_f, cx_f = owl_run(proc, model, flip)
        recs.append(dict(label_cx=r["cx"], area=r["area"],
                         score_orig=s_o, score_flip=s_f,
                         cx_orig=cx_o, cx_flip=cx_f))
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    if hf is not None:
        hf.close()

    d = recs
    so = np.array([x["score_orig"] for x in d])
    sf = np.array([x["score_flip"] for x in d])
    lab = np.array([x["label_cx"] for x in d])
    isL = lab < 0.5

    print("\n" + "=" * 68)
    print("좌우 대칭성 검정 — paired horizontal flip")
    print("=" * 68)
    print(f"\n[1] 검출 여부 대칭성 (threshold {THRESH})")
    do, df = so >= THRESH, sf >= THRESH
    print(f"  원본 검출 {do.sum()}/{len(do)} ({100*do.mean():.1f}%)  "
          f"반전 검출 {df.sum()}/{len(df)} ({100*df.mean():.1f}%)")
    print(f"  판정 불일치: {(do != df).sum()}건 ({100*(do!=df).mean():.1f}%)")
    print(f"    원본만 검출 {(do & ~df).sum()}  /  반전만 검출 {(~do & df).sum()}")

    print(f"\n[2] confidence 대칭성 — 같은 물체를 좌↔우로 옮기면?")
    dl = sf[isL] - so[isL]      # 좌측 물체 → 반전하면 우측으로 이동
    dr = sf[~isL] - so[~isL]    # 우측 물체 → 반전하면 좌측으로 이동
    print(f"  좌측 물체(n={isL.sum()}) 를 우측으로 옮김: Δscore 평균 {dl.mean():+.4f} "
          f"(중앙 {np.median(dl):+.4f})")
    print(f"  우측 물체(n={(~isL).sum()}) 를 좌측으로 옮김: Δscore 평균 {dr.mean():+.4f} "
          f"(중앙 {np.median(dr):+.4f})")
    try:
        from scipy.stats import wilcoxon
        print(f"  좌→우 이동이 0과 다른가: Wilcoxon p={wilcoxon(dl).pvalue:.2e}")
        print(f"  우→좌 이동이 0과 다른가: Wilcoxon p={wilcoxon(dr).pvalue:.2e}")
    except Exception as e:
        print("  scipy:", e)
    # ⚠️ 해석 주의: '두 Δ의 합이 0'은 대칭의 증거가 아니다.
    # 대칭이면 각 Δ가 개별적으로 0이어야 하고,
    # 한쪽 선호가 있으면 두 Δ가 부호 반대로 나오므로 합은 오히려 0에 가까워진다.
    adv = np.concatenate([-dl, dr])   # 둘 다 "좌측이 우측보다 얼마나 높은가"로 부호 정렬
    se = adv.std(ddof=1) / np.sqrt(len(adv))
    print(f"  → 대칭 판정은 '각 Δ가 0인가'로 한다 (합이 0인 것은 편향의 신호)")
    print(f"  [좌측 이점 추정] {adv.mean():+.4f} ± {se:.4f}(SE), "
          f"95%CI {adv.mean()-1.96*se:+.4f}~{adv.mean()+1.96*se:+.4f}")
    try:
        from scipy.stats import ttest_1samp
        print(f"    t-test p={ttest_1samp(adv,0).pvalue:.2e} (n={len(adv)} paired)")
    except Exception:
        pass

    print(f"\n[3] 위치 미러링 정확도 — cx_flip == 1 - cx_orig 인가")
    both = np.array([x["cx_orig"] is not None and x["cx_flip"] is not None for x in d])
    co = np.array([x["cx_orig"] if x["cx_orig"] is not None else np.nan for x in d])
    cf = np.array([x["cx_flip"] if x["cx_flip"] is not None else np.nan for x in d])
    err = np.abs(co[both] - (1 - cf[both]))
    print(f"  미러링 오차 |cx_orig - (1-cx_flip)|: 평균 {err.mean():.4f} "
          f"중앙 {np.median(err):.4f} p90 {np.percentile(err,90):.4f}  (n={both.sum()})")
    eL = np.abs(co[both & isL] - (1 - cf[both & isL]))
    eR = np.abs(co[both & ~isL] - (1 - cf[both & ~isL]))
    print(f"    좌측 물체 {eL.mean():.4f}  /  우측 물체 {eR.mean():.4f}")

    print(f"\n[4] 절대 confidence의 좌우 차이 (내용 교란 있음 — 참고용)")
    print(f"  좌측 물체 원본 score 평균 {so[isL].mean():.4f}  /  우측 {so[~isL].mean():.4f}")

    res = dict(n=len(d), threshold=THRESH,
               det_orig=float(do.mean()), det_flip=float(df.mean()),
               det_mismatch=int((do != df).sum()),
               dscore_L2R_mean=float(dl.mean()), dscore_R2L_mean=float(dr.mean()),
               dscore_sum=float(dl.mean() + dr.mean()),
               mirror_err_mean=float(err.mean()),
               score_left=float(so[isL].mean()), score_right=float(so[~isL].mean()),
               frames=d)
    (OUT / "lr_symmetry.json").write_text(json.dumps(res, indent=2))
    print(f"\n저장: {OUT/'lr_symmetry.json'}")


if __name__ == "__main__":
    main()
