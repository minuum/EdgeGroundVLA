#!/usr/bin/env python3
"""CH68 68-7 (L1 실측) — "같은 프레임 · 다른 지시문 → 다른 타겟" 을 측정한다.

배경 정정:
  CH68 68-3에서 L1(지시문→OWL 프롬프트)을 "코드 한 곳 수정으로 가능"이라 썼는데,
  코드를 확인하니 **이미 구현되어 있다**:
    stage2_v2_inference_server.py:1033
      phrase = "gray basket" if instruction == "basket" else instruction
    → :1128  self.grounder.run(image_rgb, phrase=phrase)
  즉 요청의 instruction 필드가 그대로 OWL-v2 텍스트 쿼리로 전달된다.
  따라서 L1은 "구현할 것"이 아니라 **"측정한 적이 없는 것"** 이다. 이 스크립트가 그 측정이다.

측정 대상 (사전 고정):
  V6 val 프레임 N장에 대해 phrase만 바꿔 OWL-v2를 돌린다. 이미지는 완전히 동일하다.
    ① 검출률      — phrase별 has_bbox 비율
    ② cx 분리도   — 두 phrase가 모두 검출된 프레임에서 |cx_A − cx_B|
                    이 값이 0에 가까우면 "지시문을 바꿨지만 같은 것을 보고 있다"는 뜻
    ③ 조향 방향 반전율 — sign(cx−0.5) 가 phrase에 따라 바뀌는 프레임 비율
                    **이것이 핵심 지표** — 방향이 실제로 갈리는지가 VLA 주장의 최소 조건
    ④ 부재 판정   — 장면에 없는 객체("microwave")를 요청했을 때 미검출로 응답하는가
                    (오검출하면 "지시문을 따른다"가 아니라 "아무거나 찾는다"는 뜻)

⚠️ 이 실험이 말하지 않는 것 (68-3 용어 구분 유지):
  · 이것은 **language-directed target selection** 이다. 액션 헤드는 여전히 cx·비전만
    보며 텍스트를 받지 않는다. **language-conditioned policy 가 아니다.**
  · 액션 클래스 변화는 측정하지 않는다 — 헤드 입력은 6프레임 윈도우라 단일 프레임
    교체로는 재구성되지 않는다(별도 실험 필요). 여기서는 **조향 부호까지만** 본다.
  · V6는 바스켓 주행용으로 수집됐다. 다른 객체가 프레임에 우연히 있을 뿐이므로
    "그 객체로 주행 가능"은 전혀 주장할 수 없다.

출력: docs/v5/detector/l1_target_selection.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ANN = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json"
OUT = ROOT / "docs/v5/detector/l1_target_selection.json"
SPLIT_SEED, VAL_RATIO = 42, 0.15
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESH = 0.25          # CH60 확정값. 서빙은 0.20이지만 부재판정 실험이므로 보수적으로 0.25

# 배포 phrase + 실내에 실제로 있을 법한 것 + 없는 것(부재 판정용)
PHRASES = ["gray basket", "chair", "person", "door", "microwave oven"]
ABSENT_CONTROL = "microwave oven"      # 연구실 주행 장면에 없다고 사전 판정


def val_frames(n, seed=0):
    """65-5/67-x와 동일한 seed42 val 에피소드에서 균등 표집."""
    ann = json.loads(ANN.read_text())
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(ann)))
    rng.shuffle(idx)
    val_eps = set(idx[:max(1, int(len(idx) * VAL_RATIO))])
    rows = [dict(ep=ep["episode"], fi=f["frame_idx"])
            for ei, ep in enumerate(ann) if ei in val_eps
            for f in ep["frames"] if not f["grounding_cached"] and f["detected"]]
    r2 = np.random.default_rng(seed)
    pick = r2.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in sorted(pick)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(DEV).eval()

    rows = val_frames(args.n)
    print(f"표본 {len(rows)} 프레임 (V6 val) · phrase {len(PHRASES)}종 · thresh {THRESH}",
          flush=True)

    # 서빙과 동일한 필터를 쓰면 phrase별 오버라이드가 섞이므로, 여기서는 raw 최고점 박스만
    # 사용한다(필터 없음). 목적이 "텍스트가 타겟을 바꾸는가"이지 서빙 재현이 아니다.
    det = defaultdict(dict)         # det[(ep,fi)][phrase] = dict|None
    cur, hf = None, None
    for n, r in enumerate(rows):
        if r["ep"] != cur:
            if hf is not None:
                hf.close()
            hf = h5py.File(r["ep"], "r"); cur = r["ep"]
        im = np.ascontiguousarray(np.array(hf["images"][r["fi"]])[:, :, ::-1])
        pil = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        W, H = pil.width, pil.height
        inp = proc(text=[PHRASES], images=pil, return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_object_detection(out, threshold=THRESH,
                                                target_sizes=[(H, W)])[0]
        key = (r["ep"], r["fi"])
        for pi, ph in enumerate(PHRASES):
            sel = (res["labels"] == pi).nonzero().flatten()
            if len(sel) == 0:
                det[key][ph] = None
                continue
            b = int(sel[res["scores"][sel].argmax()])
            x1, y1, x2, y2 = res["boxes"][b].cpu().tolist()
            det[key][ph] = dict(cx=(x1 + x2) / 2 / W, cy=(y1 + y2) / 2 / H,
                                area=(x2 - x1) * (y2 - y1) / (W * H),
                                score=float(res["scores"][b]))
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(rows)}", flush=True)
    if hf is not None:
        hf.close()

    keys = list(det)
    rep = {"n_frames": len(keys), "thresh": THRESH, "phrases": PHRASES,
           "absent_control": ABSENT_CONTROL, "detect_rate": {}, "pairs": {}}

    print("\n" + "=" * 84)
    print("68-7 L1 실측 — 같은 이미지, phrase만 교체 (OWL-v2)")
    print("=" * 84)
    print("\n  ① phrase별 검출률 / cx 분포")
    print(f"{'phrase':18s} {'검출률':>8s} {'cx 평균':>9s} {'cx std':>8s} {'score 중앙':>10s}")
    for ph in PHRASES:
        d = [det[k][ph] for k in keys if det[k][ph]]
        rate = len(d) / len(keys)
        cxs = np.array([x["cx"] for x in d]) if d else np.array([np.nan])
        sc = np.median([x["score"] for x in d]) if d else float("nan")
        rep["detect_rate"][ph] = dict(rate=rate, n=len(d),
                                      cx_mean=float(np.nanmean(cxs)),
                                      cx_std=float(np.nanstd(cxs)), score_med=float(sc))
        print(f"{ph:18s} {rate*100:7.1f}% {np.nanmean(cxs):9.3f} "
              f"{np.nanstd(cxs):8.3f} {sc:10.3f}")

    print(f"\n  ④ 부재 판정 — '{ABSENT_CONTROL}' 검출률 "
          f"{rep['detect_rate'][ABSENT_CONTROL]['rate']*100:.1f}% "
          f"(낮아야 정상. 높으면 '아무거나 찾는다'는 뜻)")

    print("\n  ②③ 'gray basket' 대비 — 둘 다 검출된 프레임만")
    print(f"{'phrase':18s} {'공통n':>6s} {'|Δcx| 평균':>10s} {'|Δcx| 중앙':>10s} "
          f"{'조향부호 반전율':>14s}")
    base = "gray basket"
    for ph in PHRASES:
        if ph == base:
            continue
        both = [k for k in keys if det[k][base] and det[k][ph]]
        if not both:
            print(f"{ph:18s} {0:6d}   공통 프레임 없음")
            rep["pairs"][ph] = dict(n=0)
            continue
        a = np.array([det[k][base]["cx"] for k in both])
        b = np.array([det[k][ph]["cx"] for k in both])
        d = np.abs(a - b)
        flip = float(np.mean(np.sign(a - 0.5) != np.sign(b - 0.5)))
        rep["pairs"][ph] = dict(n=len(both), dcx_mean=float(d.mean()),
                                dcx_med=float(np.median(d)), sign_flip=flip)
        print(f"{ph:18s} {len(both):6d} {d.mean():10.3f} {np.median(d):10.3f} "
              f"{flip*100:13.1f}%")

    print("\n  판정:")
    print("    · 조향부호 반전율이 유의하게 >0 인 phrase가 있으면 → 지시문이 방향을 바꾼다")
    print("      = 'language-directed target selection' 성립 (policy 아님, 68-3 용어 참조)")
    print("    · 전부 ~0% 이면 → 텍스트를 바꿔도 같은 쪽을 보므로 데모 근거가 안 된다")
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
