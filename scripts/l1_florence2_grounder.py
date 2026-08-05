#!/usr/bin/env python3
"""CH67 67-6 — Florence-2 **언어부**로 지시문 타겟 선택이 되는가 (교수님 제안의 미검정 절반).

배경:
  67-2/67-5에서 검정한 것은 Florence-2의 **비전 타워(90.4M)** 였다. 좌표 회귀에서 동등했다.
  검정하지 않은 것은 **언어부(140.2M)** 다 — CH68 68-5 미해결 질문 ①로 남겨둔 항목이고,
  교수님 제안("OWL-v2 + Florence-2 조합")의 나머지 절반이다.

  Florence-2는 텍스트 프롬프트로 태스크를 지정한다. 여기서 쓰는 태스크는
    <CAPTION_TO_PHRASE_GROUNDING> + 텍스트("chair")  → 해당 구절의 박스
  즉 **언어부가 텍스트를 해석해 위치를 내놓는 경로**다. 68-7에서 OWL-v2로 같은 것을
  측정했으므로 **동일 표본·동일 지표로 직접 비교**할 수 있다.

  ⚠️ 이 경로는 우리 실측에서 한 번 탈락한 이력이 있다 — CH59에서 Florence-2 OVD의
  예측 cx가 0.559에서 막혀 화면 우측 절반을 못 짚었다(L/R 정확도 51.8%).
  67-5에서 **비전 피처로 쓰면 그 결함이 사라짐**(pred cx max 0.8613)을 확인했으므로,
  남은 질문은 **"결함이 언어부/디코딩 경로에 있었는가"** 다. 이 스크립트가 그 판정이다.

검정 설계 (68-7과 완전히 동일한 표본·지표 — 그라운더만 교체):
  V6 val 프레임 200장(seed42, 같은 seed로 같은 200장) · phrase 5종 · 필터 미적용
    ① phrase별 검출률 / cx 분포
    ② 'gray basket' 대비 |Δcx|
    ③ 조향 부호 반전율  ← 주 지표
    ④ 부재 대조군("microwave oven") 검출률
  추가로 CH59 재현 여부:
    ⑤ **예측 cx 최대값** — 0.559 근처에서 막히면 CH59 결함이 언어부/디코딩 기인

출력: docs/v5/detector/l1_florence2_grounder.json
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
OUT = ROOT / "docs/v5/detector/l1_florence2_grounder.json"
OWL_JSON = ROOT / "docs/v5/detector/l1_target_selection.json"
SPLIT_SEED, VAL_RATIO = 42, 0.15
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "microsoft/Florence-2-base"
TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
PHRASES = ["gray basket", "chair", "person", "door", "microwave oven"]
ABSENT_CONTROL = "microwave oven"


def val_frames(n, seed=0):
    """68-7과 **같은 함수·같은 seed** — 동일한 200프레임을 보장한다."""
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

    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    rows = val_frames(args.n)
    print(f"표본 {len(rows)} 프레임 (68-7과 동일) · phrase {len(PHRASES)}종 · {TASK}", flush=True)

    det = defaultdict(dict)
    cur, hf = None, None
    for n, r in enumerate(rows):
        if r["ep"] != cur:
            if hf is not None:
                hf.close()
            hf = h5py.File(r["ep"], "r"); cur = r["ep"]
        im = np.ascontiguousarray(np.array(hf["images"][r["fi"]])[:, :, ::-1])
        pil = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        W, H = pil.width, pil.height
        key = (r["ep"], r["fi"])
        for ph in PHRASES:
            inp = proc(text=TASK + ph, images=pil, return_tensors="pt")
            with torch.no_grad():
                ids = model.generate(
                    input_ids=inp["input_ids"].to(DEV),
                    pixel_values=inp["pixel_values"].to(DEV, torch.float16),
                    max_new_tokens=128, num_beams=3, do_sample=False)
            txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = proc.post_process_generation(txt, task=TASK,
                                                  image_size=(W, H))[TASK]
            boxes = parsed.get("bboxes", [])
            if not boxes:
                det[key][ph] = None
                continue
            # 최대 면적 박스를 채택 — Florence-2는 score를 주지 않는다(OWL과 다른 점, 명시)
            best = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            x1, y1, x2, y2 = best
            det[key][ph] = dict(cx=(x1 + x2) / 2 / W, cy=(y1 + y2) / 2 / H,
                                area=(x2 - x1) * (y2 - y1) / (W * H),
                                n_boxes=len(boxes))
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(rows)}", flush=True)
    if hf is not None:
        hf.close()

    keys = list(det)
    owl = json.loads(OWL_JSON.read_text()) if OWL_JSON.exists() else None
    rep = {"n_frames": len(keys), "task": TASK, "model": MODEL,
           "phrases": PHRASES, "detect_rate": {}, "pairs": {}}

    print("\n" + "=" * 92)
    print("67-6 Florence-2 언어부 — 같은 이미지, phrase만 교체 (68-7 OWL-v2와 동일 표본)")
    print("=" * 92)
    print("\n  ① phrase별 검출률 / cx 분포   [괄호 = 68-7 OWL-v2 값]")
    print(f"{'phrase':18s} {'검출률':>8s} {'(OWL)':>8s} {'cx 평균':>9s} {'(OWL)':>8s} "
          f"{'cx std':>8s} {'cx max':>8s}")
    for ph in PHRASES:
        d = [det[k][ph] for k in keys if det[k][ph]]
        cxs = np.array([x["cx"] for x in d]) if d else np.array([np.nan])
        o = owl["detect_rate"].get(ph, {}) if owl else {}
        rep["detect_rate"][ph] = dict(rate=len(d) / len(keys), n=len(d),
                                      cx_mean=float(np.nanmean(cxs)),
                                      cx_std=float(np.nanstd(cxs)),
                                      cx_max=float(np.nanmax(cxs)))
        print(f"{ph:18s} {len(d)/len(keys)*100:7.1f}% "
              f"{o.get('rate', float('nan'))*100:7.1f}% {np.nanmean(cxs):9.3f} "
              f"{o.get('cx_mean', float('nan')):8.3f} {np.nanstd(cxs):8.3f} "
              f"{np.nanmax(cxs):8.3f}")

    allcx = np.array([det[k][ph]["cx"] for k in keys for ph in PHRASES if det[k][ph]])
    rep["pred_cx_max_all"] = float(allcx.max()) if len(allcx) else None
    print(f"\n  ⑤ 전체 예측 cx 최대값 = {allcx.max():.4f}   "
          f"(CH59 Florence-2 OVD는 0.559에서 막혔음)")
    print(f"     → 0.559 근처면 CH59 결함이 언어부/디코딩 기인, "
          f"0.85 이상이면 그 결함은 재현되지 않음")

    print(f"\n  ④ 부재 판정 — '{ABSENT_CONTROL}' 검출률 "
          f"{rep['detect_rate'][ABSENT_CONTROL]['rate']*100:.1f}% "
          f"(OWL-v2는 5.0%)")

    print("\n  ②③ 'gray basket' 대비 — 둘 다 검출된 프레임만   [괄호 = OWL-v2]")
    print(f"{'phrase':18s} {'공통n':>6s} {'|Δcx| 평균':>10s} {'조향부호 반전율':>14s} {'(OWL)':>8s}")
    base = "gray basket"
    for ph in PHRASES:
        if ph == base:
            continue
        both = [k for k in keys if det[k][base] and det[k][ph]]
        op = owl["pairs"].get(ph, {}) if owl else {}
        if not both:
            print(f"{ph:18s} {0:6d}   공통 프레임 없음")
            rep["pairs"][ph] = dict(n=0)
            continue
        a = np.array([det[k][base]["cx"] for k in both])
        b = np.array([det[k][ph]["cx"] for k in both])
        flip = float(np.mean(np.sign(a - 0.5) != np.sign(b - 0.5)))
        rep["pairs"][ph] = dict(n=len(both), dcx_mean=float(np.abs(a - b).mean()),
                                sign_flip=flip)
        of = op.get("sign_flip")
        print(f"{ph:18s} {len(both):6d} {np.abs(a-b).mean():10.3f} {flip*100:13.1f}% "
              f"{(of*100 if of is not None else float('nan')):7.1f}%")

    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
