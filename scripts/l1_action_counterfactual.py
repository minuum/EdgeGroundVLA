#!/usr/bin/env python3
"""CH68 68-8 — 지시문을 바꾸면 **액션까지** 바뀌는가 (68-7 ②의 빈칸 채우기).

68-7에서 확인한 것: 지시문만 바꾸면 OWL-v2의 cx가 바뀌고 조향 부호가 55.1%에서 뒤집힌다.
확인하지 못한 것: **그 cx 변화가 액션 헤드의 출력까지 바꾸는가.**
  당시 이유는 "헤드 입력이 6프레임 윈도우라 단일 프레임 교체로 재구성되지 않는다"였다.
  그런데 exp73 학습 캐시(exp73_v6_vis_cache.pt)가 에피소드 단위로 vis를 들고 있고,
  `swap_bboxes()`가 **vis는 그대로 두고 bbox만 교체**하는 용도로 이미 존재한다.
  → 윈도우를 정상적으로 재구성할 수 있다. 이 스크립트가 그것이다.

설계 (68-4 프로토콜 정신 유지 — 지표를 먼저 고정한다):
  같은 val 에피소드 · 같은 vis(장면 외형) · 같은 학습된 헤드.
  bbox만 두 가지로 바꿔 넣는다.
    baseline : OWL-v2 "gray basket"  (배포 구성, bbox_dataset_v6_owl.json)
    반사실   : OWL-v2 "chair"        (이 스크립트가 새로 추출)
  chair 미검출 프레임은 서빙과 동일한 fallback(cx 0.5, cy 0.6, area 0.06, has_bbox 0)을 쓴다.

사전 고정 지표:
  ① 예측 변화율            — argmax가 바뀐 결정 시점 비율
  ② 좌질량 변화 Δ(좌−우)   — softmax에서 좌계열[2,4,6] − 우계열[3,5,7]
  ③ **방향 일치율** ← 주 지표
     sign(Δ좌질량) 이 sign(cx_basket − cx_chair) 와 같은 비율.
     chair가 더 왼쪽이면 좌질량이 늘어야 한다. 우연이면 50%.
  ④ 대조군: bbox를 **셔플**(같은 에피소드 내 프레임 순서 무작위)한 경우의 ①②③
     — 변화율만 보면 "bbox를 흔들면 뭐든 변한다"와 구분이 안 되므로 필요하다.

⚠️ 주장 한계 (68-7과 동일하게 유지):
  · 이것도 **language-directed target selection**이다. 헤드는 텍스트를 받지 않는다.
  · vis(장면 외형)는 고정했으므로 "chair를 향해 주행한다"가 아니라
    "chair의 위치가 액션을 움직인다"까지만 말할 수 있다.
  · 68-7 ③의 한계(chair가 좌측에 거의 고정)가 그대로 상속된다.

출력: docs/v5/detector/l1_action_counterfactual.json
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_exp73_trackA_heads import (          # noqa: E402
    MLPActionHead, CACHE_V6, SPLIT_SEED, VAL_RATIO, NUM_CLASSES, DEVICE, BBOX_SCALE, WINDOW,
)
from scripts.exp73_held_aware_train import build_windows_hold_aware, majority  # noqa: E402
from scripts.exp73_window_cadence import swap_bboxes, OWL_ANN                  # noqa: E402

ANN = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json"
CHAIR_CACHE = ROOT / "docs/v5/detector/l1_chair_bboxes_val.json"
OUT = ROOT / "docs/v5/detector/l1_action_counterfactual.json"
HEADS = sorted((ROOT / "runs/v5_nav/mlp/exp73").glob(
    "exp73_owl_trackF_v6_mlp_holdaware_seed*.pt"))
PHRASE = "chair"
THRESH = 0.25
FALLBACK = (0.5, 0.6, 0.06, 0.0)
LEFT_CLS, RIGHT_CLS = [2, 4, 6], [3, 5, 7]


def val_episodes(base):
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(base)))
    rng.shuffle(idx)
    nv = max(1, int(len(idx) * VAL_RATIO))
    vs = set(idx[:nv])
    return [e for i, e in enumerate(base) if i in vs]


def extract_chair(eps):
    """val 에피소드의 gt_class 보유 프레임에 대해 OWL-v2 'chair' bbox를 뽑는다."""
    if CHAIR_CACHE.exists():
        print(f"chair 캐시 재사용: {CHAIR_CACHE}", flush=True)
        return json.loads(CHAIR_CACHE.read_text())

    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(DEVICE).eval()
    ann = {Path(e["episode"]).stem: e for e in json.loads(ANN.read_text())}

    out = {}
    for ei, ep in enumerate(eps, 1):
        a = ann[ep["stem"]]
        frames = [f for f in a["frames"] if f.get("gt_class") is not None]
        with h5py.File(a["episode"], "r") as h:
            imgs = np.array(h["images"])
        rec = []
        for f in frames:
            im = np.ascontiguousarray(imgs[f["frame_idx"]][:, :, ::-1])
            pil = Image.fromarray(im.astype("uint8")).convert("RGB")
            W, H = pil.width, pil.height
            inp = proc(text=[[PHRASE]], images=pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                o = model(**inp)
            r = proc.post_process_object_detection(o, threshold=THRESH,
                                                  target_sizes=[(H, W)])[0]
            if len(r["boxes"]) == 0:
                rec.append(dict(has=False))
            else:
                b = int(r["scores"].argmax())
                x1, y1, x2, y2 = r["boxes"][b].cpu().tolist()
                rec.append(dict(has=True, cx=(x1 + x2) / 2 / W, cy=(y1 + y2) / 2 / H,
                                area=(x2 - x1) * (y2 - y1) / (W * H),
                                score=float(r["scores"][b])))
        out[ep["stem"]] = rec
        if ei % 5 == 0:
            print(f"  chair 추출 {ei}/{len(eps)}", flush=True)
    CHAIR_CACHE.write_text(json.dumps(out))
    print(f"저장: {CHAIR_CACHE}", flush=True)
    return out


def apply_bboxes(eps, mode, chair):
    """mode: 'chair' | 'shuffle'. vis·gts는 건드리지 않는다."""
    rng = np.random.default_rng(0)
    out = []
    for ep in eps:
        e = dict(ep)
        if mode == "chair":
            rec = chair[ep["stem"]]
            assert len(rec) == len(ep["bboxes"]), f"{ep['stem']} 길이 불일치"
            e["bboxes"] = [(r["cx"], r["cy"], r["area"], 1.0) if r["has"] else FALLBACK
                           for r in rec]
        else:
            perm = rng.permutation(len(ep["bboxes"]))
            e["bboxes"] = [tuple(ep["bboxes"][i]) for i in perm]
        out.append(e)
    return out


@torch.no_grad()
def run_head(head, X):
    p = F.softmax(head(torch.from_numpy(X).float().to(DEVICE)), dim=1).cpu().numpy()
    return p


def cx_at_decisions(eps, stride=5):
    """build_windows_hold_aware와 동일한 결정 시점의 현재 프레임 cx / has_bbox."""
    cx, has = [], []
    for ep in eps:
        n = len(ep["gts"])
        for t in range(0, n, stride):
            cx.append(ep["bboxes"][t][0])
            has.append(ep["bboxes"][t][3])
    return np.asarray(cx), np.asarray(has)


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    base = torch.load(str(CACHE_V6), weights_only=False)
    va = val_episodes(base)
    va_owl = swap_bboxes(va, OWL_ANN)          # baseline = 배포 구성 "gray basket"
    print(f"val {len(va)}ep → OWL 주석 정렬 {len(va_owl)}ep", flush=True)
    chair = extract_chair(va_owl)

    va_chair = apply_bboxes(va_owl, "chair", chair)
    va_shuf = apply_bboxes(va_owl, "shuffle", chair)

    Xb, yb = build_windows_hold_aware(va_owl)
    Xc, _ = build_windows_hold_aware(va_chair)
    Xs, _ = build_windows_hold_aware(va_shuf)
    cx_b, _ = cx_at_decisions(va_owl)
    cx_c, has_c = cx_at_decisions(va_chair)
    assert len(Xb) == len(Xc) == len(Xs) == len(cx_b)
    print(f"결정 시점 {len(Xb)} · chair 검출된 시점 {int(has_c.sum())} "
          f"({has_c.mean()*100:.1f}%)", flush=True)

    if not HEADS:
        raise FileNotFoundError("학습된 OWL arm 헤드 체크포인트를 찾지 못했다")
    print(f"헤드 {len(HEADS)}개: {[h.name for h in HEADS]}", flush=True)

    rep = {"n_decisions": int(len(Xb)), "chair_detect_rate": float(has_c.mean()),
           "heads": [h.name for h in HEADS], "arms": {}}
    agg = {"chair": [], "shuffle": []}

    for hp in HEADS:
        head = MLPActionHead().to(DEVICE)
        sd = torch.load(str(hp), map_location=DEVICE, weights_only=False)
        # 체크포인트는 {"model": state_dict, "head": "mlp", "window":…, "bbox_scale":…} 형태.
        # 윈도우 구성이 다르면 입력 차원이 어긋나므로 조용히 넘어가지 않고 검증한다.
        assert sd["head"] == "mlp", f"{hp.name}: head={sd['head']}"
        assert sd["window"] == WINDOW and sd["bbox_scale"] == BBOX_SCALE, \
            f"{hp.name}: window/bbox_scale 불일치 {sd['window']}/{sd['bbox_scale']}"
        head.load_state_dict(sd["model"])
        head.eval()
        pb = run_head(head, Xb)
        for arm, X, cxa in [("chair", Xc, cx_c), ("shuffle", Xs, None)]:
            pa = run_head(head, X)
            # chair arm은 chair가 실제로 검출된 시점만 본다(미검출은 fallback이라 정보 없음)
            m = has_c.astype(bool) if arm == "chair" else np.ones(len(X), dtype=bool)
            chg = float((pa[m].argmax(1) != pb[m].argmax(1)).mean())
            lb = pb[m][:, LEFT_CLS].sum(1) - pb[m][:, RIGHT_CLS].sum(1)
            la = pa[m][:, LEFT_CLS].sum(1) - pa[m][:, RIGHT_CLS].sum(1)
            d_left = la - lb
            if arm == "chair":
                expect = cx_b[m] - cxa[m]        # chair가 더 왼쪽이면 >0 → 좌질량 증가 기대
                valid = np.abs(expect) > 0.02    # cx가 사실상 같은 시점은 방향 정의 불가
                agree = float((np.sign(d_left[valid]) == np.sign(expect[valid])).mean())
                nvalid = int(valid.sum())
            else:
                agree, nvalid = float("nan"), 0
            agg[arm].append(dict(change=chg, d_left_mean=float(d_left.mean()),
                                 d_left_abs=float(np.abs(d_left).mean()),
                                 agree=agree, n_valid=nvalid, n=int(m.sum())))

    print("\n" + "=" * 86)
    print("68-8 — bbox 반사실: 지시문이 액션 헤드 출력을 움직이는가")
    print("=" * 86)
    print(f"\n  결정 시점 {len(Xb)} · chair 검출 {has_c.mean()*100:.1f}% · 헤드 {len(HEADS)} seed")
    print(f"\n{'arm':10s} {'대상n':>7s} {'예측 변화율':>11s} {'Δ좌질량 평균':>13s} "
          f"{'|Δ좌질량|':>10s} {'방향 일치율':>11s}")
    for arm in ("chair", "shuffle"):
        r = agg[arm]
        f = lambda k: (np.mean([x[k] for x in r]), np.std([x[k] for x in r]))
        c, dm, da = f("change"), f("d_left_mean"), f("d_left_abs")
        ag = f("agree")
        agtxt = "—" if np.isnan(ag[0]) else f"{ag[0]*100:.1f}%±{ag[1]*100:.1f}"
        print(f"{arm:10s} {r[0]['n']:7d} {c[0]*100:9.1f}%  {dm[0]:+13.4f} "
              f"{da[0]:10.4f} {agtxt:>11s}")
        rep["arms"][arm] = dict(n=r[0]["n"], n_valid=r[0]["n_valid"],
                                change=list(c), d_left_mean=list(dm),
                                d_left_abs=list(da),
                                agree=[None if np.isnan(ag[0]) else ag[0], ag[1]])

    ca = agg["chair"][0]
    print(f"\n  방향 판정에 쓴 시점 {ca['n_valid']} (|Δcx| > 0.02 인 경우만)")
    print("\n  판정 (사전 고정):")
    print("    · 방향 일치율이 50%를 유의하게 넘으면 → 지시문 변화가 액션을 '올바른 방향으로' 움직인다")
    print("    · chair arm의 |Δ좌질량|이 shuffle arm과 비슷하면 → 방향성 없이 흔들린 것일 수 있으므로")
    print("      변화율만으로 판단하지 말고 방향 일치율을 봐야 한다")
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
