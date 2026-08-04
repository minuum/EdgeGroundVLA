#!/usr/bin/env python3
"""액션 헤드의 좌우 대칭성 검정 — 완전 미러 입력 대조 (65-8).

65-7에서 그라운더(OWL-v2)는 좌측을 +0.0118 선호하는데 실기는 좌측이 12.5%p 불리 —
부호가 반대라 그라운더로는 설명 불가. 다음 후보는 **액션 헤드 자체**다.

증명 설계 — full mirror test:
  좌우 대칭 세계라면, 입력을 완전히 미러링했을 때
  헤드의 출력도 L/R을 맞바꾼 형태로 정확히 미러링돼야 한다.
    · 이미지 좌우 반전 → 비전 피처 재추출 (256-dim)
    · bbox cx → 1 - cx
    · 예측 분포에서 LEFT↔RIGHT, FWD+L↔FWD+R, ROT_L↔ROT_R 스왑
  스왑 후 두 분포가 다르면 그 차이가 **헤드의 좌우 비대칭**이다.

부분 테스트도 함께 수행:
  bbox만 미러(비전 고정) → bbox 경로 단독 민감도
  비전만 미러(bbox 고정) → 비전 경로 단독 민감도

출력: docs/v5/grounding_analysis/head_lr_symmetry.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json"
VLM = ROOT / ".vlms/kosmos-2-patch14-224"
STAGE1 = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
HEAD = ROOT / "runs/v5_nav/mlp/exp73/exp73_owl_trackF_v6_mlp_holdaware_seed0.pt"
OUT = ROOT / "docs/v5/grounding_analysis"
OUT.mkdir(parents=True, exist_ok=True)
WINDOW, BBOX_SCALE, NUM_CLASSES = 6, 3.0, 8
SPLIT_SEED, VAL_RATIO = 42, 0.15
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 8클래스: 0 STOP · 1 FWD · 2 LEFT · 3 RIGHT · 4 FWD+L · 5 FWD+R · 6 ROT_L · 7 ROT_R
MIRROR = [0, 1, 3, 2, 5, 4, 7, 6]        # L↔R 스왑 순열
LR_PAIRS = [(2, 3), (4, 5), (6, 7)]


def val_episodes():
    ann = json.loads(ANN.read_text())
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(ann)))
    rng.shuffle(idx)
    nv = max(1, int(len(idx) * VAL_RATIO))
    return [ann[i] for i in idx[:nv]]


def build(eps, enc, limit_ep=None):
    """val 에피소드에서 (원본, 미러) 윈도 쌍 생성. 미러는 이미지 반전 후 피처 재추출."""
    if limit_ep:
        eps = eps[:limit_ep]
    out = []
    for ei, ep in enumerate(eps, 1):
        with h5py.File(ep["episode"], "r") as h:
            imgs = np.array(h["images"])
        fr = [f for f in ep["frames"] if f.get("gt_class") is not None]
        vo, vm = [], []
        for f in fr:
            im = imgs[f["frame_idx"]][:, :, ::-1]                  # BGR→RGB
            pil = Image.fromarray(im.astype("uint8"))
            vo.append(enc(pil))
            vm.append(enc(pil.transpose(Image.FLIP_LEFT_RIGHT)))
        vo = torch.stack(vo); vm = torch.stack(vm)
        bo = torch.tensor([[f["cx_det"], f["cy_det"], f["area_det"] * BBOX_SCALE,
                            1.0 if f["has_bbox"] else 0.0] for f in fr], dtype=torch.float32)
        bm = bo.clone(); bm[:, 0] = 1.0 - bm[:, 0]                 # cx 미러
        lab = np.array([f["gt_class"] for f in fr])
        for t in range(WINDOW - 1, len(fr)):
            sl = slice(t - WINDOW + 1, t + 1)
            out.append(dict(
                xo=torch.cat([bo[sl], vo[sl]], 1),                 # 원본
                xm=torch.cat([bm[sl], vm[sl]], 1),                 # 완전 미러
                xb=torch.cat([bm[sl], vo[sl]], 1),                 # bbox만 미러
                xv=torch.cat([bo[sl], vm[sl]], 1),                 # 비전만 미러
                y=int(lab[t])))
        print(f"  ep {ei}/{len(eps)} 누적 윈도 {len(out)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, default=12, help="val 에피소드 수")
    a = ap.parse_args()

    from transformers import AutoProcessor

    from robovlm_nav.serve.stage2_v2_inference_server import _load_kosmos2_vision_only
    proc = AutoProcessor.from_pretrained(str(VLM))
    vm_model = _load_kosmos2_vision_only(VLM).to(DEV).eval()
    ck = torch.load(STAGE1, map_location=DEV, weights_only=False)
    proj = torch.nn.Linear(1024, 256).to(DEV)
    proj.load_state_dict(ck["image_proj"]); proj.eval()

    @torch.no_grad()
    def enc(pil):
        pv = proc(images=pil, return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
        h = vm_model(pixel_values=pv).last_hidden_state.mean(1).float()
        return F.normalize(proj(h), dim=-1)[0].cpu()

    from scripts.train_exp73_trackA_heads import MLPActionHead
    head = MLPActionHead().to(DEV)
    hs = torch.load(HEAD, map_location=DEV, weights_only=False)
    head.load_state_dict(hs["model"] if "model" in hs else hs["state_dict"] if "state_dict" in hs else hs)
    head.eval()

    print(f"val 에피소드 {a.eps}개로 미러 쌍 생성 (이미지 반전 후 피처 재추출)")
    data = build(val_episodes(), enc, a.eps)
    print(f"윈도 {len(data)}개\n")

    def probs(key):
        with torch.no_grad():
            X = torch.stack([d[key] for d in data]).to(DEV)
            return F.softmax(head(X), dim=1).cpu().numpy()

    po, pm, pb, pv = probs("xo"), probs("xm"), probs("xb"), probs("xv")
    y = np.array([d["y"] for d in data])
    pm_s, pb_s, pv_s = pm[:, MIRROR], pb[:, MIRROR], pv[:, MIRROR]   # L/R 스왑 정렬

    print("=" * 70)
    print("액션 헤드 좌우 대칭성 — 미러 입력 후 L/R 스왑 정렬하여 비교")
    print("=" * 70)
    for lab, ps in [("완전 미러(비전+bbox)", pm_s), ("bbox만 미러", pb_s), ("비전만 미러", pv_s)]:
        l1 = np.abs(po - ps).sum(1).mean()
        agree = (po.argmax(1) == ps.argmax(1)).mean()
        print(f"\n[{lab}]")
        print(f"  분포 L1 거리 평균 {l1:.4f} (0이면 완전 대칭, 최대 2)")
        print(f"  argmax 클래스 일치율 {100*agree:.1f}%")

    print("\n" + "=" * 70)
    print("좌/우 클래스 쌍별 확률 — 대칭이면 원본의 L확률 == 미러의 R확률")
    print("=" * 70)
    names = {2: "LEFT/RIGHT", 4: "FWD+L/FWD+R", 6: "ROT_L/ROT_R"}
    res_pairs = {}
    for li, ri in LR_PAIRS:
        pl_o, pr_o = po[:, li].mean(), po[:, ri].mean()
        pl_m, pr_m = pm[:, li].mean(), pm[:, ri].mean()
        # 대칭이면 pl_o ≈ pr_m, pr_o ≈ pl_m
        d1, d2 = pl_o - pr_m, pr_o - pl_m
        print(f"\n  {names[li]}")
        print(f"    원본: P(L)={pl_o:.4f}  P(R)={pr_o:.4f}   → L−R 편향 {pl_o-pr_o:+.4f}")
        print(f"    미러: P(L)={pl_m:.4f}  P(R)={pr_m:.4f}   → L−R 편향 {pl_m-pr_m:+.4f}")
        print(f"    대칭 위반: |P_o(L)−P_m(R)|={abs(d1):.4f}  |P_o(R)−P_m(L)|={abs(d2):.4f}")
        res_pairs[names[li]] = dict(po_L=float(pl_o), po_R=float(pr_o),
                                    pm_L=float(pl_m), pm_R=float(pr_m),
                                    viol1=float(abs(d1)), viol2=float(abs(d2)))

    # 전체 L/R 질량 편향
    Lm = po[:, [2, 4, 6]].sum(1); Rm = po[:, [3, 5, 7]].sum(1)
    LmM = pm[:, [2, 4, 6]].sum(1); RmM = pm[:, [3, 5, 7]].sum(1)
    print("\n" + "=" * 70)
    print(f"  원본 입력: 좌계열 질량 {Lm.mean():.4f} / 우계열 {Rm.mean():.4f} → 편향 {Lm.mean()-Rm.mean():+.4f}")
    print(f"  미러 입력: 좌계열 질량 {LmM.mean():.4f} / 우계열 {RmM.mean():.4f} → 편향 {LmM.mean()-RmM.mean():+.4f}")
    print("  대칭이라면 두 편향의 부호가 반대이고 크기가 같아야 한다")
    print(f"    → 합 {(Lm.mean()-Rm.mean())+(LmM.mean()-RmM.mean()):+.4f}")
    print(f"       (0이 아니면 입력 내용과 무관한 '고정 좌/우 선호'가 존재)")
    fixed = ((Lm.mean() - Rm.mean()) + (LmM.mean() - RmM.mean())) / 2
    print(f"    고정 편향 추정 {fixed:+.4f}  ({'좌' if fixed>0 else '우'}측 선호)")

    (OUT / "head_lr_symmetry.json").write_text(json.dumps(dict(
        n_windows=len(data), n_eps=a.eps,
        l1_full=float(np.abs(po - pm_s).sum(1).mean()),
        l1_bbox=float(np.abs(po - pb_s).sum(1).mean()),
        l1_vis=float(np.abs(po - pv_s).sum(1).mean()),
        agree_full=float((po.argmax(1) == pm_s.argmax(1)).mean()),
        pairs=res_pairs,
        bias_orig=float(Lm.mean() - Rm.mean()),
        bias_mirror=float(LmM.mean() - RmM.mean()),
        fixed_bias=float(fixed)), indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT/'head_lr_symmetry.json'}")


if __name__ == "__main__":
    main()
