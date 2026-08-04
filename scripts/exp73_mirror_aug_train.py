#!/usr/bin/env python3
"""미러 증강 재학습 — 헤드의 고정 좌/우 편향을 제거해 실기 좌우차의 잔여분을 분리 (65-9).

배경:
  65-7 그라운더(OWL-v2)는 좌측 +0.0118 선호 → 실기 좌측 약세와 부호 반대, 원인 아님
  65-8 액션 헤드에 입력과 무관한 **고정 우측 선호 −0.0275** → 실기와 부호 일치
       원인 후보: 학습 데이터 액션 클래스가 우측 편중(좌 3384 vs 우 4122, 21.8%)
       특히 FWD+R(3117) vs FWD+L(2532)

이 실험:
  모든 학습 윈도에 대해 **미러 쌍**을 만들어 좌우 대칭을 강제한다.
    · 비전 피처: 이미지 좌우 반전 후 재추출 (256-dim)
    · bbox: cx → 1 − cx
    · 라벨: LEFT↔RIGHT, FWD+L↔FWD+R, ROT_L↔ROT_R 스왑
  결과적으로 학습 분포가 정확히 좌우 대칭이 되므로, 헤드의 고정 편향이 0으로 수렴해야 한다.

왜 중요한가:
  헤드 편향을 0으로 만든 뒤 남는 실기 좌우차가 곧 **기구/물리 편향의 크기**다.
  즉 이 실험은 로봇 없이 만들 수 있는 "기구 편향 검정의 기준선"이다.

출력: docs/v5/closed_loop_eval/exp73_mirror_aug.json
      runs/v5_nav/mlp/exp73/exp73_owl_trackF_v6_mlp_mirroraug_seed{0,1,2}.pt
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
from scripts.exp73_held_aware_train import build_windows_hold_aware, majority, train_one
from scripts.train_exp73_trackA_heads import (BBOX_SCALE, CACHE_V6, DEVICE, NUM_CLASSES,
                                              SPLIT_SEED, VAL_RATIO, WINDOW, MLPActionHead)

ANN = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json"
VLM = ROOT / ".vlms/kosmos-2-patch14-224"
STAGE1 = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
MIRROR_CACHE = ROOT / "docs/v5/closed_loop_eval/exp73_v6_vis_cache_mirror.pt"
OUT_DIR = ROOT / "runs/v5_nav/mlp/exp73"
OUT_JSON = ROOT / "docs/v5/closed_loop_eval/exp73_mirror_aug.json"
MIRROR_CLS = [0, 1, 3, 2, 5, 4, 7, 6]      # STOP FWD | LEFT↔RIGHT | FWD+L↔FWD+R | ROT_L↔ROT_R


def build_mirror_cache():
    """이미지 좌우 반전 후 비전 피처 재추출. 원본 캐시와 동일한 에피소드 순서를 유지."""
    if MIRROR_CACHE.exists():
        print(f"미러 캐시 재사용: {MIRROR_CACHE}")
        return torch.load(str(MIRROR_CACHE), weights_only=False)

    from transformers import AutoProcessor

    from robovlm_nav.serve.stage2_v2_inference_server import _load_kosmos2_vision_only
    base = torch.load(str(CACHE_V6), weights_only=False)
    ann = {Path(e["episode"]).stem: e for e in json.loads(ANN.read_text())}
    proc = AutoProcessor.from_pretrained(str(VLM))
    vm = _load_kosmos2_vision_only(VLM).to(DEVICE).eval()
    ck = torch.load(STAGE1, map_location=DEVICE, weights_only=False)
    proj = torch.nn.Linear(1024, 256).to(DEVICE)
    proj.load_state_dict(ck["image_proj"]); proj.eval()

    @torch.no_grad()
    def enc(pil):
        pv = proc(images=pil, return_tensors="pt")["pixel_values"].to(DEVICE, torch.float16)
        h = vm(pixel_values=pv).last_hidden_state.mean(1).float()
        return F.normalize(proj(h), dim=-1)[0].cpu()

    out = []
    for ei, ep in enumerate(base, 1):
        a = ann.get(ep["stem"])
        if a is None:
            raise KeyError(f"주석에 없는 에피소드: {ep['stem']}")
        fr = [f for f in a["frames"] if f.get("gt_class") is not None]
        assert len(fr) == len(ep["gts"]), f"{ep['stem']}: 프레임 수 불일치"
        with h5py.File(a["episode"], "r") as h:
            imgs = np.array(h["images"])
        vis = []
        for f in fr:
            im = imgs[f["frame_idx"]][:, :, ::-1]                     # BGR→RGB
            pil = Image.fromarray(im.astype("uint8")).transpose(Image.FLIP_LEFT_RIGHT)
            vis.append(enc(pil))
        out.append(dict(stem=ep["stem"], path_type=ep["path_type"],
                        bboxes=[[1.0 - b[0], b[1], b[2], b[3]] for b in ep["bboxes"]],
                        vis=torch.stack(vis),
                        gts=[MIRROR_CLS[g] for g in ep["gts"]],
                        acts=ep["acts"]))
        if ei % 25 == 0:
            print(f"  미러 추출 {ei}/{len(base)}", flush=True)
    torch.save(out, str(MIRROR_CACHE))
    print(f"저장: {MIRROR_CACHE}")
    return out


def val_split(base):
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(base)))
    rng.shuffle(idx)
    nv = max(1, int(len(idx) * VAL_RATIO))
    return set(idx[:nv])


@torch.no_grad()
def fixed_bias(head, Xva_o, Xva_m):
    """65-8과 동일한 지표 — 입력 내용과 무관한 고정 좌/우 편향."""
    head.eval()
    def mass(X):
        p = F.softmax(head(torch.from_numpy(X).float().to(DEVICE)), dim=1).cpu().numpy()
        return p[:, [2, 4, 6]].sum(1).mean() - p[:, [3, 5, 7]].sum(1).mean()
    bo, bm = mass(Xva_o), mass(Xva_m)
    return float(bo), float(bm), float((bo + bm) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    base = torch.load(str(CACHE_V6), weights_only=False)
    mir = build_mirror_cache()
    val_idx = val_split(base)
    tr_o = [e for i, e in enumerate(base) if i not in val_idx]
    va_o = [e for i, e in enumerate(base) if i in val_idx]
    tr_m = [e for i, e in enumerate(mir) if i not in val_idx]
    va_m = [e for i, e in enumerate(mir) if i in val_idx]
    print(f"train {len(tr_o)}ep / val {len(va_o)}ep  (미러 포함 학습 {2*len(tr_o)}ep 상당)")

    Xo, yo = build_windows_hold_aware(tr_o)
    Xm, ym = build_windows_hold_aware(tr_m)
    Xva, yva = build_windows_hold_aware(va_o)
    Xva_m, _ = build_windows_hold_aware(va_m)
    Xtr = np.concatenate([Xo, Xm]); ytr = np.concatenate([yo, ym])
    print(f"윈도: 원본 {len(Xo)} + 미러 {len(Xm)} = {len(Xtr)} / val {len(Xva)}")
    cL = int(sum((ytr == c).sum() for c in (2, 4, 6)))
    cR = int(sum((ytr == c).sum() for c in (3, 5, 7)))
    print(f"학습 라벨 좌계열 {cL} vs 우계열 {cR}  → 차 {abs(cL-cR)} (0이면 완전 대칭)")

    res = []
    for s in a.seeds:
        acc, state = train_one(Xtr, ytr, Xva, yva, s)
        head = MLPActionHead().to(DEVICE); head.load_state_dict(state)
        bo, bm, fx = fixed_bias(head, Xva, Xva_m)
        p = OUT_DIR / f"exp73_owl_trackF_v6_mlp_mirroraug_seed{s}.pt"
        torch.save(dict(model=state, val_acc=acc, head="exp73_mlp",
                        window=WINDOW, bbox_scale=BBOX_SCALE, arm="v6_mirroraug",
                        grounder="owlv2", exp="exp73", stride=5,
                        fixed_bias=fx), str(p))
        print(f"  seed{s}: val_acc {acc:.4f}  고정편향 {fx:+.4f} "
              f"(원본 {bo:+.4f} / 미러 {bm:+.4f})  → {p.name}")
        res.append(dict(seed=s, val_acc=float(acc), bias_orig=bo, bias_mirror=bm, fixed_bias=fx))

    accs = [r["val_acc"] for r in res]; fxs = [r["fixed_bias"] for r in res]
    print("\n" + "=" * 66)
    print(f"미러 증강 결과 ({len(res)} seeds)")
    print("=" * 66)
    print(f"  val_acc      {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  고정 좌/우 편향 {np.mean(fxs):+.4f} ± {np.std(fxs):.4f}")
    print(f"  대조 — 기존 holdaware seed0 고정편향: -0.0275 (우측 선호)")
    print(f"  → 편향이 0에 수렴하면 헤드 요인 제거 성공, "
          f"이후 실기 좌우차 잔여분이 기구 편향 크기")
    OUT_JSON.write_text(json.dumps(dict(
        seeds=res, val_acc_mean=float(np.mean(accs)), val_acc_std=float(np.std(accs)),
        fixed_bias_mean=float(np.mean(fxs)), fixed_bias_std=float(np.std(fxs)),
        baseline_fixed_bias=-0.0275,
        train_label_L=cL, train_label_R=cR), indent=2))
    print(f"\n저장: {OUT_JSON}")


if __name__ == "__main__":
    main()
