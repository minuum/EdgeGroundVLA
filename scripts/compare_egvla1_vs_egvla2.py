#!/usr/bin/env python3
"""EGVLA1(운영 SOTA: OWLv2+Kosmos-2+MLP 8-class, exp73_owl_stage1v3_v6_mlp) vs EGVLA2(Florence-2+FM-π0 연속) 오프라인 프레임 단위 비교.
데이터 2종(둘 다 각 모델이 학습에 안 쓴 에피소드 기준):
  v3(EGVLA2 수집): 5-fold fold0 hold-out에서 목표 위치별 2개 — EGVLA2는 fold0 학습 모델(이 에피소드 미사용), EGVLA1은 다른 수집(V6)으로 학습.
  V6(EGVLA1 수집): EGVLA1 검증 분할(seed 42, 15%)에서 목표 위치별 2개 — EGVLA2는 배포 체크포인트(v3만 학습, V6 미사용).
EGVLA1은 서버 클래스(Stage2V2Model)를 그대로 in-process 실행(OWL 임계값 0.20, 그라운딩 재사용 3, 대시보드처럼 JPEG q80), EGVLA2는 서빙 정책(EgVla2FMPolicy, K=8, EMA 0.5)과 Florence-2 캐시 특징.
출력: docs/v5/egvla2_fm/egvla1_vs_egvla2.json, docs/v5/egvla2_fm/compare/*.jpg (화살표 오버레이: 초록 사람, 빨강 EGVLA1, 노랑 EGVLA2)
실행: HF_HUB_OFFLINE=1 .venv/bin/python3 -I scripts/compare_egvla1_vs_egvla2.py
"""
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.update({"VLA_GROUNDER": "owlv2", "VLA_OWLV2_THRESH": "0.20", "VLA_GROUNDING_SKIP_N": "3",
                   "VLA_S2V2_STAGE2": str(ROOT / "runs/v5_nav/mlp/exp73_stage1v3/exp73_owl_stage1v3_v6_mlp.pt"),
                   "VLA_S2V2_STAGE1": str(ROOT / "runs/v5_nav/mlp/stage1_v3_5cls/stage1_v3_5cls_owl_projs.pt")})
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

DATA3 = ROOT / "ROS_action/mobile_vla_dataset_egvla2"; DATA6 = ROOT / "ROS_action/mobile_vla_dataset_v5"
OUTD = ROOT / "docs/v5/egvla2_fm/compare"; OUTD.mkdir(parents=True, exist_ok=True)
POS = ["strong_left", "weak_left", "center", "weak_right", "strong_right"]
A3D = {0: (0, 0, 0), 1: (1.15, 0, 0), 2: (0, 1.15, 0), 3: (0, -1.15, 0), 4: (1.15, 1.15, 0), 5: (1.15, -1.15, 0), 6: (0, 0, .25), 7: (0, 0, -.25)}
CN = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
DIRS = {1: (1, 0), 2: (0, 1), 3: (0, -1), 4: (.7071, .7071), 5: (.7071, -.7071)}


def cls_of(v, go=0.2):
    lx, ly, az = (list(v) + [0, 0, 0])[:3]
    if abs(az) > 0.1 and abs(lx) < 0.05 and abs(ly) < 0.05:
        return 6 if az > 0 else 7
    m = (lx * lx + ly * ly) ** 0.5
    if m < go:
        return 0
    return max(DIRS, key=lambda k: (lx * DIRS[k][0] + ly * DIRS[k][1]) / m)


def pick(names, per=2):
    out = []
    for p in POS:
        out += [n for n in names if f"_{p}_" in n][:per]
    return out


def arrow(dr, v, col, cx, cy, s=55):
    lx, ly = v[0], v[1]; m = (lx * lx + ly * ly) ** 0.5
    if len(v) > 2 and abs(v[2]) > 0.1 and m < 0.05:
        dr.arc([cx - 14, cy - 14, cx + 14, cy + 14], 200 if v[2] > 0 else -20, 340 if v[2] > 0 else 160, fill=col, width=4); return
    if m < 0.2:
        dr.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=col, width=3); return
    ex, ey = cx - ly / m * s * min(1, m), cy - lx / m * s * min(1, m)
    dr.line([cx, cy, ex, ey], fill=col, width=5); dr.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=col)


def prepare():
    """1단계(.venv_lerobot, Python 3.12): fold0 학습 FM-π0 체크포인트와 v3 선택 목록 저장."""
    import train_egvla2_s1 as S1
    import eval_subgoal_ablation as AB
    T = S1.T
    eps = S1.load_main(); folds = S1.make_folds(eps); cache3 = np.load(T.CACHE)
    hold = sorted(eps[i]["name"] for i in folds[0])
    E = AB.prep(eps, cache3, 3); tr_idx = [i for i in range(len(E)) if i not in folds[0]]; tr = T.to_t(E, tr_idx)
    m = AB.fit(S1.ARCHS["C_pi0_adaln"], tr, 120, 0, drop=0.3)
    ck = dict(arch="C_pi0_adaln", state=m.state_dict(), bdim=12, fdim=768, window=3, H=8, subgoal_dropout=0.3); torch.save(ck, "/tmp/egvla2_fold0.pt")
    json.dump(dict(v3_sel=pick(hold)), open("/tmp/egvla2_fold0_sel.json", "w")); print("prepared")


def main():
    """2단계(.venv, transformers): EGVLA1 서버 모델 + EGVLA2 서빙 정책으로 프레임 비교."""
    from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
    cache3 = np.load(ROOT / "docs/v5/egvla2_fm/v3_florence2_cache.npz"); v3_sel = json.load(open("/tmp/egvla2_fold0_sel.json"))["v3_sel"]
    pol3 = EgVla2FMPolicy("/tmp/egvla2_fold0.pt", k_samples=8, ema_alpha=0.5)
    pol6 = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/deploy/C_pi0_adaln_drop30_e120.pt", k_samples=8, ema_alpha=0.5)
    cache6 = np.load(ROOT / "docs/v5/bbox_nav_owl/v6_florence2_cache.npz")
    v6_val = json.load(open("/tmp/egvla1_v6_val.json")); v6_sel = pick(sorted(v6_val))
    import robovlm_nav.serve.stage2_v2_inference_server as SRV
    e1 = SRV.get_model()
    OUTJ = ROOT / "docs/v5/egvla2_fm/egvla1_vs_egvla2.json"
    res = json.loads(OUTJ.read_text()) if OUTJ.exists() else {"v3": [], "v6": []}   # 에피소드마다 저장, 재시작 시 이어서
    for ds, sel in (("v3", v3_sel), ("v6", v6_sel)):
        for name in sel:
            fn = (DATA3 / name) if ds == "v3" else (DATA6 / (name + ".h5"))
            key = fn.name if ds == "v3" else name
            ckey = key if ds == "v3" else str(fn)   # V6 캐시 키는 전체 경로
            if any(e["name"] == key for e in res[ds]):
                continue
            with h5py.File(fn, "r") as h:
                imgs = h["images"]; n = len(imgs)
                hum = np.array(h["actions_exec"] if ds == "v3" else h["actions"], float)
                hcls = np.array(h["action_classes"]) if ds == "v3" else np.array([cls_of(a) for a in hum])
                ts = np.array(h["frame_timestamps"], float) if "frame_timestamps" in h else np.arange(n) * 0.2
                c = cache3 if ds == "v3" else cache6; pol = pol3 if ds == "v3" else pol6; pol.reset(); e1.reset()
                bbs = c[f"{ckey}|bbox"]; fts = c[f"{ckey}|feat"].astype(np.float32)
                rows = []
                for t in range(n):
                    rgb = np.ascontiguousarray(imgs[t][:, :, ::-1]); buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "JPEG", quality=80)
                    t0 = time.time(); r1 = e1.predict(base64.b64encode(buf.getvalue()).decode(), "basket"); ms1 = (time.time() - t0) * 1000
                    torch.manual_seed(t); r2 = pol.step(bbs[t], fts[t])
                    v2 = list(r2["first"]); v2[2] = 0.0
                    rows.append(dict(t=t, human=hum[t].round(3).tolist(), hcls=int(hcls[t]), e1=list(map(float, r1["action_3d"])), e1cls=int(r1["predicted_class"]), e1_has=bool(r1["bbox"].get("has_bbox")),
                                     e1_ms=ms1, e1_cached=bool(r1.get("grounding_cached")), e1_stop_latched=bool(r1.get("stop_latched")), e2=[round(x, 3) for x in v2], e2cls=cls_of(v2), e2_has=bool(bbs[t][3] >= 0.5),
                                     dt=float(ts[min(t + 1, n - 1)] - ts[t]) if t + 1 < n else float(np.median(np.diff(ts)) if n > 1 else 0.2)))
                # 썸네일 6장(화살표)
                fr = np.linspace(0, n - 1, 6).astype(int)
                for k, t in enumerate(fr):
                    im = Image.fromarray(np.ascontiguousarray(imgs[t][:, :, ::-1])).resize((320, 180)); dr = ImageDraw.Draw(im)
                    rr = rows[t]
                    arrow(dr, rr["human"], (74, 222, 128), 100, 150); arrow(dr, rr["e1"], (248, 113, 113), 160, 150); arrow(dr, rr["e2"], (251, 191, 36), 220, 150)
                    im.save(OUTD / f"{ds}_{key.replace('.h5', '')}_{k}.jpg", quality=78)
            res[ds].append(dict(name=key, n=n, frames=rows, thumbs=fr.tolist()))
            OUTJ.write_text(json.dumps(res, ensure_ascii=False)); print(ds, key, n, flush=True)
    (ROOT / "docs/v5/egvla2_fm/egvla1_vs_egvla2.json").write_text(json.dumps(res, ensure_ascii=False))
    print("saved")


if __name__ == "__main__":
    prepare() if "--prepare" in sys.argv else main()
