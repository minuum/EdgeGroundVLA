#!/usr/bin/env python3
"""학습-서빙 간극(전처리) 직접 측정: 실기 세션 프레임을 학습 때와 같은 Florence-2 추출기로 다시 계산해
(1) 서버가 기록한 bbox와의 일치(서빙 경로 동등성), (2) 학습 특징 분포 대비 이동량(최근접 코사인 유사도, 학습 내 보류 에피소드와 비교),
(3) bbox 통계(cx·면적) 비교. 입력: ROS_action/dryrun_closedloop_20261022/h5, 학습 캐시 docs/v5/egvla2_fm/v3_florence2_cache.npz.
실행: HF_HUB_OFFLINE=1 .venv/bin/python3 scripts/analyze_live_vs_train_features.py   출력: docs/v5/egvla2_fm/live_vs_train.json
"""
import glob
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from robovlm_nav.serve.egvla2_fm_policy import FusedF2Extractor  # noqa: E402


def jpeg_roundtrip(rgb, q=95):
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "JPEG", quality=q)
    return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def main():
    ex = FusedF2Extractor(dtype=torch.float32)
    live_b, live_f, logged_b, meta = [], [], [], []
    for p in sorted(glob.glob(str(ROOT / "ROS_action/dryrun_closedloop_20261022/h5/*.h5"))):
        with h5py.File(p, "r") as f:
            if f["observations/images"].shape[0] < 3 or "egvla2" not in str(f.attrs.get("model_name", "")):
                continue
            imgs, gb = f["observations/images"][:], f["grounding/bbox"][:]
        for i in range(len(imgs)):
            b, ft = ex(jpeg_roundtrip(imgs[i][:, :, ::-1] if False else imgs[i]))
            live_b.append(b); live_f.append(ft); logged_b.append(gb[i]); meta.append((Path(p).stem, i))
        print(Path(p).stem, len(imgs), flush=True)
    live_b, live_f, logged_b = np.array(live_b), np.array(live_f), np.array(logged_b)
    # (1) 서버 기록 bbox와 비교 (기록 형식: cx,cy,area,has 또는 x1y1x2y2 — 형식 출력)
    out = {"n_live": int(len(live_f)), "logged_bbox_example": logged_b[:2].round(3).tolist(), "recomputed_bbox_example": live_b[:2].round(3).tolist()}
    # (2) 학습 특징 대비 최근접 코사인
    cache = np.load(ROOT / "docs/v5/egvla2_fm/v3_florence2_cache.npz")
    names = sorted({k.split("|")[0] for k in cache.files if k.endswith("|feat")})
    tr_f = [cache[f"{n}|feat"] for n in names]; ep_id = np.concatenate([[i] * len(x) for i, x in enumerate(tr_f)]); tr = np.concatenate(tr_f).astype(np.float32)
    nrm = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    trn, lvn = nrm(tr), nrm(live_f)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = torch.from_numpy(trn).to(dev)
    nn_live = (torch.from_numpy(lvn).to(dev) @ T.T).max(1).values.cpu().numpy()
    rng = np.random.default_rng(0); pick = rng.choice(len(tr), 600, replace=False)
    sims = (torch.from_numpy(trn[pick]).to(dev) @ T.T).cpu().numpy()
    for k, i in enumerate(pick):
        sims[k, ep_id == ep_id[i]] = -1                       # 같은 에피소드 제외
    nn_tr = sims.max(1)
    cos_cent = lambda x: (nrm(x) @ nrm(tr.mean(0, keepdims=True)).T).ravel()
    out.update(nn_cos_live=dict(mean=float(nn_live.mean()), p10=float(np.percentile(nn_live, 10)), p50=float(np.median(nn_live))),
               nn_cos_train_heldout_episode=dict(mean=float(nn_tr.mean()), p10=float(np.percentile(nn_tr, 10)), p50=float(np.median(nn_tr))),
               centroid_cos_live=float(cos_cent(live_f).mean()), centroid_cos_train=float(cos_cent(tr[pick]).mean()),
               feat_norm_live=float(np.linalg.norm(live_f, axis=1).mean()), feat_norm_train=float(np.linalg.norm(tr, axis=1).mean()))
    # (3) bbox 통계(학습 캐시 bbox: cx,cy,area,has)
    tb = np.concatenate([cache[f"{n}|bbox"] for n in names]); has = tb[:, 3] > 0.5
    out.update(bbox_train=dict(cx_mean=float(tb[has, 0].mean()), area_median=float(np.median(tb[has, 2])), has_rate=float(has.mean())),
               bbox_live=dict(cx_mean=float(live_b[live_b[:, 3] > 0.5, 0].mean()), area_median=float(np.median(live_b[live_b[:, 3] > 0.5, 2])), has_rate=float((live_b[:, 3] > 0.5).mean())))
    np.savez_compressed("/tmp/live_feats.npz", live_b=live_b, live_f=live_f, logged_b=logged_b, meta=np.array([f"{a}|{b}" for a, b in meta]))
    (ROOT / "docs/v5/egvla2_fm/live_vs_train.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
