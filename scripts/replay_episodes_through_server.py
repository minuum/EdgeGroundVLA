#!/usr/bin/env python3
"""드라이런(구동 없음): 수집된 에피소드 H5를 프레임 순서대로(JPEG q95) 실행 중인 서버 /predict에 보내고, 서버 출력을 사람 조작(actions_exec)과 비교한다.
에피소드 시작마다 /reset. 서버는 평활을 쓰는 설정(VLA_EGVLA2_EMA=0.5 권장)으로 띄운다. 로봇을 움직일 필요가 없다 — 이미 수집된(특히 pos2·pos3 시작 위치에서 새로 수집한) 에피소드로 오프라인 일반화를 잰다.
사용: python3 scripts/replay_episodes_through_server.py --h5 <에피소드.h5 또는 폴더> [--server http://127.0.0.1:8001] [--quality 95] [--max-frames 0] [--tag pos2] [--out /tmp/replay_server.json] [--api-key KEY]
지표: 방향 오차(°, 둘 다 이동 중), GO/STOP F1, 프레임 저크, 서버 지연, 서브골 결측 프레임 수. 프레임별 CSV도 저장(--out 옆 .csv).
"""
import argparse
import base64
import csv
import io
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--h5", required=True); ap.add_argument("--server", default="http://127.0.0.1:8001"); ap.add_argument("--quality", type=int, default=95)
ap.add_argument("--max-frames", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--out", default="/tmp/replay_server.json")
ap.add_argument("--api-key", default=os.getenv("VLA_API_KEY", ""))
A = ap.parse_args()
hdr = {"X-API-Key": A.api_key} if A.api_key else {}
p = Path(A.h5); files = [p] if p.is_file() else sorted(p.glob("*.h5"))
GO = 0.2
ang = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
def first_step(j):
    """서버 응답에서 첫 스텝 (lx,ly)를 꺼낸다. chunk_first가 없는 서버(응답 모델에서 필드가 잘리는 경우)는 action_cont의 방향×크기로 대체한다.
    정지(stop)면 방향 정보가 없어 (0,0)(크기<0.2 → 정지로 집계). 둘 다 없으면 'missing'."""
    ac = j.get("action_cont") or {}
    cf = ac.get("chunk_first") or j.get("chunk_first")
    if cf:
        return np.array(cf[:2], float), "chunk_first"
    if ac and not ac.get("error") and ac.get("model_magnitude") is not None:
        m = float(ac["model_magnitude"])
        if ac.get("stop"):
            return np.zeros(2), "action_cont_stop"
        return np.array([ac.get("dir_lx", 0.0) * m, ac.get("dir_ly", 0.0) * m]), "action_cont"
    return np.zeros(2), "missing"


rows, per_ep = [], []
for f in files:
    with h5py.File(f, "r") as h:
        im = h["observations"]["images"] if "observations" in h else h["images"]
        ex = np.array(h["actions_exec"], float)[:, :2] if "actions_exec" in h else None
        n = len(im) if not A.max_frames else min(A.max_frames, len(im))
        pos = str(h.attrs.get("cx_position", "")); path = str(h.attrs.get("cx_path", ""))
        requests.post(A.server + "/reset", headers=hdr, timeout=10)
        prev, ep_rows = None, []
        for i in range(n):
            buf = io.BytesIO(); Image.fromarray(np.ascontiguousarray(im[i][:, :, ::-1])).save(buf, "JPEG", quality=A.quality)
            t0 = time.time(); r = requests.post(A.server + "/predict", json={"image": base64.b64encode(buf.getvalue()).decode(), "instruction": "basket"}, headers=hdr, timeout=60); ms = (time.time() - t0) * 1000
            r.raise_for_status(); j = r.json(); ac = j.get("action_cont") or {}
            v, src = first_step(j)
            d = dict(file=f.name, tag=A.tag, pos=pos, path=path, frame=i, server_ms=ms, pred_lx=float(v[0]), pred_ly=float(v[1]), drive=json.dumps(j.get("drive")), state=j.get("controller_state"),
                     pred_src=src, subgoal_missing=j.get("subgoal_missing"), human_lx=float(ex[i][0]) if ex is not None else None, human_ly=float(ex[i][1]) if ex is not None else None,
                     jitter=float(np.linalg.norm(v - prev)) if prev is not None else None)
            prev = v; ep_rows.append(d)
        rows += ep_rows
        if ex is not None:
            H = np.array([[r["human_lx"], r["human_ly"]] for r in ep_rows]); P = np.array([[r["pred_lx"], r["pred_ly"]] for r in ep_rows])
            pm, hm = np.linalg.norm(P, axis=1), np.linalg.norm(H, axis=1); both = (pm > GO) & (hm > GO)
            a = [ang(P[k], H[k]) for k in np.where(both)[0]]
            tp, fp, fn = int(((pm > GO) & (hm > GO)).sum()), int(((pm > GO) & (hm <= GO)).sum()), int(((pm <= GO) & (hm > GO)).sum())
            per_ep.append(dict(file=f.name, pos=pos, path=path, n=n, angle_deg=float(np.mean(a)) if a else None, go_f1=2 * tp / max(2 * tp + fp + fn, 1), server_ms=float(np.mean([r["server_ms"] for r in ep_rows]))))
            print(per_ep[-1], flush=True)
ok = [e["angle_deg"] for e in per_ep if e["angle_deg"] is not None]
summ = dict(tag=A.tag, episodes=len(per_ep), angle_deg_mean=float(np.mean(ok)) if ok else None, go_f1_mean=float(np.mean([e["go_f1"] for e in per_ep])) if per_ep else None,
            jitter_mean=float(np.mean([r["jitter"] for r in rows if r["jitter"] is not None])) if rows else None, server_ms_mean=float(np.mean([r["server_ms"] for r in rows])) if rows else None,
            pred_src_counts={k: int(sum(1 for r in rows if r['pred_src'] == k)) for k in sorted({r['pred_src'] for r in rows})}, subgoal_missing_frames=int(sum(1 for r in rows if r["subgoal_missing"])), frames=len(rows), quality=A.quality)
if summ.get('pred_src_counts', {}).get('missing'):
    print('WARNING: 응답에서 방향을 못 읽은 프레임이 있습니다(pred_src=missing). 서버 InferenceResponse에 chunk_first/action_cont 필드가 있는지 확인하세요.')
print(json.dumps(summ, indent=1, ensure_ascii=False))
Path(A.out).write_text(json.dumps(dict(summary=summ, episodes=per_ep), indent=1, ensure_ascii=False))
with open(Path(A.out).with_suffix(".csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
