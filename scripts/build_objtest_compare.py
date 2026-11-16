#!/usr/bin/env python3
"""11/13 물체 실험 65세션 — 원본 실기 → A3 재생 → V4 → V4R → A3C-S30 을 같은 프레임에서 동시에 비교하는 페이지.
입력: docs/v5/egvla2_fm/objtest_data.json(A3·V4·V4R 재생), 세션 H5(기록 명령·박스), runs/egvla2_fm/a3c/A3C_S30_s0.pt + t13_obj_cache.npz(S30 재생)
출력: docs/v5/object_test_compare_1113.html, docs/v5/egvla2_fm/objtest_compare_data.json. 프레임 이미지는 docs/v5/figures/objtest/{sid}_{i}.jpg(기존, gitignore 대상).
주의: 재생은 열린 루프(기록된 프레임·박스를 모델에 넣은 첫 명령)이고 실제 주행 결과를 예측하지 않는다."""
import json, sys
from pathlib import Path
import h5py, numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
D = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json")); C = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz")
POL = {"A3": EgVla2FMPolicy(ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt", k_samples=8), "S30": EgVla2FMPolicy(ROOT / "runs/egvla2_fm/a3c/A3C_S30_s0.pt", k_samples=8)}
def replay(sid, key):
    pol = POL[key]
    vis = C[f"{sid}|vis"].astype(np.float32); raw = C[f"{sid}|bbox"][:len(vis)].astype(np.float32); bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0; o = []
    for b, v in zip(bw, vis):
        torch.manual_seed(0); o.append(HD.sample_chunk(pol.model, torch.tensor(b[None], device=pol.device), torch.tensor(v[None], device=pol.device), 8).mean(1)[0, 0, :2].cpu().numpy())
    return np.array(o)
r2 = lambda a: np.round(np.asarray(a, float), 3).tolist()
S = []
for x in D:
    sid = x["sid"]
    with h5py.File(ROOT / f"docs/inference_sessions/session_{sid}.h5", "r") as h: n = h["observations/images"].shape[0]; act = h["actions"][:n, :2]
    m = {"원본 실기(기록 명령)": act, "A3 재생": replay(sid, "A3"), "V4 재생": np.array(x["V4_ins_gb"]), "V4R 재생": np.array(x["V4R_ins_gb"]), "A3C-S30 재생": replay(sid, "S30")}
    S.append(dict(sid=sid, ins=x["ins"], obj=x["obj"], form=x["form"], pos=x["pos"], res=x["res"], n=n, rb=r2(x["bbox"]), gb=r2(x["gb_bbox"]) if x.get("gb_bbox") else None, m={k: r2(v[:n]) for k, v in m.items()}))
    print(sid, n, flush=True)
# 로봇 서버 JSON(docs/inference_reports)에 남은 모델 원출력(chunk_first): 수신한 3세션만 있음
for s_ in S:
    f = ROOT / f"_received_1113_env_check/session_{s_['sid']}.json"
    if f.exists():
        hist = json.load(open(f))["history"]; s_["m"]["실기 모델 원출력(3세션만)"] = r2([h["chunk_first"][:2] for h in hist][:s_["n"]])
        rep = np.array(s_["m"]["A3 재생"]); raw = np.array(s_["m"]["실기 모델 원출력(3세션만)"]); k = min(len(rep), len(raw))
        print("실기 원출력 vs A3 재생", s_["sid"], "mean abs diff %.3f" % np.abs(rep[:k] - raw[:k]).mean())
json.dump(S, open(ROOT / "docs/v5/egvla2_fm/objtest_compare_data.json", "w"), ensure_ascii=False)
# 충실도: 기록 명령(원본 실기)과 A3 헤드 재생의 차이
d = np.concatenate([np.abs(np.array(s["m"]["원본 실기(기록 명령)"]) - np.array(s["m"]["A3 재생"])).ravel() for s in S]); print("recorded vs A3 replay abs diff mean %.4f max %.4f" % (d.mean(), d.max()))
# 부호 점검: 기록 명령 ly와 A3 재생 ly가 같은 방향인가
a = np.concatenate([np.array(s["m"]["원본 실기(기록 명령)"])[:, 1] for s in S]); b = np.concatenate([np.array(s["m"]["A3 재생"])[:, 1] for s in S]); print("ly corr recorded vs A3 replay", float(np.corrcoef(a, b)[0, 1]))
HTML = open(ROOT / "scripts/objtest_compare_template.html", encoding="utf-8").read().replace("/*DATA*/[]", json.dumps(S, ensure_ascii=False))
(ROOT / "docs/v5/object_test_compare_1113.html").write_text(HTML, encoding="utf-8"); print("html", len(HTML))
# 단독 실행판: 실기 프레임(로봇 서버가 보낸 1280×720 원본)을 448×252 JPEG로 줄여 HTML 안에 넣는다
import base64, io
from PIL import Image
FR = {}
for s_ in S:
    FR[s_["sid"]] = []
    for i in range(s_["n"]):
        im = Image.open(ROOT / f"docs/v5/figures/objtest/{s_['sid']}_{i}.jpg").convert("RGB").resize((448, 252), Image.LANCZOS); b = io.BytesIO(); im.save(b, "JPEG", quality=70)
        FR[s_["sid"]].append("data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode())
ST = HTML.replace("/*FRAMES*/null", json.dumps(FR, separators=(",", ":")), 1)
assert ST != HTML
(ROOT / "docs/v5/object_test_compare_1113_standalone.html").write_text(ST, encoding="utf-8"); print("standalone", len(ST))
