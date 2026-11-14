#!/usr/bin/env python3
"""11/13 물체 실험 65세션 분석·재생 페이지. 입력: 세션 H5, logs/episode_log.csv, runs/egvla2_fm/t13_obj_replay.json
출력: docs/v5/object_test_1113.html, docs/v5/egvla2_fm/objtest_data.json, docs/v5/figures/objtest/{sid}_{i}.jpg"""
import csv, glob, json
from pathlib import Path
import h5py, numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent; GO = 0.2
R = {r["session_id"]: r for r in csv.DictReader(open(ROOT / "logs/episode_log.csv"))}
RP = json.load(open(ROOT / "runs/egvla2_fm/t13_obj_replay.json"))
OBJ = {"go to the gray basket": "바구니", "flower pot": "화분", "go to the brown wooden chair": "의자", "small desk": "작은 책상", "go to the small desk": "작은 책상"}
FORM = {"go to the gray basket": "문장", "flower pot": "단어", "go to the brown wooden chair": "문장", "small desk": "단어", "go to the small desk": "문장"}
POS = {"trackA_weak_left": "왼쪽(G2로 추정)", "trackF_center": "가운데(G3)", "trackA_weak_right": "오른쪽(G4로 추정)"}
out = ROOT / "docs/v5/figures/objtest"; out.mkdir(parents=True, exist_ok=True)
nrm = lambda a: np.linalg.norm(np.array(a)[..., :2], axis=-1)
S = []
for f in sorted(glob.glob(str(ROOT / "docs/inference_sessions/session_20261113_*.h5"))):
    sid = Path(f).stem.replace("session_", ""); r = R.get(sid, {})
    with h5py.File(f, "r") as h:
        a = dict(h.attrs); imgs = h["observations/images"][:]; bb = h["grounding/bbox"][:]; act = h["actions"][:len(imgs), :2]; bb = bb[:len(imgs)]
    for i, im in enumerate(imgs): Image.fromarray(np.ascontiguousarray(im).astype(np.uint8)).convert("RGB").resize((384, 216)).save(out / f"{sid}_{i}.jpg", quality=72)
    ins = str(a["instruction"]); n = len(act); mv = nrm(act) >= GO; first = int(np.argmax(mv)) if mv.any() else -1
    hes = int((~mv[first:]).sum()) if first >= 0 else n
    rp = RP[sid]; rec = {"sid": sid, "ins": ins, "obj": OBJ[ins], "form": FORM[ins], "route": r.get("경로") or "(CSV 기록 없음)", "pos": POS.get(r.get("경로"), "기록 없음"),
         "res": r.get("결과") or "기록 없음", "n": n, "first": first, "hes": hes, "go": round(float(mv.mean()), 2), "area0": round(float(bb[0, 2]), 3), "areaN": round(float(bb[-1, 2]), 3),
         "cx0": round(float(bb[0, 0]), 2), "vis": round(float((bb[:, 3] > 0.5).mean()), 2), "bbox": bb.round(3).tolist(), "a3": rp["a3"]}
    for k in ("V0_ins", "V4_ins", "V4_gray", "V4R_ins", "V4R_gray", "V4_ins_gb", "V4R_ins_gb"):
        if k in rp:
            x = np.array(rp[k]); rec[k] = x.mean(0).round(3).tolist(); rec[k + "_stop"] = [round(float(v), 2) for v in (nrm(x) < GO).mean(0)]
    rec["gb_vis"] = round(float(np.mean(rp["gb_has"])), 2) if "gb_has" in rp else None; rec["gb_bbox"] = rp.get("gb_bbox")
    S.append(rec)
json.dump(S, open(ROOT / "docs/v5/egvla2_fm/objtest_data.json", "w"), ensure_ascii=False)
# ---- 집계표 ----
def agg(sel):
    g = [s for s in S if sel(s)]; ok = sum(s["res"] == "성공" for s in g); return len(g), ok
rows = []
for o in ("바구니", "의자", "화분", "작은 책상"):
    c = []
    for p in ("trackA_weak_left", "trackF_center", "trackA_weak_right"):
        n, k = agg(lambda s: s["obj"] == o and s["route"] == p); c.append(f"{k}/{n}" if n else "–")
    rows.append((o, c))
side = agg(lambda s: s["route"] in ("trackA_weak_left", "trackA_weak_right")); cen = agg(lambda s: s["route"] == "trackF_center"); allr = agg(lambda s: True)
noRow = [s for s in S if s["res"] == "기록 없음"]
def m(sel, key): 
    v = [x for s in S if sel(s) for x in s[key]]; return 100 * float(np.mean(v)) if v else float("nan")
cen_fail = lambda s: s["route"] == "trackF_center" and s["res"] == "실패"
G = {"centerF": [s for s in S if cen_fail(s)]}
cs = lambda sel: [s for s in S if sel(s)]
def hrow(name, sel):
    g = cs(sel); 
    if not g: return ""
    return (f"<tr><td>{name}</td><td>{len(g)}</td><td>{np.mean([s['go'] for s in g]):.2f}</td><td>{np.mean([s['hes'] for s in g]):.1f}</td><td>{np.mean([s['area0'] for s in g]):.3f}</td>"
            f"<td>{np.mean([s['cx0'] for s in g]):.2f}</td><td>{100*np.mean([1-np.mean(s['V4R_ins_stop']) for s in g]):.0f}%</td><td>{100*np.mean([1-np.mean(s['V4R_gray_stop']) for s in g]):.0f}%</td><td>{100*np.mean([1-np.mean(s['V0_ins_stop']) for s in g]):.0f}%</td><td>{100*np.mean([s['gb_vis'] for s in g]):.0f}%</td><td>{100*np.mean([1-np.mean(s['V4R_ins_gb_stop']) for s in g]):.0f}%</td><td>{100*np.mean([1-np.mean(s['V4_ins_gb_stop']) for s in g]):.0f}%</td></tr>")
cond = [("바구니 가운데 (성공)", lambda s: s["obj"] == "바구니" and s["route"] == "trackF_center"), ("바구니 옆", lambda s: s["obj"] == "바구니" and s["route"] != "trackF_center"),
        ("의자 가운데 (전부 실패)", lambda s: s["obj"] == "의자" and s["route"] == "trackF_center"), ("의자 옆", lambda s: s["obj"] == "의자" and s["route"] != "trackF_center"),
        ("책상 가운데", lambda s: s["obj"] == "작은 책상" and s["route"] == "trackF_center"), ("책상 옆", lambda s: s["obj"] == "작은 책상" and s["route"] != "trackF_center"),
        ("화분 가운데 성공", lambda s: s["obj"] == "화분" and s["res"] == "성공"), ("화분 가운데 실패", lambda s: s["obj"] == "화분" and s["res"] == "실패")]
htab = "".join(hrow(n, f) for n, f in cond)
html_rows = "".join(f"<tr><td>{o}</td>{''.join(f'<td>{x}</td>' for x in c)}</tr>" for o, c in rows)
(ROOT / "docs/v5/egvla2_fm/objtest_summary.json").write_text(json.dumps({"side": side, "center": cen, "all": allr, "no_csv_row": [s["sid"] for s in noRow]}, ensure_ascii=False))
TPL = (ROOT / "scripts/objtest_template.html").read_text()
page = (TPL.replace("@@ROWS@@", html_rows).replace("@@HTAB@@", htab).replace("@@SIDE@@", f"{side[1]}/{side[0]}").replace("@@CEN@@", f"{cen[1]}/{cen[0]}").replace("@@ALL@@", f"{allr[1]}/{allr[0]}")
        .replace("@@NOROW@@", ", ".join(s["sid"] + f"({s['obj']}, {s['pos']})" for s in noRow) or "없음"))
FP = ROOT / "scripts/objtest_findings.html"; page = page.replace("@@FINDINGS@@", FP.read_text() if FP.exists() else "")
(ROOT / "docs/v5/object_test_1113.html").write_text(page); print("ok", len(page) // 1024, "KB", side, cen, allr)
