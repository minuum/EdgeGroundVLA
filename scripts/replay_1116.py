#!/usr/bin/env python3
"""11/16 a3-s2 실기 프레임(로봇 서버가 보낸 JPG, figures/objtest_1116)을 이 서버에서 재생한다.
1) 각 프레임을 Florence-2(fp32)에 지시어 문구로 넣어 박스(cx,cy,면적,has)와 이미지 특징(768)을 얻는다(soda의 phrase_from_request 동작 재현).
   검증: 다시 찾은 박스 면적을 로봇이 기록한 면적 시계열(real_1116_sessions_bbox_area.json)과 비교.
2) 박스 창(3프레임)+이미지 특징을 a3·a3-s1·a3-s2 헤드에 넣어 첫 명령을 구하고 앙상블(3)을 만든다.
출력: runs/egvla2_fm/t16_cache.npz, docs/v5/egvla2_fm/a3c/replay_1116.json. 열린 루프 재생이며 실기 결과를 예측하지 않는다."""
import glob, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import precompute_v6_florence2_fused as P
import eval_subgoal_ablation as AB
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy, FusedF2Extractor
A = ROOT / "docs/v5/egvla2_fm/a3c"; FR = ROOT / "docs/v5/figures/objtest_1116"; S = json.load(open(A / "real_1116_sessions_bbox_area.json")); CACHE = ROOT / "runs/egvla2_fm/t16_cache.npz"
frames = {sid: sorted(int(os.path.basename(f).rsplit("_", 1)[1][:-4]) for f in glob.glob(str(FR / f"{sid}_*.jpg"))) for sid in S}
if not CACHE.exists():
    ext = FusedF2Extractor(dtype=torch.float32); model, proc = ext.model, ext.proc; out = {}; t0 = time.time()
    @torch.no_grad()
    def ground(pil, phrase):
        inp = proc(text=[P.TASK + phrase], images=[pil], return_tensors="pt"); pv = inp["pixel_values"].to(ext.device, torch.float32); ids = inp["input_ids"].to(ext.device)
        img = model._encode_image(pv); emb, _ = model._merge_input_ids_with_image_features(img, model.get_input_embeddings()(ids))
        gen = model.language_model.generate(input_ids=None, inputs_embeds=emb, max_new_tokens=128, num_beams=3, do_sample=False)
        text = proc.batch_decode(gen, skip_special_tokens=False)[0]; parsed = proc.post_process_generation(text, task=P.TASK, image_size=(pil.width, pil.height))[P.TASK]
        b = P.florence2_output_to_bbox(parsed.get("bboxes", []), pil.width, pil.height)
        return np.array([b["cx"], b["cy"], b["area"], float(b["has_bbox"])], np.float32), img.float().mean(1)[0].cpu().numpy()
    for k, (sid, s) in enumerate(S.items()):
        bb, vv = [], []
        for i in frames[sid]:
            pil = Image.open(FR / f"{sid}_{i}.jpg").convert("RGB"); b, v = ground(pil, s["instruction"]); bb.append(b); vv.append(v)
        out[f"{sid}|bbox"] = np.stack(bb); out[f"{sid}|vis"] = np.stack(vv).astype(np.float32); print(k, sid, len(bb), f"{time.time()-t0:.0f}s", flush=True)
    np.savez(CACHE, **out)
C = np.load(CACHE)
H = {"a3": ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt", "a3-s1": ROOT / "runs/egvla2_fm/a3c/A3C_D30_s1.pt", "a3-s2": ROOT / "runs/egvla2_fm/a3c/A3C_D30_s2.pt"}
pols = {k: EgVla2FMPolicy(v, k_samples=8) for k, v in H.items()}
def run(pol, bw, vis):
    o = []
    for b, v in zip(bw, vis):
        torch.manual_seed(0); o.append(HD.sample_chunk(pol.model, torch.tensor(b[None], device=pol.device), torch.tensor(v[None], device=pol.device), 8).mean(1)[0, 0, :2].cpu().numpy())
    return np.array(o)
R = {}; dif = []; hasd = []
for sid, s in S.items():
    raw = C[f"{sid}|bbox"]; vis = C[f"{sid}|vis"]; rec = np.array(s["bbox_area_seq"], float)[:len(raw)]
    dif += list(np.abs(raw[:, 2] - rec)); hasd += list((raw[:, 3] > .5).astype(float))
    bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0; cmd = {k: run(p, bw, vis) for k, p in pols.items()}; cmd["a3-ens3"] = np.mean([cmd[k] for k in H], 0)
    R[sid] = dict(instruction=s["instruction"], frames=frames[sid], area_rec=[round(float(a), 4) for a in rec], area_re=[round(float(a), 4) for a in raw[:, 2]], has=[int(h > .5) for h in raw[:, 3]], cx=[round(float(c), 3) for c in raw[:, 0]], cy=[round(float(c), 3) for c in raw[:, 1]], cmd={k: np.round(v, 3).tolist() for k, v in cmd.items()})
R["_validation"] = dict(area_abs_diff_median=round(float(np.median(dif)), 4), area_abs_diff_mean=round(float(np.mean(dif)), 4), area_abs_diff_p90=round(float(np.percentile(dif, 90)), 4), has_bbox_rate=round(float(np.mean(hasd)), 3), n_frames=len(dif))
json.dump(R, open(A / "replay_1116.json", "w"), ensure_ascii=False); print(R["_validation"])
