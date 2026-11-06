#!/usr/bin/env python3
"""문구별 Florence-2 그라운딩을 실제 사진 위에 그려 사람이 확인하는 그림 + 융합 인코더 은닉의 문구 민감도(T0 보강).
plan: docs/plans/plan_20261106_text_conditioned_e2e_vla.md §2 T0. 실행: .venv/bin/python3 scripts/eval_text_visual.py"""
import glob, json, sys
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import precompute_v6_florence2_fused as P
from transformers import AutoModelForCausalLM, AutoProcessor

PHR = ["gray basket", "gray bin", "gray container", "the gray basket", "go to the gray bin", "basket", "white box", "red chair", "person"]
model = AutoModelForCausalLM.from_pretrained(P.MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(P.DEV).eval()
proc = AutoProcessor.from_pretrained(P.MODEL, trust_remote_code=True)

@torch.no_grad()
def run(pil, phrase):
    inp = proc(text=[P.TASK + phrase], images=[pil], return_tensors="pt")
    pv, ids = inp["pixel_values"].to(P.DEV, torch.float16), inp["input_ids"].to(P.DEV)
    img = model._encode_image(pv)
    emb, att = model._merge_input_ids_with_image_features(img, model.get_input_embeddings()(ids))
    enc = model.language_model.get_encoder()(inputs_embeds=emb, attention_mask=att).last_hidden_state
    out = model.language_model.generate(input_ids=None, inputs_embeds=emb, max_new_tokens=128, num_beams=3, do_sample=False)
    parsed = proc.post_process_generation(proc.batch_decode(out, skip_special_tokens=False)[0], task=P.TASK, image_size=(pil.width, pil.height))[P.TASK]
    return parsed.get("bboxes", []), enc.float().mean(1)[0].cpu().numpy(), img.float().mean(1)[0].cpu().numpy()

fs = sorted(glob.glob(str(ROOT / "docs/inference_sessions/session_20261029_*.h5")))[::4][:24]
fr = []
for f in fs:
    with h5py.File(f, "r") as h:
        n = h["observations/images"].shape[0]
        for q in (0.25, 0.5, 0.75):
            fr.append((Path(f).stem[-6:], int(n * q), Image.fromarray(h["observations/images"][int(n * q)]).convert("RGB")))
import os
FIG_ONLY = os.environ.get("FIG_ONLY") == "1"   # 1이면 그림용 4장만(민감도 json 재계산 생략)
PICK = [1, 13, 28, 49]   # 서로 다른 세션 4장(시작 위치·바구니 위치가 다양하도록 균등 간격)
cos = {p: [] for p in PHR}; cosimg = {p: [] for p in PHR}; boxes = {}
for k, (sid, j, pil) in enumerate(fr):
    if FIG_ONLY and k not in PICK:
        continue
    ref = None
    for p in PHR:
        b, e, i = run(pil, p)
        if p == PHR[0]: ref = (e, i)
        cos[p].append(float(e @ ref[0] / (np.linalg.norm(e) * np.linalg.norm(ref[0]))))
        cosimg[p].append(float(i @ ref[1] / (np.linalg.norm(i) * np.linalg.norm(ref[1]))))
        boxes[(k, p)] = b
    print(k, flush=True)
if not FIG_ONLY:
  summ = {p: dict(fused_cos_mean=float(np.mean(cos[p])), fused_cos_min=float(np.min(cos[p])), img_cos_mean=float(np.mean(cosimg[p]))) for p in PHR}
  (ROOT / "docs/v5/egvla2_fm/text_fused_sensitivity.json").write_text(json.dumps(summ, indent=1))
  print(json.dumps(summ, indent=1))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
plt.rcParams["font.family"] = "NanumGothic" if any("Nanum" in f.name for f in fm.fontManager.ttflist) else "DejaVu Sans"
pick = PICK
fig, ax = plt.subplots(len(PHR), len(pick), figsize=(4.2 * len(pick), 2.55 * len(PHR)))
for r, p in enumerate(PHR):
    for c, k in enumerate(pick):
        sid, j, pil = fr[k]; a = ax[r, c]; a.imshow(pil); a.set_xticks([]); a.set_yticks([])
        bs = boxes[(k, p)]
        for bx in bs:
            a.add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec="#ff2d2d", lw=2.5))
        full = bool(bs) and all((b_[2] - b_[0]) * (b_[3] - b_[1]) > 0.9 * pil.width * pil.height for b_ in bs)
        a.set_title(f"{sid} f{j}: " + ("화면 전체 박스(=미검출로 처리)" if full else f"박스 {len(bs)}개" if bs else "검출 없음"), fontsize=9, color="#c00" if (full or not bs) else "k")
        if c == 0: a.set_ylabel(f'"{p}"', fontsize=11, rotation=0, ha="right", va="center", labelpad=8)
fig.suptitle("문구별 Florence-2 그라운딩(빨간 박스) — 10/29 실제 H5 프레임, 같은 모델·같은 사진에서 문구만 변경", fontsize=12)
fig.tight_layout(rect=(0.06, 0, 1, 0.98))
out = ROOT / "docs/v5/figures/ch79/text_phrases_grid.png"; fig.savefig(out, dpi=80); print("saved", out)
