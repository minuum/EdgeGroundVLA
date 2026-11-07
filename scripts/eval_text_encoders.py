#!/usr/bin/env python3
"""인코더 부분별 문구 민감도(T0): ① 이미지 인코더 풀링 ② 융합 인코더 전체 평균 ③ 융합 은닉의 이미지 토큰 평균 ④ 융합 은닉의 텍스트 토큰 평균.
기준 문구(gray basket) 대비 코사인. 10/29 H5 72프레임(24세션×3, RGB 그대로). 출력: docs/v5/egvla2_fm/text_encoder_parts.json, docs/v5/figures/ch79/text_encoder_parts.png
실행: .venv/bin/python3 scripts/eval_text_encoders.py"""
import glob, json, sys
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import precompute_v6_florence2_fused as P
from transformers import AutoModelForCausalLM, AutoProcessor

PHR = ["gray basket", "gray bin", "gray container", "the gray basket", "go to the gray bin", "basket", "white box", "red chair", "person", "blue bottle"]
model = AutoModelForCausalLM.from_pretrained(P.MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(P.DEV).eval()
proc = AutoProcessor.from_pretrained(P.MODEL, trust_remote_code=True)
fs = sorted(glob.glob(str(ROOT / "docs/inference_sessions/session_20261029_*.h5")))[::4][:24]
fr = []
for f in fs:
    with h5py.File(f, "r") as h:
        n = h["observations/images"].shape[0]
        fr += [Image.fromarray(h["observations/images"][int(n * q)]).convert("RGB") for q in (0.25, 0.5, 0.75)]
tok = [proc(text=[P.TASK + p], images=[Image.new("RGB", (64, 64))], return_tensors="pt")["input_ids"].to(P.DEV) for p in PHR]
cs = lambda a, b: float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
acc = {p: {k: [] for k in ("vision", "fused_all", "fused_img", "fused_text")} for p in PHR}
with torch.no_grad():
    for pil in fr:
        pv = proc(text=[P.TASK + PHR[0]], images=[pil], return_tensors="pt")["pixel_values"].to(P.DEV, torch.float16)
        img = model._encode_image(pv); ni = img.shape[1]; vis = img.float().mean(1)[0].cpu().numpy(); ref = None
        for k, p in enumerate(PHR):
            emb, att = model._merge_input_ids_with_image_features(img, model.get_input_embeddings()(tok[k]))
            enc = model.language_model.get_encoder()(inputs_embeds=emb, attention_mask=att).last_hidden_state.float()[0]
            v = dict(vision=vis, fused_all=enc.mean(0).cpu().numpy(), fused_img=enc[:ni].mean(0).cpu().numpy(), fused_text=enc[ni:].mean(0).cpu().numpy())
            if k == 0: ref = v
            for n_ in v: acc[p][n_].append(cs(v[n_], ref[n_]))
summ = {p: {n_: float(np.mean(x)) for n_, x in d.items()} for p, d in acc.items()}
(ROOT / "docs/v5/egvla2_fm/text_encoder_parts.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False)); print(json.dumps(summ, indent=1))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
plt.rcParams["font.family"] = "NanumGothic" if any("Nanum" in f.name for f in fm.fontManager.ttflist) else "DejaVu Sans"
names = [("vision", "① 이미지 인코더"), ("fused_all", "② 융합 인코더 전체 평균"), ("fused_img", "③ 융합 은닉 중 이미지 토큰"), ("fused_text", "④ 융합 은닉 중 텍스트 토큰")]
fig, ax = plt.subplots(figsize=(11, 4.6)); w = 0.2; x = np.arange(len(PHR) - 1)
for j, (k, lab) in enumerate(names):
    ax.bar(x + (j - 1.5) * w, [summ[p][k] for p in PHR[1:]], w, label=lab)
ax.set_xticks(x); ax.set_xticklabels(PHR[1:], rotation=20, ha="right"); ax.set_ylim(0.5, 1.005); ax.set_ylabel("기준 문구(gray basket) 대비 cos")
ax.set_title("인코더 부분별 문구 민감도 — 낮을수록 문구가 특징에 많이 실림 (72프레임 평균)"); ax.legend(fontsize=9, loc="lower left"); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(ROOT / "docs/v5/figures/ch79/text_encoder_parts.png", dpi=110)
