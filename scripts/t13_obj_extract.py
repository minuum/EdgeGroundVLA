#!/usr/bin/env python3
"""11/13 물체 실험 65세션: 프레임별 Florence-2 이미지 풀링(vis) + 문구별 텍스트 평균(txt). 문구 = 실제 사용한 instruction + 'go to the gray basket'.
출력 runs/egvla2_fm/t13_obj_cache.npz ({sid}|vis,{sid}|txt(n,P,768),{sid}|bbox,{sid}|act, _phrases)"""
import glob, sys
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import precompute_v6_florence2_fused as P
from transformers import AutoModelForCausalLM, AutoProcessor
files = sorted(glob.glob(str(ROOT / "docs/inference_sessions/session_20261113_*.h5")))
ins = sorted({h5py.File(f).attrs["instruction"] for f in files}); PHR = ["go to the gray basket"] + [x for x in ins if x != "go to the gray basket"]
print(PHR, len(files), flush=True)
model = AutoModelForCausalLM.from_pretrained(P.MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(P.DEV).eval()
proc = AutoProcessor.from_pretrained(P.MODEL, trust_remote_code=True)
tok = [proc(text=[P.TASK + ph], images=[Image.new("RGB", (64, 64))], return_tensors="pt")["input_ids"].to(P.DEV) for ph in PHR]
out = {"_phrases": np.array(PHR)}
with torch.no_grad():
    for f in files:
        sid = Path(f).stem.replace("session_", "")
        with h5py.File(f, "r") as h: imgs = h["observations/images"][:]; out[f"{sid}|bbox"] = h["grounding/bbox"][:].astype(np.float32); out[f"{sid}|act"] = h["actions"][:].astype(np.float32)
        pils = [Image.fromarray(np.ascontiguousarray(i).astype(np.uint8)).convert("RGB") for i in imgs]
        pv = proc(text=[P.TASK + PHR[0]] * len(pils), images=pils, return_tensors="pt")["pixel_values"].to(P.DEV, torch.float16); img = model._encode_image(pv)
        out[f"{sid}|vis"] = img.float().mean(1).cpu().numpy().astype(np.float16); t = []
        for k in range(len(PHR)):
            emb = model.get_input_embeddings()(tok[k].expand(len(pils), -1)); emb, att = model._merge_input_ids_with_image_features(img, emb)
            t.append(model.language_model.get_encoder()(inputs_embeds=emb, attention_mask=att).last_hidden_state.float()[:, img.shape[1]:].mean(1).cpu().numpy())
        out[f"{sid}|txt"] = np.stack(t, 1).astype(np.float16)
np.savez(ROOT / "runs/egvla2_fm/t13_obj_cache.npz", **out); print("saved", flush=True)
