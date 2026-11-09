#!/usr/bin/env python3
"""T7 (1): A3 세션 프레임의 Florence-2 이미지 풀링(vis) + 문구별 융합 인코더 텍스트 평균(txt). 생성 없음. 세션 이미지는 이미 RGB.
출력 runs/egvla2_fm/t7_session_cache.npz: {name}|vis, {name}|txt(n,P,768), {name}|bbox, {name}|act, _phrases"""
import glob, json, os, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import precompute_v6_florence2_fused as P
import text_vocab as TV
from transformers import AutoModelForCausalLM, AutoProcessor
PHR = [TV.ALL[0]] + TV.HELD
OUT = ROOT / "runs/egvla2_fm/t7_session_cache.npz"; BATCH = 16


@torch.no_grad()
def main():
    model = AutoModelForCausalLM.from_pretrained(P.MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(P.DEV).eval()
    proc = AutoProcessor.from_pretrained(P.MODEL, trust_remote_code=True)
    tok = [proc(text=[P.TASK + ph], images=[Image.new("RGB", (64, 64))], return_tensors="pt")["input_ids"].to(P.DEV) for ph in PHR]
    files = []
    for f in sorted(glob.glob(str(ROOT / "docs/inference_sessions/session_*.h5"))):
        with h5py.File(f, "r") as h:
            rc = json.loads(h.attrs["runtime_config"]) if "runtime_config" in h.attrs else {}
        if rc.get("ckpt_name") == "C_pi0_servolabel1.0_drop30_e120.pt" and rc.get("servo_blend") == 0.0: files.append(f)
    if os.environ.get("SMOKE"): files = files[:2]
    print("세션", len(files), flush=True)
    cache, t0 = {}, time.time()
    for si, f in enumerate(files):
        name = Path(f).name
        with h5py.File(f, "r") as h:
            imgs = h["observations/images"][:]; cache[f"{name}|bbox"] = h["grounding/bbox"][:].astype(np.float32); cache[f"{name}|act"] = h["actions"][:].astype(np.float32)
        pils = [Image.fromarray(np.ascontiguousarray(im).astype(np.uint8)).convert("RGB") for im in imgs]
        vis, txt = [], [[] for _ in PHR]
        for s in range(0, len(pils), BATCH):
            b = pils[s:s + BATCH]
            pv = proc(text=[P.TASK + PHR[0]] * len(b), images=b, return_tensors="pt")["pixel_values"].to(P.DEV, torch.float16)
            img = model._encode_image(pv); vis.append(img.float().mean(1).cpu().numpy())
            for k in range(len(PHR)):
                emb = model.get_input_embeddings()(tok[k].expand(len(b), -1))
                emb, att = model._merge_input_ids_with_image_features(img, emb)
                enc = model.language_model.get_encoder()(inputs_embeds=emb, attention_mask=att).last_hidden_state.float()
                txt[k].append(enc[:, img.shape[1]:].mean(1).cpu().numpy())
        cache[f"{name}|vis"] = np.concatenate(vis).astype(np.float16)
        cache[f"{name}|txt"] = np.stack([np.concatenate(t) for t in txt], 1).astype(np.float16)
        if si % 10 == 0: print(f"[{si + 1}/{len(files)}] {(time.time() - t0) / 60:.1f}m", flush=True)
    cache["_phrases"] = np.array(PHR)
    np.savez(OUT if not os.environ.get("SMOKE") else OUT.with_name("t7_session_cache_smoke.npz"), **cache); print("saved", flush=True)


main()
