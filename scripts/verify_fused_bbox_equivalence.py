#!/usr/bin/env python3
"""B(분리 generate) vs C(fused) bbox 출력이 bit-exact로 동일한지 실제 V6 프레임 여러 장으로 검증."""
import sys, glob, h5py, numpy as np, torch
from PIL import Image
sys.path.insert(0, "scripts")
from florence2_bbox_adapter import florence2_output_to_bbox
from transformers import AutoModelForCausalLM, AutoProcessor

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float32
TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
PHRASE = "gray basket"

m = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=DT).to(DEV).eval()
p = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

def decode(out, W, H):
    t = p.batch_decode(out, skip_special_tokens=False)[0]
    parsed = p.post_process_generation(t, task=TASK, image_size=(W, H))[TASK]
    return florence2_output_to_bbox(parsed.get("bboxes", []), W, H)

@torch.no_grad()
def run_b(pil, W, H):
    inp = p(text=TASK + PHRASE, images=pil, return_tensors="pt")
    ids, pv = inp["input_ids"].to(DEV), inp["pixel_values"].to(DEV, DT)
    out = m.generate(input_ids=ids, pixel_values=pv, max_new_tokens=128, num_beams=3, do_sample=False)
    return decode(out, W, H)

@torch.no_grad()
def run_c(pil, W, H):
    inp = p(text=TASK + PHRASE, images=pil, return_tensors="pt")
    ids, pv = inp["input_ids"].to(DEV), inp["pixel_values"].to(DEV, DT)
    feats = m._encode_image(pv)
    emb = m.get_input_embeddings()(ids)
    emb, _ = m._merge_input_ids_with_image_features(feats, emb)
    out = m.language_model.generate(input_ids=None, inputs_embeds=emb, max_new_tokens=128, num_beams=3, do_sample=False)
    return decode(out, W, H)

files = sorted(glob.glob("ROS_action/mobile_vla_dataset_v5/episode_20260716_*.h5"))[:5]
n_match, n_total = 0, 0
for fp in files:
    with h5py.File(fp, "r") as f:
        imgs = f["images"]
        for idx in [imgs.shape[0]//4, imgs.shape[0]//2, 3*imgs.shape[0]//4]:
            arr = imgs[idx]
            pil = Image.fromarray(arr)
            W, H = pil.size
            b = run_b(pil, W, H)
            c = run_c(pil, W, H)
            match = (b == c)
            n_total += 1
            n_match += int(match)
            print(f"{fp.split('/')[-1]}[{idx}] match={match} B={b} C={c}", flush=True)

print(f"\n{n_match}/{n_total} bit-exact 일치")
