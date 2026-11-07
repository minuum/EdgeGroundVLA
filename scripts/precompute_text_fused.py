#!/usr/bin/env python3
"""T1/T2용 문구별 인코더 특징 캐시. 프레임마다 Florence-2 이미지 인코더는 1회, 문구마다 융합(BART) 인코더만 통과(디코더 생성 없음).
저장(fp16): {ep}|vis(ceil(n/3),768)=이미지 인코더 풀링(문구 무관), {ep}|txt|{k}(n,768)=융합 인코더 은닉의 텍스트 토큰 평균, {ep}|fimg|{k}=융합 은닉의 이미지 토큰 평균.
입력 규칙은 precompute_egvla2_continuous_features와 같다(데이터셋 H5는 BGR → RGB 뒤집기). plan: docs/plans/plan_20261106_text_conditioned_e2e_vla.md §2 T1~T2
실행: .venv/bin/python3 scripts/precompute_text_fused.py   (SMOKE=1: 2에피소드)"""
import json, os, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import precompute_v6_florence2_fused as P
from transformers import AutoModelForCausalLM, AutoProcessor

POS = ["gray basket", "gray bin", "gray container", "the gray basket", "go to the gray bin", "basket"]          # 0=기준, 1~5=패러프레이즈
NEG = ["red chair", "person", "blue bottle"]                                                                     # 6~8: 장면에 없는 물체(음성 지시)
PHR = POS + NEG
OUT = ROOT / "runs/egvla2_fm/text_fused_cache.npz"
BATCH = 16
STRIDE = 3   # 계산 시간(0.4초/프레임) 때문에 3프레임마다 1개만. 같은 부분집합으로 모든 변형을 비교한다(frames=0,3,6,..)


@torch.no_grad()
def main():
    model = AutoModelForCausalLM.from_pretrained(P.MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(P.DEV).eval()
    proc = AutoProcessor.from_pretrained(P.MODEL, trust_remote_code=True)
    tok = []
    for ph in PHR:
        t = proc(text=[P.TASK + ph], images=[Image.new("RGB", (64, 64))], return_tensors="pt")["input_ids"].to(P.DEV)
        tok.append(t)
    meta = json.loads((ROOT / "docs/v5/egvla2_fm/meta/egvla2_v3_main.json").read_text())["files"]
    if os.environ.get("SMOKE"): meta = meta[:2]
    ddir = ROOT / "ROS_action/mobile_vla_dataset_egvla2"
    cache, t0 = {}, time.time()
    for ei, d in enumerate(meta):
        name = d["name"]
        imgs = P.load_images(ddir / name)[::STRIDE]
        pils = [Image.fromarray(np.ascontiguousarray(im[:, :, ::-1]).astype(np.uint8)).convert("RGB") for im in imgs]
        vis, txt, fimg = [], [[] for _ in PHR], [[] for _ in PHR]
        for s in range(0, len(pils), BATCH):
            b = pils[s:s + BATCH]
            pv = proc(text=[P.TASK + PHR[0]] * len(b), images=b, return_tensors="pt")["pixel_values"].to(P.DEV, torch.float16)
            img = model._encode_image(pv)
            vis.append(img.float().mean(1).cpu().numpy())
            for k in range(len(PHR)):
                emb = model.get_input_embeddings()(tok[k].expand(len(b), -1))
                emb, att = model._merge_input_ids_with_image_features(img, emb)
                enc = model.language_model.get_encoder()(inputs_embeds=emb, attention_mask=att).last_hidden_state.float()
                ni = img.shape[1]
                fimg[k].append(enc[:, :ni].mean(1).cpu().numpy()); txt[k].append(enc[:, ni:].mean(1).cpu().numpy())
        cache[f"{name}|vis"] = np.concatenate(vis).astype(np.float16)
        for k in range(len(PHR)):
            cache[f"{name}|txt|{k}"] = np.concatenate(txt[k]).astype(np.float16)
            cache[f"{name}|fimg|{k}"] = np.concatenate(fimg[k]).astype(np.float16)
        print(f"[{ei + 1}/{len(meta)}] {name[-30:]} {len(pils)}f {(time.time() - t0) / 60:.1f}m", flush=True)
    cache["_phrases"] = np.array(PHR)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT if not os.environ.get("SMOKE") else OUT.with_name("text_fused_cache_smoke.npz"), **cache)
    print("saved", OUT, flush=True)


main()
