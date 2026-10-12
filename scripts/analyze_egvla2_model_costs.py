#!/usr/bin/env python3
"""EGVLA2 — 구성 A/B/C가 왜 이 지연을 내는지 설명하는 정적 분석(파라미터·토큰·GFLOPs·메모리).

  A: OWLv2(google/owlv2-base-patch16-ensemble, 960px) + Kosmos-2 vision_model(224px) + image_proj (프로덕션)
  B/C: Florence-2-base (DaViT 비전 + 인코더-디코더 LM, 768px)
GFLOPs = torch.utils.flop_counter.FlopCounterMode(2×MAC) fp16 단일 프레임. 지연이 아니라 '계산량' 근거이므로
GPU 점유 여부와 무관. 출력: docs/v5/detector/egvla2_model_cost_breakdown.json
실행: HF_HUB_OFFLINE=1 .venv/bin/python3 scripts/analyze_egvla2_model_costs.py
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.flop_counter import FlopCounterMode
from PIL import Image

os.environ.setdefault("VLA_OWLV2_FP16", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
DEV = torch.device("cuda")
FRAME = ROOT / "docs/v5/egvla2_florence2/bench_frame.jpg"
pil = Image.open(FRAME).convert("RGB"); W, H = pil.size
M = lambda n: round(n / 1e6, 2)
cnt = lambda mod: mod.numel() if isinstance(mod, torch.nn.Parameter) else sum(p.numel() for p in mod.parameters())


def gflops(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return round(fc.get_total_flops() / 1e9, 1)


def mem_peak(fn):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); fn(); torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


out = {}
# ---------------- OWLv2 ----------------
from transformers import Owlv2Processor, Owlv2ForObjectDetection
base0 = torch.cuda.memory_allocated()
op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
om = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", torch_dtype=torch.float16).to(DEV).eval()
w_owl = torch.cuda.memory_allocated() - base0
inp = op(text=[["gray basket"]], images=pil, return_tensors="pt").to(DEV)
inp["pixel_values"] = inp["pixel_values"].half()
with torch.no_grad():
    run = lambda: om(**inp)
    out["owlv2"] = dict(
        params_M=dict(total=M(cnt(om)), vision_tower=M(cnt(om.owlv2.vision_model)), text_tower=M(cnt(om.owlv2.text_model)),
                      class_head=M(cnt(om.class_head)), box_head=M(cnt(om.box_head)), objectness_head=M(cnt(om.objectness_head))),
        vision=dict(layers=om.config.vision_config.num_hidden_layers, hidden=om.config.vision_config.hidden_size,
                    heads=om.config.vision_config.num_attention_heads, input_px=960, patch=om.config.vision_config.patch_size,
                    tokens=(960 // om.config.vision_config.patch_size) ** 2 + 1, attention="전역 self-attention (O(N²))"),
        gflops_forward=gflops(run), gflops_vision_only=gflops(lambda: om.owlv2.vision_model(pixel_values=inp["pixel_values"])),
        weights_MB=round(w_owl / 2**20, 1), peak_MB=mem_peak(run))
del om; torch.cuda.empty_cache()

# ---------------- Kosmos-2 vision (프로덕션 Stage1Encoder 경로) ----------------
from robovlm_nav.serve.stage2_v2_inference_server import _load_kosmos2_vision_only, DEFAULT_VLM, resize_for_vlm
from transformers import AutoProcessor
base0 = torch.cuda.memory_allocated()
kv = _load_kosmos2_vision_only(Path(str(DEFAULT_VLM))).to(DEV).eval()
w_k = torch.cuda.memory_allocated() - base0
kp = AutoProcessor.from_pretrained(str(DEFAULT_VLM))
pv = kp(images=resize_for_vlm(pil), return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
with torch.no_grad():
    run = lambda: kv(pixel_values=pv)
    cfg = kv.config
    out["kosmos2_vision"] = dict(
        params_M=dict(vision_model=M(cnt(kv)), image_proj_256=round((1024 * 256 + 256) / 1e6, 3)),
        vision=dict(layers=cfg.num_hidden_layers, hidden=cfg.hidden_size, heads=cfg.num_attention_heads,
                    input_px=int(pv.shape[-1]), patch=cfg.patch_size, tokens=(int(pv.shape[-1]) // cfg.patch_size) ** 2 + 1,
                    attention="전역 self-attention (CLIP ViT-L)"),
        gflops_forward=gflops(run), weights_MB=round(w_k / 2**20, 1), peak_MB=mem_peak(run))
del kv; torch.cuda.empty_cache()

# ---------------- Florence-2 ----------------
from transformers import AutoModelForCausalLM
from florence2_bbox_adapter import florence2_output_to_bbox
base0 = torch.cuda.memory_allocated()
fm = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
fp = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
w_f = torch.cuda.memory_allocated() - base0
fin = fp(text="<CAPTION_TO_PHRASE_GROUNDING>gray basket", images=pil, return_tensors="pt")
ids, fpv = fin["input_ids"].to(DEV), fin["pixel_values"].to(DEV, torch.float16)
lm = fm.language_model
with torch.no_grad():
    feats = fm._encode_image(fpv)
    emb = fm.get_input_embeddings()(ids)
    emb, _ = fm._merge_input_ids_with_image_features(feats, emb)
    gen = lambda: lm.generate(input_ids=None, inputs_embeds=emb, max_new_tokens=128, num_beams=3, do_sample=False)
    g = gen(); n_tok = int(g.shape[1])
    enc_len = int(emb.shape[1])
    g_enc = gflops(lambda: lm.get_encoder()(inputs_embeds=emb))
    g_vis = gflops(lambda: fm._encode_image(fpv))
    g_all = gflops(gen)
    run_c = lambda: lm.generate(input_ids=None, inputs_embeds=emb, max_new_tokens=128, num_beams=3, do_sample=False)
    ecfg, dcfg = lm.config.encoder_layers, lm.config.decoder_layers
    out["florence2_base"] = dict(
        params_M=dict(total=M(cnt(fm)), vision_tower_DaViT=M(cnt(fm.vision_tower)),
                      image_projection_and_pos=M(sum(cnt(getattr(fm, n)) for n in ("image_projection","image_proj_norm","image_pos_embed","visual_temporal_embed") if hasattr(fm, n))),
                      lm_encoder=M(cnt(lm.get_encoder())), lm_decoder=M(cnt(lm.get_decoder())), embeddings_shared=M(cnt(lm.get_input_embeddings()))),
        vision=dict(layers="DaViT 4 stage", input_px=768, tokens=int(feats.shape[1]), feat_dim=int(feats.shape[2]),
                    attention="윈도우+채널 attention (토큰 수에 선형)"),
        lm=dict(enc_layers=ecfg, dec_layers=dcfg, d_model=lm.config.d_model, enc_seq_len=enc_len, generated_tokens=n_tok, beams=3),
        gflops=dict(vision_encode=g_vis, lm_encoder=g_enc, generate_total_incl_encoder=g_all, decoder_steps_approx=round(g_all - g_enc, 1)),
        weights_MB=round(w_f / 2**20, 1), peak_MB=mem_peak(run_c))
p = Path(ROOT / "docs/v5/detector/egvla2_model_cost_breakdown.json")
p.write_text(json.dumps(out, indent=2, ensure_ascii=False)); print(json.dumps(out, indent=2, ensure_ascii=False))
