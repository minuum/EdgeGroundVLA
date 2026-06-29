"""
고수준 비전 레이어만 파인튜닝 ablation

방법: 비전 인코더 전체 frozen → 마지막 N개 블록만 unfreeze → cx 회귀 head 학습

비교:
  ft_last1 : 마지막 1 블록 + head
  ft_last2 : 마지막 2 블록 + head
  ft_last4 : 마지막 4 블록 + head
  [참고용] lora, lp_patch 도 같은 데이터로 재실행

Data:
  Train: V5 H5 (free 포함, 140개) frame 0, PG2 cx_det
  Eval:  V5 hold-out (80/20 × 5 seeds) + 6/26 inference session

Usage:
  .venv/bin/python3 -u scripts/ablate_last_layers.py
  .venv/bin/python3 -u scripts/ablate_last_layers.py --models clip kosmos --layers 1 2 4
"""

import argparse, gc, io, json, random, time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE      = "cuda"
PHRASE      = "gray basket"
ANN_FILE    = Path("docs/v5/bbox_frame_level/bbox_dataset_pg2_cx_freehsv.json")
DATA_DIR    = Path("ROS_action/mobile_vla_dataset_v5")
SESS_DIR    = Path("docs/inference_sessions")
OUT_FILE    = Path("docs/v5/ablate_last_layers.json")
N_SEEDS     = 5
TRAIN_RATIO = 0.8
EPOCHS      = 60       # last-layer FT는 수렴이 느릴 수 있어서 약간 더
LR          = 2e-5     # 낮은 LR + weight_decay로 오버피팅 방지
WD          = 1e-3
BATCH       = 4


# ── 데이터 ──────────────────────────────────────────────────────────────────

def _raw_to_pil(raw):
    if raw.ndim == 1:
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    return Image.fromarray(raw.astype(np.uint8)).convert("RGB")


def load_v5_frames():
    ann = json.load(open(ANN_FILE))
    frames = []
    for ep in ann:
        ep_path = DATA_DIR / Path(ep["episode"]).name
        if not ep_path.exists(): continue
        f0 = ep["frames"][0]
        if not f0.get("has_bbox", False): continue
        cx = float(f0.get("cx_det", f0.get("cx", 0.5)))
        with h5py.File(ep_path) as f:
            raw = f["observations"]["images"][0]
        frames.append({"img": _raw_to_pil(raw), "cx": cx})
    cxs = [f["cx"] for f in frames]
    n_l = sum(1 for c in cxs if c < 0.4)
    n_c = sum(1 for c in cxs if 0.4 <= c <= 0.6)
    n_r = sum(1 for c in cxs if c > 0.6)
    print(f"[V5] {len(frames)}개  L={n_l} C={n_c} R={n_r}  cx_mean={np.mean(cxs):.3f}")
    return frames


def load_session_frames():
    frames = []
    for sp in sorted(SESS_DIR.glob("session_20260626*.h5")):
        with h5py.File(sp) as f:
            if "grounding/bbox" not in f: continue
            bbox = f["grounding/bbox"][:]
            if len(bbox) < 2 or bbox[1, 3] < 0.5: continue
            cx = float(bbox[1, 0])
            try: raw = f["observations"]["images"][0]
            except: continue
        frames.append({"img": _raw_to_pil(raw), "cx": cx})
    cxs = [f["cx"] for f in frames]
    print(f"[sess] {len(frames)}개  FORWARD={sum(1 for c in cxs if c<=0.5)} ROT_R={sum(1 for c in cxs if c>0.5)}")
    return frames


def split_seeds(frames, n_seeds=N_SEEDS, ratio=TRAIN_RATIO):
    n = len(frames)
    out = []
    for seed in range(n_seeds):
        rng = random.Random(seed)
        idx = list(range(n)); rng.shuffle(idx)
        cut = int(n * ratio)
        out.append((idx[:cut], idx[cut:]))
    return out


def eval_preds(preds, frames):
    mae, dir_ok, det = [], [], []
    for p, fr in zip(preds, frames):
        det.append(p is not None)
        if p is not None:
            mae.append(abs(p - fr["cx"]))
            dir_ok.append((p > 0.5) == (fr["cx"] > 0.5))
    return {
        "det":     float(np.mean(det)),
        "cx_mae":  float(np.mean(mae))    if mae    else 1.0,
        "dir_bin": float(np.mean(dir_ok)) if dir_ok else 0.0,
    }


def _agg(lst):
    if not lst: return None
    return {k: float(np.mean([r[k] for r in lst])) for k in lst[0]}


def _pr(tag, rv5, rss=None):
    s = f"  {tag:35s} v5: dir={rv5['dir_bin']:.1%} mae={rv5['cx_mae']:.3f}"
    if rss: s += f"  |  sess: dir={rss['dir_bin']:.1%} mae={rss['cx_mae']:.3f}"
    print(s)


# ── 공통 FT 루프 ──────────────────────────────────────────────────────────────

def run_last_layers(fresh_vm_fn, feat_fn, feat_dim, n_unfreeze_blocks, get_blocks_fn,
                    pvs, frames, splits, sess_pvs, sess_frames, label):
    """
    fresh_vm_fn()  → 새 vision model (frozen, loaded once, state_dict reset per seed)
    feat_fn(vm, pv) → [B, feat_dim]
    get_blocks_fn(vm) → list of nn.Module (transformer blocks)
    n_unfreeze_blocks: 마지막 N개 블록만 unfreeze
    """
    tag = f"{label} ft_last{n_unfreeze_blocks}"
    print(f"  [{tag}] EPOCHS={EPOCHS} LR={LR} WD={WD} batch={BATCH}...")

    vm = fresh_vm_fn().to(DEVICE)
    # 초기 가중치 CPU 백업
    init_sd = {k: v.cpu().clone() for k, v in vm.state_dict().items()}

    v5_res, ss_res = [], []
    for s_i, (tr_idx, va_idx) in enumerate(splits):
        if s_i > 0:
            vm.load_state_dict(init_sd)

        # 전체 frozen
        for p in vm.parameters(): p.requires_grad_(False)

        # 마지막 N 블록 + post_layernorm unfreeze
        blocks = get_blocks_fn(vm)
        for blk in blocks[-n_unfreeze_blocks:]:
            for p in blk.parameters(): p.requires_grad_(True)
        # post layernorm (마지막 norm)
        for name, mod in vm.named_modules():
            if "post_layernorm" in name or "layernorm" in name.lower() and "post" in name.lower():
                for p in mod.parameters(): p.requires_grad_(True)

        trainable = sum(p.numel() for p in vm.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in vm.parameters())
        if s_i == 0:
            print(f"    trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({trainable/total*100:.1f}%)")

        head = nn.Linear(feat_dim, 1).to(DEVICE)
        opt  = torch.optim.AdamW(
            [p for p in vm.parameters() if p.requires_grad] + list(head.parameters()),
            lr=LR, weight_decay=WD)

        tr_pv = [pvs[i] for i in tr_idx]
        tr_cx = [frames[i]["cx"] for i in tr_idx]
        va_pv = [pvs[i] for i in va_idx]

        for _ in range(EPOCHS):
            vm.train(); head.train()
            perm = list(range(len(tr_idx))); random.shuffle(perm)
            for b in range(0, len(perm), BATCH):
                bi   = perm[b:b+BATCH]
                pv_b = torch.cat([tr_pv[i] for i in bi]).to(DEVICE)
                cx_b = torch.tensor([tr_cx[i] for i in bi], dtype=torch.float32, device=DEVICE).unsqueeze(1)
                opt.zero_grad()
                F.mse_loss(head(feat_fn(vm, pv_b)), cx_b).backward()
                opt.step()

        vm.eval(); head.eval()
        def _inf(pv_list):
            out = []
            with torch.no_grad():
                for b in range(0, len(pv_list), BATCH):
                    pv_b = torch.cat(pv_list[b:b+BATCH]).to(DEVICE)
                    out.extend(head(feat_fn(vm, pv_b)).squeeze(1).cpu().tolist())
            return out

        v5_res.append(eval_preds(_inf(va_pv), [frames[i] for i in va_idx]))
        if sess_frames and sess_pvs:
            ss_res.append(eval_preds(_inf(sess_pvs), sess_frames))
        print(f"    seed {s_i+1}/{N_SEEDS} v5_dir={v5_res[-1]['dir_bin']:.1%}")
        del head, opt; gc.collect(); torch.cuda.empty_cache()
        
    del vm; gc.collect(); torch.cuda.empty_cache()

    return _agg(v5_res), _agg(ss_res) if ss_res else None


# ── CLIP ─────────────────────────────────────────────────────────────────────

def run_clip(frames, splits, sess_frames, n_layers_list):
    from transformers import CLIPModel, CLIPProcessor
    print("\n[CLIP] 로딩 + pixel_values 추출...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    pvs_v5 = [proc(images=fr["img"], return_tensors="pt")["pixel_values"].cpu() for fr in frames]
    sess_pvs = [proc(images=fr["img"], return_tensors="pt")["pixel_values"].cpu() for fr in sess_frames] if sess_frames else None
    del model; torch.cuda.empty_cache()

    results = {}
    feat_dim = 1024

    def fresh_vm():
        return CLIPModel.from_pretrained("openai/clip-vit-large-patch14").vision_model

    def feat_fn(vm, pv):
        return vm(pixel_values=pv).last_hidden_state[:,1:,:].mean(1).float()

    def get_blocks(vm):
        return list(vm.encoder.layers)

    for n in n_layers_list:
        rv5, rss = run_last_layers(fresh_vm, feat_fn, feat_dim, n, get_blocks,
                                   pvs_v5, frames, splits, sess_pvs, sess_frames, "CLIP")
        results[f"ft_last{n}"] = {"v5": rv5, "session": rss}
        _pr(f"[CLIP ft_last{n}]", rv5, rss)
    return results


# ── Kosmos-2 ─────────────────────────────────────────────────────────────────

def run_kosmos(frames, splits, sess_frames, n_layers_list):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    LOCAL = Path(".vlms/kosmos-2-patch14-224")
    print("\n[Kosmos-2] 로딩 + pixel_values 추출...")
    model = AutoModelForVision2Seq.from_pretrained(str(LOCAL)).to(DEVICE)
    proc  = AutoProcessor.from_pretrained(str(LOCAL))

    pvs_v5 = [proc(text="<grounding>", images=fr["img"], return_tensors="pt")["pixel_values"].cpu() for fr in frames]
    sess_pvs = [proc(text="<grounding>", images=fr["img"], return_tensors="pt")["pixel_values"].cpu() for fr in sess_frames] if sess_frames else None
    del model; torch.cuda.empty_cache()

    results = {}
    feat_dim = 1024

    def fresh_vm():
        return AutoModelForVision2Seq.from_pretrained(str(LOCAL)).vision_model

    def feat_fn(vm, pv):
        return vm(pixel_values=pv).last_hidden_state[:,1:,:].mean(1).float()

    def get_blocks(vm):
        return list(vm.model.encoder.layers)

    for n in n_layers_list:
        rv5, rss = run_last_layers(fresh_vm, feat_fn, feat_dim, n, get_blocks,
                                   pvs_v5, frames, splits, sess_pvs, sess_frames, "Kosmos-2")
        results[f"ft_last{n}"] = {"v5": rv5, "session": rss}
        _pr(f"[Kosmos-2 ft_last{n}]", rv5, rss)
    return results


# ── OWL-v2 ───────────────────────────────────────────────────────────────────

def run_owlv2(frames, splits, sess_frames, n_layers_list):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    print("\n[OWL-v2] 로딩 + pixel_values 추출...")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEVICE)
    proc  = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")

    pvs_v5 = [proc(text=[[PHRASE]], images=fr["img"], return_tensors="pt")["pixel_values"].cpu() for fr in frames]
    sess_pvs = [proc(text=[[PHRASE]], images=fr["img"], return_tensors="pt")["pixel_values"].cpu() for fr in sess_frames] if sess_frames else None
    del model; torch.cuda.empty_cache()

    results = {}

    def fresh_vm():
        return Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").owlv2.vision_model

    def feat_fn(vm, pv):
        return vm(pixel_values=pv).last_hidden_state[:,1:,:].mean(1).float()

    def get_blocks(vm):
        return list(vm.encoder.layers)

    # OWL-v2 feat_dim 확인
    tmp = fresh_vm().to(DEVICE)
    pv_tmp = pvs_v5[0].to(DEVICE)
    with torch.no_grad(): feat_dim = feat_fn(tmp, pv_tmp).shape[1]
    del tmp; torch.cuda.empty_cache()
    print(f"  OWL-v2 feat_dim={feat_dim}")

    for n in n_layers_list:
        rv5, rss = run_last_layers(fresh_vm, feat_fn, feat_dim, n, get_blocks,
                                   pvs_v5, frames, splits, sess_pvs, sess_frames, "OWL-v2")
        results[f"ft_last{n}"] = {"v5": rv5, "session": rss}
        _pr(f"[OWL-v2 ft_last{n}]", rv5, rss)
    return results


# ── Florence-2 ───────────────────────────────────────────────────────────────

def run_florence(frames, splits, sess_frames, n_layers_list):
    from transformers import AutoProcessor, AutoModelForCausalLM
    print("\n[Florence-2] 로딩 + pixel_values 추출...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(DEVICE)
    proc  = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

    pvs_v5 = [proc(text="<MORE_DETAILED_CAPTION>", images=fr["img"], return_tensors="pt")["pixel_values"].cpu() for fr in frames]
    sess_pvs = [proc(text="<MORE_DETAILED_CAPTION>", images=fr["img"], return_tensors="pt")["pixel_values"].cpu() for fr in sess_frames] if sess_frames else None
    del model; torch.cuda.empty_cache()

    results = {}

    def fresh_vm():
        return AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).vision_tower

    def feat_fn(vm, pv):
        return vm(pv)[:,1:,:].mean(1).float()

    # Florence-2 블록 구조 확인
    tmp_vm = fresh_vm()
    blocks_found = None
    for attr in ["encoder", "model", "layers"]:
        sub = getattr(tmp_vm, attr, None)
        if sub is not None and hasattr(sub, "layers"):
            blocks_found = lambda vm, _attr=attr: list(getattr(getattr(vm, _attr), "layers"))
            print(f"  Florence-2 blocks via .{attr}.layers ({len(list(sub.layers))}개)")
            break
    if blocks_found is None:
        # DaViT 등 다른 구조
        named = [(n,m) for n,m in tmp_vm.named_modules() if "stage" in n.lower() or "block" in n.lower()]
        print(f"  Florence-2 block-like modules: {[n for n,_ in named[:8]]}")
        blocks_found = lambda vm: [blk for stage in vm.stages for blk in stage.blocks]
    del tmp_vm

    feat_dim_tmp = fresh_vm().to(DEVICE)
    with torch.no_grad(): feat_dim = feat_fn(feat_dim_tmp, pvs_v5[0].to(DEVICE)).shape[1]
    del feat_dim_tmp; torch.cuda.empty_cache()
    print(f"  Florence-2 feat_dim={feat_dim}")

    for n in n_layers_list:
        rv5, rss = run_last_layers(fresh_vm, feat_fn, feat_dim, n, blocks_found,
                                   pvs_v5, frames, splits, sess_pvs, sess_frames, "Florence-2")
        results[f"ft_last{n}"] = {"v5": rv5, "session": rss}
        _pr(f"[Florence-2 ft_last{n}]", rv5, rss)
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["clip","kosmos","owlv2","florence"])
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 4])
    args = parser.parse_args()

    v5_frames   = load_v5_frames()
    sess_frames = load_session_frames()
    splits      = split_seeds(v5_frames)

    baseline_v5 = eval_preds([0.5]*len(v5_frames), v5_frames)
    baseline_ss = eval_preds([0.5]*len(sess_frames), sess_frames)
    print(f"[baseline] v5: dir={baseline_v5['dir_bin']:.1%}  sess: dir={baseline_ss['dir_bin']:.1%}\n")

    all_results = {"_baseline": {"v5": baseline_v5, "session": baseline_ss}}
    runners = {"clip": run_clip, "kosmos": run_kosmos, "owlv2": run_owlv2, "florence": run_florence}

    for name in args.models:
        if name not in runners: continue
        try:
            all_results[name] = runners[name](v5_frames, splits, sess_frames, args.layers)
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results[name] = {"error": str(e)}
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUT_FILE.write_text(json.dumps(all_results, indent=2))
        print(f"  → 저장: {OUT_FILE}")

    print("\n========== 요약 ==========")
    print(f"baseline: v5 dir={baseline_v5['dir_bin']:.1%}  sess dir={baseline_ss['dir_bin']:.1%}")
    print(f"{'모델':10s} {'방법':15s} {'v5_dir':>8s} {'v5_mae':>7s} {'ss_dir':>8s} {'ss_mae':>7s}")
    print("─"*58)
    for model, mres in all_results.items():
        if model.startswith("_") or not isinstance(mres, dict) or "error" in mres: continue
        for method, r in mres.items():
            if not isinstance(r, dict): continue
            rv5 = r.get("v5", {}); rss = r.get("session") or {}
            if "dir_bin" not in rv5: continue
            print(f"{model:10s} {method:15s} {rv5['dir_bin']:>7.1%}  {rv5['cx_mae']:>6.3f}"
                  f"  {rss.get('dir_bin',0):>7.1%}  {rss.get('cx_mae',0):>6.3f}")
    print(f"\n저장: {OUT_FILE}")

if __name__ == "__main__":
    main()
