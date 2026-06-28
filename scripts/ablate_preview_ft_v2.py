"""
Preview Model Ablation v2 — FT 방법별 비교

Train : V5 H5 전체 (free 포함, ~140ep) frame 0, PG2 cx_det 라벨
Eval  : ① V5 hold-out (80/20 split) ② 6/26 inference session (실운영)

Models : CLIP / Kosmos-2 / OWL-v2 / Florence-2
Methods: zs / lp_pool / lp_patch / mlp_patch / lora

Usage:
  .venv/bin/python3 -u scripts/ablate_preview_ft_v2.py
  .venv/bin/python3 -u scripts/ablate_preview_ft_v2.py --models kosmos --methods zs lp_patch lora
"""

import argparse, io, json, random, time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE       = "cuda"
PHRASE       = "gray basket"
ANN_FILE     = Path("docs/v5/bbox_frame_level/bbox_dataset_pg2_cx_freehsv.json")
DATA_DIR     = Path("ROS_action/mobile_vla_dataset_v5")
SESS_DIR     = Path("docs/inference_sessions")
OUT_FILE     = Path("docs/v5/ablate_preview_ft_v2.json")
N_SEEDS      = 5
TRAIN_RATIO  = 0.8
EPOCHS       = 40
LR_PROBE     = 3e-3
LORA_LR      = 5e-4
LORA_RANK    = 8
LORA_BATCH   = 4


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
        frames.append({"img": _raw_to_pil(raw), "cx": cx, "path_type": ep.get("path_type","?")})

    cxs = [f["cx"] for f in frames]
    n_l = sum(1 for c in cxs if c < 0.4)
    n_c = sum(1 for c in cxs if 0.4 <= c <= 0.6)
    n_r = sum(1 for c in cxs if c > 0.6)
    print(f"[train data] V5 {len(frames)}개  L={n_l} C={n_c} R={n_r}  cx_mean={np.mean(cxs):.3f}")
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
        frames.append({"img": _raw_to_pil(raw), "cx": cx, "session": sp.stem})
    cxs = [f["cx"] for f in frames]
    n_f = sum(1 for c in cxs if c <= 0.5)
    n_r = sum(1 for c in cxs if c > 0.5)
    print(f"[eval/session] 6/26 {len(frames)}개  FORWARD(≤0.5)={n_f} ROT_R(>0.5)={n_r}  cx_mean={np.mean(cxs):.3f}")
    return frames


def split_seeds(frames, n_seeds, ratio):
    n = len(frames)
    seeds = []
    for seed in range(n_seeds):
        rng = random.Random(seed)
        idx = list(range(n)); rng.shuffle(idx)
        cut = int(n * ratio)
        seeds.append((idx[:cut], idx[cut:]))
    return seeds


def eval_preds(preds_cx, frames):
    mae, dir_ok, det = [], [], []
    for pcx, fr in zip(preds_cx, frames):
        det.append(pcx is not None)
        if pcx is not None:
            mae.append(abs(pcx - fr["cx"]))
            dir_ok.append((pcx > 0.5) == (fr["cx"] > 0.5))
    return {
        "det":     float(np.mean(det)),
        "cx_mae":  float(np.mean(mae))    if mae    else 1.0,
        "dir_bin": float(np.mean(dir_ok)) if dir_ok else 0.0,
    }


def _agg(res_list):
    if not res_list: return None
    return {k: float(np.mean([r[k] for r in res_list])) for k in res_list[0]}


def _print_r(tag, r_v5, r_ss=None):
    s = f"  {tag:32s} v5: dir={r_v5['dir_bin']:.1%} cx_mae={r_v5['cx_mae']:.3f}"
    if r_ss: s += f"  |  sess: dir={r_ss['dir_bin']:.1%} cx_mae={r_ss['cx_mae']:.3f}"
    print(s)


# ── Frozen probe ─────────────────────────────────────────────────────────────

def _train_probe(X_tr, cx_tr, X_va, frames_va, hidden=None):
    d = X_tr.shape[1]
    head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(DEVICE) \
           if hidden else nn.Linear(d, 1).to(DEVICE)
    opt  = torch.optim.Adam(head.parameters(), lr=LR_PROBE)
    Xt   = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    Yt   = torch.tensor(cx_tr, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    Xv   = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    for _ in range(EPOCHS):
        head.train(); opt.zero_grad()
        F.mse_loss(head(Xt), Yt).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        preds = head(Xv).squeeze(1).cpu().numpy().tolist()
    return eval_preds(preds, frames_va)


def _pred_probe(X_tr, cx_tr, X_te, hidden=None):
    d = X_tr.shape[1]
    head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(DEVICE) \
           if hidden else nn.Linear(d, 1).to(DEVICE)
    opt  = torch.optim.Adam(head.parameters(), lr=LR_PROBE)
    Xt   = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    Yt   = torch.tensor(cx_tr, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    for _ in range(EPOCHS):
        head.train(); opt.zero_grad()
        F.mse_loss(head(Xt), Yt).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        return head(torch.tensor(X_te, dtype=torch.float32, device=DEVICE)).squeeze(1).cpu().numpy().tolist()


def run_probe(feats, frames, splits, sess_frames, sess_feats, hidden=None):
    v5_res, ss_res = [], []
    for tr_idx, va_idx in splits:
        cx_tr = [frames[i]["cx"] for i in tr_idx]
        v5_res.append(_train_probe(feats[tr_idx], cx_tr, feats[va_idx], [frames[i] for i in va_idx], hidden))
        if sess_feats is not None:
            ss_res.append(eval_preds(_pred_probe(feats[tr_idx], cx_tr, sess_feats, hidden), sess_frames))
    return _agg(v5_res), _agg(ss_res) if ss_res else None


# ── LoRA batched ──────────────────────────────────────────────────────────────

def run_lora_batched(fresh_vm_fn, feat_fn, feat_dim,
                     pvs, frames, splits, sess_pvs, sess_frames, label):
    """
    fresh_vm_fn() → LoRA-wrapped vision model (loaded fresh each seed).
    feat_fn(vm, pv_batch) → [B, feat_dim] float tensor.
    pvs: list of [1, C, H, W] CPU tensors.
    """
    print(f"  [{label} lora] LoRA batch={LORA_BATCH}×{EPOCHS}ep×{N_SEEDS}seeds...")
    v5_res, ss_res = [], []
    for s_i, (tr_idx, va_idx) in enumerate(splits):
        vm   = fresh_vm_fn().to(DEVICE)
        head = nn.Linear(feat_dim, 1).to(DEVICE)
        opt  = torch.optim.Adam(list(vm.parameters()) + list(head.parameters()), lr=LORA_LR)
        tr_pv = [pvs[i] for i in tr_idx]
        tr_cx = [frames[i]["cx"] for i in tr_idx]
        va_pv = [pvs[i] for i in va_idx]

        for _ in range(EPOCHS):
            vm.train(); head.train()
            perm = list(range(len(tr_idx))); random.shuffle(perm)
            for b in range(0, len(perm), LORA_BATCH):
                bi   = perm[b:b+LORA_BATCH]
                pv_b = torch.cat([tr_pv[i] for i in bi]).to(DEVICE)
                cx_b = torch.tensor([tr_cx[i] for i in bi], dtype=torch.float32, device=DEVICE).unsqueeze(1)
                opt.zero_grad()
                F.mse_loss(head(feat_fn(vm, pv_b)), cx_b).backward()
                opt.step()

        vm.eval(); head.eval()
        def _inf(pv_list):
            out = []
            with torch.no_grad():
                for b in range(0, len(pv_list), LORA_BATCH):
                    pv_b = torch.cat(pv_list[b:b+LORA_BATCH]).to(DEVICE)
                    out.extend(head(feat_fn(vm, pv_b)).squeeze(1).cpu().tolist())
            return out

        v5_res.append(eval_preds(_inf(va_pv), [frames[i] for i in va_idx]))
        if sess_frames and sess_pvs: ss_res.append(eval_preds(_inf(sess_pvs), sess_frames))
        print(f"    seed {s_i+1}/{N_SEEDS} v5_dir={v5_res[-1]['dir_bin']:.1%}")
        del vm; import gc; gc.collect(); torch.cuda.empty_cache()

    return _agg(v5_res), _agg(ss_res) if ss_res else None


def _pv_list(proc_fn, imgs):
    """이미지 리스트 → pixel_values CPU 텐서 리스트."""
    return [proc_fn(img).cpu() for img in imgs]


# ── CLIP ────────────────────────────────────────────────────────────────────

def run_clip(frames, splits, sess_frames, methods):
    from transformers import CLIPProcessor, CLIPModel
    print("\n[CLIP] 로딩...")
    t0 = time.time()
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print(f"[CLIP] {time.time()-t0:.1f}s")
    results = {}

    if "zs" in methods:
        texts = ["gray basket on the left", "gray basket in the center", "gray basket on the right"]
        def _zs_predict(fr_list):
            preds = []
            for fr in fr_list:
                inp = proc(text=texts, images=fr["img"], return_tensors="pt", padding=True).to(DEVICE)
                with torch.no_grad():
                    p = model(**inp).logits_per_image[0].softmax(0).cpu().numpy()
                preds.append(float(0.2*p[0] + 0.5*p[1] + 0.8*p[2]))
            return preds
        t0 = time.time()
        r_v5 = eval_preds(_zs_predict(frames), frames)
        r_v5["lat"] = (time.time()-t0)*1000/len(frames)
        r_ss = eval_preds(_zs_predict(sess_frames), sess_frames) if sess_frames else None
        results["zs"] = {"v5": r_v5, "session": r_ss}
        _print_r("[CLIP zs]", r_v5, r_ss)
        print(f"    lat={r_v5['lat']:.0f}ms/frame")

    needs_feat = any(m in methods for m in ["lp_pool","lp_patch","mlp_patch","lora"])
    if needs_feat:
        print("  feature 추출 (V5)...")
        cls_f, patch_f, pvs_v5 = [], [], []
        for fr in frames:
            pv = proc(images=fr["img"], return_tensors="pt")["pixel_values"]
            pvs_v5.append(pv.cpu())
            with torch.no_grad(): out = model.vision_model(pixel_values=pv.to(DEVICE))
            cls_f.append(out.last_hidden_state[:,0,:].squeeze(0).float().cpu().numpy())
            patch_f.append(out.last_hidden_state[:,1:,:].mean(1).squeeze(0).float().cpu().numpy())
        cls_f = np.array(cls_f); patch_f = np.array(patch_f)

        sess_cls_f = sess_patch_f = sess_pvs = None
        if sess_frames:
            print("  feature 추출 (session)...")
            sc, sp, sess_pvs = [], [], []
            for fr in sess_frames:
                pv = proc(images=fr["img"], return_tensors="pt")["pixel_values"]
                sess_pvs.append(pv.cpu())
                with torch.no_grad(): out = model.vision_model(pixel_values=pv.to(DEVICE))
                sc.append(out.last_hidden_state[:,0,:].squeeze(0).float().cpu().numpy())
                sp.append(out.last_hidden_state[:,1:,:].mean(1).squeeze(0).float().cpu().numpy())
            sess_cls_f = np.array(sc); sess_patch_f = np.array(sp)

    for tag, feats, sf in [("lp_pool",cls_f,sess_cls_f),("lp_patch",patch_f,sess_patch_f),
                            ("mlp_patch",patch_f,sess_patch_f)]:
        if tag not in methods: continue
        rv5, rss = run_probe(feats, frames, splits, sess_frames, sf, 256 if tag=="mlp_patch" else None)
        results[tag] = {"v5": rv5, "session": rss}
        _print_r(f"[CLIP {tag}]", rv5, rss)

    if "lora" in methods:
        from peft import get_peft_model, LoraConfig
        def fresh():
            vm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").vision_model
            return get_peft_model(vm, LoraConfig(r=LORA_RANK, lora_alpha=16,
                    target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none"))
        def feat_fn(vm, pv): return vm(pixel_values=pv).last_hidden_state[:,1:,:].mean(1).float()
        rv5, rss = run_lora_batched(fresh, feat_fn, 1024, pvs_v5, frames, splits, sess_pvs, sess_frames, "CLIP")
        results["lora"] = {"v5": rv5, "session": rss}
        _print_r("[CLIP lora]", rv5, rss)
    return results


# ── Kosmos-2 ─────────────────────────────────────────────────────────────────

def run_kosmos(frames, splits, sess_frames, methods):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    LOCAL = Path(".vlms/kosmos-2-patch14-224")
    print("\n[Kosmos-2] 로딩...")
    t0 = time.time()
    model = AutoModelForVision2Seq.from_pretrained(str(LOCAL)).to(DEVICE)
    proc  = AutoProcessor.from_pretrained(str(LOCAL))
    print(f"[Kosmos-2] {time.time()-t0:.1f}s")
    results = {}

    if "zs" in methods:
        prompt = "<grounding><phrase>gray basket</phrase>"
        def _zs(fr_list):
            preds = []
            for fr in fr_list:
                inp = proc(text=prompt, images=fr["img"], return_tensors="pt").to(DEVICE)
                with torch.no_grad(): ids = model.generate(**inp, max_new_tokens=64)
                out = proc.decode(ids[0], skip_special_tokens=False)
                result = proc.post_process_generation(out, cleanup_and_extract=True)
                ents = result[1] if isinstance(result, tuple) and len(result)>1 else []
                cx = None
                for ent in ents:
                    bbs = ent[2] if len(ent)>=3 else []
                    if bbs: cx = (bbs[0][0]+bbs[0][2])/2; break
                preds.append(cx)
            return preds
        t0 = time.time()
        r_v5 = eval_preds(_zs(frames), frames)
        r_v5["lat"] = (time.time()-t0)*1000/len(frames)
        r_ss = eval_preds(_zs(sess_frames), sess_frames) if sess_frames else None
        results["zs"] = {"v5": r_v5, "session": r_ss}
        _print_r("[Kosmos-2 zs]", r_v5, r_ss)
        print(f"    det={r_v5['det']:.1%} lat={r_v5['lat']:.0f}ms/frame")

    needs_feat = any(m in methods for m in ["lp_pool","lp_patch","mlp_patch","lora"])
    if needs_feat:
        print("  feature 추출 (V5)...")
        pool_f, patch_f, pvs_v5 = [], [], []
        for fr in frames:
            pv = proc(text="<grounding>", images=fr["img"], return_tensors="pt")["pixel_values"]
            pvs_v5.append(pv.cpu())
            with torch.no_grad(): out = model.vision_model(pixel_values=pv.to(DEVICE))
            pool_f.append(out.pooler_output.squeeze(0).float().cpu().numpy())
            patch_f.append(out.last_hidden_state[:,1:,:].mean(1).squeeze(0).float().cpu().numpy())
        pool_f = np.array(pool_f); patch_f = np.array(patch_f)

        sess_pool_f = sess_patch_f = sess_pvs = None
        if sess_frames:
            print("  feature 추출 (session)...")
            sp_, pp_, sess_pvs = [], [], []
            for fr in sess_frames:
                pv = proc(text="<grounding>", images=fr["img"], return_tensors="pt")["pixel_values"]
                sess_pvs.append(pv.cpu())
                with torch.no_grad(): out = model.vision_model(pixel_values=pv.to(DEVICE))
                sp_.append(out.pooler_output.squeeze(0).float().cpu().numpy())
                pp_.append(out.last_hidden_state[:,1:,:].mean(1).squeeze(0).float().cpu().numpy())
            sess_pool_f = np.array(sp_); sess_patch_f = np.array(pp_)

    for tag, feats, sf in [("lp_pool",pool_f,sess_pool_f),("lp_patch",patch_f,sess_patch_f),
                            ("mlp_patch",patch_f,sess_patch_f)]:
        if tag not in methods: continue
        rv5, rss = run_probe(feats, frames, splits, sess_frames, sf, 256 if tag=="mlp_patch" else None)
        results[tag] = {"v5": rv5, "session": rss}
        _print_r(f"[Kosmos-2 {tag}]", rv5, rss)

    if "lora" in methods:
        from peft import get_peft_model, LoraConfig
        feat_dim = patch_f.shape[1]
        def fresh():
            vm = AutoModelForVision2Seq.from_pretrained(str(LOCAL)).vision_model
            return get_peft_model(vm, LoraConfig(r=LORA_RANK, lora_alpha=16,
                    target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none"))
        def feat_fn(vm, pv): return vm(pixel_values=pv).last_hidden_state[:,1:,:].mean(1).float()
        rv5, rss = run_lora_batched(fresh, feat_fn, feat_dim, pvs_v5, frames, splits, sess_pvs, sess_frames, "Kosmos-2")
        results["lora"] = {"v5": rv5, "session": rss}
        _print_r("[Kosmos-2 lora]", rv5, rss)
    return results


# ── OWL-v2 ───────────────────────────────────────────────────────────────────

def run_owlv2(frames, splits, sess_frames, methods):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    print("\n[OWL-v2] 로딩...")
    t0 = time.time()
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEVICE)
    proc  = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    print(f"[OWL-v2] {time.time()-t0:.1f}s")
    results = {}

    if "zs" in methods:
        def _zs(fr_list):
            preds = []
            for fr in fr_list:
                W, H = fr["img"].width, fr["img"].height
                inp = proc(text=[[PHRASE]], images=fr["img"], return_tensors="pt").to(DEVICE)
                with torch.no_grad(): out = model(**inp)
                res = proc.post_process_object_detection(out, threshold=0.1, target_sizes=[(H,W)])[0]
                boxes = res["boxes"]
                if len(boxes)==0: preds.append(None)
                else:
                    best = int(res["scores"].argmax())
                    x1,_,x2,_ = boxes[best].cpu().tolist()
                    preds.append((x1+x2)/2/W)
            return preds
        t0 = time.time()
        r_v5 = eval_preds(_zs(frames), frames)
        r_v5["lat"] = (time.time()-t0)*1000/len(frames)
        r_ss = eval_preds(_zs(sess_frames), sess_frames) if sess_frames else None
        results["zs"] = {"v5": r_v5, "session": r_ss}
        _print_r("[OWL-v2 zs]", r_v5, r_ss)
        print(f"    det={r_v5['det']:.1%} lat={r_v5['lat']:.0f}ms/frame")

    needs_feat = any(m in methods for m in ["lp_patch","mlp_patch","lora"])
    if needs_feat:
        print("  feature 추출 (V5)...")
        patch_f, pvs_v5 = [], []
        for fr in frames:
            pv = proc(text=[[PHRASE]], images=fr["img"], return_tensors="pt")["pixel_values"]
            pvs_v5.append(pv.cpu())
            with torch.no_grad(): out = model.owlv2.vision_model(pixel_values=pv.to(DEVICE))
            patch_f.append(out.last_hidden_state[:,1:,:].mean(1).squeeze(0).float().cpu().numpy())
        patch_f = np.array(patch_f)

        sess_patch_f = sess_pvs = None
        if sess_frames:
            print("  feature 추출 (session)...")
            sp_, sess_pvs = [], []
            for fr in sess_frames:
                pv = proc(text=[[PHRASE]], images=fr["img"], return_tensors="pt")["pixel_values"]
                sess_pvs.append(pv.cpu())
                with torch.no_grad(): out = model.owlv2.vision_model(pixel_values=pv.to(DEVICE))
                sp_.append(out.last_hidden_state[:,1:,:].mean(1).squeeze(0).float().cpu().numpy())
            sess_patch_f = np.array(sp_)

    for tag, feats, sf in [("lp_patch",patch_f,sess_patch_f),("mlp_patch",patch_f,sess_patch_f)]:
        if tag not in methods: continue
        rv5, rss = run_probe(feats, frames, splits, sess_frames, sf, 256 if tag=="mlp_patch" else None)
        results[tag] = {"v5": rv5, "session": rss}
        _print_r(f"[OWL-v2 {tag}]", rv5, rss)

    if "lora" in methods:
        from peft import get_peft_model, LoraConfig
        feat_dim = patch_f.shape[1]
        def fresh():
            vm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").owlv2.vision_model
            return get_peft_model(vm, LoraConfig(r=LORA_RANK, lora_alpha=16,
                    target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none"))
        def feat_fn(vm, pv): return vm(pixel_values=pv).last_hidden_state[:,1:,:].mean(1).float()
        rv5, rss = run_lora_batched(fresh, feat_fn, feat_dim, pvs_v5, frames, splits, sess_pvs, sess_frames, "OWL-v2")
        results["lora"] = {"v5": rv5, "session": rss}
        _print_r("[OWL-v2 lora]", rv5, rss)
    return results


# ── Florence-2 ───────────────────────────────────────────────────────────────

def run_florence(frames, splits, sess_frames, methods):
    from transformers import AutoProcessor, AutoModelForCausalLM
    print("\n[Florence-2] 로딩...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base",
                trust_remote_code=True).to(DEVICE)
    proc  = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    print(f"[Florence-2] {time.time()-t0:.1f}s")
    results = {}
    TASK = "<OPEN_VOCABULARY_DETECTION>"

    if "zs" in methods:
        def _zs(fr_list):
            preds = []
            for fr in fr_list:
                inp = proc(text=TASK+PHRASE, images=fr["img"], return_tensors="pt").to(DEVICE)
                with torch.no_grad(): ids = model.generate(**inp, max_new_tokens=128)
                out_text = proc.decode(ids[0], skip_special_tokens=False)
                res = proc.post_process_generation(out_text, task=TASK,
                          image_size=(fr["img"].height, fr["img"].width))
                bbs = res.get(TASK, {}).get("bboxes", [])
                if not bbs: preds.append(None)
                else:
                    x1,_,x2,_ = bbs[0]
                    preds.append((x1+x2)/2/fr["img"].width)
            return preds
        t0 = time.time()
        r_v5 = eval_preds(_zs(frames), frames)
        r_v5["lat"] = (time.time()-t0)*1000/len(frames)
        r_ss = eval_preds(_zs(sess_frames), sess_frames) if sess_frames else None
        results["zs"] = {"v5": r_v5, "session": r_ss}
        _print_r("[Florence-2 zs]", r_v5, r_ss)
        print(f"    det={r_v5['det']:.1%} lat={r_v5['lat']:.0f}ms/frame")

    needs_feat = any(m in methods for m in ["lp_patch","mlp_patch","lora"])
    if needs_feat:
        print("  feature 추출 (V5)...")
        patch_f, pvs_v5 = [], []
        for fr in frames:
            pv = proc(text="<MORE_DETAILED_CAPTION>", images=fr["img"], return_tensors="pt")["pixel_values"]
            pvs_v5.append(pv.cpu())
            with torch.no_grad(): enc = model.vision_tower(pv.to(DEVICE))
            patch_f.append(enc[:,1:,:].mean(1).squeeze(0).float().cpu().numpy())
        patch_f = np.array(patch_f)

        sess_patch_f = sess_pvs = None
        if sess_frames:
            print("  feature 추출 (session)...")
            sp_, sess_pvs = [], []
            for fr in sess_frames:
                pv = proc(text="<MORE_DETAILED_CAPTION>", images=fr["img"], return_tensors="pt")["pixel_values"]
                sess_pvs.append(pv.cpu())
                with torch.no_grad(): enc = model.vision_tower(pv.to(DEVICE))
                sp_.append(enc[:,1:,:].mean(1).squeeze(0).float().cpu().numpy())
            sess_patch_f = np.array(sp_)

    for tag, feats, sf in [("lp_patch",patch_f,sess_patch_f),("mlp_patch",patch_f,sess_patch_f)]:
        if tag not in methods: continue
        rv5, rss = run_probe(feats, frames, splits, sess_frames, sf, 256 if tag=="mlp_patch" else None)
        results[tag] = {"v5": rv5, "session": rss}
        _print_r(f"[Florence-2 {tag}]", rv5, rss)

    if "lora" in methods:
        print("  [Florence-2 lora] DaViT attributes lookup 및 PEFT 호환성 문제로 lora 기법은 건너뜁니다.")
        results["lora"] = {"v5": {"det": 0.0, "cx_mae": 1.0, "dir_bin": 0.0}, "session": None}
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",  nargs="+", default=["clip","kosmos","owlv2","florence"])
    parser.add_argument("--methods", nargs="+", default=["zs","lp_pool","lp_patch","mlp_patch","lora"])
    args = parser.parse_args()

    v5_frames   = load_v5_frames()
    sess_frames = load_session_frames()
    splits = split_seeds(v5_frames, N_SEEDS, TRAIN_RATIO)

    baseline_v5 = eval_preds([0.5]*len(v5_frames), v5_frames)
    baseline_ss = eval_preds([0.5]*len(sess_frames), sess_frames)
    print(f"[baseline always-center]  v5: dir={baseline_v5['dir_bin']:.1%} cx_mae={baseline_v5['cx_mae']:.3f}"
          f"  |  sess: dir={baseline_ss['dir_bin']:.1%} cx_mae={baseline_ss['cx_mae']:.3f}")

    all_results = {"_baseline": {"v5": baseline_v5, "session": baseline_ss}}
    runners = {"clip": run_clip, "kosmos": run_kosmos, "owlv2": run_owlv2, "florence": run_florence}

    for name in args.models:
        if name not in runners: print(f"unknown model: {name}"); continue
        try:
            all_results[name] = runners[name](v5_frames, splits, sess_frames, set(args.methods))
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results[name] = {"error": str(e)}
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUT_FILE.write_text(json.dumps(all_results, indent=2))
        print(f"  → 저장: {OUT_FILE}")

    print("\n========== 최종 요약 ==========")
    print(f"baseline: v5 dir={baseline_v5['dir_bin']:.1%}  sess dir={baseline_ss['dir_bin']:.1%}")
    print(f"{'모델':10s} {'방법':12s} {'v5_dir':>8s} {'v5_mae':>7s} {'ss_dir':>8s} {'ss_mae':>7s}")
    print("─"*55)
    for model, mres in all_results.items():
        if model.startswith("_") or not isinstance(mres, dict) or "error" in mres: continue
        for method, r in mres.items():
            if not isinstance(r, dict): continue
            rv5 = r.get("v5", {}); rss = r.get("session") or {}
            if "dir_bin" not in rv5: continue
            print(f"{model:10s} {method:12s} {rv5['dir_bin']:>7.1%}  {rv5['cx_mae']:>6.3f}"
                  f"  {rss.get('dir_bin',0):>7.1%}  {rss.get('cx_mae',0):>6.3f}")
    print(f"\n저장: {OUT_FILE}")

if __name__ == "__main__":
    main()
