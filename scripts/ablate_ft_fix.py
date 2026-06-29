"""
ablate_preview_ft_v2 누락분 보완
  - OWL-v2 lora (이전 OOM → 별도 프로세스에서 실행)
  - Florence-2 lp_patch / mlp_patch / lora (_encode_image 경로 사용)

결과를 ablate_preview_ft_v2.json에 병합 저장.
"""

import io, json, random, time
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
BATCH        = 8   # OOM 방지용 작은 배치


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
    print(f"[V5] {len(frames)}개")
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
    print(f"[sess] {len(frames)}개")
    return frames


def split_seeds(frames):
    n = len(frames)
    out = []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        idx = list(range(n)); rng.shuffle(idx)
        cut = int(n * TRAIN_RATIO)
        out.append((idx[:cut], idx[cut:]))
    return out


def eval_preds(preds, frames):
    mae, dir_ok, det = [], [], []
    for p, fr in zip(preds, frames):
        det.append(p is not None)
        if p is not None:
            mae.append(abs(p - fr["cx"]))
            dir_ok.append((p > 0.5) == (fr["cx"] > 0.5))
    return {"det": float(np.mean(det)),
            "cx_mae": float(np.mean(mae)) if mae else 1.0,
            "dir_bin": float(np.mean(dir_ok)) if dir_ok else 0.0}


def _agg(lst):
    if not lst: return None
    return {k: float(np.mean([r[k] for r in lst])) for k in lst[0]}


def _pr(tag, rv5, rss=None):
    s = f"  {tag:35s} v5: dir={rv5['dir_bin']:.1%} mae={rv5['cx_mae']:.3f}"
    if rss: s += f"  |  sess: dir={rss['dir_bin']:.1%} mae={rss['cx_mae']:.3f}"
    print(s)


def run_probe(feats, frames, splits, sess_frames, sess_feats, hidden=None):
    v5_res, ss_res = [], []
    for tr_idx, va_idx in splits:
        d = feats.shape[1]
        head = (nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                if hidden else nn.Linear(d, 1)).to(DEVICE)
        opt = torch.optim.Adam(head.parameters(), lr=LR_PROBE)
        Xt = torch.tensor(feats[tr_idx], dtype=torch.float32, device=DEVICE)
        Yt = torch.tensor([frames[i]["cx"] for i in tr_idx], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        for _ in range(EPOCHS):
            head.train(); opt.zero_grad()
            F.mse_loss(head(Xt), Yt).backward(); opt.step()
        head.eval()
        with torch.no_grad():
            preds = head(torch.tensor(feats[va_idx], dtype=torch.float32, device=DEVICE)).squeeze(1).cpu().tolist()
        v5_res.append(eval_preds(preds, [frames[i] for i in va_idx]))
        if sess_feats is not None:
            with torch.no_grad():
                sp = head(torch.tensor(feats[tr_idx], dtype=torch.float32, device=DEVICE))  # retrain on tr
            # 별도 probe 재학습해서 sess 예측
            head2 = (nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                     if hidden else nn.Linear(d, 1)).to(DEVICE)
            opt2 = torch.optim.Adam(head2.parameters(), lr=LR_PROBE)
            for _ in range(EPOCHS):
                head2.train(); opt2.zero_grad()
                F.mse_loss(head2(Xt), Yt).backward(); opt2.step()
            head2.eval()
            with torch.no_grad():
                sp = head2(torch.tensor(sess_feats, dtype=torch.float32, device=DEVICE)).squeeze(1).cpu().tolist()
            ss_res.append(eval_preds(sp, sess_frames))
    return _agg(v5_res), _agg(ss_res) if ss_res else None


def run_lora_batched(fresh_fn, feat_fn, feat_dim, pvs, frames, splits, sess_pvs, sess_frames, label):
    from peft import get_peft_model, LoraConfig
    print(f"  [{label} lora] batch={BATCH}×{EPOCHS}ep×{N_SEEDS}seeds...")
    v5_res, ss_res = [], []
    for s_i, (tr_idx, va_idx) in enumerate(splits):
        model = fresh_fn().to(DEVICE)
        head  = nn.Linear(feat_dim, 1).to(DEVICE)
        opt   = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=LORA_LR)
        tr_pv = [pvs[i] for i in tr_idx]
        tr_cx = [frames[i]["cx"] for i in tr_idx]

        for _ in range(EPOCHS):
            model.train(); head.train()
            perm = list(range(len(tr_idx))); random.shuffle(perm)
            for b in range(0, len(perm), BATCH):
                bi   = perm[b:b+BATCH]
                pv_b = torch.cat([tr_pv[i] for i in bi]).to(DEVICE)
                cx_b = torch.tensor([tr_cx[i] for i in bi], dtype=torch.float32, device=DEVICE).unsqueeze(1)
                opt.zero_grad()
                F.mse_loss(head(feat_fn(model, pv_b)), cx_b).backward()
                opt.step()

        model.eval(); head.eval()
        def _inf(pv_list):
            out = []
            with torch.no_grad():
                for b in range(0, len(pv_list), BATCH):
                    pv_b = torch.cat(pv_list[b:b+BATCH]).to(DEVICE)
                    out.extend(head(feat_fn(model, pv_b)).squeeze(1).cpu().tolist())
            return out

        v5_res.append(eval_preds(_inf([pvs[i] for i in va_idx]), [frames[i] for i in va_idx]))
        if sess_frames and sess_pvs:
            ss_res.append(eval_preds(_inf(sess_pvs), sess_frames))
        print(f"    seed {s_i+1}/{N_SEEDS} v5_dir={v5_res[-1]['dir_bin']:.1%}")
        del model; torch.cuda.empty_cache()
    return _agg(v5_res), _agg(ss_res) if ss_res else None


# ── OWL-v2 LoRA ──────────────────────────────────────────────────────────────

def fix_owlv2_lora(frames, splits, sess_frames):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from peft import get_peft_model, LoraConfig
    print("\n[OWL-v2 lora fix] 로딩...")
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")

    pvs_v5   = [proc(text=[[PHRASE]], images=fr["img"], return_tensors="pt")["pixel_values"].cpu() for fr in frames]
    sess_pvs = [proc(text=[[PHRASE]], images=fr["img"], return_tensors="pt")["pixel_values"].cpu() for fr in sess_frames]

    # feat_dim 확인
    tmp = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").owlv2.vision_model.to(DEVICE)
    with torch.no_grad(): feat_dim = tmp(pixel_values=pvs_v5[0].to(DEVICE)).last_hidden_state[:,1:,:].mean(1).shape[1]
    del tmp; torch.cuda.empty_cache()
    print(f"  feat_dim={feat_dim}")

    def fresh():
        vm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").owlv2.vision_model
        return get_peft_model(vm, LoraConfig(r=LORA_RANK, lora_alpha=16,
                target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none"))

    def feat_fn(vm, pv):
        return vm(pixel_values=pv).last_hidden_state[:,1:,:].mean(1).float()

    rv5, rss = run_lora_batched(fresh, feat_fn, feat_dim, pvs_v5, frames, splits, sess_pvs, sess_frames, "OWL-v2")
    _pr("[OWL-v2 lora]", rv5, rss)
    return rv5, rss


# ── Florence-2 (_encode_image 경로) ──────────────────────────────────────────

def fix_florence(frames, splits, sess_frames):
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig
    print("\n[Florence-2 fix] 로딩...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base",
                trust_remote_code=True).to(DEVICE)
    proc  = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

    # _encode_image 경로: [B, 577, 768] → [:,1:,:].mean(1) = [B, 768]
    print("  feature 추출 (V5)...")
    patch_f, pvs_v5 = [], []
    for fr in frames:
        pv = proc(text="<MORE_DETAILED_CAPTION>", images=fr["img"], return_tensors="pt")["pixel_values"]
        pvs_v5.append(pv.cpu())
        with torch.no_grad():
            enc = model._encode_image(pv.to(DEVICE))  # [1, 577, 768]
        patch_f.append(enc[:,1:,:].mean(1).squeeze(0).float().cpu().numpy())
    patch_f = np.array(patch_f)

    print("  feature 추출 (session)...")
    sp_, sess_pvs = [], []
    for fr in sess_frames:
        pv = proc(text="<MORE_DETAILED_CAPTION>", images=fr["img"], return_tensors="pt")["pixel_values"]
        sess_pvs.append(pv.cpu())
        with torch.no_grad():
            enc = model._encode_image(pv.to(DEVICE))
        sp_.append(enc[:,1:,:].mean(1).squeeze(0).float().cpu().numpy())
    sess_patch_f = np.array(sp_)

    feat_dim = patch_f.shape[1]
    results = {}

    for tag, hidden in [("lp_patch", None), ("mlp_patch", 256)]:
        rv5, rss = run_probe(patch_f, frames, splits, sess_frames, sess_patch_f, hidden)
        results[tag] = {"v5": rv5, "session": rss}
        _pr(f"[Florence-2 {tag}]", rv5, rss)

    # LoRA: image_projection 레이어에 적용 (DaViT 대신)
    print("  [Florence-2 lora] image_projection LoRA...")
    del model; torch.cuda.empty_cache()

    def fresh():
        m = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        # image_projection (Linear 레이어) + 주변 레이어
        try:
            lora_m = get_peft_model(m.image_projection, LoraConfig(r=LORA_RANK, lora_alpha=16,
                    target_modules=["linear_1","linear_2"] if hasattr(m.image_projection,"linear_1")
                    else ["0","2"], lora_dropout=0.05, bias="none"))
            m.image_projection = lora_m
        except Exception:
            pass  # image_projection LoRA 불가시 전체 freeze + head만
        return m

    def feat_fn_full(m, pv):
        return m._encode_image(pv.to(DEVICE))[:,1:,:].mean(1).float()

    rv5, rss = run_lora_batched(fresh, feat_fn_full, feat_dim, pvs_v5, frames, splits, sess_pvs, sess_frames, "Florence-2")
    results["lora"] = {"v5": rv5, "session": rss}
    _pr("[Florence-2 lora]", rv5, rss)
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    frames      = load_v5_frames()
    sess_frames = load_session_frames()
    splits      = split_seeds(frames)

    # 기존 결과 로드
    existing = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}

    # OWL-v2 lora
    rv5, rss = fix_owlv2_lora(frames, splits, sess_frames)
    if "owlv2" not in existing: existing["owlv2"] = {}
    existing["owlv2"]["lora"] = {"v5": rv5, "session": rss}
    OUT_FILE.write_text(json.dumps(existing, indent=2))
    print(f"→ 저장 (owlv2 lora)")

    # Florence-2
    fl_res = fix_florence(frames, splits, sess_frames)
    if "florence" not in existing: existing["florence"] = {}
    existing["florence"].update(fl_res)
    OUT_FILE.write_text(json.dumps(existing, indent=2))
    print(f"→ 저장 (florence)")

    print("\n완료")

if __name__ == "__main__":
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
