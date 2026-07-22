#!/usr/bin/env python3
"""
exp73: 트랙A(V6 180ep) 중간 점검 학습 — 2-arm × multi-head × multi-seed.

Arms:
  v6    — V6 트랙A 180ep 단독
  v6v5  — V6 train + 레거시 V5 150ep(colorfixed 캐시) 혼합
  ※ 두 arm 모두 val = V6 val(고정 15%, split seed 42) → arm 간 직접 비교 가능

Heads (모두 window=6, bbox_scale=3.0 — 배포 규격):
  transformer — 배포 아키텍처 (exp71_window6_bboxscale_final 동일)
  mlp         — flatten(window×260) → FC
  cxgeom      — temporal branch + 현재 프레임 geometric branch (exp72 아이디어)

V6 vis 캐시는 BGR→RGB 반전 + F.normalize (colorfixed 규격)로 생성.

Usage:
  .venv/bin/python3 scripts/train_exp73_trackA_heads.py            # 전체
  .venv/bin/python3 scripts/train_exp73_trackA_heads.py --heads transformer --arms v6
"""
import json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VLM_PATH   = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_PT  = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
ANN_V6     = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json"   # --ann-v6로 교체 가능 (owl ablation)
CACHE_V6   = ROOT / "docs/v5/closed_loop_eval/exp73_v6_vis_cache.pt"           # vis는 그라운더와 무관 — 공유
CACHE_V5   = ROOT / "docs/v5/closed_loop_eval/exp71_vis_cache_colorfixed.pt"
OUT_DIR    = ROOT / "runs/v5_nav/mlp/exp73"
OUT_LOG    = ROOT / "docs/v5/closed_loop_eval/exp73_trackA_heads.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 6
BBOX_SCALE = 3.0
NUM_CLASSES = 8
PROJ_DIM = 256
VIS_DIM = 1024
FRAME_DIM = 4 + PROJ_DIM  # 260
VAL_RATIO = 0.15
SPLIT_SEED = 42


# ── 인코더 (exp71 FrozenCLIPV2 + 배치 처리) ──────────────────────────────
class FrozenCLIPV2(nn.Module):
    def __init__(self, vlm_path, stage1_pt, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ckpt = torch.load(str(stage1_pt), map_location=device, weights_only=False)
        self.processor = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vm = base.vision_model.to(device).eval()
        self.proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.proj.load_state_dict(ckpt["image_proj"])
        self.proj.eval()

    @torch.no_grad()
    def encode(self, pil_imgs, batch=32):
        feats = []
        for b in range(0, len(pil_imgs), batch):
            inp = self.processor(images=pil_imgs[b:b + batch], return_tensors="pt")
            pv = inp["pixel_values"].to(DEVICE, dtype=torch.float16)
            f = self.vm(pixel_values=pv).last_hidden_state.mean(1).float()
            feats.append(self.proj(f))
        return torch.cat(feats)


def build_v6_cache():
    """V6 주석 → colorfixed 규격 캐시 (BGR→RGB, F.normalize)."""
    with open(ANN_V6) as f:
        ann = json.load(f)
    print("[CACHE] FrozenCLIPV2 로드...", flush=True)
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_PT, DEVICE).eval()
    episodes = []
    for i, ep in enumerate(ann):
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            continue
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        if not frames:
            continue
        with h5py.File(str(h5_path), "r") as f:
            imgs_np = (f["observations"]["images"] if "observations" in f else f["images"])[:]
            acts_np = f["actions"][:] if "actions" in f else None
        pil_imgs = [Image.fromarray(imgs_np[fr["frame_idx"]][:, :, ::-1].astype("uint8"))
                    for fr in frames]
        vis = F.normalize(enc.encode(pil_imgs), dim=-1).cpu()
        bboxes = [(fr.get("cx_det", 0.5), fr.get("cy_det", 0.5),
                   fr.get("area_det", 0.05), float(fr.get("has_bbox", False))) for fr in frames]
        gts = [fr["gt_class"] for fr in frames]
        # 연속 회귀 헤드용 raw 액션 (lx, ly, az)
        acts = ([tuple(float(v) for v in acts_np[fr["frame_idx"]]) for fr in frames]
                if acts_np is not None else None)
        episodes.append({"stem": h5_path.stem, "path_type": ep["path_type"],
                          "bboxes": bboxes, "vis": vis, "gts": gts, "acts": acts})
        if (i + 1) % 20 == 0:
            print(f"  encoded {i+1}/{len(ann)}", flush=True)
    torch.save(episodes, str(CACHE_V6))
    del enc
    torch.cuda.empty_cache()
    print(f"[CACHE] 저장 → {CACHE_V6} ({len(episodes)}ep)", flush=True)
    return episodes


# ── 헤드 정의 ────────────────────────────────────────────────────────────
class TransformerActionHead(nn.Module):
    """배포 아키텍처 (exp71_window6_bboxscale_final 동일)."""
    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW, nhead=4, num_layers=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, frame_dim))
        self.pos_emb = nn.Embedding(window + 1, frame_dim)
        el = nn.TransformerEncoderLayer(d_model=frame_dim, nhead=nhead, dim_feedforward=512,
                                         dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, 128), nn.ReLU(),
                                   nn.Dropout(0.1), nn.Linear(128, NUM_CLASSES))

    def forward(self, x):
        B = x.size(0)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        pos = torch.arange(x.size(1), device=x.device)
        return self.head(self.encoder(x + self.pos_emb(pos))[:, 0])


class MLPActionHead(nn.Module):
    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(frame_dim * window, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, NUM_CLASSES))

    def forward(self, x):
        return self.net(x.flatten(1))


class CxGeomHead(nn.Module):
    """temporal branch(flatten) + 현재 프레임 bbox geometric branch."""
    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW):
        super().__init__()
        self.branch_t = nn.Sequential(
            nn.Linear(frame_dim * window, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU())
        self.branch_g = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.merge = nn.Sequential(
            nn.Linear(128 + 32, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES))

    def forward(self, x):
        # x: (B, window, 260); 마지막 프레임의 bbox 4채널이 geometric 입력
        t = self.branch_t(x.flatten(1))
        g = self.branch_g(x[:, -1, :4])
        return self.merge(torch.cat([t, g], dim=1))


class ContRegHead(nn.Module):
    """연속 회귀 헤드 — 배포 transformer trunk 동일, 출력만 (lx,ly,az) 3-dim.
    이산 vs 연속 비교용 (MoNa-pi flow 계열의 최소 버전)."""
    ACTION_SCALE = 1.15  # 조이스틱 최대값

    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW, nhead=4, num_layers=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, frame_dim))
        self.pos_emb = nn.Embedding(window + 1, frame_dim)
        el = nn.TransformerEncoderLayer(d_model=frame_dim, nhead=nhead, dim_feedforward=512,
                                         dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, 128), nn.ReLU(),
                                   nn.Dropout(0.1), nn.Linear(128, 3), nn.Tanh())

    def forward(self, x):
        B = x.size(0)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        pos = torch.arange(x.size(1), device=x.device)
        return self.head(self.encoder(x + self.pos_emb(pos))[:, 0])  # (B,3) in [-1,1]


def cont_to_class_t(a):
    """(B,3) raw 액션 → 8-class (nav_h5_dataset_impl.py 규칙, 텐서 벡터화)."""
    x, y, az = a[:, 0], a[:, 1], a[:, 2]
    cls = torch.zeros(len(a), dtype=torch.long, device=a.device)
    is_x, is_y = x.abs() > 0.3, y.abs() > 0.3
    neither = ~is_x & ~is_y
    cls[neither & (az > 0.1)] = 6
    cls[neither & (az < -0.1)] = 7
    fwd = x > 0.3
    cls[fwd & (y > 0.3)] = 4
    cls[fwd & (y < -0.3)] = 5
    cls[fwd & (y.abs() <= 0.3)] = 1
    lat = (x.abs() <= 0.3)
    cls[lat & (y > 0.3)] = 2
    cls[lat & (y < -0.3)] = 3
    return cls


class FlowMatchingHead(nn.Module):
    """경량 rectified-flow 헤드 — MoNa-pi(AdaLN-Zero flow) 핵심 아이디어의 최소 버전.
    trunk(transformer encoder)로 window 컨텍스트를 CLS 벡터 c로 압축한 뒤,
    velocity field v_theta(x_t, t, c)를 작은 MLP로 예측. 학습은 rectified flow
    (x_t = (1-t)x0 + t*a_target, target = a_target - x0), 추론은 Euler ODE 적분."""
    ACTION_SCALE = 1.15
    N_STEPS = 10  # 추론 시 ODE 적분 스텝 수

    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW, nhead=4, num_layers=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, frame_dim))
        self.pos_emb = nn.Embedding(window + 1, frame_dim)
        el = nn.TransformerEncoderLayer(d_model=frame_dim, nhead=nhead, dim_feedforward=512,
                                         dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=num_layers)
        self.context_norm = nn.LayerNorm(frame_dim)
        self.velocity_net = nn.Sequential(
            nn.Linear(frame_dim + 3 + 1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3))

    def encode_context(self, x):
        B = x.size(0)
        seq = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        pos = torch.arange(seq.size(1), device=seq.device)
        return self.context_norm(self.encoder(seq + self.pos_emb(pos))[:, 0])

    def velocity(self, x_t, t, c):
        t_ = t.view(-1, 1).expand(x_t.size(0), 1)
        return self.velocity_net(torch.cat([x_t, t_, c], dim=-1))

    @torch.no_grad()
    def forward(self, x, x0=None):
        """추론: x0에서 시작해 N_STEPS Euler 적분으로 action 예측."""
        c = self.encode_context(x)
        B = x.size(0)
        cur = torch.zeros(B, 3, device=x.device) if x0 is None else x0
        dt = 1.0 / self.N_STEPS
        for i in range(self.N_STEPS):
            t = torch.full((B,), i * dt, device=x.device)
            cur = cur + self.velocity(cur, t, c) * dt
        return cur


def train_flow(X_tr, A_tr, X_va, y_va, A_va, seed, epochs=300, lr=5e-4):
    """rectified flow 학습 — MSE(velocity_pred, target_velocity), 평가는 ODE 적분 후
    8-class 변환 val_acc (다른 헤드와 직접 비교)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    scale = FlowMatchingHead.ACTION_SCALE
    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    A_tr_t = torch.tensor(A_tr / scale, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE)
    y_va_t = torch.tensor(y_va, device=DEVICE)
    A_va_t = torch.tensor(A_va, device=DEVICE)

    model = FlowMatchingHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc, best_state, best_mse = 0.0, None, float("inf")

    def eval_acc():
        model.eval()
        with torch.no_grad():
            gen = torch.Generator(device=DEVICE).manual_seed(seed)
            x0 = torch.randn(len(X_va_t), 3, device=DEVICE, generator=gen)
            pred = model(X_va_t, x0=x0) * scale
            acc = (cont_to_class_t(pred) == y_va_t).float().mean().item()
            mse = F.mse_loss(pred, A_va_t).item()
        return acc, mse, pred

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            xb, ab = X_tr_t[b], A_tr_t[b]
            t = torch.rand(len(b), device=DEVICE)
            x0 = torch.randn_like(ab)
            x_t = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * ab
            target_v = ab - x0
            c = model.encode_context(xb)
            v_pred = model.velocity(x_t, t, c)
            loss = F.mse_loss(v_pred, target_v)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            acc, mse, _ = eval_acc()
            if acc >= best_acc:
                best_acc, best_mse = acc, mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, _, pred = eval_acc()
    pred_cls = cont_to_class_t(pred).cpu().numpy()
    per_class = {}
    for c in range(NUM_CLASSES):
        m = (y_va == c)
        if m.sum() > 0:
            per_class[c] = float((pred_cls[m] == c).mean())
    return best_acc, best_state, per_class, best_mse


class HybridHead(nn.Module):
    """lx,ly는 원본이 이미 {-1.15,0,1.15} 3값 이산 신호(실측 확인됨) — 6-way 분류로 충분.
    az만 실제 연속 스펙트럼(33+ 고유값) — 별도 회귀 브랜치. mlp와 동일 trunk 크기."""
    LAT_FWD_CLASSES = 6  # STOP,F,L,R,FL,FR (az 무관, lx/ly만으로 결정)

    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(frame_dim * window, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 128), nn.ReLU())
        self.disc_head = nn.Linear(128, self.LAT_FWD_CLASSES)
        self.az_head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x):
        h = self.trunk(x.flatten(1))
        return self.disc_head(h), self.az_head(h).squeeze(-1)


def hybrid_targets(y, A, az_scale=1.15):
    """8-class gt → (6-way lat/fwd label, az_norm) 쌍. ROT_L/R(6,7)은 lat/fwd 관점에서 STOP(0)."""
    lat_fwd = np.where(y >= 6, 0, y).astype(np.int64)
    az_norm = (A[:, 2] / az_scale).astype(np.float32)
    return lat_fwd, az_norm


def hybrid_combine(lat_fwd_pred, az_pred, az_thresh=0.1):
    """(6-way 예측, 연속 az 예측) → 최종 8-class. nav_h5_dataset_impl.py 규칙과 동일 threshold."""
    cls = lat_fwd_pred.clone()
    is_stop = lat_fwd_pred == 0
    cls[is_stop & (az_pred > az_thresh)] = 6
    cls[is_stop & (az_pred < -az_thresh)] = 7
    return cls


def train_hybrid(X_tr, y_tr, A_tr, X_va, y_va, A_va, seed, epochs=300, lr=5e-4, az_scale=1.15):
    torch.manual_seed(seed)
    np.random.seed(seed)
    lat_fwd_tr, az_tr = hybrid_targets(y_tr, A_tr, az_scale=az_scale)

    cls_counts = np.bincount(lat_fwd_tr, minlength=HybridHead.LAT_FWD_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = 1.0 / cls_counts
    weights = weights / weights.sum() * HybridHead.LAT_FWD_CLASSES
    weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    lat_fwd_tr_t = torch.tensor(lat_fwd_tr, device=DEVICE)
    az_tr_t = torch.tensor(az_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE)
    y_va_t = torch.tensor(y_va, device=DEVICE)
    az_va = (A_va[:, 2] / az_scale).astype(np.float32)
    az_va_t = torch.tensor(az_va, device=DEVICE)

    model = HybridHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc, best_state = 0.0, None

    def eval_acc():
        model.eval()
        with torch.no_grad():
            disc_logit, az_pred = model(X_va_t)
            lat_fwd_pred = disc_logit.argmax(1)
            final_pred = hybrid_combine(lat_fwd_pred, az_pred, az_thresh=0.1 / az_scale)
            acc = (final_pred == y_va_t).float().mean().item()
        return acc, final_pred

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            disc_logit, az_pred = model(X_tr_t[b])
            loss = (F.cross_entropy(disc_logit, lat_fwd_tr_t[b], weight=weights_t)
                    + F.mse_loss(az_pred, az_tr_t[b]))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            acc, _ = eval_acc()
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, pred = eval_acc()
    pred = pred.cpu().numpy()
    per_class = {}
    for c in range(NUM_CLASSES):
        m = (y_va == c)
        if m.sum() > 0:
            per_class[c] = float((pred[m] == c).mean())
    return best_acc, best_state, per_class


CHUNK_K = 4  # action-chunk 크기(offset 0..3 동시 예측) — ACT식 temporal ensembling용


class ActionChunkHead(nn.Module):
    """window 프레임 컨텍스트로 향후 K프레임(offset 0..K-1)의 8-class 액션을 동시 예측.
    63-11 정정 이후 발견: closed-loop 실패가 프레임 단위 오분류가 아니라 "구간 전체의
    방향 오판"에서 옴(청크 최빈값 acc≈프레임 acc, 즉 오류가 국소적이지 않고 구간
    전체에 걸쳐 일관되게 틀림) — 여러 시점에서 겹쳐 예측한 청크를 앙상블하면 이
    구간형 오류가 평균화되어 줄어들 것이라는 가설. trunk는 mlp와 동일."""
    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW, chunk_k=CHUNK_K):
        super().__init__()
        self.chunk_k = chunk_k
        self.trunk = nn.Sequential(
            nn.Linear(frame_dim * window, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 128), nn.ReLU())
        self.head = nn.Linear(128, chunk_k * NUM_CLASSES)

    def forward(self, x):
        h = self.trunk(x.flatten(1))
        return self.head(h).view(-1, self.chunk_k, NUM_CLASSES)


def build_chunk_targets(eps, chunk_k=CHUNK_K):
    """build_windows()와 동일 순서로 순회 — X는 build_windows() 결과를 그대로 재사용 가능.
    경계(에피소드 끝)는 마지막 프레임 클래스를 반복(ACT 관례)."""
    y_chunk = []
    for ep in eps:
        gts = ep["gts"]
        n = len(gts)
        for t in range(n):
            y_chunk.append([gts[min(t + o, n - 1)] for o in range(chunk_k)])
    return np.asarray(y_chunk, dtype=np.int64)


def train_chunk(X_tr, y_chunk_tr, X_va, y_chunk_va, seed, epochs=300, lr=5e-4, chunk_k=CHUNK_K):
    """offset별 CE 평균으로 학습. val_acc는 offset-0(=다른 헤드와 동일 정의, 즉시 다음
    프레임 예측)만 기준 — 헤드 간 비교 가능하게 유지. 실제 이점은 closed-loop 앙상블에서 검증."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    y0_tr = y_chunk_tr[:, 0]
    cls_counts = np.bincount(y0_tr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = 1.0 / cls_counts
    weights = weights / weights.sum() * NUM_CLASSES
    weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_chunk_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE)
    y_va_t = torch.tensor(y_chunk_va, device=DEVICE)

    model = ActionChunkHead(chunk_k=chunk_k).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            logits = model(X_tr_t[b])  # (B, K, C)
            loss = sum(F.cross_entropy(logits[:, o], y_tr_t[b, o], weight=weights_t)
                       for o in range(chunk_k)) / chunk_k
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(X_va_t)[:, 0].argmax(1) == y_va_t[:, 0]).float().mean().item()
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred0 = model(X_va_t)[:, 0].argmax(1).cpu().numpy()
    per_class = {}
    for c in range(NUM_CLASSES):
        m = (y_chunk_va[:, 0] == c)
        if m.sum() > 0:
            per_class[c] = float((pred0[m] == c).mean())
    return best_acc, best_state, per_class


HEADS = {"transformer": TransformerActionHead, "mlp": MLPActionHead, "cxgeom": CxGeomHead}


# ── 데이터 빌드 (exp71 build_windows 동일) ──────────────────────────────
def build_windows(eps, window=WINDOW, bbox_scale=BBOX_SCALE):
    """returns X, y(8-class), A(raw 액션 — 없는 에피소드는 NaN)"""
    X, y, A = [], [], []
    for ep in eps:
        bboxes, vis, gts = ep["bboxes"], ep["vis"], ep["gts"]
        acts = ep.get("acts")
        for t in range(len(gts)):
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                seq.append([v * bbox_scale for v in bboxes[idx]] + vis[idx].tolist())
            X.append(seq)
            y.append(gts[t])
            A.append(acts[t] if acts is not None else (float("nan"),) * 3)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64),
            np.asarray(A, dtype=np.float32))


def train_contreg(X_tr, A_tr, X_va, y_va, A_va, seed, epochs=300, lr=5e-4):
    """연속 회귀 학습 — MSE(raw/1.15), 평가는 클래스 변환 후 val_acc (이산 헤드와 직접 비교)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    scale = ContRegHead.ACTION_SCALE
    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    A_tr_t = torch.tensor(A_tr / scale, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE)
    y_va_t = torch.tensor(y_va, device=DEVICE)
    A_va_t = torch.tensor(A_va, device=DEVICE)

    model = ContRegHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc, best_state, best_mse = 0.0, None, float("inf")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            loss = F.mse_loss(model(X_tr_t[b]), A_tr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pred = model(X_va_t) * scale
                acc = (cont_to_class_t(pred) == y_va_t).float().mean().item()
                mse = F.mse_loss(pred, A_va_t).item()
            if acc >= best_acc:
                best_acc = acc
                best_mse = mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = cont_to_class_t(model(X_va_t) * scale).cpu().numpy()
    per_class = {}
    for c in range(NUM_CLASSES):
        m = (y_va == c)
        if m.sum() > 0:
            per_class[c] = float((pred[m] == c).mean())
    return best_acc, best_state, per_class, best_mse


def train_one(head_cls, X_tr, y_tr, X_va, y_va, seed, epochs=300, lr=5e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cls_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = 1.0 / cls_counts
    weights = weights / weights.sum() * NUM_CLASSES
    weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE)
    y_va_t = torch.tensor(y_va, device=DEVICE)

    model = head_cls().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            loss = F.cross_entropy(model(X_tr_t[b]), y_tr_t[b], weight=weights_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(X_va_t).argmax(1) == y_va_t).float().mean().item()
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    # per-class acc (best state 기준)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X_va_t).argmax(1).cpu().numpy()
    per_class = {}
    for c in range(NUM_CLASSES):
        m = (y_va == c)
        if m.sum() > 0:
            per_class[c] = float((pred[m] == c).mean())
    return best_acc, best_state, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="transformer,mlp,cxgeom,contreg")
    ap.add_argument("--arms", default="v6,v6v5")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--ann-v6", default=str(ANN_V6),
                    help="V6 주석 json — owl ablation 시 bbox_dataset_v6_owl.json 지정")
    ap.add_argument("--tag", default="pg448",
                    help="결과/체크포인트 접미사 (그라운더 구분)")
    ap.add_argument("--exclude-trackf", action="store_true",
                    help="트랙F(center_*, 45ep) 제외하고 트랙A(180ep)만 사용 — CACHE_V6가 "
                         "225ep로 덮어써진 뒤에도 원래 'v6'(트랙A only) 조건을 재현하기 위함. "
                         "2026-07-22: CACHE_V6 파일이 180ep→225ep로 재빌드되면서 'arm=v6'가 "
                         "더 이상 트랙A만을 의미하지 않게 된 버그 발견 후 추가.")
    args = ap.parse_args()

    # V6 캐시 (vis는 그라운더와 무관 — 최초 1회만 인코딩)
    if CACHE_V6.exists():
        v6_eps = torch.load(str(CACHE_V6), weights_only=False)
        print(f"[CACHE] V6 캐시 로드: {len(v6_eps)}ep")
    else:
        v6_eps = build_v6_cache()

    # 선택한 주석으로 bbox 교체 (vis 재사용)
    if Path(args.ann_v6) != ANN_V6:
        with open(args.ann_v6) as f:
            alt = json.load(f)
        alt_by_stem = {Path(e["episode"]).stem: e for e in alt}
        replaced = 0
        for ep in v6_eps:
            src = alt_by_stem.get(ep["stem"])
            if src is None:
                continue
            frames = [fr for fr in src["frames"] if fr.get("gt_class") is not None]
            ep["bboxes"] = [(fr.get("cx_det", 0.5), fr.get("cy_det", 0.5),
                             fr.get("area_det", 0.05), float(fr.get("has_bbox", False)))
                            for fr in frames]
            replaced += 1
        print(f"[ANN] bbox 교체({args.tag}): {replaced}/{len(v6_eps)}ep")
    v5_eps = torch.load(str(CACHE_V5), weights_only=False)
    print(f"[CACHE] V5 레거시 캐시: {len(v5_eps)}ep")

    if args.exclude_trackf:
        before = len(v6_eps)
        v6_eps = [ep for ep in v6_eps if not ep["path_type"].startswith("center")]
        print(f"[FILTER] --exclude-trackf: {before}ep → {len(v6_eps)}ep (트랙A only)")

    # V6 split (고정) — 두 arm 공통 val
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(v6_eps)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO))
    val_eps  = [v6_eps[i] for i in idx[:n_val]]
    v6_train = [v6_eps[i] for i in idx[n_val:]]
    print(f"[SPLIT] V6 train={len(v6_train)} / val={len(val_eps)} (공통 val)")

    X_va, y_va, A_va = build_windows(val_eps)
    y_chunk_va = build_chunk_targets(val_eps)
    arm_train = {
        "v6":   v6_train,
        "v6v5": v6_train + v5_eps,
    }

    results = {}
    class_names = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
    for arm in args.arms.split(","):
        X_tr, y_tr, A_tr = build_windows(arm_train[arm])
        print(f"\n=== ARM {arm}: train {len(X_tr)} / val {len(X_va)} samples ===", flush=True)
        for head in args.heads.split(","):
            if head in ("contreg", "flow", "hybrid") and np.isnan(A_tr).any():
                print(f"  [{arm}/{head}] raw 액션 없는 에피소드 포함(레거시 V5) → 스킵", flush=True)
                continue
            accs = []
            best_overall, best_state_overall = 0.0, None
            for seed in [int(s) for s in args.seeds.split(",")]:
                if head == "contreg":
                    acc, state, per_class, mse = train_contreg(
                        X_tr, A_tr, X_va, y_va, A_va, seed, epochs=args.epochs)
                elif head == "flow":
                    acc, state, per_class, mse = train_flow(
                        X_tr, A_tr, X_va, y_va, A_va, seed, epochs=args.epochs)
                elif head == "hybrid":
                    acc, state, per_class = train_hybrid(
                        X_tr, y_tr, A_tr, X_va, y_va, A_va, seed, epochs=args.epochs)
                elif head == "chunk":
                    y_chunk_tr = build_chunk_targets(arm_train[arm])
                    acc, state, per_class = train_chunk(
                        X_tr, y_chunk_tr, X_va, y_chunk_va, seed, epochs=args.epochs)
                else:
                    acc, state, per_class = train_one(HEADS[head], X_tr, y_tr, X_va, y_va,
                                                       seed, epochs=args.epochs)
                accs.append(acc)
                if acc > best_overall:
                    best_overall, best_state_overall = acc, state
                    best_per_class = per_class
                print(f"  [{arm}/{head}] seed={seed} val_acc={acc*100:.1f}%", flush=True)
            key = f"{args.tag}/{arm}/{head}"
            results[key] = {
                "val_acc_mean": float(np.mean(accs)),
                "val_acc_std": float(np.std(accs)),
                "val_acc_best": best_overall,
                "seeds": accs,
                "per_class_best": {class_names[c]: v for c, v in best_per_class.items()},
            }
            ckpt = OUT_DIR / f"exp73_{args.tag}_{arm}_{head}.pt"
            torch.save({"model": best_state_overall, "val_acc": best_overall,
                        "head": head, "arm": arm, "window": WINDOW,
                        "bbox_scale": BBOX_SCALE, "exp": "exp73"}, str(ckpt))
            print(f"  [{arm}/{head}] mean={np.mean(accs)*100:.1f}±{np.std(accs)*100:.1f}% "
                  f"best={best_overall*100:.1f}% → {ckpt.name}", flush=True)

    # 기존 결과에 병합 (그라운더 태그별 누적)
    merged = json.loads(OUT_LOG.read_text()) if OUT_LOG.exists() else {}
    merged.update(results)
    OUT_LOG.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"\n결과 저장 → {OUT_LOG}")
    print("\n=== 요약 ===")
    for k, v in results.items():
        print(f"  {k:22s} {v['val_acc_mean']*100:5.1f}±{v['val_acc_std']*100:.1f}%  best {v['val_acc_best']*100:.1f}%")


if __name__ == "__main__":
    main()
