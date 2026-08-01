#!/usr/bin/env python3
"""검출기 Step 1+2 — 데이터셋 구축 + (C) Kosmos-2 patch 피처 실현가능성.

계획서: docs/plans/plan_20260801_specialized_detector.md

Step 1: V6 주석에서 **LIVE & detected 라벨만** 추출(캐시 상속본 10,847건 제외).
Step 2: 이미 매 프레임 돌고 있는 Kosmos-2 vision_model의 patch 피처(16x16x1024) 위에
        경량 heatmap 헤드를 얹어 (cx, cy, area)를 회귀. 되면 검출기 추가 비용이 거의 0.

go/no-go 기준(계획서 6절 준용):
  - cx 오차가 실기 조향에 쓸 만한가 (OWL-v2 라벨 대비 MAE)
  - area 0.05~0.09 구간(우선 개선 대상)에서 무너지지 않는가

분할: V6 SPLIT_SEED=42 / VAL_RATIO=0.15 를 그대로 재사용(액션 헤드와 동일 val).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json"
VLM = ROOT / ".vlms/kosmos-2-patch14-224"
CACHE = ROOT / "docs/v5/detector/step2_patch_feats.npz"
OUT = ROOT / "docs/v5/detector"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT_SEED, VAL_RATIO = 42, 0.15
GRID = 16
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- Step 1
def collect_labels():
    """LIVE & detected 프레임만. 캐시 상속본은 독립 라벨이 아니므로 제외."""
    ann = json.loads(ANN.read_text())
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(ann)))
    rng.shuffle(idx)
    nv = max(1, int(len(idx) * VAL_RATIO))
    val_eps = set(idx[:nv])
    rows = []
    for ei, ep in enumerate(ann):
        for f in ep["frames"]:
            if f["grounding_cached"] or not f["detected"]:
                continue
            rows.append(dict(ep=ep["episode"], fi=f["frame_idx"], ei=ei,
                             cx=f["cx_det"], cy=f["cy_det"], area=f["area_det"],
                             is_val=ei in val_eps))
    return rows


def extract(rows, limit=None):
    """Kosmos-2 patch 피처(16x16x1024) 추출. 이미 서빙에서 돌던 그 연산."""
    from transformers import AutoProcessor

    from robovlm_nav.serve.stage2_v2_inference_server import _load_kosmos2_vision_only
    proc = AutoProcessor.from_pretrained(str(VLM))
    vm = _load_kosmos2_vision_only(VLM).to(DEV).eval()
    if limit:
        rows = rows[:limit]
    feats = np.zeros((len(rows), GRID * GRID, 1024), dtype=np.float16)
    cur, imgs_cache = None, None
    from PIL import Image
    for n, r in enumerate(rows):
        if r["ep"] != cur:
            if imgs_cache is not None:
                imgs_cache.close()
            imgs_cache = h5py.File(r["ep"], "r")
            cur = r["ep"]
        im = np.array(imgs_cache["images"][r["fi"]])[:, :, ::-1]   # BGR→RGB (주석 생성과 동일)
        pv = proc(images=Image.fromarray(im.astype("uint8")),
                  return_tensors="pt")["pixel_values"].to(DEV, dtype=torch.float16)
        with torch.no_grad():
            o = vm(pixel_values=pv).last_hidden_state[0, 1:]        # CLS 제외 → (256,1024)
        feats[n] = o.cpu().numpy().astype(np.float16)
        if (n + 1) % 500 == 0:
            print(f"  {n+1}/{len(rows)}", flush=True)
    if imgs_cache is not None:
        imgs_cache.close()
    y = np.array([[r["cx"], r["cy"], r["area"]] for r in rows], dtype=np.float32)
    v = np.array([r["is_val"] for r in rows], dtype=bool)
    np.savez(CACHE, feats=feats, y=y, is_val=v)
    print(f"저장: {CACHE}  feats{feats.shape}")
    return feats, y, v


# ---------------------------------------------------------------- Step 2
class PatchHead(nn.Module):
    """16x16x1024 patch 피처 → heatmap + area. soft-argmax로 서브패치 정밀도 확보."""

    def __init__(self, dim=1024, hid=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(dim, hid, 1), nn.ReLU(),
                                  nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU())
        self.heat = nn.Conv2d(hid, 1, 1)
        self.area = nn.Sequential(nn.Conv2d(hid, 1, 1), nn.Flatten(),
                                  nn.Linear(GRID * GRID, 1))
        gy, gx = torch.meshgrid(torch.linspace(0, 1, GRID),
                                torch.linspace(0, 1, GRID), indexing="ij")
        self.register_buffer("gx", gx.reshape(-1))
        self.register_buffer("gy", gy.reshape(-1))

    def forward(self, x):                       # x: (B,256,1024)
        B = x.shape[0]
        h = x.transpose(1, 2).reshape(B, -1, GRID, GRID)
        h = self.proj(h)
        logit = self.heat(h).reshape(B, -1)     # (B,256)
        p = F.softmax(logit, dim=1)
        cx = (p * self.gx).sum(1)
        cy = (p * self.gy).sum(1)
        area = self.area(h).squeeze(1)
        return cx, cy, area, logit


def train(feats, y, is_val, epochs=60, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(feats[~is_val]).float()
    Xva = torch.from_numpy(feats[is_val]).float()
    ytr = torch.from_numpy(y[~is_val]); yva = torch.from_numpy(y[is_val])
    print(f"train {len(Xtr)} / val {len(Xva)}")
    m = PatchHead().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = 1e9, None
    bs = 64
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            xb = Xtr[b].to(DEV); yb = ytr[b].to(DEV)
            cx, cy, area, _ = m(xb)
            loss = (F.l1_loss(cx, yb[:, 0]) + F.l1_loss(cy, yb[:, 1])
                    + 2.0 * F.l1_loss(area, yb[:, 2]))
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pc, py_, pa = [], [], []
            for i in range(0, len(Xva), 256):
                a, b, c, _ = m(Xva[i:i + 256].to(DEV))
                pc.append(a.cpu()); py_.append(b.cpu()); pa.append(c.cpu())
            pc = torch.cat(pc); py_ = torch.cat(py_); pa = torch.cat(pa)
            mae_cx = (pc - yva[:, 0]).abs().mean().item()
            mae_cy = (py_ - yva[:, 1]).abs().mean().item()
            mae_a = (pa - yva[:, 2]).abs().mean().item()
            sc = mae_cx + mae_cy + 2 * mae_a
        if sc < best:
            best, best_state = sc, {k: v.clone() for k, v in m.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"  ep{ep+1:3d} val cx {mae_cx:.4f}  cy {mae_cy:.4f}  area {mae_a:.4f}")
    m.load_state_dict(best_state)
    return m, (Xva, yva)


def report(m, Xva, yva):
    m.eval()
    with torch.no_grad():
        pc, py_, pa = [], [], []
        for i in range(0, len(Xva), 256):
            a, b, c, _ = m(Xva[i:i + 256].to(DEV))
            pc.append(a.cpu()); py_.append(b.cpu()); pa.append(c.cpu())
        pc = torch.cat(pc); py_ = torch.cat(py_); pa = torch.cat(pa)
    ex = (pc - yva[:, 0]).abs(); ey = (py_ - yva[:, 1]).abs(); ea = (pa - yva[:, 2]).abs()
    print("\n" + "=" * 62)
    print("Step 2 결과 — Kosmos-2 patch 피처 위 경량 헤드 (val)")
    print("=" * 62)
    print(f"  cx MAE {ex.mean():.4f}  중앙 {ex.median():.4f}  p90 {ex.quantile(0.9):.4f}")
    print(f"  cy MAE {ey.mean():.4f}  area MAE {ea.mean():.4f}")
    print(f"  파라미터 {sum(p.numel() for p in m.parameters())/1e6:.3f}M")
    print("\n  [area 구간별 cx MAE] — 우선 개선 대상 0.05~0.09 포함")
    a = yva[:, 2]
    for lo, hi, lab in [(0, 0.05, "<0.05 (먼 객체)"), (0.05, 0.09, "0.05~0.09 (목표구간)"),
                        (0.09, 9, ">=0.09 (가까움)")]:
        s = (a >= lo) & (a < hi)
        if s.sum():
            print(f"    {lab:22s} n={int(s.sum()):4d}  cx MAE {ex[s].mean():.4f}  "
                  f"area MAE {ea[s].mean():.4f}")
    print("\n  [참고] cx 오차 기준 — 16x16 그리드 양자화 한계는 ±0.031")
    for t in [0.02, 0.05, 0.10]:
        print(f"    cx 오차 < {t:.2f} 비율: {100*(ex<t).float().mean():.1f}%")
    return dict(cx_mae=float(ex.mean()), cy_mae=float(ey.mean()), area_mae=float(ea.mean()),
                params=sum(p.numel() for p in m.parameters()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reuse", action="store_true", help="캐시된 피처 재사용")
    a = ap.parse_args()
    rows = collect_labels()
    print(f"Step 1 — LIVE & detected 라벨 {len(rows)}건 "
          f"(val 에피소드 {sum(r['is_val'] for r in rows)}건)")
    if a.reuse and CACHE.exists():
        z = np.load(CACHE); feats, y, v = z["feats"], z["y"], z["is_val"]
        print(f"캐시 재사용 {feats.shape}")
    else:
        print("Step 2 — Kosmos-2 patch 피처 추출")
        feats, y, v = extract(rows, a.limit)
    m, (Xva, yva) = train(feats, y, v)
    res = report(m, Xva, yva)
    torch.save({"state": m.state_dict(), "res": res}, OUT / "step2_patch_head.pt")
    (OUT / "step2_result.json").write_text(json.dumps(res, indent=2))
    print(f"\n저장: {OUT/'step2_patch_head.pt'}")
