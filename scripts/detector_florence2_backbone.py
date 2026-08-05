#!/usr/bin/env python3
"""CH67 67-2 — Florence-2를 비전 백본으로 쓸 수 있는가 (교수님 제안 검정).

배경(2026-08-04 논의):
  교수님 제안 "추후 OWL-v2 + Florence-2가 경량 VLA를 만들기에 더 좋은 조합인 것 같아"
  파라미터 근거는 강함 — Florence-2-base 231.4M(vision 90.4M) vs Kosmos-2 1664.5M(vision 303.2M)

  단 우리 실측 2건에서 Florence-2가 탈락한 이력이 있다:
    CH59  L/R 정확도 51.8% — cx max=0.559로 화면 우측 절반을 예측 못 함(구조적 우편향)
    CH55계열  v5 85.7% → 실주행 15.0% 급락(일반화 취약)
  그러나 둘 다 **OVD(검출) 출력**에 대한 것이고, 백본으로 쓰면 검출은 OWL-v2가 담당하므로
  좌표 결함이 경로에서 빠진다 → **아직 측정한 적 없는 조합**이다.

검정 설계 (apples-to-apples):
  CH65 65-5와 **같은 라벨·같은 분할(V6 seed42)·같은 헤드**로 백본만 교체한다.
    Kosmos-2 : vision_model → (16×16, 1024),  cx MAE 0.0020, 53.7ms  [측정 완료]
    Florence2: forward_features_unpool → (24×24, 1024)                [본 실험]

판정 기준 (사전 고정, 사후 변경 금지):
  · cx MAE가 Kosmos-2와 동등 이내 → 비전 백본으로 유효
  · 유의하게 나쁘면 → CH59 결함이 OVD 헤드가 아니라 피처 자체에서 온다는 뜻
  · **area 구간별·좌우별 필수 보고** — 전체 평균만 보면 우편향 같은 구조적 결함을 놓친다

출력: docs/v5/detector/florence2_backbone.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json"
CACHE = ROOT / "docs/v5/detector/florence2_patch_feats.npz"
OUT = ROOT / "docs/v5/detector"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT_SEED, VAL_RATIO = 42, 0.15
GRID = 24                      # Florence-2: 576 = 24×24 (Kosmos-2는 16×16)
DIM = 1024
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchHead(nn.Module):
    """65-5와 동일 구조. GRID만 24로(파라미터 수는 grid와 무관)."""

    def __init__(self, dim=DIM, hid=128, grid=GRID):
        super().__init__()
        self.grid = grid
        self.proj = nn.Sequential(nn.Conv2d(dim, hid, 1), nn.ReLU(),
                                  nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU())
        self.heat = nn.Conv2d(hid, 1, 1)
        self.area = nn.Sequential(nn.Conv2d(hid, 1, 1), nn.Flatten(),
                                  nn.Linear(grid * grid, 1))
        gy, gx = torch.meshgrid(torch.linspace(0, 1, grid),
                                torch.linspace(0, 1, grid), indexing="ij")
        self.register_buffer("gx", gx.reshape(-1))
        self.register_buffer("gy", gy.reshape(-1))

    def forward(self, x):
        B = x.shape[0]
        h = self.proj(x.transpose(1, 2).reshape(B, -1, self.grid, self.grid))
        logit = self.heat(h).reshape(B, -1)
        p = F.softmax(logit, dim=1)
        return (p * self.gx).sum(1), (p * self.gy).sum(1), self.area(h).squeeze(1)


def collect_labels():
    """65-5와 완전히 동일한 표본 — LIVE & detected만, 같은 seed42 분할."""
    ann = json.loads(ANN.read_text())
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(ann)))
    rng.shuffle(idx)
    val_eps = set(idx[:max(1, int(len(idx) * VAL_RATIO))])
    rows = []
    for ei, ep in enumerate(ann):
        for f in ep["frames"]:
            if f["grounding_cached"] or not f["detected"]:
                continue
            rows.append(dict(ep=ep["episode"], fi=f["frame_idx"],
                             cx=f["cx_det"], cy=f["cy_det"], area=f["area_det"],
                             is_val=ei in val_eps))
    return rows


def extract(rows):
    if CACHE.exists():
        z = np.load(CACHE)
        print(f"캐시 재사용: {CACHE}")
        return z["feats"], z["y"], z["is_val"]
    from transformers import AutoModelForCausalLM, AutoProcessor
    print("[Florence-2-base] 로딩")
    m = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True,
        torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    vt = m.vision_tower
    feats = np.zeros((len(rows), GRID * GRID, DIM), dtype=np.float16)
    cur, hf = None, None
    for n, r in enumerate(rows):
        if r["ep"] != cur:
            if hf is not None:
                hf.close()
            hf = h5py.File(r["ep"], "r"); cur = r["ep"]
        im = np.array(hf["images"][r["fi"]])[:, :, ::-1]           # BGR→RGB (주석과 동일)
        pv = proc(images=Image.fromarray(im.astype("uint8")), text="<OD>",
                  return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
        with torch.no_grad():
            o = vt.forward_features_unpool(pv)[0]                  # (576, 1024)
        feats[n] = o.cpu().numpy().astype(np.float16)
        if (n + 1) % 500 == 0:
            print(f"  {n+1}/{len(rows)}", flush=True)
    if hf is not None:
        hf.close()
    y = np.array([[r["cx"], r["cy"], r["area"]] for r in rows], dtype=np.float32)
    v = np.array([r["is_val"] for r in rows], dtype=bool)
    np.savez(CACHE, feats=feats, y=y, is_val=v)
    print(f"저장: {CACHE}  {feats.shape}")
    return feats, y, v


def train(feats, y, is_val, epochs=60, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(feats[~is_val]).float(); Xva = torch.from_numpy(feats[is_val]).float()
    ytr = torch.from_numpy(y[~is_val]); yva = torch.from_numpy(y[is_val])
    print(f"train {len(Xtr)} / val {len(Xva)}")
    m = PatchHead().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bs = 1e9, None, 48
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            xb, yb = Xtr[b].to(DEV), ytr[b].to(DEV)
            cx, cy, ar = m(xb)
            loss = F.l1_loss(cx, yb[:, 0]) + F.l1_loss(cy, yb[:, 1]) + 2.0 * F.l1_loss(ar, yb[:, 2])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pc, py_, pa = [], [], []
            for i in range(0, len(Xva), 128):
                a, b2, c = m(Xva[i:i + 128].to(DEV))
                pc.append(a.cpu()); py_.append(b2.cpu()); pa.append(c.cpu())
            pc, py_, pa = torch.cat(pc), torch.cat(py_), torch.cat(pa)
            sc = ((pc - yva[:, 0]).abs().mean() + (py_ - yva[:, 1]).abs().mean()
                  + 2 * (pa - yva[:, 2]).abs().mean()).item()
        if sc < best:
            best, best_state = sc, {k: v.clone() for k, v in m.state_dict().items()}
        if (ep + 1) % 15 == 0:
            print(f"  ep{ep+1:3d} val cx {(pc-yva[:,0]).abs().mean():.4f}", flush=True)
    m.load_state_dict(best_state)
    return m, Xva, yva


def measure_latency():
    from transformers import AutoModelForCausalLM, AutoProcessor
    m = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True,
        torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    img = Image.fromarray(np.zeros((720, 1280, 3), dtype=np.uint8))
    pv = proc(images=img, text="<OD>", return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
    vt = m.vision_tower
    for _ in range(10):
        with torch.no_grad():
            vt.forward_features_unpool(pv)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50):
        with torch.no_grad():
            vt.forward_features_unpool(pv)
    torch.cuda.synchronize()
    return (time.time() - t) / 50 * 1000


def main():
    ap = argparse.ArgumentParser(); ap.parse_args()
    rows = collect_labels()
    print(f"표본 {len(rows)} (val {sum(r['is_val'] for r in rows)}) — 65-5와 동일 분할")
    feats, y, v = extract(rows)
    m, Xva, yva = train(feats, y, v)
    m.eval()
    with torch.no_grad():
        pc, pa = [], []
        for i in range(0, len(Xva), 128):
            a, _, c = m(Xva[i:i + 128].to(DEV))
            pc.append(a.cpu()); pa.append(c.cpu())
        pc, pa = torch.cat(pc), torch.cat(pa)
    ex = (pc - yva[:, 0]).abs(); ea = (pa - yva[:, 2]).abs()
    lat = measure_latency()
    nparam = sum(p.numel() for p in m.parameters())

    print("\n" + "=" * 68)
    print("67-2 결과 — Florence-2 비전 백본 (Kosmos-2와 동일 조건)")
    print("=" * 68)
    print(f"  cx MAE {ex.mean():.4f}  중앙 {ex.median():.4f}  p90 {ex.quantile(0.9):.4f}")
    print(f"  area MAE {ea.mean():.4f}   헤드 파라미터 {nparam/1e6:.3f}M   그리드 {GRID}x{GRID}")
    print(f"  비전 백본 지연 {lat:.1f}ms   (대조: Kosmos-2 53.7ms)")
    print(f"\n  [대조] Kosmos-2 65-5: cx MAE 0.0020 · 비전 303.2M · 53.7ms · 16x16")
    print(f"         Florence-2   : cx MAE {ex.mean():.4f} · 비전 90.4M · {lat:.1f}ms · 24x24")

    print("\n  [area 구간별 cx MAE]")
    a = yva[:, 2]
    seg = {}
    for lo, hi, lab in [(0, 0.05, "<0.05 (먼 객체)"), (0.05, 0.09, "0.05~0.09 (목표구간)"),
                        (0.09, 9, ">=0.09 (가까움)")]:
        s = (a >= lo) & (a < hi)
        if s.sum():
            print(f"    {lab:22s} n={int(s.sum()):4d}  cx MAE {ex[s].mean():.4f}")
            seg[lab] = float(ex[s].mean())

    print("\n  [좌/우 구간별] — CH59의 우편향(cx max=0.559) 재현 여부 확인")
    gt = yva[:, 0]
    for lo, hi, lab in [(0, 0.5, "좌측 (cx<0.5)"), (0.5, 1.01, "우측 (cx>=0.5)")]:
        s = (gt >= lo) & (gt < hi)
        if s.sum():
            print(f"    {lab:16s} n={int(s.sum()):4d}  cx MAE {ex[s].mean():.4f}  "
                  f"예측 cx 범위 {pc[s].min():.3f}~{pc[s].max():.3f}")
    print(f"    ★ 예측 cx 전체 최대값 {pc.max():.4f}  "
          f"(CH59 Florence-2 OVD는 0.559에서 막혔음 — 0.9 이상이면 피처는 정상)")

    json.dump(dict(cx_mae=float(ex.mean()), cx_median=float(ex.median()),
                   area_mae=float(ea.mean()), latency_ms=float(lat),
                   head_params=int(nparam), grid=GRID,
                   pred_cx_max=float(pc.max()), pred_cx_min=float(pc.min()),
                   by_area=seg,
                   cx_mae_left=float(ex[gt < 0.5].mean()),
                   cx_mae_right=float(ex[gt >= 0.5].mean()),
                   baseline_kosmos2=dict(cx_mae=0.0020, latency_ms=53.7,
                                         vision_params=303_200_000, grid=16)),
              open(OUT / "florence2_backbone.json", "w"), indent=2)
    print(f"\n저장: {OUT/'florence2_backbone.json'}")


if __name__ == "__main__":
    main()
