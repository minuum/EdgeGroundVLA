#!/usr/bin/env python3
"""CH67 67-5 — 비전 백본 스윕: "얼마나 작아져도 좌표가 유지되는가".

배경:
  67-2에서 Florence-2 비전(90.4M)이 Kosmos-2 비전(303.2M)보다 cx MAE가 나빠지지
  않음을 확인했다(0.0015 vs 0.0020). 그러면 자연스러운 다음 질문은
  **"어디까지 줄일 수 있는가"** 다. 경량화가 논문 방향이므로 이 곡선이 곧 근거가 된다.

  단 67-2·65-5는 각각 **단일 seed** 였다. 0.0015 vs 0.0020 같은 작은 차이를 순위로
  주장하려면 seed 변동을 알아야 한다. 그래서 이 스윕은 **6개 백본 전부를 3 seed로
  다시 돌린다** — 기존 두 결과도 포함해 재측정한다(캐시된 피처 재사용, 헤드만 재학습).

검정 설계 (apples-to-apples — 백본만 바뀐다):
  · 라벨      : V6 pg448 cx, LIVE & detected only (65-5와 동일)
  · 분할      : seed42 에피소드 단위 15% val (동일)
  · 헤드      : PatchHead (Conv 1x1 → Conv 3x3 → heat 1ch, soft-argmax) — 구조 동일
                파라미터 수는 grid와 무관, 입력 dim에만 의존 → dim 차이는 표에 명시
  · 학습      : AdamW lr 1e-3, 60 epoch, cosine, L1(cx)+L1(cy)+2·L1(area)
  · seed      : 0, 1, 2  → mean±std 보고

사전 고정한 판정 기준 (사후 변경 금지):
  ① 주 지표는 **cx MAE** (조향이 cx만 쓰므로)
  ② 좌우/area 구간별 필수 보고 — 전체 평균은 구조적 편향을 가린다
  ③ **예측 cx 최대값이 라벨 최대값(0.8612)을 추종하는가** — CH59 Florence-2 OVD가
     0.559에서 막혔던 실패 모드. 추종하지 못하면 피처가 화면 한쪽을 못 본다는 뜻
  ④ 지연은 **비전 백본 forward만** 측정(검출기 제외), 720x1280 원본 1장 기준
  ⑤ "작은 백본이 충분하다"는 결론은 cx MAE가 큰 백본의 mean+1std 이내일 때만

⚠️ 이 실험이 말하지 않는 것:
  cx MAE는 **탐지 성공 프레임에서의 좌표 정확도**다. has_bbox 판정(미검출 인지)은
  라벨이 없어 평가 불가(CH65 65-6). 따라서 "작은 백본으로 교체 가능"이 아니라
  "좌표 회귀에서는 작은 백본으로도 동등"까지만 주장할 수 있다.

출력: docs/v5/detector/backbone_sweep.json
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ANN = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json"
OUT = ROOT / "docs/v5/detector"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT_SEED, VAL_RATIO = 42, 0.15
SEEDS = [0, 1, 2]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 백본 등록표. cache= 가 있으면 기존 피처 재사용(추출 생략).
BACKBONES = {
    "mobilenetv3s": dict(kind="tv", name="mobilenet_v3_small", img=448, vis_params=0.927),
    "effb0":        dict(kind="tv", name="efficientnet_b0",    img=448, vis_params=4.008),
    "florence2":    dict(kind="cache", cache="florence2_patch_feats.npz", img=768, vis_params=90.4),
    "clipL":        dict(kind="hf_clip", name="openai/clip-vit-large-patch14", img=224, vis_params=303.2),
    "kosmos2":      dict(kind="cache", cache="step2_patch_feats.npz", img=224, vis_params=303.2),
    "siglip400m":   dict(kind="hf_siglip", name="google/siglip-so400m-patch14-384", img=384, vis_params=428.2),
}
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PatchHead(nn.Module):
    """65-5/67-2와 동일 구조. grid·dim만 백본에 맞춘다."""

    def __init__(self, dim, grid, hid=128):
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

    def forward(self, x):                       # x: (B, N, dim)
        B = x.shape[0]
        h = self.proj(x.transpose(1, 2).reshape(B, -1, self.grid, self.grid))
        logit = self.heat(h).reshape(B, -1)
        p = F.softmax(logit, dim=1)
        return (p * self.gx).sum(1), (p * self.gy).sum(1), self.area(h).squeeze(1)


def collect_labels():
    """65-5/67-2와 완전히 동일한 표본·순서 — 캐시 재사용의 전제."""
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


# ---------------------------------------------------------------- 백본 로더
def build_backbone(spec):
    """(forward_fn, grid, dim) 반환. forward_fn: uint8 RGB 배치 → (B, N, dim) fp16."""
    img = spec["img"]

    if spec["kind"] == "tv":
        import torchvision.models as tvm
        ctor = getattr(tvm, spec["name"])
        net = ctor(weights="IMAGENET1K_V1").features.to(DEV, torch.float16).eval()

        def fwd(batch):                                   # (B,H,W,3) uint8 RGB
            x = np.stack([np.asarray(Image.fromarray(b).resize((img, img),
                          Image.BILINEAR), dtype=np.float32) / 255.0 for b in batch])
            x = (x - IMNET_MEAN) / IMNET_STD
            t = torch.from_numpy(x).permute(0, 3, 1, 2).to(DEV, torch.float16)
            with torch.no_grad():
                o = net(t)                                # (B, C, g, g)
            return o.flatten(2).transpose(1, 2)           # (B, g*g, C)

    elif spec["kind"] == "hf_clip":
        from transformers import CLIPImageProcessor, CLIPVisionModel
        net = CLIPVisionModel.from_pretrained(spec["name"], torch_dtype=torch.float16).to(DEV).eval()
        proc = CLIPImageProcessor.from_pretrained(spec["name"])

        def fwd(batch):
            pv = proc(images=[Image.fromarray(b) for b in batch],
                      return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
            with torch.no_grad():
                o = net(pixel_values=pv).last_hidden_state
            return o[:, 1:, :]                            # CLS 제거

    elif spec["kind"] == "hf_siglip":
        from transformers import SiglipImageProcessor, SiglipVisionModel
        net = SiglipVisionModel.from_pretrained(spec["name"], torch_dtype=torch.float16).to(DEV).eval()
        proc = SiglipImageProcessor.from_pretrained(spec["name"])

        def fwd(batch):
            pv = proc(images=[Image.fromarray(b) for b in batch],
                      return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
            with torch.no_grad():
                o = net(pixel_values=pv).last_hidden_state
            return o                                      # SigLIP은 CLS 없음
    else:
        raise ValueError(spec["kind"])

    probe = fwd([np.zeros((720, 1280, 3), dtype=np.uint8)])
    n, dim = probe.shape[1], probe.shape[2]
    grid = int(round(n ** 0.5))
    assert grid * grid == n, f"패치 수 {n}가 정사각 그리드가 아니다"
    return fwd, grid, dim, net


def extract(key, spec, rows, bs=8):
    """피처 추출(또는 캐시 재사용). 반환 (feats fp16, grid, dim, latency_ms)."""
    if spec["kind"] == "cache":
        z = np.load(OUT / spec["cache"])
        f = z["feats"]
        grid = int(round(f.shape[1] ** 0.5))
        assert len(f) == len(rows), f"{key}: 캐시 {len(f)} vs 라벨 {len(rows)} 불일치"
        print(f"[{key}] 캐시 재사용 {f.shape}  grid {grid}", flush=True)
        return f, grid, f.shape[2], None

    cache = OUT / f"sweep_{key}.npz"
    fwd, grid, dim, net = build_backbone(spec)
    # 지연: 백본 forward만, 원본 720x1280 1장
    one = [np.zeros((720, 1280, 3), dtype=np.uint8)]
    for _ in range(10):
        fwd(one)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30):
        fwd(one)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    lat = (time.time() - t0) / 30 * 1000

    if cache.exists():
        f = np.load(cache)["feats"]
        print(f"[{key}] 캐시 재사용 {f.shape}", flush=True)
        del net
        torch.cuda.empty_cache()
        return f, grid, dim, lat

    print(f"[{key}] 추출 시작 grid {grid} dim {dim} 지연 {lat:.1f}ms", flush=True)
    feats = np.zeros((len(rows), grid * grid, dim), dtype=np.float16)
    cur, hf, buf, bidx = None, None, [], []
    for n, r in enumerate(rows):
        if r["ep"] != cur:
            if hf is not None:
                hf.close()
            hf = h5py.File(r["ep"], "r")
            cur = r["ep"]
        buf.append(np.ascontiguousarray(np.array(hf["images"][r["fi"]])[:, :, ::-1]))
        bidx.append(n)
        if len(buf) == bs:
            feats[bidx] = fwd(buf).cpu().numpy().astype(np.float16)
            buf, bidx = [], []
        if (n + 1) % 1000 == 0:
            print(f"  {n+1}/{len(rows)}", flush=True)
    if buf:
        feats[bidx] = fwd(buf).cpu().numpy().astype(np.float16)
    if hf is not None:
        hf.close()
    np.savez(cache, feats=feats)
    del net
    torch.cuda.empty_cache()
    return feats, grid, dim, lat


# ---------------------------------------------------------------- 헤드 학습
def train_head(feats, y, is_val, grid, dim, seed, epochs=60, lr=1e-3, bs=48):
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(feats[~is_val])          # fp16 유지 → 배치에서만 float
    Xva = torch.from_numpy(feats[is_val])
    ytr = torch.from_numpy(y[~is_val])
    yva = torch.from_numpy(y[is_val])
    m = PatchHead(dim, grid).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = 1e9, None
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            xb = Xtr[b].to(DEV).float()
            yb = ytr[b].to(DEV)
            cx, cy, ar = m(xb)
            loss = (F.l1_loss(cx, yb[:, 0]) + F.l1_loss(cy, yb[:, 1])
                    + 2.0 * F.l1_loss(ar, yb[:, 2]))
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pc, pa = [], []
            for i in range(0, len(Xva), 128):
                a, _, c = m(Xva[i:i + 128].to(DEV).float())
                pc.append(a.cpu()); pa.append(c.cpu())
            pc, pa = torch.cat(pc), torch.cat(pa)
            sc = ((pc - yva[:, 0]).abs().mean() + 2 * (pa - yva[:, 2]).abs().mean()).item()
        if sc < best:
            best, best_state = sc, (pc.clone(), pa.clone())
    pc, pa = best_state
    return pc, pa, yva, sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="일부 백본만")
    args = ap.parse_args()

    rows = collect_labels()
    y = np.array([[r["cx"], r["cy"], r["area"]] for r in rows], dtype=np.float32)
    v = np.array([r["is_val"] for r in rows], dtype=bool)
    print(f"표본 {len(rows)} (val {int(v.sum())}) · 라벨 cx 범위 "
          f"{y[:,0].min():.4f}~{y[:,0].max():.4f}", flush=True)
    label_cx_max = float(y[v, 0].max())

    keys = args.only or list(BACKBONES)
    res = {}
    for key in keys:
        spec = BACKBONES[key]
        feats, grid, dim, lat = extract(key, spec, rows)
        runs = []
        for s in SEEDS:
            pc, pa, yva, nparam = train_head(feats, y, v, grid, dim, s)
            ex = (pc - yva[:, 0]).abs()
            ea = (pa - yva[:, 2]).abs()
            gt = yva[:, 0]
            runs.append(dict(
                cx_mae=float(ex.mean()), cx_med=float(ex.median()),
                cx_p90=float(ex.quantile(0.9)), area_mae=float(ea.mean()),
                cx_mae_left=float(ex[gt < 0.5].mean()),
                cx_mae_right=float(ex[gt >= 0.5].mean()),
                pred_cx_max=float(pc.max()), pred_cx_min=float(pc.min()),
                nparam=nparam,
                seg={lab: float(ex[(yva[:, 2] >= lo) & (yva[:, 2] < hi)].mean())
                     for lo, hi, lab in [(0, .05, "far"), (.05, .09, "mid"), (.09, 9, "near")]},
            ))
            print(f"  [{key}] seed{s} cx MAE {runs[-1]['cx_mae']:.4f} "
                  f"pred cx max {runs[-1]['pred_cx_max']:.4f}", flush=True)
        agg = lambda f: (float(np.mean([f(r) for r in runs])), float(np.std([f(r) for r in runs])))
        res[key] = dict(
            vis_params_M=spec["vis_params"], grid=grid, dim=dim, input=spec["img"],
            head_params_M=runs[0]["nparam"] / 1e6, latency_ms=lat, seeds=runs,
            cx_mae=agg(lambda r: r["cx_mae"]), area_mae=agg(lambda r: r["area_mae"]),
            cx_mae_left=agg(lambda r: r["cx_mae_left"]),
            cx_mae_right=agg(lambda r: r["cx_mae_right"]),
            pred_cx_max=agg(lambda r: r["pred_cx_max"]),
            seg={k: agg(lambda r, k=k: r["seg"][k]) for k in ("far", "mid", "near")},
        )
        del feats
        (OUT / "backbone_sweep.json").write_text(json.dumps(
            dict(label_cx_max=label_cx_max, n=len(rows), n_val=int(v.sum()),
                 seeds=SEEDS, results=res), indent=2, ensure_ascii=False))

    # ------------------------------------------------------------ 보고
    print("\n" + "=" * 96)
    print("67-5 비전 백본 스윕 — 같은 라벨·같은 분할·같은 헤드, 3 seed")
    print("=" * 96)
    print(f"{'백본':14s} {'비전(M)':>8s} {'grid':>5s} {'dim':>5s} "
          f"{'cx MAE (mean±std)':>20s} {'area MAE':>10s} {'지연ms':>8s} {'pred cx max':>12s}")
    for k in sorted(res, key=lambda k: res[k]["vis_params_M"]):
        r = res[k]
        lat = f"{r['latency_ms']:.1f}" if r["latency_ms"] else "미측정"
        print(f"{k:14s} {r['vis_params_M']:8.1f} {r['grid']:5d} {r['dim']:5d} "
              f"{r['cx_mae'][0]:12.4f}±{r['cx_mae'][1]:.4f} {r['area_mae'][0]:10.4f} "
              f"{lat:>8s} {r['pred_cx_max'][0]:9.4f}±{r['pred_cx_max'][1]:.4f}")
    print(f"\n  ★ 라벨 val cx 최대값 = {label_cx_max:.4f} — pred cx max가 이를 추종해야 정상")
    print("\n  [좌/우 · area 구간별 cx MAE]")
    print(f"{'백본':14s} {'좌(cx<.5)':>10s} {'우(cx>=.5)':>11s} "
          f"{'far':>8s} {'mid':>8s} {'near':>8s}")
    for k in sorted(res, key=lambda k: res[k]["vis_params_M"]):
        r = res[k]
        print(f"{k:14s} {r['cx_mae_left'][0]:10.4f} {r['cx_mae_right'][0]:11.4f} "
              f"{r['seg']['far'][0]:8.4f} {r['seg']['mid'][0]:8.4f} {r['seg']['near'][0]:8.4f}")

    # 판정 ⑤: 가장 큰 백본 기준 동등성
    big = max(res, key=lambda k: res[k]["vis_params_M"])
    thr = res[big]["cx_mae"][0] + res[big]["cx_mae"][1]
    print(f"\n  판정 ⑤ — 기준: {big} mean+1std = {thr:.4f} 이내면 '좌표 회귀 동등'")
    for k in sorted(res, key=lambda k: res[k]["vis_params_M"]):
        ok = res[k]["cx_mae"][0] <= thr
        print(f"    {k:14s} {res[k]['cx_mae'][0]:.4f}  {'동등' if ok else '열등'}")
    print(f"\n저장: {OUT/'backbone_sweep.json'}")


if __name__ == "__main__":
    main()
