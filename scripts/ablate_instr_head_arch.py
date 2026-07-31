#!/usr/bin/env python3
"""③ VLA 사다리 후속 — 텍스트 조건화가 헤드 구조(MLP/LSTM/Transformer)에 따라
다른 의미를 갖는지 검증. 같은 데이터(bbox_dataset_owl, window=6, ablation 최적값)에
같은 방식(instruction emb 512 broadcast-concat)으로 텍스트를 주입하되 헤드만 바꿔서:

  1. PM (5-seed): 텍스트 이득이 헤드마다 다른가
  2. Permutation: 텍스트 의존도가 헤드마다 다른가 (구조가 텍스트를 더/덜 쓰는가)
  3. Counterfactual: 명령 순응(방향 지시 따라가기)이 헤드마다 다른가
     — LSTM/Transformer는 시퀀스 구조라 텍스트를 "매 스텝 반복 신호"로 받는데,
       이게 MLP의 "1회성 concat"보다 명령 신호를 더 잘 붙잡는지가 핵심 가설

Usage: .venv/bin/python3 scripts/ablate_instr_head_arch.py
출력: docs/v5/bbox_nav_owl/instr_head_arch_compare.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("step2", ROOT / "scripts" / "test_v5_bbox_nav_step2.py")
step2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step2)

spec2 = importlib.util.spec_from_file_location("instr", ROOT / "scripts" / "train_step2_instr_head.py")
instr = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(instr)

DATASET = ROOT / "docs" / "v5" / "bbox_nav_owl" / "bbox_dataset_owl.json"
OUT = ROOT / "docs" / "v5" / "bbox_nav_owl" / "instr_head_arch_compare.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2, 3, 4]
WINDOW = 6  # ablate_instr_window.json 최적값
IMG_DIM = 256  # 16x16
BBOX_DIM = 4
TEXT_DIM = 512
NUM_CLASSES = 8
LEFT_CLASSES = {2, 4, 6}
RIGHT_CLASSES = {3, 5, 7}


def build_seq(eps, embs, mode, window):
    """(N, window, BBOX_DIM+IMG_DIM[+TEXT_DIM]) 시퀀스 + y + meta.
    MLP는 flatten해서 쓰고, LSTM/Transformer는 시퀀스 그대로 사용."""
    rng = np.random.default_rng(7)
    pts = list(instr.INSTRUCTIONS)
    seqs, y, meta = [], [], []
    for ep in eps:
        frames = ep["frames"]
        img_feats = [step2.frame_to_small_feature(_get_img(ep["episode"], f["frame_idx"])) for f in frames]
        pt = ep["path_type"]
        tvec = None
        if mode != "none":
            use_pt = pt if mode == "real" else pts[rng.integers(len(pts))]
            tvec = embs[use_pt]
        for t in range(len(frames)):
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                f = frames[idx]
                bbox = np.array([f["cx"], f["cy"], f["area"], float(f["has_bbox"])], dtype=np.float32)
                step_feat = np.concatenate([bbox, img_feats[idx]])
                if tvec is not None:
                    step_feat = np.concatenate([step_feat, tvec])
                seq.append(step_feat)
            seqs.append(np.stack(seq))
            y.append(frames[t]["gt_class"])
            meta.append({"path_type": pt, "episode": ep["episode"]})
    return np.stack(seqs).astype(np.float32), np.asarray(y, dtype=np.int64), meta


_img_cache = {}
def _get_img(ep_stem, frame_idx):
    import h5py
    key = (ep_stem, frame_idx)
    if key not in _img_cache:
        h5p = next((ROOT / "ROS_action" / "mobile_vla_dataset_v5").glob(f"{ep_stem}.h5"))
        with h5py.File(str(h5p)) as f:
            _img_cache[key] = np.array(f["observations"]["images"][frame_idx]).astype(np.uint8)
    return _img_cache[key]


# ── 헤드 3종 ──────────────────────────────────────────────────────────────

class MLPHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, NUM_CLASSES),
        )
    def forward(self, x):  # x: (B, window, D) → flatten
        return self.net(x.reshape(x.size(0), -1))


class LSTMHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.lstm = nn.LSTM(d_in, 256, 2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(256, NUM_CLASSES)
    def forward(self, x):  # (B, window, D)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class TransformerHead(nn.Module):
    def __init__(self, d_in, window, nh=4, nl=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_in))
        self.pos_emb = nn.Embedding(window + 1, d_in)
        el = nn.TransformerEncoderLayer(d_model=d_in, nhead=nh, dim_feedforward=512,
                                        dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=nl)
        self.head = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 128), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(128, NUM_CLASSES))
    def forward(self, x):
        B = x.size(0)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        pos = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_emb(pos)
        return self.head(self.encoder(x)[:, 0])


HEADS = {"mlp": MLPHead, "lstm": LSTMHead, "transformer": TransformerHead}
# transformer의 d_model은 nhead(4)로 나눠떨어져야 함 — 260/774는 안 나눠짐 → 4의 배수로 pad
def pad_to_multiple(d, k=4):
    return d if d % k == 0 else d + (k - d % k)


def make_head(name, d_in, window):
    if name == "transformer":
        return TransformerHead(d_in, window)
    if name == "mlp":
        return MLPHead(d_in * window)  # MLP는 시퀀스를 flatten하므로 총 차원 필요
    return HEADS[name](d_in)


def train_one(name, X_tr, y_tr, X_te, y_te, window, seed, epochs=150):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d_in = X_tr.shape[-1]
    model = make_head(name, d_in, window).to(DEVICE)
    w = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    w = np.where(w == 0, 1.0, w)
    wt = torch.tensor(1.0 / w, device=DEVICE)
    wt = wt / wt.sum() * NUM_CLASSES
    loss_fn = nn.CrossEntropyLoss(weight=wt)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt = torch.tensor(X_tr, device=DEVICE)
    yt = torch.tensor(y_tr, device=DEVICE)
    Xe = torch.tensor(X_te, device=DEVICE)
    ye = torch.tensor(y_te, device=DEVICE)
    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            loss = loss_fn(model(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xe).argmax(-1) == ye).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def causal_eval(model, X_te, y_te, meta_te, embs, window):
    """permutation + counterfactual — text 부분(마지막 TEXT_DIM)만 교체."""
    n_notext = BBOX_DIM + IMG_DIM
    Xe = torch.tensor(X_te, device=DEVICE)
    ye = torch.tensor(y_te, device=DEVICE)
    with torch.no_grad():
        base_acc = (model(Xe).argmax(-1) == ye).float().mean().item()

    rng = np.random.default_rng(0)
    pts = list(instr.INSTRUCTIONS)
    Xp = X_te.copy()
    for i, m in enumerate(meta_te):
        others = [p for p in pts if p != m["path_type"]]
        other_pt = others[rng.integers(len(others))]
        Xp[i, :, n_notext:] = embs[other_pt]  # 모든 타임스텝의 text 부분 교체
    with torch.no_grad():
        perm_acc = (model(torch.tensor(Xp, device=DEVICE)).argmax(-1) == ye).float().mean().item()

    def flip(pt):
        Xc = X_te.copy()
        Xc[:, :, n_notext:] = embs[pt]
        with torch.no_grad():
            return model(torch.tensor(Xc, device=DEVICE)).argmax(-1).cpu().numpy()

    p_left = flip("left_left")
    p_right = flip("right_right")
    return {
        "base_acc": base_acc, "perm_acc": perm_acc, "perm_drop": base_acc - perm_acc,
        "cf_left_to_leftcls": float(np.mean([c in LEFT_CLASSES for c in p_left])),
        "cf_right_to_rightcls": float(np.mean([c in RIGHT_CLASSES for c in p_right])),
        "cf_changed_rate": float(np.mean(p_left != p_right)),
    }


def main():
    dataset = json.loads(DATASET.read_text())
    embs = instr.embed_instructions()
    train_eps, test_eps = step2.make_episode_split(dataset)

    results = {}
    for head_name in ["mlp", "lstm", "transformer"]:
        results[head_name] = {}
        print(f"\n=== HEAD: {head_name} ===")
        for mode in ["none", "real", "shuffled"]:
            X_tr, y_tr, _ = build_seq(train_eps, embs, mode, WINDOW)
            X_te, y_te, meta_te = build_seq(test_eps, embs, mode, WINDOW)
            accs = []
            last_model = None
            for seed in SEEDS:
                model, acc = train_one(head_name, X_tr, y_tr, X_te, y_te, WINDOW, seed)
                accs.append(acc)
                last_model = model
            results[head_name][mode] = {"pm_mean": float(np.mean(accs)), "pm_std": float(np.std(accs))}
            print(f"  [{mode}] PM {100*np.mean(accs):.1f}% ± {100*np.std(accs):.1f}%")
            if mode == "real":
                causal = causal_eval(last_model, X_te, y_te, meta_te, embs, WINDOW)
                results[head_name]["causal"] = causal
                print(f"  permutation: {causal['base_acc']:.3f} → {causal['perm_acc']:.3f} "
                      f"(Δ{causal['perm_drop']:+.3f})  cf_left→L:{100*causal['cf_left_to_leftcls']:.1f}% "
                      f"cf_right→R:{100*causal['cf_right_to_rightcls']:.1f}% "
                      f"changed:{100*causal['cf_changed_rate']:.1f}%")

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")

    print(f"\n{'head':<12}{'none':>10}{'real':>10}{'shuffled':>10}{'perm_drop':>11}{'cf_changed':>11}")
    for h in ["mlp", "lstm", "transformer"]:
        r = results[h]
        print(f"{h:<12}{100*r['none']['pm_mean']:>9.1f}%{100*r['real']['pm_mean']:>9.1f}%"
              f"{100*r['shuffled']['pm_mean']:>9.1f}%{100*r['causal']['perm_drop']:>10.1f}%"
              f"{100*r['causal']['cf_changed_rate']:>10.1f}%")


if __name__ == "__main__":
    main()
