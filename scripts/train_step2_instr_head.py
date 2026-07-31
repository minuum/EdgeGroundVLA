#!/usr/bin/env python3
"""② 언어→정책 조건화 — instruction CLIP 임베딩을 Step2 헤드에 concat.

path_type(9종)별 합성 instruction을 OWL-v2 내장 텍스트 인코더로 임베딩(512d)해서
bbox+image feature에 붙여 학습. 비교군 3개 × 5-seed:
  1. no_text      : 기존 78.4% 레시피 그대로 (재실측)
  2. with_text    : +text emb 512
  3. shuffled_text: 학습 시 에피소드간 임베딩 무작위 교환 (텍스트가 노이즈면 2==3)

인과 검증 (with_text 모델):
  a. permutation: test에서 임베딩을 다른 path_type 것으로 바꿔치기 → acc 하락폭
  b. counterfactual: 같은 test 프레임에 left/right instruction → 예측 방향 이동률

Usage: .venv/bin/python3 scripts/train_step2_instr_head.py
출력: docs/v5/bbox_nav_owl/instr_head_results.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("step2", ROOT / "scripts" / "test_v5_bbox_nav_step2.py")
step2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step2)

DATASET = ROOT / "docs" / "v5" / "bbox_nav_owl" / "bbox_dataset_owl.json"
OUT = ROOT / "docs" / "v5" / "bbox_nav_owl" / "instr_head_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2, 3, 4]

INSTRUCTIONS = {
    "center_straight": "go straight ahead to the basket in front of you",
    "center_left":     "the basket is ahead, curve to the left side to reach it",
    "center_right":    "the basket is ahead, curve to the right side to reach it",
    "left_straight":   "the basket is on your left, go straight toward it",
    "left_left":       "the basket is on your left, curve left to reach it",
    "left_right":      "the basket is on your left, curve right to reach it",
    "right_straight":  "the basket is on your right, go straight toward it",
    "right_left":      "the basket is on your right, curve left to reach it",
    "right_right":     "the basket is on your right, curve right to reach it",
}
LEFT_CLASSES = {2, 4, 6}   # LEFT, FWD+LEFT, ROT_L
RIGHT_CLASSES = {3, 5, 7}  # RIGHT, FWD+RIGHT, ROT_R


def embed_instructions():
    """OWL-v2 내장 CLIP 텍스트 타워로 9문장 임베딩 — 추가 모델 로드 없음(같은 ckpt)."""
    from transformers import Owlv2Processor, Owlv2Model
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2Model.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEVICE).eval()
    embs = {}
    for pt, sent in INSTRUCTIONS.items():
        inp = proc(text=[sent], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            e = model.get_text_features(input_ids=inp["input_ids"],
                                        attention_mask=inp["attention_mask"])
        e = torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy()
        embs[pt] = e.astype(np.float32)
    del model
    torch.cuda.empty_cache()
    print(f"임베딩 {len(embs)}개, dim={len(next(iter(embs.values())))}")
    return embs


def build(dataset, embs, text_mode):
    """text_mode: none | real | shuffled"""
    X, y, meta = step2.build_windows(dataset)
    if text_mode == "none":
        return X, y, meta
    rng = np.random.default_rng(7)
    pts = list(INSTRUCTIONS)
    tvecs = []
    for m in meta:
        pt = m["path_type"]
        if text_mode == "shuffled":
            pt = pts[rng.integers(len(pts))]
        tvecs.append(embs[pt])
    return np.concatenate([X, np.stack(tvecs)], axis=1), y, meta


def main():
    dataset = json.loads(DATASET.read_text())
    embs = embed_instructions()
    train_eps, test_eps = step2.make_episode_split(dataset)

    results = {}
    trained_variant_preds = {}
    for mode in ["none", "real", "shuffled"]:
        X_tr, y_tr, _ = build(train_eps, embs, mode)
        X_te, y_te, meta_te = build(test_eps, embs, mode)
        accs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            acc, preds = step2.train_eval(X_tr, y_tr, X_te, y_te)
            accs.append(acc)
        results[mode] = {"pm_mean": float(np.mean(accs)), "pm_std": float(np.std(accs)),
                          "pm_seeds": accs}
        print(f"[{mode}] PM {100*np.mean(accs):.1f}% ± {100*np.std(accs):.1f}%")

    # ── 인과 검증: real 모드 마지막 seed 모델을 다시 학습해 고정 (재현용 seed 0) ──
    torch.manual_seed(0)
    np.random.seed(0)
    X_tr, y_tr, _ = build(train_eps, embs, "real")
    X_te, y_te, meta_te = build(test_eps, embs, "real")
    d_in = X_tr.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, 256), torch.nn.ReLU(), torch.nn.Dropout(0.25),
        torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8),
    ).to(DEVICE)
    # step2.train_eval와 동일 루프 (모델 반환이 없어 여기서 직접)
    w = np.bincount(y_tr, minlength=8).astype(np.float32)
    w = np.where(w == 0, 1.0, w)
    wt = torch.tensor(1.0 / w, device=DEVICE)
    wt = wt / wt.sum() * 8
    loss_fn = torch.nn.CrossEntropyLoss(weight=wt)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt = torch.tensor(X_tr, device=DEVICE)
    yt = torch.tensor(y_tr, device=DEVICE)
    Xe = torch.tensor(X_te, device=DEVICE)
    ye = torch.tensor(y_te, device=DEVICE)
    best_acc, best_state = 0.0, None
    for ep in range(220):
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
    print(f"검증용 고정 모델 acc={best_acc:.3f}")

    # a) permutation: text 부분을 다른 path_type 임베딩으로 교체
    n_img = X_te.shape[1] - 512
    rng = np.random.default_rng(0)
    pts = list(INSTRUCTIONS)
    X_perm = X_te.copy()
    for i, m in enumerate(meta_te):
        others = [p for p in pts if p != m["path_type"]]
        X_perm[i, n_img:] = embs[others[rng.integers(len(others))]]
    with torch.no_grad():
        acc_perm = (model(torch.tensor(X_perm, device=DEVICE)).argmax(-1) == ye).float().mean().item()
    print(f"permutation acc: {best_acc:.3f} → {acc_perm:.3f} (Δ {best_acc-acc_perm:+.3f})")

    # b) counterfactual: 전 test 프레임에 left_left vs right_right instruction 주입
    def flip_stats(pt):
        Xc = X_te.copy()
        Xc[:, n_img:] = embs[pt]
        with torch.no_grad():
            p = model(torch.tensor(Xc, device=DEVICE)).argmax(-1).cpu().numpy()
        return p

    p_left = flip_stats("left_left")
    p_right = flip_stats("right_right")
    left_rate = float(np.mean([c in LEFT_CLASSES for c in p_left]))
    right_rate = float(np.mean([c in RIGHT_CLASSES for c in p_right]))
    cross_l = float(np.mean([c in RIGHT_CLASSES for c in p_left]))
    cross_r = float(np.mean([c in LEFT_CLASSES for c in p_right]))
    changed = float(np.mean(p_left != p_right))
    print(f"counterfactual: left지시→LEFT류 {100*left_rate:.1f}% (RIGHT류 {100*cross_l:.1f}%) | "
          f"right지시→RIGHT류 {100*right_rate:.1f}% (LEFT류 {100*cross_r:.1f}%) | 예측변화 {100*changed:.1f}%")

    results["causal"] = {
        "fixed_model_acc": best_acc, "perm_acc": acc_perm,
        "perm_drop": best_acc - acc_perm,
        "cf_left_to_leftcls": left_rate, "cf_right_to_rightcls": right_rate,
        "cf_left_to_rightcls": cross_l, "cf_right_to_leftcls": cross_r,
        "cf_changed_rate": changed,
    }
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
