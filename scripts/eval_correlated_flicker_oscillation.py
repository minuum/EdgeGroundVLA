#!/usr/bin/env python3
"""
scripts/train_owl_flicker_robustness.py의 IID 랜덤 flicker는 정적 PM에서 baseline도
멀쩡해서 실제 문제(진동)를 못 잡았다. soda 관찰 패턴 그대로 재현:
"area가 임계값(근접) 넘으면 바로 다음 프레임 검출 끊길 확률이 급증"하는 상관형 flicker를
주고, per-episode 순차 예측에서 실제로 반대방향 클래스(LEFT류<->RIGHT류) 진동이
나오는지, 변형별로 그 진동이 줄어드는지 측정.

같은 5개 변형 모델을 재사용(train_owl_flicker_robustness의 train_variant를 import)하고
평가만 새로 한다.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("flicker", ROOT / "scripts" / "train_owl_flicker_robustness.py")
flicker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flicker)

LEFT_CLASSES = {2, 4, 6}
RIGHT_CLASSES = {3, 5, 7}
AREA_NEAR_THRESH = 0.15   # soda 예시: step11 area=0.2923에서 임계값(0.25) 초과 직후 끊김
P_BASE = 0.1              # 평소 flicker 확률(멀리 있을 때)
P_NEAR = 0.85             # 근접 직후 flicker 확률 (soda 관찰: 거의 항상 끊김)


def correlated_flicker(bboxes, rng):
    """area가 AREA_NEAR_THRESH를 넘은 바로 다음 프레임부터 높은 확률로 fallback."""
    out = []
    was_near = False
    for b in bboxes:
        cx, cy, area, has = b
        p = P_NEAR if was_near else P_BASE
        if rng.random() < p:
            out.append(flicker.FALLBACK)
        else:
            out.append(b)
        was_near = (has > 0.5 and area > AREA_NEAR_THRESH)
    return out


def sequential_predict(model, ep, window, rng):
    """실제 production처럼 프레임을 순서대로 밀어넣으며(윈도우는 관측된 bbox 스트림 기준)
    프레임별 예측 클래스를 얻는다."""
    bboxes = correlated_flicker(ep["bboxes"], rng)
    img_feats = ep["img_feats"]
    n = len(bboxes)
    preds = []
    for t in range(n):
        feat = []
        for k in range(window):
            idx = max(0, t - (window - 1 - k))
            feat.extend(list(bboxes[idx]))
        feat.extend(img_feats[t].tolist())
        x = torch.tensor([feat], dtype=torch.float32, device=flicker.DEVICE)
        with torch.no_grad():
            pred = model(x).argmax(dim=-1).item()
        preds.append(pred)
    return preds, bboxes


def oscillation_rate(preds):
    """연속된 예측이 반대방향(LEFT류<->RIGHT류)으로 바로 뒤집히는 비율."""
    flips = 0
    pairs = 0
    for a, b in zip(preds[:-1], preds[1:]):
        a_side = "L" if a in LEFT_CLASSES else ("R" if a in RIGHT_CLASSES else None)
        b_side = "L" if b in LEFT_CLASSES else ("R" if b in RIGHT_CLASSES else None)
        if a_side and b_side:
            pairs += 1
            if a_side != b_side:
                flips += 1
    return (flips / pairs) if pairs else 0.0


def run_variant(name, train_eps, test_eps, window, aug_p, sticky, cx_aux, n_trials=5):
    torch.manual_seed(flicker.SEED)
    rng_train = np.random.default_rng(flicker.SEED)
    X_tr, y_tr, _ = flicker.windows_from_episodes(train_eps, window, flicker_p=aug_p, sticky=sticky, rng=rng_train)
    X_te, y_te, _ = flicker.windows_from_episodes(test_eps, window, flicker_p=0.0)

    d_in = X_tr.shape[1]
    model = flicker.make_model(d_in)
    cls_counts = np.bincount(y_tr, minlength=flicker.NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = torch.tensor(1.0 / cls_counts, dtype=torch.float32, device=flicker.DEVICE)
    weights = weights / weights.sum() * flicker.NUM_CLASSES
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    X_tr_t = torch.tensor(X_tr, device=flicker.DEVICE)
    y_tr_t = torch.tensor(y_tr, device=flicker.DEVICE)
    X_te_t = torch.tensor(X_te, device=flicker.DEVICE)
    y_te_t = torch.tensor(y_te, device=flicker.DEVICE)

    best_acc, best_state = 0.0, None
    for ep in range(180):
        model.train()
        idx = torch.randperm(len(X_tr_t))
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            logits = model(X_tr_t[b])
            loss = loss_fn(logits, y_tr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X_te_t).argmax(dim=-1) == y_te_t).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    osc_rates = []
    for trial in range(n_trials):
        rng = np.random.default_rng(200 + trial)
        for ep in test_eps:
            preds, bboxes = sequential_predict(model, ep, window, rng)
            osc_rates.append(oscillation_rate(preds))

    result = {
        "clean_pm": float(best_acc),
        "oscillation_rate_mean": float(np.mean(osc_rates)),
        "oscillation_rate_std": float(np.std(osc_rates)),
    }
    print(f"[{name:14s}] clean_pm={best_acc:.1%}  진동율(상관flicker,{n_trials}trial)="
          f"{np.mean(osc_rates):.1%}±{np.std(osc_rates):.1%}")
    return result


def main():
    dataset = flicker.load_dataset()
    train_ds, test_ds = flicker.make_episode_split(dataset)
    train_eps6 = flicker.build_raw_windows(train_ds, window=6)
    test_eps6 = flicker.build_raw_windows(test_ds, window=6)
    train_eps3 = flicker.build_raw_windows(train_ds, window=3)
    test_eps3 = flicker.build_raw_windows(test_ds, window=3)

    results = {}
    results["1_baseline"] = run_variant("baseline", train_eps6, test_eps6, 6, 0.0, False, False)
    results["2_dropout_aug"] = run_variant("dropout_aug", train_eps6, test_eps6, 6, 0.5, False, False)
    results["3_sticky_aug"] = run_variant("sticky_aug", train_eps6, test_eps6, 6, 0.5, True, False)
    results["4_window3"] = run_variant("window3", train_eps3, test_eps3, 3, 0.0, False, False)
    results["5_cx_aux_loss"] = run_variant("cx_aux_loss", train_eps6, test_eps6, 6, 0.0, False, True)

    out = ROOT / "docs" / "v5" / "closed_loop_eval" / "correlated_flicker_oscillation.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
