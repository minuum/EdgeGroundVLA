# 3-Tier Grounding Fallback Ablation

검증일: 2026-05-05  
대상 작업: 2026-05-04 낮 `proxy_inference_server.py` 3-tier grounding 통합  
검증 데이터: `docs/v5/bbox_truth_mini.json` (72 gold annotations · L=18 / C=38 / R=16)

## 0. 배경

5/4 낮 `proxy_inference_server.py`(15:52 변경, +131줄)에 3-tier grounding fallback이 통합됐다:

- **Tier 1** — LoRA-merged Kosmos-2가 `<phrase>gray basket</phrase>` entity로 직접 출력
  - 자산: `docs/v5/bbox_nav_step1/grounding_lora/` (9.7MB safetensors, 16:31 학습)
- **Tier 2** — 13가지 caption 방향 패턴 매칭 (5→13 확장)
  - 자산: 코드 내 `_CAPTION_DIRECTION_PATTERNS`
- **Tier 3** — frozen Kosmos-2 vision feature → Linear(1024→3) → cx
  - 자산: `docs/v5/bbox_nav_step1/coarse_direction_clf.pt` (23KB, 13:34 학습)

학습 직후 정량 검증 보고서 없이 통합 후 끝났기에 사후 ablation 수행.

## 1. A — Smoke Test (3 frames)

`GroundingBackend.run()` 직접 호출, 3 path types (center / left / right) 중간 프레임:

| 프레임 | caption | 발동 Tier | cx |
|---|---|---|---|
| center_straight | `''` (empty) | Tier 3 | 0.25 |
| center_left | `''` | Tier 3 | 0.25 |
| center_right | `''` | Tier 3 | 0.25 |

→ LoRA merge가 caption을 brick하고 모든 답이 Tier 3 단독에서 나옴을 확인.

## 2. B — 4-Mode Ablation (72 frames)

| Mode | Tier1 (entity) | Tier2 (caption) | Tier3 (coarse) | NONE | DirAcc | IoU |
|---|---|---|---|---|---|---|
| M0_base (5-pat caption) | 1.4% | 37.5% | — | **61.1%** | 23.6% | 0.96 (n=1) |
| M1_caption13 | 1.4% | 44.4% | — | 54.2% | 25.0% | 0.96 (n=1) |
| M2_coarse (M1 + clf) | 1.4% | 44.4% | 54.2% | 0% | **73.6%** | 0.96 (n=1) |
| M3_LoRA (M2 + adapter) | **0%** | **0%** | **100%** | 0% | **86.1%** | 0.00 (n=0) |

핵심:
- caption 5→13 패턴 확장은 +6.9%p 매칭률, +1.4%p dir_acc — **거의 효과 없음**
- coarse_clf 추가가 dir_acc 25→73.6% (+48.6%p) — 가장 큰 기여
- LoRA merge는 caption을 0%로 무력화
- M3가 dir_acc 1위지만 bbox IoU=0 — Tier 3는 cx만 출력, bbox 좌표 없음

## 3. C — 개별 자산 검증

### C1. LoRA Adapter Eval-Only (`scripts/finetune_kosmos2_grounding.py --eval_only`)

```
Entity match : 0.000  (72/72)
Direction acc: 0.000  (72/72)
```

→ **어댑터 broken 확정.** LoRA merge 안 한 PeftModel 상태에서도 출력 망가짐. 학습 코드의 `target_text` (patch_index 토큰 포함)가 generate를 brick한 것으로 추정.

### C2. coarse_clf 진짜 일반화 측정

| 평가 모드 | 정확도 | 비고 |
|---|---|---|
| Shipped ckpt (972 samples, mini 포함) | **0.861** | train acc — 기억력 |
| LOO mini-only (mini 72만으로 학습+평가) | 0.639 | mini 내부 LOO |
| Full minus mini eps (900 train) → mini 평가 | **0.361** | **진짜 unseen 일반화** |
| Random baseline | 0.333 | — |

LOO confusion matrix (gold mini):
```
          pred_L  pred_C  pred_R
GT=LEFT    11      4       3
GT=CENTER   7     25       6
GT=RIGHT    2      4      10
```

Unseen confusion (full→mini 평가):
```
          pred_L  pred_C  pred_R
GT=LEFT    12      5       1     (66.7%)
GT=CENTER  25     13       0     (34.2%)
GT=RIGHT    7      8       1     (6.2%)
```

→ Silver(`bbox_dataset_full`)의 cx≷0.35/0.65 자동 라벨이 gold mini의 manual `coarse_position`과 정렬 안 됨. RIGHT class는 silver에 24개뿐이라 flip-aug로 9배 부풀린 합성 데이터로 학습됨 → unseen RIGHT 6%.

## 4. 종합 평가

| 자산 | 단독 정확도 | 평가 |
|---|---|---|
| Tier 1 (LoRA adapter) | 0% | ❌ broken — 폐기 |
| Tier 2 (caption 13-pat) | 25% (M1 dir_acc) | △ 보조용, 5→13 효과 미미 |
| Tier 3 (coarse_clf) | 36% unseen / 86% train | ❌ 거의 random — silver/gold mismatch |

5/4 낮 작업으로 만든 3-tier 모두 실용 가치 없음. proxy_inference_server.py의 통합 코드는 mini 72에 의존한 평가에서만 좋아 보였고, unseen에서는 random baseline 수준.

## 5. 적용된 수정

### `proxy_inference_server.py` — LoRA 자동 merge 비활성화

```python
# Optional LoRA grounding adapter — DISABLED BY DEFAULT.
# The 5/4 fine-tune (grounding_lora/) collapses caption to "" and zeros
# entity matching (verified 0/72 on bbox_truth_mini, 2026-05-05).
# Set VLA_ENABLE_GROUNDING_LORA=1 to opt back in once a working adapter
# is trained. See docs/v5/grounding_3tier_ablation.md for the diagnosis.
if _GROUNDING_LORA.exists() and os.getenv("VLA_ENABLE_GROUNDING_LORA") == "1":
    ...
```

→ 디폴트 OFF, 환경변수로 명시적 opt-in. caption + entity 정상 출력 복구 확인.

## 6. 미해결 / 다음 단계

1. **Tier 3 재학습**: gold-only로는 데이터 부족(72), silver는 라벨 노이즈. 두 가지 옵션:
   - (a) gold annotations 확장 (현재 72 → 200~300 manual)
   - (b) silver의 cx threshold를 gold에 맞춰 재정의 후 silver-only 학습 + gold val
2. **Tier 1 재시도**: 학습 hyperparam 재검토 (epoch, lr, target_text 토큰화 방식)
3. **proxy_inference_server.py 미커밋**: 현재 변경(+131줄 + LoRA-disable) 전부 working tree에만 존재
4. master_memory 미해결 #4 "Gray basket prompted grounding 재실행 필요" → "Tier 3 재학습 + Tier 1 재시도"로 구체화

## 7. 산출물

| 파일 | 위치 |
|---|---|
| Smoke test | `/tmp/smoke_test_grounding.py` |
| 4-mode ablation | `/tmp/ablate_grounding_tiers.py` + `/tmp/ablate_results.json` |
| LOO 측정 | `/tmp/loo_coarse_clf.py` + `/tmp/loo_results.json` |
| Q2 unseen 측정 | `/tmp/retrain_clf_no_mini.py` + `/tmp/q2_results.json` + `/tmp/coarse_clf_no_mini.pt` |
| 본 보고서 | `docs/v5/grounding_3tier_ablation.md` |
