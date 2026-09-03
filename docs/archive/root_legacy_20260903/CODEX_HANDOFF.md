# Codex Handoff — MoNaVLA Exp18 (2026-04-21)

## 현재 상황 요약

**Exp18 VLM + LoRA + Text Embedding 학습이 진행 중이다.**

- 시작: 2026-04-21 07:32
- 예상 종료: 11:00 ~ 16:00 (early stopping 여부에 따라)
- 로그: `logs/exp18_vla_training.log`

---

## 실험 히스토리 (중요)

| 실험 | 방식 | PM | Closed-loop | 결론 |
|------|------|----|----|------|
| Exp11 | VLM end-to-end | 58.6% | 0% | FORWARD collapse |
| Exp14 Step2 | MLP (BBox+Image) | 75.9% | **66.7%** ✅ | **현재 최강 baseline** |
| Exp17 | VLM end-to-end (33/33/33 균등) | 76.95% | 11.1% | 누적 오류 폭발 |
| **Exp18** | **VLM + LoRA + Text Embedding** | 진행 중 | 진행 중 | 목표: 70%+ |

**핵심 인사이트:**
- PM이 높아도 closed-loop은 실패할 수 있다 (Exp17: PM 77% → closed-loop 11%)
- Decomposition(작은 MLP)이 end-to-end VLM보다 실제 성능이 압도적으로 높다
- Text attention = 0% (Google-robot post-train이 text pathway 망가뜨림)

---

## 이번 Exp18이 이전과 다른 점

**무엇을 추가했는가:**
1. Dataset에 text embedding 추가 (`docs/v5/v5_dataset_with_text_embeddings.json`)
   - 150개 에피소드 전체에 path_type별 1024D text embedding 포함
2. `NavRoboKosMos.forward_continuous`에 `text_embedding` 인자 추가
   - frozen text embedding을 직접 instruction embedding으로 사용 가능
3. Dataset `__getitem__`에 `text_embedding` 반환 추가

**변경된 파일:**
- `robovlm_nav/datasets/nav_h5_dataset_impl.py` — text_embedding 로드 및 반환
- `robovlm_nav/models/nav_robokosmos.py` — text_embedding kwarg 처리
- `configs/mobile_vla_v5_exp18_vla_finetuned.json` — 새 실험 config
- `docs/v5/v5_dataset_with_text_embeddings.json` — 150개 에피소드 text embedding

**학습 설정:**
- 백본: Google-robot pretrained Kosmos-2 + LoRA
- 데이터: 150 episodes, 33/33/34 (left/straight/right) 균등 비율
- Epochs: 최대 20 (early stopping patience=5)
- LR: 5e-5, batch_size: 4, accumulate_grad: 16

---

## Codex가 할 일 (학습 완료 후)

### Step 1: 학습 완료 확인

```bash
# 프로세스 확인
ps aux | grep "robovlm_nav/train" | grep -v grep

# 로그에서 완료 메시지 확인
tail -50 logs/exp18_vla_training.log | grep -E "val_loss|Epoch|completed|best"

# 체크포인트 확인
ls -lh runs/v5_nav/kosmos/mobile_vla_v5_exp18/*/v5-exp18-vla-text-fusion/*.ckpt
```

### Step 2: PM 평가

```bash
# Best checkpoint 찾기
CKPT=$(ls runs/v5_nav/kosmos/mobile_vla_v5_exp18/*/v5-exp18-vla-text-fusion/*.ckpt | head -1)
echo "Using: $CKPT"

# PM 평가 실행
python3 scripts/test_v5_pm_dm.py \
    --model_path "$CKPT" \
    --config configs/mobile_vla_v5_exp18_vla_finetuned.json \
    --output_dir docs/v5/bbox_nav_step3/exp18_pm_results.json
```

### Step 3: Closed-loop 평가

```bash
# 먼저 inference_server.py에 exp18 모델 지원 추가 확인
# scripts/sim/evaluate_closed_loop_v5.py의 --model 옵션 확인

python3 scripts/sim/evaluate_closed_loop_v5.py \
    --model exp18 \
    --checkpoint "$CKPT"
```

### Step 4: 결과 비교 및 문서화

기록할 것:
- Exp18 PM 수치
- Exp18 Closed-loop 성공률 (FPE 포함)
- Exp14 Step2와 비교: 66.7% 넘었는가?
- 실패/성공 원인 분석

결과를 `docs/v5/bbox_nav_step3/` 에 저장하고 `docs/index.html` Hero 버튼에 링크 추가.

### Step 5: Git 커밋

```bash
git add robovlm_nav/datasets/nav_h5_dataset_impl.py \
        robovlm_nav/models/nav_robokosmos.py \
        configs/mobile_vla_v5_exp18_vla_finetuned.json \
        docs/v5/ \
        scripts/

git commit -m "exp18: VLM+LoRA+text embedding fusion training and evaluation"
git push
```

---

## 핵심 파일 위치

| 파일 | 역할 |
|------|------|
| `logs/exp18_vla_training.log` | 진행 중인 학습 로그 |
| `configs/mobile_vla_v5_exp18_vla_finetuned.json` | Exp18 학습 설정 |
| `docs/v5/v5_dataset_with_text_embeddings.json` | 150 episodes + text embedding |
| `scripts/sim/evaluate_closed_loop_v5.py` | Closed-loop 평가 스크립트 |
| `scripts/test_v5_pm_dm.py` | PM 평가 스크립트 |
| `robovlm_nav/models/nav_robokosmos.py` | VLM backbone (text_embedding 추가됨) |
| `robovlm_nav/datasets/nav_h5_dataset_impl.py` | Dataset (text_embedding 반환) |

---

## 주의사항

1. **`generate()` 절대 호출 금지** — Google-robot backbone은 "Tin Tin Tin..." 무한 반복
2. **`third_party/RoboVLMs/` 수정 금지** — 우리 코드는 모두 `robovlm_nav/` 에
3. **closed-loop 결과가 PM보다 중요** — Exp17처럼 PM 높고 closed-loop 실패 사례 있음
4. **목표: closed-loop 70%** 이상 (현재 baseline Exp14 Step2: 66.7%)

---

## 현재 학습 모니터링

```bash
tail -f logs/exp18_vla_training.log
```

Epoch 당 약 26분, 최대 20 epochs (Early stopping patience=5).
