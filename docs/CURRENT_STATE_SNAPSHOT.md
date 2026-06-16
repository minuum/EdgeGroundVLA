# CURRENT_STATE_SNAPSHOT

**마지막 업데이트**: 2026-05-29 (Antigravity Agent)

---

## 1. Latest Commit & Modifications

- **수정사항**: `scripts/eval_exp59_closedloop.py` 내의 BBox 히스토리 누적 시 과거 프레임의 실제 검출 여부와 상관없이 현재 프레임 상태로 덮어쓰여지던 `has_bbox` 버그를 교정함. 추가로, BBox 지터 억제를 위한 EMA 필터 기능(`--ema-alpha` 옵션)을 구현함.
- **평가 실행**: validation 세트 22개 에피소드에 대해 오류 수정 후 및 EMA($\alpha=0.5$) 적용 조건 하에 closed-loop 평가를 다시 수행하고 [exp59_closedloop_result.json](file:///home/minum/26CS/MoNaVLA/docs/v5/closed_loop_eval/exp59_closedloop_result.json)을 업데이트함.
- **검증 결과**: 버그 수정 및 스무딩 조치에도 성공률은 **4.5% (1/22)**로 정체됨. 이는 프레임 간 지터(Jitter)보다 VLM 그라운더의 예측 BBox가 지닌 GT 대비 계통 편향(Systematic Bias/Offset)이 Stage2 MLP에 OOD로 작용하고 있기 때문임이 검증됨.

---

## 2. Active Model (Champion)

- **그라운더**: **Exp59 (PaliGemma2-3B LoRA Grounder)**
  - Hard Negative 학습 데이터 반영으로 R2-3(오탐) 문제 극복.
  - 타겟(gray basket) 탐지 **95.0%**, Negative 3종(pot/ball/person) 오탐 **0.0%** 기록.
- **제어 헤드**: Stage2 MLP (Exp54 학습 가중치 사용)
  - 8-class discrete action prediction.

---

## 3. Next Steps (향후 과제)

1. **BBox Noise & Offset Augmentation 학습 반영**:
   - VLM BBox와 HSV GT BBox의 계통 오차(Offset)를 모사하여 BBox 노이즈 증강(Augmentation)을 MLP 학습 데이터셋에 주입하고, Stage2 MLP를 재학습시켜 OOD 대응력을 극복해야 함.

2. **실로봇 연동 배포**:
   - `robovlm_nav/serve/inference_server.py` 에 Exp59 PaliGemma2 그라운더와 Stage2 MLP 동작 코드를 연동하고, 배포 서버(`soda@100.85.118.58:~/MoNaVLA`)에 가중치를 업로드하여 실제 물리 테스트를 준비해야 함.

---

## 4. Known Issues & Caveats

- **MLP 오프셋 민감도**: Stage2 MLP가 노이즈가 없는 완벽한 HSV GT BBox에 과적합되어, VLM이 예측한 미세한 스케일/오프셋 차이에도 극단적인 행동을 출력해 drift가 발생함.
- **VLM 추론 속도**: PaliGemma2 3B 모델 추론 시간으로 인해 22개 에피소드 전체 평가에 약 3~4분이 소요됨.
