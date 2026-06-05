# 교수님 미팅 Q&A 준비 — 2026-06-03

## 30초 핵심 요약 (먼저 말할 것)
1. **객체 인식 증명됨** — 텍스트로 basket을 특정(gray basket 100% / red ball 0%), 가리면 행동 바뀜(masking 인과).
2. **파이프라인 동작** — Grounding(VLM) → cx → Action MLP. 시뮬 CL **base/exp59 × b1 = 100%**, B1(243ep) 70%.
3. **정직한 새 발견** — 우리 LoRA fine-tuning이 grounding을 **오히려 악화**시켰고(base zero-shot이 더 안정), STOP은 윈도우 규칙이 최선.

---

## Q1. "모델이 진짜 객체(basket)를 인식하는가?"
- **텍스트 선택성**: 동일 이미지에 phrase만 교체 → `gray basket` 100% vs `red ball` 0% / `person` 3% (**gap 98.3%p**, Exp57).
- **마스킹 인과**: basket 영역을 가리면 Stage1 confidence가 떨어지고 행동이 바뀜 → "복도 암기"가 아니라 basket을 본다.
- **base PG2 probe**: bottle/red ball/container = **0/12 거부**, basket류(gray basket·laundry basket·hamper) = 12/12 검출. 없는 건 안 만든다.

## Q2. "bbox는 위치 좌표일 뿐, 인식이 아니지 않나?"
- bbox는 **텍스트 쿼리에 조건부**로만 생성됨 — "red ball" 넣으면 `<eos>`(아무것도 안 나옴). 색상필터(HSV)는 텍스트 무관 → **메커니즘이 다름**.
- 즉 `<loc>` 토큰은 텍스트 토큰이 이미지 패치에 attend한 결과 = "인식→위치" 순서의 증거.

## Q3. "다른 물체 넣으면 다른 행동을 하는가?"
- 같은 이미지 + 다른 쿼리 → 다른 grounding → 다른 cx → 다른 action. Exp59가 gray basket만 검출(FP 0%)하도록 분리.
- **정직하게**: 아직 같은 환경의 1개 타겟(basket)만 충분히 수집됨. brown pot/chair 등 다물체 주행 데이터는 다음 단계(R3 근본 해결).

## Q4. "도착은 어떻게 알고 멈추는가?" (R4)
- **area 기반 STOP 규칙**: 최근 W프레임 basket area 평균 > 0.5 & 중앙 → STOP 래치. 도착 90% 탐지, 조기오발 0%.
- CL: 규칙 래치 **68.8%** (학습 STOP은 53%로 정체 — area 신호 강해 학습은 쉬우나 precision 부족).
- Y-center 게이트는 효과 없음(도착 시 basket이 수직 중앙이라 무변별) — 실측으로 확인.

## Q5. "왜 fine-tuning(LoRA)했는데 성능이 더 나빠졌나?" (← 우리가 먼저 정직하게 꺼낼 것)
- **핵심 발견**: base PaliGemma2(zero-shot)가 LoRA(exp59)보다 **더 안정** — cx_std 0.070 vs 0.134, full-frame 폭발 0.2% vs 5.9%.
- 원인 3가지: ① 소규모(243ep) 도메인 **과적합** ② Hard-Negative `<eos>` 패널티가 정상 검출 박스도 위축 ③ 1024 이산 bin **양자화 지터**.
- exp58(2-class)은 **full-frame 53% 폭발**로 사실상 붕괴.
- → **결론: base PG2로 단순화하면 grounding 더 안정 + CL 동일(100%).** 무거운 LoRA 불필요.

## Q6. "그래서 closed-loop 성능은?"
| Grounding × Control | CL | 비고 |
|---|---|---|
| HSV GT(상한) × b1 | 96.9% | 알고리즘 GT |
| base PG2 × b1 | **100%** | LoRA 없이 |
| exp59 × b1 | 100% | 현재 |
| exp58 × b1 | 56% | 붕괴 |
| 모든 grounding × exp54(HSV학습) | 43.8% | HSV→PG2 전이 실패 |
- **정직하게**: 이 100%는 시뮬(expert=gt_class, pg2_cx val) 기준이라 낙관적. 라이브 grounding 기준 보고치는 70%. **실로봇 검증 아직**.

## Q7. "center 경로는 왜 실패하나?"
- center_straight는 basket이 정중앙이라 cx≈0.5 근처에서 **미세 jitter가 좌/우 조향으로 증폭** → drift. 소프트웨어 6종(EMA/노이즈/오버샘플) 효과 없었음.
- 해결: center 경로 **추가 수집**(동기식) 또는 실로봇(물리 피드백). 구조적 한계로 정직히 인정.

## Q8. "데이터 수집은 신뢰할 만한가?" (STOP 아티팩트)
- 비동기 조이스틱 수집(25Hz)으로 **중간 STOP 유령 프레임 84개** 생김 → 삭제 대신 **재라벨**(프레임 보존), STOP은 마지막만.
- 동기식(PRE_CACHE, s_t→a_t lock-step) 재수집이 근본 해결.

## Q9. "E2E VLA는 안 되나?"
- exp63 순수 Kosmos E2E = **18.8%** (회전 경로 FPE>3m). Decomposed(70~100%)가 소규모 데이터에서 압승 → 현재 분해 구조가 옳음.

## Q10. "다른 VLM은 더 낫나?" (Moondream2 대조)
- Moondream2(2B)는 cx 안정(0.075)이나 **선택성 0** — basket이든 아니든 다 검출(오탐 100%) → CL 0%. PaliGemma의 텍스트 선택성이 핵심 강점.

---

## 정직하게 인정할 한계 (먼저 말하면 신뢰↑)
- 실로봇 물리 검증 아직 (시뮬 CL 기반).
- 단일 타겟(basket) 환경 — 다물체 goal-conditioned는 다음 단계.
- center 경로 구조적 실패 — 추가 수집 필요.

## 미팅 후 다음 계획 (질문 대비)
1. **base PG2 grounding으로 파이프라인 단순화** 확정 (LoRA 제거, 더 안정)
2. center 경로 동기식 추가 수집
3. 실로봇 배포(SODA) — base×b1 + STOP 규칙
4. 다물체(brown pot/chair) goal-conditioned 데이터 수집
