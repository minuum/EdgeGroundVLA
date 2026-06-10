# V5-2 Chair 수집 궤적 Taxonomy 설계 플랜

> 작성 2026-06-10. 승인 전 수집 본격 개시 금지 — 메모 반영 후 갱신.
> 전제: 조이스틱 휠로 자유 궤적 주행. 시간 제약 큼. 객체 = chair (V5와 완전 별개).
> 기존 [PRETRAINED_OBJECT_REPLACEMENT_PLAN.md](PRETRAINED_OBJECT_REPLACEMENT_PLAN.md)의 메인70%/복원30% + 실패 taxonomy 위에 **궤적 카테고리 축**을 정의한다.

---

## 0. 왜 taxonomy가 필요한가 (문제 정의)

조이스틱으로 "자유롭게" 그리면 **데이터 분포가 조종자 습관에 쏠린다.** V5 결과가 그 증거:
- FORWARD 70%, ROT_L/R 합 2% → **FORWARD collapse**, 회전·복원 행동 학습 실패.
- closed-loop에서 누적 방향오류로 붕괴 (offline PM 높아도 rollout 실패).

→ taxonomy의 목적은 "예쁜 분류"가 아니라 **정책이 테스트에서 마주칠 상태분포를 의도적으로 커버**하고 **액션 균형을 강제**하는 것.

---

## 1. 다른 VLA 수집 사례 비교 (핵심 교훈만)

| 데이터셋 | 규모 | 분류 축 | 우리에게 주는 교훈 |
|---------|------|--------|-------------------|
| **RT-1** (Google) | 130k ep, 700+ task | skill 동사 × 객체 (pick/place/move-near) | 언어로 task 다양화 — 단 대규모 전제. 우리는 단일 task라 **분류 축을 행동(조향)으로** 잡아야 함 |
| **BridgeData V2** | 60k traj, 24 env | task × scene/viewpoint | **개수보다 시점/환경 다양성**이 일반화에 더 중요 |
| **DROID** | 76k traj, 564 scene | scene × distractor | 장면·방해물 다양성. 표준화된 obs/action 포맷 |
| **MobilityVLA** (nav) | demo tour + topological graph | 목적지 도달 데모 | **내비게이션은 다양한 출발자세→목표도달 궤적**이 핵심 |
| **GNM/ViNT/NoMaD** (nav FM) | 다로봇/다환경 | goal-conditioned + exploration | goal-conditioned 다양성 + 약간의 탐색/복원 |
| **DAgger** (Ross 2011) | — | 복원/교정 데이터 | **모방학습 공변량 변화 → 복원 궤적 필수** (이미 30% 반영) |

**소규모(300~500ep) 단일-객체 visual-servoing nav에 맞춰 압축한 원칙 4가지:**
1. **개수보다 커버리지** — 테스트 상태분포(출발자세 × 화면 내 chair 위치 × 접근각)를 빠짐없이.
2. **액션 균형 강제** — FORWARD collapse 방지. LEFT/RIGHT/ROT/대각을 만드는 궤적을 의도 할당.
3. **복원 데이터(DAgger)** — closed-loop 성공의 결정 요인. 30% 유지.
4. **시각 다양성** — 조명/거리/방해물/시점. chair는 pretrained 인지 강해 grounding 부담은 낮음 → 시각 다양성은 "있으면 좋음" 우선순위.

---

## 2. V5-2 궤적 Taxonomy — 4개 직교 축

조이스틱 운전자가 "무엇을 바꿔가며 운전해야 하는지"를 직교 축으로 정의한다. (기존 Gradio 9-path 시나리오와 호환)

### 축 A — 출발 시 chair의 화면 위치 (grounding 분포 결정)
`Left` / `Center` / `Right` — episode 시작 프레임에서 chair가 cx 어디에 보이는가.

### 축 B — 필요한 접근 곡률 (action 분포 결정 = 교수님 33/33/33)
`Straight` / `Curve→L` / `Curve→R` — 목표까지 직진인가, 좌/우로 꺾어 접근하는가.

> **A × B = 9 path types** = 기존 Gradio 시나리오 [1]~[9]. 이게 구조화된 **메인 경로(70%)**.

### 축 C — 복원 (covariate shift 방어, 30%)
정상 궤적에 "고의 탈선 → 복귀"를 오버레이. chair를 cx<0.25 또는 cx>0.75로 밀어낸 뒤 반대로 꺾어 중앙 복귀(`FWD+L`/`FWD+R`/`ROT`). → 자유 수집(FL/FC/FR) 슬롯 활용.

### 축 D — 시각/방해 다양성 (A~C에 직교 적용)
- 조명 7:2:1 (형광 : 저조도 : 자연광)
- 거리: 근(0.5~1m) / 원(1.5~2.5m)
- 시점: 정면 / 좌30° / 우30°
- (선택) 방해물/distractor 1개

---

## 3. 시간-효율 Tier 수집 (시간 없을 때 우선순위)

시간이 끊겨도 **앞 Tier만으로 사용 가능**하도록 누적 설계.

| Tier | 목표 | 카테고리 | 예상 ep | 비고 |
|------|------|---------|---------|------|
| **T1 (필수, 1일차)** | closed-loop 최소동작 | 직진 3종(C/L/R 출발) + 복원 일부 | 60~90 | 이것만으로도 rollout 테스트 가능 |
| **T2 (2일차)** | 균형 데이터 | 전체 9-path + 복원 30% | +150 (누적 ~220) | 교수님 33/33/33 충족 |
| **T3 (여유 시)** | 일반화 | 축 D 다양성 오버레이 | +100 (누적 ~320) | 조명/거리/방해물 |

> chair는 grounding 강건 → T3 시각 다양성은 후순위. **T1·T2 먼저 끝내고 평가** 후 T3 결정.

### 카테고리별 ep 예산 (T2 기준 목표)
| 시나리오 | 메인 | 복원 | 합 |
|---------|------|------|-----|
| 직진 (C/L/R) | 20×3=60 | 8×3=24 | 84 |
| 좌곡 (C/L/R) | 12×3=36 | 5×3=15 | 51 |
| 우곡 (C/L/R) | 12×3=36 | 5×3=15 | 51 |
| **합** | **132** | **54** | **~186** |

목표 액션 분포: FORWARD ≤ 55%, (LEFT+RIGHT+대각) ≥ 35%, ROT ≥ 5%, STOP(합성) ~10%.
→ Gradio 하단 **전체 데이터셋 액션 분포** 위젯으로 실시간 모니터링하며 쏠리면 부족 카테고리 보충.

---

## 4. Hz 설계 (세션 타이밍 측정 연동)

1. chair 한 세션을 **SYNC / ASYNC 각각 1회** 같은 경로로 수집 → `⏱️ 마지막 세션` 위젯에서 소요 초 T, 프레임 N, 실측 Hz 확인.
2. 설계 기준:
   - 추론 상한 = **13.3Hz** (GoalNav exp49 실측, [goalnav_exp49_latency_20260610](../inference_reports/goalnav_exp49_latency_20260610.md)).
   - 수집 Hz = 추론 Hz와 **일치**시켜 train/inference gap 제거.
   - 한 세션 T초 × Hz = 프레임 수. window=8 학습이므로 에피소드당 ≥ 20~40 프레임 권장.
3. 후보:
   - **SYNC(0.45s≈2.2Hz 캡처)**: T=10s면 ~22프레임. 적지만 V5 호환. 추론도 저Hz로 맞춰야.
   - **ASYNC(10Hz 캡처)**: T=10s면 ~100프레임. 촘촘, 추론 10Hz와 정합. **권장 후보**.
   - → 두 모드 세션 길이 비교 후 확정. (사용자 측정값 입력 대기)

---

## 5. 품질 게이트 (수집 즉시 폐기 기준 — 기존 플랜 재확인)
- `collision_fail` / `forward_collapse`(bbox 3프레임↑ 소실) / `mid_stop`(주행 중 유령정지) / `overshoot_fail`(FPE>0.15m) → 즉시 Discard.
- Action-Image Lag: s_t ← a_{t+1} (100ms 시프트, ASYNC).
- STOP 합성: 도착 직전 area>0.65 & cy>0.5 구간 자동 STOP 치환 (+ 수집 종료 시 stop_inject_n).

---

## 6. 미해결/결정 필요
- [ ] 수집 모드 확정: SYNC vs ASYNC (세션 타이밍 측정 후).
- [ ] T1 최소 직진 데이터로 먼저 closed-loop 돌려볼지(빠른 검증 루프) vs T2까지 모으고 한번에.
- [ ] 방해물/distractor를 T3에 넣을지 (chair 단일 task 수렴 우선이면 제외).
- [ ] STOP 합성 후처리 스크립트를 수집 파이프라인에 넣을지(현재 stop_inject_n 5프레임만).
