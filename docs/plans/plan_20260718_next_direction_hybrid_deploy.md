# 플랜: exp73 이후 방향성 — hybrid 헤드 실전화 + 트랙C 대기 중 할 일

> 작성일: 2026-07-18
> 근거: CH61(수집 캠페인 설계) → CH62(closed-loop 함정: offline 지표 ≠ 배포 성능) →
> CH63(exp73 ablation, hybrid 헤드 84.8% success 최종 1위)
> 상태: **검토 대기 — 승인 전 구현 금지**

## 0. 현황 분석 (최근 CH 종합)

| CH | 핵심 결론 | 남긴 숙제 |
|---|---|---|
| CH61 | 데이터에 신호 없으면 재수집이 유일한 해법. 트랙 A/B/C/F 설계 | 트랙C(64ep) 물리 수집 미착수 |
| CH62 | offline PM 1위가 closed-loop 0%일 수 있음(exp11). FPE/TLD/Success가 배포 판단 기준 | 평가는 kinematic replay — 실기 검증 아님 |
| CH63 | exp73: V6 225ep 단독 학습이 V5 혼합보다 우수. **hybrid 헤드(이산 lx/ly + 연속 az)가 offline 78.1% / closed-loop 84.8%로 최종 1위**. offline 1위(cxgeom)는 closed-loop 최하위 재확인 | hybrid는 아직 학습 스크립트 안에만 존재 — 추론 서버 미탑재 |

**블로킹 요인 (soda 대기)**: ① 트랙C 64ep 물리 수집, ② lx/ly 연속화 답변.
→ 이 플랜은 **minum 단독으로 지금 가능한 것**을 우선순위화한다.

## 1. 방향성 판단

세 갈래 중 선택:

- **(가) hybrid 헤드 실전화** — CH62의 교훈("offline 지표만 믿지 마라")을 그대로 적용하면,
  84.8%도 어디까지나 kinematic replay 수치. **실기(Jetson) closed-loop 확인이 다음 관문**이고,
  그 전제조건인 추론 서버 통합은 minum이 지금 할 수 있음. ← **권장 1순위**
- **(나) hybrid 헤드 자체 개선** — 연속 az를 이산화해 버리지 않고 궤적 적분에 직접 사용,
  az_thresh 스윕, seed 분산 확인. 코드 몇십 줄, 반나절. ← **권장 2순위 (가와 병행 가능)**
- **(다) 트랙C 도착 대비 재학습 파이프라인 정비** — 289ep 도착 즉시 원버튼 재학습+재평가.
  ← 3순위 (트랙C 일정 확정 후)

언어조건화(트랙B)는 CH61 결론(text 경로 구조적 사망)대로 계속 보류.

## 2. 작업 항목 (우선순위순)

### A. 연속 az를 살린 closed-loop 재평가 (0.5일, 코드 소규모)

현재 `hybrid_combine()`은 연속 az 예측을 ROT_L/ROT_R **이산 클래스로 되돌려서** 평가한다
— hybrid의 핵심 장점(연속 회전)을 평가에서 스스로 버리는 셈. 실기 배포 시엔 cmd_vel에
연속 az를 그대로 실을 수 있으므로, 평가도 그렇게 해야 배포 성능 예측이 맞다.

변경 (`scripts/sim/evaluate_closed_loop_exp73.py`만):

```python
# 신규: 클래스의 (lx,ly)는 ACTION_VEL에서, az는 모델 연속 예측을 그대로 사용
def build_trajectory_hybrid(lat_fwd_pred, az_pred, az_scale=1.15, dt=0.1):
    poses = [Pose(0, 0, 0)]
    for cls, az_n in zip(lat_fwd_pred, az_pred):
        lx, ly, _ = ACTION_VEL.get(int(cls), (0.0, 0.0, 0.0))
        poses.append(pose_step(poses[-1], lx, ly, float(az_n) * az_scale, dt))
    return poses
```

- `--az-mode {discrete,continuous}` 플래그로 두 방식 비교 저장.
- 기대: FPE 추가 감소(회전 미세보정 가능). 악화되면 az 회귀 품질 문제로 진단.
- 결과는 CH63에 63-10 카드로 추가.

### B. 추론 서버에 hybrid 헤드 탑재 (리서치 완료 — 아래는 구체 계획, 구현은 승인 후)

`inference_server.py`(3227줄, 운영 중인 FastAPI 서버)를 직접 리서치한 결과, **exp73은
이미 존재하는 `GoalNavMLPInference`(exp49~55 계열) 패턴에 신규 variant로 얹는 것으로
충분**하다는 걸 확인함 — 새 서브시스템이 필요 없음:

- `GoalNavMLPInference._DEFAULT_CKPTS`의 `exp54_s2v2`/`exp55`가 이미 exp73과 동일한
  `stage1_v2_projs.pt`(FrozenCLIPV2, VIS_DIM=1024→PROJ_DIM=256)를 인코더로 공유하고
  있음 — exp73(`train_exp73_trackA_heads.py`)도 정확히 같은 `STAGE1_PT` 경로를 사용.
- `CLASS_NAMES`/`CLASS_ACTIONS`(8-class → linear_x/y/angular_z 매핑)도 이미 서버에
  존재 — hybrid의 6-way+연속az 출력을 8-class로 결합(`hybrid_combine`, 63-10 검증 결과
  az_mode=**discrete** 유지)한 뒤 그대로 재사용 가능.
- **차이점**(신규 구현 필요한 부분만): (1) `WINDOW=6`(서버 기본은 8) — variant별
  분기 필요, (2) `BBOX_SCALE=3.0` 적용 위치 확인, (3) `HybridHead`(disc_head 6-way +
  az_head 연속) 클래스 자체를 서버에 import, (4) 결합 로직(`hybrid_combine`)을
  `az_mode=discrete`, `az_thresh=0.1`(63-10에서 민감하지 않음 확인)로 고정.

**계획된 변경**: `GoalNavMLPInference`에 `variant="exp73_hybrid"`를
`_DEFAULT_CKPTS`/`_GOAL_VARIANTS`/`_PROJ_VARIANTS` 판정 로직에 추가하고, `WINDOW`를
variant별 값으로 분기(현재 클래스 상수라 인스턴스 속성으로 변경 필요 — 다른 variant
영향 없는지 확인 필수), forward 시 `HybridHead` 사용 + `hybrid_combine` 결합.

⚠️ 실행 전 재확인 필요:
1. soda PG2 동시로드 크래시 메모리 — 운영 서버 떠있을 때 이 variant 로드 테스트는
   API 레벨(모델 로드+forward 1회)까지만, 실제 서버 재기동/실기 구동은 soda와 일정
   조율 필요.
2. `WINDOW`를 클래스 상수→인스턴스 속성으로 바꾸는 게 기존 exp49~55 variant 동작에
   영향 없는지 diff 확인.

**승인 대기** — 위 변경은 운영 중인 서버 파일(3227줄)에 대한 수정이라 CLAUDE.md
5단계 워크플로우(리서치 완료 → 이 계획 → 승인 → 구현)를 따름. 사용자 검토 후 진행.

### C. hybrid 견고성 확인 (0.5일, A와 같은 스크립트)

- az_thresh {0.05, 0.1, 0.2}/1.15 스윕 — 현재 0.1은 임의값.
- seed 3개 체크포인트 각각 closed-loop 평가 → success 분산 확인 (offline은 이미
  77.0/78.1/78.0으로 안정, closed-loop 분산은 미확인).

### D. 트랙C 도착 대비 원버튼 재학습 스크립트 (0.5일, 트랙C 일정 확정 후)

- `train_exp73_trackA_heads.py`에 신규 에피소드 디렉토리 추가만으로 289ep 재학습 +
  A의 closed-loop 재평가까지 이어지는 실행 순서를 플랜/스크립트로 고정.
- §5 평가지표(VSC/오버슈트회복률) 재검증 훅 포함.

## 3. 하지 않을 것

- soda 답변 전 lx/ly 연속화 관련 어떤 구현도 착수하지 않음 (수집 소프트웨어는 soda 소관).
- 트랙B(언어조건화) 관련 작업 일절 없음.
- `mona_dashboard.py` 등 soda 코드 수정 없음 (문서 요청만).

## 4. DoD

- [x] A: continuous-az closed-loop 결과 (discrete 대비 비교표) + CH63 63-10 기록 — 2026-07-18.
      **부정 결과**: continuous 모드가 오히려 악화(Success 84.8%→48.5%). az_mode=discrete
      유지로 결론. 원인: hybrid_combine의 discrete 결합이 비-STOP 프레임에서 az를
      암묵적으로 0-클램프하는 정규화 역할을 하고 있었음.
- [x] C(부분): az_thresh {0.05,0.1,0.2} 스윕 완료 — 결과 사실상 불변, 임계값 민감하지
      않음 확인. seed 3개 전체 closed-loop 분산은 **미착수**(학습 스크립트가 best-of-3만
      저장 — 재학습 필요, 배포 결정 안 바뀔 것으로 판단해 낮은 우선순위로 보류)
- [x] B: `GoalNavMLPInference`에 `variant="exp73_hybrid"` 추가, API 레벨(모델 로드+forward+
      reset) 검증 완료 — 2026-07-18. d_in=260/window=6 정상 판정, 8-class 예측 정상
      동작(CPU 스모크 테스트), `EXP73_AZ_THRESH=0.1`(63-10 근거)로 고정. env var
      `VLA_GOALNAV_VARIANT=exp73_hybrid`로 선택 가능(기본값은 여전히 exp49 — 실기
      전환은 soda와 별도 조율). **부수 발견(범위 밖)**: 기존 exp49 체크포인트 로드가
      이 환경 torch 버전(weights_only 기본값 변경)에서 실패 — 내 diff와 무관한 기존
      코드 줄이라 별도 이슈로 분리, 이번 작업 범위에서 수정하지 않음.
- [ ] D: 289ep 원버튼 재학습 절차 문서화
- [ ] soda 실기 테스트 일정 문의 동기화 (monavla-driving 문서)
