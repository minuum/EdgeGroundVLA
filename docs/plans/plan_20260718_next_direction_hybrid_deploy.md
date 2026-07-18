# 플랜: exp73 이후 방향성 — hybrid 헤드 실전화 + 트랙C 대기 중 할 일

> 작성일: 2026-07-18
> 근거: CH61(수집 캠페인 설계) → CH62(closed-loop 함정: offline 지표 ≠ 배포 성능) →
> CH63(exp73 ablation, hybrid 헤드 84.8% success 최종 1위)
>
> ⚠️ **2026-07-19 정정**: `evaluate_closed_loop_exp73.py`의 val split이 학습 스크립트와
> 다른 랜덤 API(`RandomState` vs `default_rng`)를 써서 val 33ep 중 27ep가 실제로는
> 학습 데이터였음이 드러남 — 이 플랜에 기록된 "hybrid 84.8%" 등 A/B/C 항목의 수치는
> 전부 오염됨. 수정 후 재검증 결과는 CH63 63-11 및 아래 각 항목의 [정정] 표시 참고.
> **정정된 실제 1위는 hybrid가 아니라 pg448/v6/mlp(트랙F 없음, Success 60.6%)**.
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

### D. 트랙C 도착 대비 289ep 재학습 런북 (리서치 완료 — soda 물리수집 완료 후 실행)

기존 exp73 파이프라인(`scripts/run_exp73_pipeline.sh`) 리서치 결과, 트랙C(64ep) 추가는
**신규 파이프라인이 아니라 기존 4단계에 소스 파일 갱신만 끼워 넣으면 된다**:

1. **frame-level 소스 갱신** (`scripts/gen_v6_frame_level.py`)
   - 현재 `DATA.glob("episode_2026071*.h5")`로 V6 225ep(2026-07-1x 수집분)만 하드코딩
     매칭 중. 트랙C가 수집되면 날짜 prefix가 다를 것이므로(예: `2026072*`),
     `--src-glob` 인자를 추가하거나 glob 패턴을 트랙C 날짜까지 포함하도록 수정 필요
     — **파일 자체가 이미 트랙A+F(225ep) 전용으로 하드코딩돼 있어 트랙C를 그냥
     넣으면 씹힘. 실행 전 반드시 glob 패턴부터 확인.**
   - 출력을 `bbox_dataset_v6c_frame_level.json`(신규 파일명, 기존 225ep본 보존)으로 분리 저장 권장.
   - `to_class()` 8-class 규칙은 트랙C도 동일 임계값(|x|>0.3, az>±0.1)이라 수정 불필요.

2. **PG448 그라운딩 주석** (`scripts/gen_v6_pg448_annotation.py`)
   - `--src bbox_dataset_v6c_frame_level.json --out bbox_dataset_v6c_pg448_cx.json`
   - 신규 트랙C 프레임만 그라운딩되므로 기존 225ep 결과 재계산 불필요(`--resume` 지원 확인됨).

3. **vis 캐시 재생성 + 학습** (`train_exp73_trackA_heads.py`)
   - `CACHE_V6`(`exp73_v6_vis_cache.pt`)는 225ep 전용 — 289ep용 신규 캐시 경로 필요.
     `--ann-v6 .../bbox_dataset_v6c_pg448_cx.json` 지정 시 캐시가 없으면 자동 재인코딩됨
     (§ANN_V6 캐시 로직 확인됨, 코드 수정 불필요 — 인자만 다르게 실행).
   - `--heads hybrid --arms v6 --tag pg448_trackC` 로 hybrid 헤드만 재학습(다른 헤드는
     이미 결론 났으므로 재확인 불필요, 63-7 결론 참고).
   - 3-seed 그대로 유지, best-of-3 저장 방식도 기존과 동일.

4. **closed-loop 재검증** (`scripts/sim/evaluate_closed_loop_exp73.py`, 이번 세션에 완성)
   - `--ckpt runs/v5_nav/mlp/exp73/exp73_pg448_trackC_v6_hybrid.pt --head hybrid`
     (az_mode는 discrete 고정 — 63-10 결론).
   - §5 평가지표(VSC/오버슈트회복률/반응지연) 중 **오버슈트회복률**이 트랙C 추가의
     직접 목적이므로, 이 지표만큼은 `evaluate_closed_loop_exp73.py`에 없는 별도 계산이
     필요 — 트랙C(overshoot_left_recover/overshoot_right_recover) episode만 필터링해
     별도 success/FPE 집계 스크립트 추가가 이 단계에서 신규로 필요함(기존 스크립트
     재사용 불가, 사전 확인 완료).

**실행 시점**: 트랙C 64ep 물리 수집(soda) 완료 통보 후 착수. 현재는 실행하지 않음.

## 3. 하지 않을 것

- soda 답변 전 lx/ly 연속화 관련 어떤 구현도 착수하지 않음 (수집 소프트웨어는 soda 소관).
- 트랙B(언어조건화) 관련 작업 일절 없음.
- `mona_dashboard.py` 등 soda 코드 수정 없음 (문서 요청만).

## 4. DoD

- [x] A: continuous-az closed-loop 결과 (discrete 대비 비교표) + CH63 63-10 기록 — 2026-07-18.
      **부정 결과(상대 순위는 정정 후에도 유지)**: continuous 모드가 오히려 악화.
      최초 측정 discrete 84.8%→continuous 48.5%는 val split 버그로 오염(아래 정정 참고),
      수정된 split 재검증 결과 discrete 39.4% > continuous 33.3% — discrete 우위
      결론은 그대로 유효, 절대 수치만 하향 정정.
- [x] C(부분): az_thresh {0.05,0.1,0.2} 스윕 완료 — 결과 사실상 불변, 임계값 민감하지
      않음 확인(이 결론은 split 버그와 무관하게 유효). seed 3개 전체 closed-loop 분산은
      **미착수**(학습 스크립트가 best-of-3만 저장 — 재학습 필요, 낮은 우선순위로 보류)
- [x] B: `GoalNavMLPInference`에 `variant="exp73_hybrid"` 추가, API 레벨(모델 로드+forward+
      reset) 검증 완료 — 2026-07-18. d_in=260/window=6 정상 판정, 8-class 예측 정상
      동작(CPU 스모크 테스트). **[2026-07-19 정정] 아래 val split 버그로 hybrid가
      "최종 1위"라는 선정 근거 자체가 무효 — 코드 인프라(variant 확장 메커니즘)는
      재사용 가능하나 실제 서버에 태울 체크포인트는 `exp73_pg448_v6_mlp.pt`(트랙F
      없는 mlp, 정정된 진짜 1위)로 재검토 필요. 서버에 별도 `exp73_mlp` variant 추가는
      후속 작업으로 분리.** 부수 발견(범위 밖, 무관): 기존 exp49 체크포인트 로드가
      이 환경 torch 버전(weights_only 기본값 변경)에서 실패 — 별도 이슈로 분리.
- [x] D: 289ep 재학습 런북 문서화 완료 — 2026-07-18. 기존 4단계 파이프라인 재사용
      가능함을 확인, 단 (a) `gen_v6_frame_level.py`의 날짜 glob 하드코딩은 트랙C 수집 후
      반드시 먼저 수정 필요, (b) 오버슈트회복률 집계는 기존 스크립트로 안 되고 신규
      스크립트 필요함을 사전 확인, **(c) [2026-07-19 추가] 재학습 후 재평가 시
      `evaluate_closed_loop_exp73.py`의 `val_split()`이 학습 스크립트와 동일한
      `np.random.default_rng` 기반인지 반드시 재확인할 것(이번에 발견된 버그 재발 방지)**.
      실행은 트랙C 물리 수집 완료 후.
- [x] soda 실기 테스트 일정 문의 동기화 — 2026-07-18, `monavla-driving`
      `docs/DATASET_V6_STATUS.md`에 exp73_hybrid 서버 통합 완료 + 실기 테스트 일정
      문의 추가. **[2026-07-19] 정정 메시지 추가 동기화 완료(아래)**.

## 5. [2026-07-19 정정] val split 버그 및 재평가 결과

`evaluate_closed_loop_exp73.py`의 `val_split()`이 `np.random.RandomState(42)`(레거시 API)를
쓰고 있었는데, 학습 스크립트(`train_exp73_trackA_heads.py` `main()`)는
`np.random.default_rng(42)`를 씀 — **같은 seed=42라도 다른 셔플 순서**가 나와서
"val 33ep" 중 27ep가 실제로는 학습 데이터였다(6ep만 진짜 겹침). A/B/C 전 항목에
보고된 closed-loop 수치가 오염됨.

`np.random.default_rng`로 통일 후 exp73 전 체크포인트 재평가 — 정정된 리더보드는
CH63 63-11 참고. 핵심 변경: **hybrid(구 84.8%)가 아니라 pg448/v6/mlp(트랙F 없음,
60.6%)가 진짜 1위**. 트랙F 추가가 closed-loop를 개선한다던 결론도 반대로 뒤집힘.

이 버그는 minum이 스스로 발견한 게 아니라, 사용자가 "정확도 낮은 부분"을 물어봐서
클래스/청크 단위로 파고들다 val split 불일치를 우연히 포착한 것 — 향후 두 스크립트의
split 로직을 단일 유틸 함수로 통합해 재발을 막는 게 안전(§D에 반영).
