# Plan — 통합 메인 서버 + 실측 + Hz 정합 + 의자 객체 교체

> 작성 2026-06-10. 승인 전 구현 금지. 사용자 메모 반영 후 갱신.

## 0. 배경 / 리서치 결론

이번 실측 시도에서 **두 개의 프로덕션 버그**가 드러났고, 이게 지난 세션 "STOP loop 회귀"의 근본 원인이다.

### 버그 A — VLA `/predict` 기본값이 MLP 가중치를 로드 (회귀 스모킹건)
- 쉘 환경(프로필 `proxy_default`, `scripts/vla_profile.py`가 export)이:
  ```
  VLA_CHECKPOINT_PATH = runs/v5_nav/mlp/exp49/exp49_mlp.pt   # GoalNav MLP(2.8MB)
  VLA_CONFIG_PATH     = configs/brown_pot_left.json          # Kosmos VLA config
  ```
- `inference_server.py`의 VLA 경로(`MobileVLAInference`)가 Kosmos-2 VLM(31.75M)을 빌드한 뒤 exp49_mlp.pt를 로드 → `act_head.*`, `backbone.image_to_text_projection.*` 가중치가 **전부 missing → 랜덤 초기화** → 출력 garbage/STOP.
- health는 200을 주지만 실제 추론은 망가진 상태. (증거: `logs/measure_8082.log`의 missing-keys set)

### 버그 B — 통합서버 GoalNav `exp49` variant d_in mismatch
- exp49/50/51 체크포인트 첫 레이어 = `(512, 1059)` = **1056 + goal 3-dim**.
- `inference_server.py:GoalNavMLPInference`는 variant별 `D_IN`을 하드코딩(exp49=1056, goal 누락) → `size mismatch 1059 vs 1056` 로드 실패.
- 반면 `proxy_inference_server.py`(8001)는 `d_in`을 **체크포인트에서 동적으로 읽어** 빌드 → exp49/50/51 정상.

### 현 서버 구조 (사실)
- `inference_server.py`(8082)는 **이미 통합 서버**: `/predict`(VLA) + `/goalnav/predict`(GoalNav MLP) + `/predict_mlp`(Exp47 MLP).
  - GoalNav variant 테이블: exp49 / exp54_s2v2 / exp55 만 등록 (exp50~53 누락).
  - exp49는 위 버그 B로 깨짐. 동작하는 건 exp54_s2v2 / exp55 (288-dim, goal 없음).
- `proxy_inference_server.py`(8001)는 GoalNav 전용: exp49/50/51/52/53 + exp53 CLIP LoRA. d_in 동적. **이쪽이 exp49 정상 경로**.
- 즉 두 서버에 GoalNav 로직이 **중복**되어 있고, 8082쪽이 미완성.

### 실험 현황 (학습 완료, 모두 GoalNav MLP 계열 = decomposition 노선)
| Exp | d_in | PM/val | 비고 |
|-----|------|--------|------|
| exp49 | 1059 (goal+) | 96.4% val / CL 100% | 운영 추천(proxy) |
| exp50/51 | 1059 | 92~93% | |
| exp52 | (파일 위치 불명, 재확인) | 93.9% | lang+vis |
| exp53 | nested(mlp/d_in) + CLIP LoRA | 94.7% | grounding 최고 |
| exp54_s2v2 | 288 | 미평가 | 2-stage contrastive (8082 동작) |
| exp55 | 288 | — | 경량 |
- ch23~28 결론: E2E vision-LoRA는 구조적으로 depth grounding 불가 → **decomposition 정답** 확정.

---

## 1. 작업 범위 (우선순위 순)

### Task 1 — 버그 수정 (실측·운영의 전제, 최우선)
1A. **GoalNav d_in 동적화** — `inference_server.py:GoalNavMLPInference`가 체크포인트에서 d_in을 읽어 MLP를 빌드하도록 변경 (proxy 방식 이식). exp49(1059, goal 포함)/exp50/51 등록.
   - exp49 입력 = `bbox_hist(32) + vis(1024) + goal(3)` = 1059. goal 3-dim을 추론 입력에 포함해야 함 → proxy의 goal 계산 로직 확인 후 이식.

1B. **VLA가 MLP 체크포인트를 거부** — `MobileVLAInference` 로드 시 state_dict가 numeric-only(MLP) 이면 명확한 에러로 막거나, env가 MLP 경로면 무시. 최소한 `vla_profile.py`의 `proxy_default` 프로필이 VLA 경로에 MLP를 넣지 않도록 분리.
   - 옵션: `proxy_default` 프로필을 `goalnav` 런타임으로 명시 → 서버가 `VLA_GOALNAV_ONLY=1`로 기동되게.

```python
# 1A 핵심 변경 (GoalNavMLPInference._build_mlp / _load_weights)
sd = torch.load(mlp_path, map_location="cpu")
if isinstance(sd, dict) and "mlp" in sd:          # exp53/54/55 nested 포맷
    d_in = sd.get("d_in"); state = sd["mlp"]
else:                                              # exp49/50/51 plain Sequential
    state = sd
    d_in = state["0.weight"].shape[1]              # 1059 동적 추출
self._mlp = self._build_mlp(d_in)                  # 하드코딩 D_IN 제거
self._mlp.load_state_dict(state)
```

### Task 2 — 통합 메인 서버 일원화
- 8082를 **단일 메인 서버**로 확정. proxy(8001)의 GoalNav variant(exp49~53, CLIP LoRA)를 8082로 흡수.
- `/goalnav/predict`가 `VLA_GOALNAV_VARIANT` 또는 요청 파라미터로 exp49~55 + exp53(CLIP LoRA) 스위칭.
- 8001 proxy는 **deprecate**(주석/문서). start_all.sh는 이미 8082 단일.
- 대시보드(7865)는 8082로 연결. `vlm_model` 파라미터로 GoalNav/VLA/monapi 라우팅.
- **결정 필요(아래 질문)**: VLA full 경로(`/predict`)를 유지할지(현재 깨짐 + 폐기 노선) vs 8082를 GoalNav 전용으로 만들지.

### Task 3 — 실측 (Task 1 완료 후)
- `tools/measure_latency.py`로 `/goalnav/predict`(exp49) 1회 추론 latency 측정 → 10Hz(100ms) 물리적 가능 여부 확정.
- 카메라 서비스 프레임 획득 주기도 함께 측정 (GetImage 서비스 왕복).
- 결과를 `docs/inference_reports/`에 기록.

### Task 4 — Hz 정합
- 데이터 수집 = 10Hz async (확정, Action Lag 1프레임 시프트 검증 완료).
- `api_client_node` = 10Hz (이미 일치).
- `vla_inference_node.py:38` `inference_interval = 0.5`(2Hz) → **실측 latency 기반으로 0.1(10Hz) 또는 가능한 최소값으로 조정**. 추론 latency가 100ms 초과면 그 값에 맞춤(예: 150ms면 ~6.7Hz). → 실측 후 결정.

### Task 5 — 의자 객체 교체 데이터 수집
- 객체 = **의자/스툴** (PaliGemma 인지 98%). `docs/v5/PRETRAINED_OBJECT_REPLACEMENT_PLAN.md` 프로토콜 사용.
- 수집: 10Hz async, 메인경로 70% / 복원경로 30%, 진입각 정면·±30°, 조명 7:2:1.
- 수집 후 grounding 재주석(의자 캡션) → GoalNav 재학습(Exp60+).

### Task 6 — 조이스틱 수집 쾌적성 + 관측 (✅ 1차 완료 2026-06-10)
- 조이스틱 코드 분석 결과 (이상 없음 + 잠재 이슈 3):
  - 구조 정상: 25Hz 폴링, 재연결, SYNC/ASYNC, 300ms Jitter Hold 모두 동작.
  - 이슈1 ASYNC neutral 시 프레임 미캡처 → 보류 (수집 영향 검증 후 결정).
  - 이슈2 회전 미세보정 불가(az 0.5 bang-bang) → **✅ ROT 임계값 슬라이더 옵션화 (0.1~0.7, 기본 0.5)**.
  - 이슈3 Action Lag +1 이중보정 위험 → 보류 (walkthrough 검증됨, 설계 존중).
- ✅ 관측 위젯 4종 추가 (read-only, 기존 수집 로직 무수정):
  1. 라이브 액션 타임라인 (최근 28프레임 기호 + 실측 캡처 Hz)
  2. 이번 에피소드 8-class 분포 막대
  3. 최근 캡처 프레임 썸네일 스트립 (gr.Gallery, 액션 라벨)
  4. 전체 데이터셋 8-class 누적 분포 (파일 수 변동 시 캐시 갱신)
  - 8-class 분류는 nav_h5_dataset_impl.py와 동일 임계값(|x|>0.3,|y|>0.3,|az|>0.1), WASD 8종 검증 통과.
  - 스모크 테스트: 8081 기동 HTTP 200, 실데이터 289ep/5482frames FORWARD 70% 정상 표시.
- 후속(실수집하며 튜닝): 가감속 ramp, 데드존 조정, neutral 캡처(이슈1) 재검토.

---

## 2. 수정 대상 파일
| 파일 | 변경 |
|------|------|
| `robovlm_nav/serve/inference_server.py` | GoalNavMLPInference d_in 동적화, exp49~53 등록, CLIP LoRA, VLA-MLP 오로드 차단 |
| `scripts/vla_profile.py` / 프로필 정의 | `proxy_default`를 goalnav 런타임으로 분리(VLA 경로에 MLP 금지) |
| `scripts/start_all.sh` | 8082 기동 시 `VLA_GOALNAV_ONLY=1` + `VLA_GOALNAV_VARIANT=exp49` 명시 |
| `scripts/gradio_inference_dashboard.py` | 8082 연결 확인, GoalNav variant 선택 UI |
| `ROS_action/.../vla_inference_node.py` | `inference_interval` 실측 기반 조정 |
| `scripts/gradio_data_collector.py` | (Task 6) ramp/데드존/HUD — 실수집하며 튜닝 |
| `robovlm_nav/serve/proxy_inference_server.py` | deprecate 표기 |

## 3. 트레이드오프 / 리스크
- VLA full 경로 비활성화 시: 혹시 모를 VLA 데모 필요성 상실. → 환경변수로 on-demand 로드 유지하면 해소.
- exp49 goal-3dim 추론 입력: goal 벡터를 추론 시 어떻게 채우는지(grounded goal) proxy 로직 정확 이식 필요. 잘못하면 또 mismatch. → 1A 구현 전 proxy의 goal 계산부 정독.
- Jetson 16GB 통합메모리: Kosmos-2 vision encoder(GoalNav) + (옵션)VLA full 동시 로드 시 OOM 위험 → GoalNav 전용 권장.

## 4. 진행 순서
1. (승인 후) Task 1A/1B 버그 수정 → 서버 정상 기동 확인
2. Task 3 실측 → latency 확정
3. Task 4 Hz 조정 (실측값 기반)
4. Task 2 서버 일원화 마무리 + 대시보드 연결
5. Task 5/6 의자 수집 시작 (실수집하며 조이스틱 튜닝)
