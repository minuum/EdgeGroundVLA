# Plan — 학습/추론 이미지 처리 통일 + STOP 거리/latency 튜닝

> 작성: 2026-06-22 · 상태: **승인됨 — §2(1+2) 구현 진행, 3/4/5는 실측 데이터 선행 필요해 보류**
> 동기: 사용자가 5가지 가설/액션아이템 제시 — 학습/추론 이미지 리사이즈 통일(1280x720→224x224), latency 4초→1초 이내, STOP 거리 40~50cm, hidden state-이미지 해상도 alignment 의심.

---

## 0. 리서치 결과 — 사용자 가설을 코드로 검증한 결과 (정직하게)

| 가설 | 검증 결과 |
|---|---|
| 1. 학습 시 1280x720을 224x224로 바꿔야 함 | **부분적으로 이미 그렇다.** `train_exp54_stage2_v2_action.py`/`train_hidden_state_action.py`(이번 세션 CH40~43에 쓴 스크립트)는 h5 원본 이미지를 그대로 HuggingFace `AutoProcessor`에 넘기고, **processor가 내부적으로 224x224로 리사이즈**한다(Kosmos-2/PG2 둘 다 모델명 자체가 `-224`). 명시적으로 우리가 리사이즈 코드를 쓴 게 아니라 processor의 암묵적 동작에 의존하고 있다는 게 차이. |
| 2. 추론 시도 224x224로 리사이즈해서 넣기 | **이미 그렇다.** `stage2_v2_inference_server.py`의 `Stage1Encoder.encode_image()`/`PG2Grounder.run()`도 똑같이 원본을 PIL로만 변환하고 같은 `AutoProcessor`에 넘긴다 — **학습과 추론이 같은 코드(같은 processor)를 쓴다.** 즉 "리사이즈 크기"가 다른 게 아니다. |
| 3. latency 4초 → 1초 이내 | **이번 세션 GB10 실측은 560~620ms(warm)였다 — 이미 1초 이내.** "4초"가 어디서 나온 수치인지 명확하지 않음(soda/Jetson 실측일 수도, PG2 cold-start 43초와 다른 것). **확인 필요**: soda에서 직접 측정한 적 없음. |
| 4. STOP 40~50cm | 현재 `GOAL_AREA_THRESHOLD=0.25`(정규화 면적, cm 아님) — **cm 단위로 바꾸려면 카메라-거리 캘리브레이션 데이터가 필요**(현재 없음). |
| 5. HS와 입력 해상도 alignment 의심 | 위 1/2 검증 결과, 적어도 **Stage1/Stage2 decomposition 파이프라인(Exp54~66, CH39~43)에서는 학습/추론이 같은 224x224 경로**를 쓰므로 "해상도 자체"가 mismatch 원인일 가능성은 낮다. **다만 "리사이즈 크기"가 아니라 "실제 카메라 캡처 vs h5에 저장된 이미지"의 화질/색공간/압축 차이**는 코드만 봐서는 확인 불가 — 직접 비교가 필요(범위 밖, 별도 검증). |

**중요**: `robovlm_nav/datasets/nav_h5_dataset_impl.py`의 `_resize_image()`는 **letterbox(여백 패딩)/squish(찌그러뜨림) 직접 구현**을 갖고 있는데, 이건 **다른 학습 스크립트(end-to-end 계열, Exp01~16 추정)에서만 쓰이고 CH39~43이 쓴 decomposition 스크립트는 안 씀** — 두 계열이 섞이면 진짜 리사이즈 방식 불일치가 생길 수 있으니, 어느 파이프라인을 디버깅 중인지 명확히 할 필요가 있다.

---

## 1. 제안 (검증 결과 기반으로 사용자 5개 항목 재구성)

| 항목 | 제안 | 비고 |
|---|---|---|
| 1+2. 리사이즈 통일 | "이미 같다"는 걸 명시적으로 만들기 — processor의 암묵적 리사이즈에 의존하지 말고, **train/inference 양쪽에 동일한 명시적 `resize(224,224)` 전처리 함수를 하나 만들어 공유**(현재처럼 "어쩌다 같다"가 아니라 "코드로 강제 같다"로). | 동작은 안 바뀜, 안전망 추가 |
| 3. latency | soda(실제 로봇)에서 동일 방식으로 직접 측정 — "4초"가 어디서 나왔는지 먼저 확인. GB10(560~620ms)과 다르면 하드웨어 차이(Jetson vs GB10)로 설명될 수 있음. | soda 접속 필요, [[soda-pg2-concurrent-load-crash]] 준수 |
| 4. STOP 40~50cm | 카메라 거리-면적 캘리브레이션 먼저 — 로봇이 바스켓에서 40cm/50cm 떨어진 실측 사진 몇 장 찍어서 그때 area가 얼마인지 확인한 다음 GOAL_AREA_THRESHOLD를 cm 기준으로 역산. | 실측 데이터 필요(범위 밖, 별도 진행) |
| 5. HS-해상도 alignment | 코드 검증으로는 "해상도"가 원인은 아닌 것으로 보임 — 대신 **실제 카메라 프레임 1장과 같은 장면의 h5 저장 프레임 1장을 나란히 놓고 픽셀 비교**(색공간, 압축, 밝기)를 다음 단계로 제안. | 실측 비교 필요 |

---

## 2. 변경 파일 (1+2만 — 나머지는 실측/캘리브레이션 선행 필요)

| 파일 | 작업 |
|---|---|
| `robovlm_nav/image_preprocess.py` (신규) | `resize_for_vlm(pil_img, size=224) -> PIL.Image` 공유 함수 |
| `robovlm_nav/serve/stage2_v2_inference_server.py` | `Stage1Encoder.encode_image()`, `PG2Grounder.run()` 호출 전에 위 함수로 명시적 리사이즈 |
| `scripts/train_hidden_state_action.py`(및 동일 계열) | `load_images()`에서 같은 함수로 리사이즈 |

## 3. 트레이드오프

- 1/2은 "이미 똑같이 동작하는 걸 명시적으로 만드는" 안전 조치라 위험 낮음, 바로 진행 가능.
- 3/4/5는 **실측 데이터(soda 접속, 거리 캘리브레이션, 카메라-h5 픽셀 비교)가 먼저 필요** — 이번 plan에서 코드만으로는 더 못 감. soda 접속이 필요한 부분은 별도 승인 필요.

## 4. 완료 기준
- [ ] `image_preprocess.py` 공유 함수 작성, 학습/추론 양쪽에 적용
- [ ] (보류) soda latency 실측 — "4초"의 출처 확인
- [ ] (보류) STOP 거리 캘리브레이션 — 실측 데이터 확보 후
- [ ] (보류) 카메라 vs h5 픽셀 비교 — 실측 후
