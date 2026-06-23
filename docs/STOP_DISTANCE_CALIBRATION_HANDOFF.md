# STOP 거리(40~50cm) 캘리브레이션 — soda 현장 작업 핸드오프

> 작성: 2026-06-24 (minum 서버) · 대상: soda(Jetson AGX, 실로봇) 세션
> 목적: "바스켓 40~50cm 앞에서 정지"를 실측 기반으로 캘리브레이션. 현재는 area>0.25(코드 기본값)가
> cm 단위 근거 없이 박혀있음 — 이번 작업으로 실제 거리에 매핑.

---

## 0. 현재 상태 — 정정 필요한 불일치

| 출처 | 값 | 근거 |
|---|---|---|
| `stage2_v2_inference_server.py` `GOAL_AREA_THRESHOLD` | **0.25** | S6~S8 실주행 세션에서 경험적으로 정해짐(주석: "바스켓 실주행으로 캘리브레이션된 값") — **cm 단위 매핑 기록 없음** |
| `docs/MONAPI_INTEGRATION.md` | 0.18 ≈ 0.5m, 0.30 ≈ 0.3m | 누군가의 **대략적 추정치**로 보임 — 실측 근거 불명, 코드 기본값(0.25)과도 안 맞음 |
| `configs/goal_area_map.json` | 파일 없음(비어있음) | "gray basket" phrase의 개별 오버라이드 아직 없음 — 지금 GOAL_AREA_THRESHOLD가 그대로 적용 중 |

→ 이번 작업으로 위 표의 추정치들을 **실측값 하나로 통일**한다.

---

## 1. 이미 준비된 도구 (코드 작성 불필요, soda에서 바로 실행)

| 스크립트 | 역할 |
|---|---|
| `scripts/calibrate_goal_area.py` | 물체를 정지거리에 직접 놓고 `/ground`로 area 측정 → `configs/goal_area_map.json`에 저장(이미 존재, 프로젝트 표준 방식) |
| `scripts/eval/calibrate_stop_distance.py` | 2개 이상의 (거리,area) 쌍으로 핀홀모델(area=k/distance²) 피팅 → 임의 거리의 area 역산(같은 측정값 재사용 가능) |

**둘 다 같은 측정 데이터를 쓸 수 있음** — 한 번의 현장 측정으로 충분.

---

## 2. 수집 목표 (soda에서 할 일)

### 2-1. 측정 거리 (4곳 권장)

```
30cm, 40cm, 45cm, 50cm
```
- 40cm/50cm: 목표 정지거리 양 끝(직접 보강)
- 30cm: 외삽 방지용 하한
- 45cm: 중간점 — 핀홀모델(area∝1/d²) 선형성 검증용

### 2-2. 측정 방법

1. 바스켓을 로봇 카메라 정면, **줄자로 정확히 측정한 거리**에 고정(바닥에 테이프로 거리 표시해두면 반복 측정 편함).
2. 카메라 높이/각도는 **실제 주행 시와 동일**하게 유지(로봇에 거치된 그대로, 손으로 들지 말 것).
3. 각 거리에서 3회 반복 측정(조명/미세 위치 흔들림 평균화):

```bash
# soda에서, 운영 서버(8001) 띄운 상태로
cd /home/soda/MoNaVLA
python3 scripts/calibrate_goal_area.py \
  --instruction "gray basket" \
  --camera /dev/video0 \
  --n 3 \
  --dry-run   # 4곳 다 측정할 때까지는 --dry-run으로 저장 안 함, 출력값만 기록
```

4. 매 거리마다 출력되는 `median_area` 값을 아래처럼 기록(이 핸드오프 문서나 별도 메모 어디든 상관없음):

```
30cm: area=0.XXXX
40cm: area=0.XXXX
45cm: area=0.XXXX
50cm: area=0.XXXX
```

> `--camera` 대신 이미 찍은 사진 파일이 있으면 `--image-glob "/tmp/calib_*.jpg"`로 대체 가능.

---

## 3. 측정 후 — minum으로 결과만 보내주면 정리

4쌍의 (거리,area)를 보내주면 이쪽에서:

```bash
.venv/bin/python3 scripts/eval/calibrate_stop_distance.py \
  --measurements 30:<area> 40:<area> 45:<area> 50:<area> \
  --targets 40 45 50
```

- 핀홀모델 피팅 잔차(residual_std)로 측정 신뢰도 확인
- 40~50cm 구간에 대한 최종 권장 threshold 산출
- `configs/goal_area_map.json`에 `"gray basket": <threshold>` 반영(또는 `GOAL_AREA_THRESHOLD` 코드 기본값 자체를 갱신)
- `research_story.html` TODO 항목(CH44-3) 갱신 + 새 챕터로 결과 기록

---

## 4. 주의사항

- **서버 재시작 필요**: `goal_area_map.json`은 모듈 로드 시점에 1회만 읽음 — 측정 반영 후 `go.sh --stop && go.sh --server`로 재시작해야 적용됨.
- soda PG2 동시로드 크래시 주의(기존 메모): 운영 서버 떠있는 상태에서 별도로 PG2를 또 로드하지 말 것 — `calibrate_goal_area.py`는 운영 서버의 `/ground` 엔드포인트를 호출만 하므로 별도 모델 로드 없음, 안전.
- 측정 중 바스켓 외 다른 물체가 화면에 들어오지 않도록(오탐 방지).
