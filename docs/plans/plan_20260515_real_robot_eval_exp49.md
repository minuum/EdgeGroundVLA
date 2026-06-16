# Plan — Exp49 실로봇 평가 프로토콜

작성: 2026-05-15  
브랜치: 계획 `inference-integration` / 구현 `monavla-driving`

---

## 0. 한 줄 목표

Exp49(GoalNav MLP)의 오프라인 CL 96.7%가 실로봇 환경에서도 재현되는지 확인한다.

---

## 1. 배경

| 항목 | 값 |
|------|-----|
| 모델 | Exp49 GoalNav MLP (d_in=1059, bbox+vision+goal) |
| 오프라인 val acc | 96.4% (bootstrap CI: [94.7%, 97.9%]) |
| 오프라인 CL 성공 | 96.7% (9개 path type 전부 포함) |
| 서버 상태 | VLA_MODEL=exp49, port 8001, 실행 중 |
| 이전 기록 | Exp14 Step2 66.7%, Exp25(end-to-end) 55.6% |

---

## 2. 테스트 구성

### 핵심 아이디어

Exp49의 goal은 **cx0** (첫 프레임에서 Kosmos-2가 grounding한 바스켓 x위치).  
9개 path type은 수집 분류일 뿐 — 실로봇 eval은 **바스켓 위치 3종**만으로 충분하다.

| 바스켓 위치 | cx0 범위 | 기대 초기 액션 | 오프라인 CL 성공 |
|------------|---------|--------------|----------------|
| **LEFT** | ≈ 0.1~0.38 | LEFT / FWD+L / ROT_L | 9/9 = 100% |
| **CENTER** | ≈ 0.38~0.62 | FORWARD | 12/12 = 100% |
| **RIGHT** | ≈ 0.62~0.9 | RIGHT / FWD+R / ROT_R | 8/9 = 88.9% |

로봇의 **시작 방향**은 중요하지 않다 — 바스켓이 어디 보이느냐가 goal을 결정한다.

### 2.1 테스트 구성 (3위치 × 최소 3회 = 9회)

| 바스켓 위치 | 설치 방법 | 최소 횟수 |
|------------|---------|---------|
| LEFT | 바스켓을 로봇 카메라 좌측 1/3 구역에 배치 | 3회 |
| CENTER | 바스켓을 카메라 중앙 1/3 구역에 배치 | 3회 |
| RIGHT | 바스켓을 카메라 우측 1/3 구역에 배치 | 3회 |

총 최소 9회. 여유 있으면 각 위치에서 **로봇 시작 각도를 다르게** 해서 3회씩 추가.

### 2.2 우선 확인 (10분 사전 체크)

```bash
# 서버 상태 확인
curl http://100.85.118.58:8001/health
# 기대: {"status":"healthy","model_name":"exp49",...}

# grounding 동작 확인 (단일 이미지)
# Gradio 대시보드 http://100.85.118.58:7860 에서 이미지 1장 전송 → bbox 표시 확인
```

---

## 3. 실행 절차

### Step 1 — 서버 확인 (로봇 켜기 전)
```bash
# soda 서버에서
curl http://localhost:8001/health | python3 -m json.tool
# model_name이 exp49인지 확인
# model_loaded가 false이면 첫 요청 시 자동 로드 (30초 대기)
```

### Step 2 — Gradio 세션 eval로 사전 체크
```bash
# 로봇 실행 전, 오늘 수집된 최근 H5 에피소드로 세션 평가
python3 scripts/gradio_session_eval.py --port 7861
# 브라우저: http://100.85.118.58:7861
# → grounding_success_rate 확인 (< 50% 이면 조명/바스켓 위치 조정)
```

### Step 3 — Trial Logger 실행 (별도 터미널)
```bash
# soda 서버에서 (모니터 또는 laptop 브라우저로 접근)
python3 scripts/real_robot_trial_logger.py --port 7862
# 브라우저: http://100.85.118.58:7862
```

Trial Logger UI:
- **Path Type** 드롭다운 → **성공/실패** 라디오 → **실패 원인** (실패 시 표시) → **기록** 버튼
- 오른쪽 패널에 오프라인 CL vs 실로봇 비교 테이블 실시간 갱신
- 자동 저장: `docs/v5/eval/real_robot_exp49_YYYYMMDD_HHMMSS.json`

### Step 4 — 로봇 실행 + 기록

각 시도 순서:
1. Gradio 메인 대시보드(7865)에서 path_type instruction 설정 → **Inference (Auto)** 시작
2. 로봇 완주 또는 중단 확인
3. Trial Logger(7862)에서 결과 기록 → 📝 기록 버튼

---

## 4. 기록 형식 (자동 저장)

```json
{
  "model": "exp49",
  "date": "2026-05-15",
  "server": "soda@100.85.118.58:8001",
  "total_trials": 18,
  "trials": [
    {
      "trial_id": 1,
      "timestamp": "2026-05-15T14:23:11",
      "path_type": "center_straight",
      "success": true,
      "failure_reason": null,
      "notes": ""
    }
  ]
}
```

저장 위치: `docs/v5/eval/real_robot_exp49_YYYYMMDD_HHMMSS.json`  
구현: `scripts/real_robot_trial_logger.py` (monavla-driving 브랜치)

---

## 5. 성공 기준

| 지표 | 기준 |
|------|------|
| 전체 성공률 | ≥ 80% (오프라인 96.7%의 ±20% 허용) |
| center_straight | ≥ 1/2 성공 |
| 곡선 경로 | ≥ 1/2 성공 |
| straight+rot | ≥ 1/2 성공 |

오프라인과 실로봇 차이(sim-to-real gap)가 > 20%p 이면 원인 분석 필요.

---

## 6. 실패 시 대응

| 증상 | 원인 후보 | 대응 |
|------|----------|------|
| 항상 FORWARD | grounding 실패 (bbox 없음) | 조명 조정, 바스켓 위치 변경 |
| 방향 반대 | 카메라 좌우 반전? | goal cx0 = 1-cx 로 테스트 |
| 초반 정상, 후반 드리프트 | 시간차 오류 누적 | step 수 줄이기 |
| 완전 정지 | STOP 클래스 과예측 | goal area 점검 |

---

## 7. 체크리스트

**사전 준비 (soda에서)**
- [ ] `curl localhost:8001/health` → model_name=exp49 확인
- [ ] `python3 scripts/real_robot_trial_logger.py --port 7862` 실행
- [ ] Gradio session eval (7861)로 최근 H5 품질 사전 점검

**테스트 중**
- [ ] 9개 path type × 2회(최소) 실시
- [ ] 매 시도 후 Trial Logger에 즉시 기록

**테스트 후**
- [ ] Trial Logger에서 JSON 경로 확인 버튼 → 파일 커밋
- [ ] 오프라인 CL과 비교: 차이 > 20%p 이면 원인 분석
- [ ] `inference-integration` 브랜치에 결과 JSON 커밋

## 8. 실로봇 결과 커밋 방법

```bash
# 테스트 완료 후 (minum 로컬에서)
scp soda@100.85.118.58:MoNaVLA/docs/v5/eval/real_robot_exp49_*.json \
    docs/v5/eval/
git add docs/v5/eval/real_robot_exp49_*.json
git commit -m "eval(real-robot): Exp49 실로봇 결과 YYYYMMDD"
git push origin inference-integration
```
