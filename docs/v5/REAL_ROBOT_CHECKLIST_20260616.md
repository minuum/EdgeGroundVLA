# Exp66 실기 테스트 체크리스트 (2026-06-16/17)

> 모델: Exp66 ActionMLP w=8, Stage1 shared, PG2 grounding  
> 목표: CL 성공률 확인 + PG2 STOP 로직 실환경 검증

---

## 0. 사전 드라이런 (로봇 전)

```bash
cd /home/minum/26CS/MoNaVLA
.venv/bin/python3 scripts/dryrun_stop_logic.py
```
- [ ] 전체 5장 이미지 테스트 PASS (특히 center_area0.79 → STOP, center_area0.05 → 주행)

---

## 1. 서버 시작 (Billy 머신)

```bash
# Billy 접속 후
cd ~/MoNaVLA
VLA_S2V2_STAGE1=runs/v5_nav/mlp/shared/stage1_v2_projs.pt \
VLA_S2V2_STAGE2=runs/v5_nav/mlp/exp66/action_mlp.pt \
  .venv/bin/python3 robovlm_nav/serve/stage2_v2_inference_server.py --port 8001
```

- [ ] `PG2Grounder: ready` 로그 확인
- [ ] `ActionMLP` 로드 확인 (`head=mlp, window=8`)
- [ ] Health check: `curl http://localhost:8001/health`

### 체크포인트 경로 확인

```bash
ls -lh runs/v5_nav/mlp/shared/stage1_v2_projs.pt     # Stage1
ls -lh runs/v5_nav/mlp/exp66/action_mlp.pt            # Stage2 (Exp66)
```

---

## 2. 테스트 매트릭스

| 경로 타입 | 목표 횟수 | 성공 기준 | 비고 |
|----------|---------|---------|-----|
| center_straight | 3회 | TLD∈[0.7,1.5], FPE<0.5m | STOP 트리거 필수 |
| left_diagonal   | 3회 | TLD∈[0.7,1.5], FPE<0.5m | |
| right_diagonal  | 3회 | TLD∈[0.7,1.5], FPE<0.5m | sim 67% → 실환경 확인 |
| center_curve    | 2회 | TLD∈[0.7,1.5], FPE<0.5m | 옵션 |
| **합계**        | **11회** | **≥7/11 = 63.6% 목표** | |

> **right_diagonal 우선**: sim에서 2/3 실패(FPE=0.517m). 실환경에서 재현되는지 확인 필요.

---

## 3. STOP 로직 검증 항목

각 에피소드에서 바스켓 근처(~1m 이내)에서:

- [ ] 로봇이 멈추는가? (STOP 출력)
- [ ] 서버 로그에 `[PROXIMITY STOP]` 표시되는가?
- [ ] 로그의 `area` 값이 ≥0.25인가?
- [ ] 로그의 `|cx-0.5|` 값이 ≤0.35인가?
- [ ] 너무 일찍 멈추지 않는가? (바스켓 안 보이는데 STOP)
- [ ] 너무 늦게 멈추지 않는가? (바스켓이 프레임을 채우는데 계속 전진)

> STOP이 안 나오면: `VLA_STOP_AREA=0.20` 로 threshold 낮추고 재시도

---

## 4. 매 에피소드 기록표

```
에피소드 #:
경로 타입:
시작 위치:
성공/실패:
STOP 발생?: Y / N
마지막 PG2 area:             cx:
FPE 추정(육안):              m
특이사항:
```

---

## 5. 로그 수집

서버 로그는 파일로 저장:

```bash
VLA_S2V2_STAGE2=runs/v5_nav/mlp/exp66/action_mlp.pt \
  .venv/bin/python3 robovlm_nav/serve/stage2_v2_inference_server.py \
  2>&1 | tee logs/realtest_20260616.log
```

각 에피소드 후:
- [ ] 로그에서 `PROXIMITY STOP` 줄 추출: `grep "PROXIMITY STOP" logs/realtest_20260616.log`
- [ ] `area`, `cx`, `near_frames` 값 확인

---

## 6. 실패 시 디버그 순서

1. **아무것도 안 함 (로봇 정지)**: Stage2 ckpt 로드 실패 → `/health` 재확인
2. **계속 FORWARD만**: Stage1 인코더 문제 → bbox 값 로그 확인
3. **STOP 안 나옴**: PG2가 바스켓 못 찾음 → `area` 값 확인, threshold 0.20으로 조정
4. **너무 일찍 STOP**: threshold 올리기 `VLA_STOP_AREA=0.30`
5. **right_diagonal 계속 실패**: sim과 같은 패턴(FPE>0.5m)이면 경로 편향 문제 — 별도 분석

---

## 7. 성공 판정

| 지표 | 기준 | 실제 |
|-----|-----|-----|
| 전체 성공률 | ≥63.6% (7/11) | / |
| center_straight 성공률 | 3/3 (100%) | / |
| right_diagonal 성공률 | ≥2/3 (67%) | / |
| STOP 정확도 | 모든 성공 ep에서 goal 앞 STOP | / |
| FPE (성공 ep 평균) | <0.3m | / |

---

## 8. 테스트 후 처리

```bash
# 결과 커밋
git add logs/realtest_20260616.log
git commit -m "feat(eval): Exp66 real robot test 결과 2026-06-16"
git push origin inference-integration
```

- [ ] `docs/v5/closed_loop_eval/rollout_metrics.json` 에 실환경 결과 추가
- [ ] 실패 에피소드 프레임 저장 (로봇 카메라 영상)
