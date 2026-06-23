# 드리프트/그라운딩 latency 분석 — 이전 시행·수집세션 대비 변경점 (2026-06-23)

> 관련: CH44("4초"=누적 드리프트 정식 정리), [REAL_ROBOT_CHECKLIST_20260616.md](REAL_ROBOT_CHECKLIST_20260616.md), [PROF_QA_PREP_20260603.md](PROF_QA_PREP_20260603.md)
> 모델: Exp66 Stage2 v2 (port 8001) — Stage1 shared + ActionMLP w=8 + PG2 zero-shot grounding

## 1. 이번에 새로 확인한 것

### 1-1. 카메라 CPU 컨텐션은 부차적 요인 (가설 기각)
정적 테스트 이미지로 `/predict` n=20 반복:

| 조건 | mean latency |
|---|---|
| `usb_camera_service_server` ON (CPU 45%) | 1297ms |
| 동일 프로세스 OFF | 1208ms |

차이는 89ms(7%)뿐. CH44에서 본 1356ms baseline → 라이브 1719~1767ms로 튄 구간은 카메라 경쟁이 아니라 다른 원인.

### 1-2. 진짜 원인: 그라운딩(PaliGemma2) 자체가 latency의 95% 차지
`/ground` 엔드포인트로 그라운딩만 단독 측정:
```
latency_ms: 1232~1250
raw_output: '<loc0000><loc0000><loc1011><loc1022> gray basket<eos>'
```
- 출력은 이미 8토큰 내 EOS — `max_new_tokens=48` 한도까지 안 감. 토큰 수/이미지 크기(224×224, bf16) 둘 다 이미 최소치.
- `/predict` 전체(그라운딩+Stage1+Stage2 MLP) 평균 1717~1722ms 중 그라운딩이 1635~1647ms(95.5%), MLP는 ~77ms.
- **결론: Jetson Orin에서 PaliGemma2(~3B) 자체의 forward pass 비용이 하한선.** 코드 레벨(프롬프트 길이, 이미지 크기) 튜닝 여지 없음 — 줄이려면 TensorRT/INT8 양자화나 더 작은 그라운딩 모델 교체(기술 선택, 별도 결정 필요).

### 1-3. SYNC 루프는 이미 "이동 후 그라운딩" 구조 — 캐싱/스킵은 부적합
`gradio_inference_dashboard.py`의 `infer_move_mode == "SYNC"`(기본값)는:
```
move → robust_stop → 150ms settle → stable_frame 캡처(이동 후 프레임)
→ 다음 스텝에서 그 frame으로 그라운딩 fresh 호출
```
서버의 `VLA_GROUNDING_SKIP_N`(그라운딩 캐싱/스킵 옵션)도 기본값 1(스킵 없음)로 운영 중. V5 데이터셋 분석상 이동 시 초점이 명확히 바뀌므로 이전 bbox 재사용은 부적합하다는 점이 코드 설계와도 일치함 — **여기는 바꿀 게 없다.**

### 1-4. 드리프트 탭 가정시간 기준을 3종으로 재보정
기존엔 "1fps(1.0s) 가정"만 있어서 드리프트가 항상 커 보였음. 실측 기반 3개 기준 추가(`scripts/gradio_inference_dashboard.py` drift 탭):

| 기준 | 산출 근거 | 실측 drift (4프레임 누적, latency~1.75s/frame) |
|---|---|---|
| 1.0s (1fps 운영) | 기존 가정, 근거 없음 | +2.99s |
| 1.35s (학습 수집 cadence) | `auto_play_core()` 0.4s 이동타이머+0.15s stop펄스+0.8s rest 실측 | +1.59s |
| **1.92s (SYNC 실측 풀사이클)** | `move_and_stop_ramped` ramp(0.05s)+settle(0.15s)+그라운딩·MLP(~1.717s) | **-0.17s** |

1.92s 기준에서 drift가 거의 0(±0.2s)로 수렴 — **"4초 드리프트"는 운영 루프 결함이 아니라 1fps라는 잘못된 가정과 비교한 결과였다.**

## 2. 이전 시행/수집세션과의 차이

| 항목 | 이전 (4/9~5/22 수집, CH44 이전 분석) | 지금 |
|---|---|---|
| 학습 데이터 수집 cadence | "1.2s"로 추정(0.4+0.8만 합산) | **1.35s**로 정정 (`timer.join()`이 `timed_stop()` 내부 0.15s까지 포함하는 걸 누락했었음) |
| 데이터셋 cadence 커버리지 | 전체 289개 동일 가정 | 260408(4/8) 50개(17%)는 `time.sleep(0.8)` 도입 커밋(78dabd6a, 4/9 07:30)보다 먼저 수집 — 해당 cadence 미검증 상태로 남음 |
| "4초 드리프트" 원인 | latency 문서 baseline(1356ms)과 라이브(1719~1767ms) 차이가 미해명 | 그라운딩 자체 비용(95%)으로 확인, 카메라 컨텐션은 7%뿐 |
| 드리프트 비교 기준 | 1.0s 단일 | 1.0s/1.35s/1.92s 3종 동시 비교 가능 (전체 비교 모드) |
| 그라운딩 호출 정책 | 매 스텝 fresh(코드상 기본값) — 검증 안 됨 | `VLA_GROUNDING_SKIP_N=1` 기본 확인, SYNC 루프의 post-move 캡처 구조와 일치함을 확인 |

## 3. 후속 검토 필요 항목 (미착수)

- [ ] PG2 그라운딩 TensorRT/INT8 양자화 가능성 검토 (latency 하한선을 낮추는 유일한 레버, 기술 선택 필요)
- [ ] 260408(4/8) 수집 50개 episode의 실제 collection cadence 재확인 — git에서 이전 버전 파일 추적 실패, H5에 timestamp 메타데이터 없어 사후 검증 불가능. 영향 범위(전체 학습에서 17%)만 인지하고 넘어간 상태
- [ ] git push (monavla-driving + inference-integration 양쪽) 완료 여부 확인
