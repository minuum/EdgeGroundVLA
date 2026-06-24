# Plan — 에피소드 초반 N프레임 한정 2배 줌 재그라운딩 + 세션 기반 ablation

> 작성: 2026-06-24 · 상태: **§2 ablation 스크립트 실행 완료(n=2, 결론 보류) — §1 운영 코드 반영은 보류**
> 결과: research_story.html CH51 참고. 표본 2건으로 효과 방향 불명(1건 개선/1건 악화) — 운영 코드
> 반영 안 함, 세션 누적 후 재검토.
> 동기: 조이스틱 체감 테스트 중 "맨 처음이 너무 멀어서 (그라운딩이) 안 될 때도 있다"는 실측 보고.
> CH50(전체 데이터셋 area&lt;0.05 영구 재주석)은 5-seed 검증 결과 노이즈로 정정됐으나, 이번 요청은
> **스코프가 다른 시나리오**(에피소드 시작 후 첫 N프레임에만 조건부 적용)다 — 사용자 지시에 따라
> "CH50 결론을 재검토하는 셋업"으로 취급한다.

---

## 0. 리서치 결과

### 0-1. 실시간 추론 서버의 grounding 호출 경로

`robovlm_nav/serve/stage2_v2_inference_server.py`:
- `Stage2V2Model.predict()` (L542~) 매 프레임마다 `self.grounder.run(image_rgb, phrase=phrase)` 호출
  (단, `grounding_skip_n>1`이고 캐시 재사용 조건이면 스킵 — `CH49`에서 다룬 그 로직, L555~573)
- `PG2Grounder.run()` (L355~414): `resize_for_vlm`로 224×224 축소 후 PaliGemma2 `generate()`,
  결과 `{cx, cy, area, has_bbox, x1,y1,x2,y2}`. `min_area`(기본 0.01)/`min_cy`(기본 0.35) 필터를
  통과 못하면 `has_bbox=False` + 고정 fallback(`cx=0.5, cy=0.6, area=0.06`) 반환.
- `self.inference_count`: `reset()` 호출 시 0으로 리셋, `predict()` 끝에서 매 프레임 +1
  (에피소드 시작 후 경과 프레임 수를 그대로 추적 — 이번 기능의 "첫 N프레임" 판단 기준으로 그대로 쓸 수 있음).
- `self.history`: 프레임별 `{cx,cy,area,has_bbox,vis_feat,...}` 누적 리스트 — 직전 프레임의 cx,cy를
  "현재 줌 크롭의 중심"으로 쓸 수 있는 기존 자료구조.

### 0-2. CH50과의 관계 — 같은 기법, 다른 스코프

| | CH50(기존, 노이즈 판정) | 이번 요청 |
|---|---|---|
| 적용 대상 | **학습 데이터셋 전체**의 area&lt;0.05 프레임(36.2%, 949/2621) | **실시간 추론**, 에피소드 시작 후 **첫 N프레임만**(조건부) |
| 적용 방식 | 오프라인 1회 재주석 → 모델 재학습 | 운영 코드 내 실시간 조건 분기, 재학습 불필요 |
| 검증 결과 | 5-seed 평균 SR 89.7%±3.8%p — baseline보다 낮음, "노이즈" | (아직 없음 — 이 plan의 목적) |
| latency 영향 | 전체 36% 프레임에 항상 적용 → grounding 호출 2배, `CH49`의 skip_n=3 예산과 충돌 가능 | 에피소드당 최대 N번(기본 5번)만, 조건(작거나 실패) 만족 시만 → latency 영향 거의 없음(아래 0-4) |

**핵심 차이**: CH50은 "효과가 노이즈였다"는 결론이지만, 그 결론은 *학습 데이터 전체 재주석*이라는
큰 스코프에서 나온 것. 이번처럼 *런타임에 첫 몇 프레임만* 조건부로 재시도하는 것은 학습 자체를
건드리지 않으므로 noise-prone training-variance 문제(5-seed 이슈)가 원천적으로 없다 — PG2Grounder의
`generate(do_sample=False)`는 deterministic이라 같은 이미지를 다시 넣으면 항상 같은 결과.

### 0-3. 실제 세션 증거 (오늘 soda에서 풀링된 4개 조이스틱 체감 테스트)

`docs/inference_sessions/session_20260624_{104411,105123,150541,151056}.h5` (rsync 직후 즉시 분석):

| 세션 | n_frames | status | 첫 5프레임 area | 비고 |
|---|---|---|---|---|
| 104411 | 3 | manual_stop | 0.052, 0.045 | 둘 다 area&lt;0.05 |
| 105123 | 2 | manual_stop | 0.066 | 경계값 근처 |
| 150541 | 8 | manual_stop | 0.076,0.053,0.063,0.096,0.034 | frame1에 22.3초 레이턴시 이상치(콜드스타트로 추정), frame5 area=0.034로 가장 작음 |
| 151056 | 1 | manual_stop | (그라운딩 호출 전 정지) | — |

frame0은 항상 `cx=0.5,cy=0.6,area=0,has_bbox=0,cached=-1` placeholder(아직 실제 grounding 호출 전
초기값) — 실패가 아니라 "호출 전" 상태. **4세션 중 2세션에서 첫 5프레임 내 area&lt;0.05 프레임 존재**,
다만 이번 표본 어디서도 `has_bbox=False`(완전 실패)는 관측되지 않음 — 사용자가 말한 "안 될 때도 있다"는
이 4개보다 더 많은(또는 다른) 체감 테스트 회차에서 나온 보고로 추정, **현재 가진 표본만으로는
완전 실패 사례를 직접 재현하지 못했다**(환각 방지 차원에서 명시).

### 0-4. latency 예산 재검토 — CH49 결론과 충돌하지 않음

CH49는 "그라운딩 호출을 프레임마다 2배로 늘리면 안 된다"(skip_n=3 예산 깨짐)는 결론이었다. 이번 기능은
- 에피소드 전체가 아니라 **첫 N프레임(기본 5)에서만** 작동 가능
- 그 중에서도 **area&lt;임계값이거나 has_bbox=False인 경우에만** 추가 호출 발생
이므로 최악의 경우에도 에피소드당 추가 PG2 호출은 최대 5회. 실측 레이턴시(CH49-4, ~1.3~1.4s/call)
기준으로도 초반 5프레임 한정 추가 비용은 에피소드 전체 latency 예산에 미치는 영향이 미미함 — CH49의
"skip_n=3 유지" 결정과 양립 가능.

---

## 1. 제안 로직 (구현 시 — 아직 코드 작성 안 함)

`stage2_v2_inference_server.py`의 `predict()`, L566~573 부근에 조건부 재시도 추가:

```python
FIRST_N_ZOOM = int(os.getenv("VLA_FIRST_N_ZOOM", "5"))
ZOOM_AREA_THRESHOLD = 0.05  # CH46-4/CH50과 동일 기준 — 일관성 유지

...
bbox = self.grounder.run(image_rgb, phrase=phrase, return_hidden=use_hidden)
if (self.inference_count < FIRST_N_ZOOM
        and not use_cache
        and (not bbox["has_bbox"] or bbox["area"] < ZOOM_AREA_THRESHOLD)):
    # frame0(history 비어있음)은 prior bbox가 없으므로 화면 중앙 기준
    anchor_cx = self.history[-1]["cx"] if self.history else 0.5
    anchor_cy = self.history[-1]["cy"] if self.history else 0.5
    crop, x0, y0, cw, ch, W, H = zoom_crop(image_rgb, anchor_cx, anchor_cy, zoom=2.0)
    zb = self.grounder.run(crop, phrase=phrase)
    if zb["has_bbox"]:
        bbox = remap_zoom_bbox(zb, x0, y0, cw, ch, W, H)  # 크롭 좌표 -> 원본 정규화 좌표
    # zb["has_bbox"]==False면 원본 bbox 유지(악화 방지, regroun_zoom_small.py와 동일 폴백 원칙)
```

`zoom_crop`/좌표 역변환 함수는 `scripts/eval/regroun_zoom_small.py`에 이미 구현된 로직을 그대로
재사용(import 또는 동일 함수 복제) — 새로 설계하지 않음.

---

## 2. 평가 — 세션 기반 ablation (사용자 추가 요청: "테스트 가능한 세션 정보로 ablation화")

운영 코드를 건드리기 전, **저장된 세션의 raw 이미지를 오프라인 리플레이**해서 효과를 먼저 측정한다
(서버 재배포·실주행 없이 가능).

### 신규 스크립트: `scripts/eval/ablate_first_frame_zoom.py`

1. 대상: `docs/inference_sessions/*.h5` 중 `grounding/bbox`가 있는 세션 전체(현재 4개, 앞으로 누적되는
   세션도 같은 스크립트로 재사용 가능)
2. 각 세션의 frame 0~`FIRST_N_ZOOM-1` 중 `area<0.05 or has_bbox=False`인 프레임만 추출
3. 해당 프레임의 원본 이미지(`observations/images`)를 PG2Grounder로 다시 한 번, 이번엔 2배 줌 크롭
   적용 후 재그라운딩 → "교체 전 vs 교체 후" `(cx,cy,area,has_bbox)` 비교
4. 측정 지표(사용자가 명시한 "그라운딩 인식률의 변화"):
   - `has_bbox` False→True 전환 건수/비율
   - area 평균 증가량(작게 잡힌 프레임이 줌 후 얼마나 커지는지)
   - 좌표(cx,cy) 이동량(줌으로 그라운딩 위치 자체가 달라지는지 — CH50처럼 거의 안 변하는지 확인)
5. 출력: `docs/v5/first_frame_zoom_ablation.json` + research_story.html에 새 챕터(CH51 예정)로 정리

### 한계 (미리 명시)

- 표본이 현재 4세션(첫 5프레임 기준 약 10~15개 그라운딩 프레임)뿐 — 통계적 결론을 내리기엔 작음.
  앞으로 쌓이는 체감 테스트 세션을 같은 스크립트로 누적 재실행하는 방식으로 보강 가능.
- 이 ablation은 grounding 품질(인식률)까지만 측정 가능 — 실주행 세션에는 정답 액션(`gt_class`)이
  없으므로 CH50처럼 "closed-loop SR/FPE"까지 측정하려면 별도로 액션 헤드 추론까지 돌려야 하고,
  그 경우에도 "정답"이 없어 정량적 정오 비교가 아니라 "예측 클래스가 줌 전후 달라지는 빈도" 정도의
  정성적 지표로 제한됨 — 필요 시 추가 논의.
- frame0의 "prior bbox 없음 → 화면 중앙 크롭" 가정은 객체가 화면 가장자리에 있으면 틀릴 수 있음
  (이 경우 줌 크롭이 객체를 오히려 잘라낼 위험) — ablation 결과에서 이 실패 모드가 보이는지 별도 확인.

---

## 3. 다음 단계 (승인 대기)

1. (코드 없음, 분석만) `scripts/eval/ablate_first_frame_zoom.py` 작성 + 현재 4세션으로 1차 실행
2. 결과를 research_story.html CH51로 문서화 — CH50과 명확히 구분해서 "스코프가 다른 재검토"임을 표기
3. ablation 결과가 명확한 grounding 인식률 개선을 보이면, 그 다음 단계로 운영 코드(`predict()`)
   반영 여부를 별도로 승인받음 — **이번 plan에서는 운영 코드 수정까지는 포함하지 않음**
4. 표본 누적을 위해 앞으로의 조이스틱 체감 테스트 세션도 같은 스크립트로 주기적 재실행 권장
