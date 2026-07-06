# 발견: 조건 달라도 액션 패턴이 똑같아짐 — OWL-v2 detection flicker로 인한 입력 붕괴 의심

**보낸 쪽:** soda
**대상 세션 (7개, episode_log #72~78):**
```
session_20260706_231153.h5  (1)
session_20260706_231407.h5  (2)
session_20260706_231523.h5  (3)
session_20260706_231638.h5  (4)
session_20260706_233159.h5  (5)
session_20260706_233327.h5  (6)
session_20260706_233424.h5  (7)
```
1~5는 동일 조건, 6·7은 각각 다른 조건으로 수집. **조건이 다른데도 액션 시퀀스 패턴이
거의 동일하게 나와서, STOP 임계값 문제라기보다 학습/입력 자체의 문제로 의심됨.**

---

## 1. 액션이 4종류로 수렴 (STOP 포함 다른 클래스는 전혀 안 나옴)

7개 세션 105스텝 전체에서 관측된 액션은 `FORWARD / FWD+L / RIGHT / FWD+R` 뿐.
`STOP`, `LEFT`, `ROT_L`, `ROT_R`은 105스텝 중 단 한 번도 예측되지 않음.

| session | steps | action_label_counts |
|---|---|---|
| 231153 | 15 | FWD+L 11, RIGHT 4 |
| 231407 | 16 | RIGHT 8, FWD+L 8 |
| 231523 | 17 | FORWARD 17 |
| 231638 | 16 | FORWARD 8, FWD+R 8 |
| 233159 | 15 | FWD+L 10, RIGHT 5 |
| 233327 | 12 | RIGHT 5, FWD+L 7 |
| 233424 | 10 | RIGHT 9, FWD+L 1 |

7개 중 5개가 정확히 "RIGHT ↔ FWD+L 왕복" 패턴, 2개(231523, 231638)만 FORWARD 위주.
서로 다른 조건(6, 7)도 결과적으로 1~5와 같은 왕복 패턴에 수렴.

## 2. OWL-v2 검출이 프레임마다 flicker — has_bbox가 40~60% 확률로 꺼짐

세션 하나 예시(`231153`, cx/cy/area/has_bbox 순):
```
step0:  cx=0.500 area=0.0600 has_bbox=0   ← fallback(합성값)
step1:  cx=0.680 area=0.0851 has_bbox=1
step2:  cx=0.644 area=0.0924 has_bbox=1
step3:  cx=0.500 area=0.0600 has_bbox=0   ← fallback
step4:  cx=0.571 area=0.1097 has_bbox=1
step5:  cx=0.521 area=0.1213 has_bbox=1
step6:  cx=0.450 area=0.1391 has_bbox=1
step7:  cx=0.390 area=0.1543 has_bbox=1
step8:  cx=0.341 area=0.1768 has_bbox=1
step9:  cx=0.285 area=0.2021 has_bbox=1
step10: cx=0.237 area=0.2311 has_bbox=1
step11: cx=0.179 area=0.2923 has_bbox=1   ← 가장 가까움 (임계값 0.25 초과)
step12: cx=0.500 area=0.0600 has_bbox=0   ← 바로 다음 프레임에 검출 끊김
step13: cx=0.500 area=0.0600 has_bbox=0
step14: cx=0.500 area=0.0600 has_bbox=0
```
7개 세션 전부 비슷한 패턴: 접근하면서 area가 꾸준히 커지다가, 가장 가까워지는
바로 그 순간(가장 정보가 중요한 순간) 검출이 끊기고 `cx=0.5, cy=0.6, area=0.06`
합성 fallback으로 리셋됨. `has_bbox=False`인 프레임이 세션당 40~60%.

## 3. 왜 "조건이 달라도 패턴이 똑같아지는가" — 가설

- 각 세션 프레임의 절반 가까이가 **완전히 동일한 합성 입력값**(cx=0.5, area=0.06)임.
  실제 씬 정보가 담긴 프레임은 세션당 절반 정도밖에 안 됨.
- window=6 transformer 헤드 입장에서는, 서로 다른 조건이라도 입력 시퀀스의 상당
  부분이 "검출 안 됨" 동일 신호로 채워지므로, 실제 조건 차이보다 이 반복되는
  fallback 패턴에 더 크게 반응하는 것으로 보임 → 결과적으로 조건 무관하게 비슷한
  RIGHT↔FWD+L 왕복이 나오는 것으로 추정.
- STOP 클래스가 전혀 안 나오는 것도 같은 맥락: area가 임계값을 넘는 바로 그 프레임
  직후 검출이 끊겨서, "충분히 길게 안정적으로 가까움"을 모델이 볼 기회 자체가 없음.
  (단, 이번 조사에서 STOP 자체는 급한 이슈 아님 — 조건별 액션 패턴 붕괴가 더 근본 문제로 보임)

## 4. 서버 설정 (7개 세션 공통, H5 attrs.runtime_config 확인)
```
grounder_model: OWL-v2
owlv2_thresh: 0.25
owlv2_area_scale: 3.0   (soda에서 오늘 추가한 보정계수, 기본값 그대로 사용 중)
preview_enabled: False
grounding_skip_n: 1
checkpoint_path: runs/v5_nav/mlp/exp71_window6/action_transformer.pt
```
PG2 대비 OWL-v2가 훨씬 자주 detection을 놓치는 것(threshold/모델 특성 때문일 수도,
바스켓이 카메라 프레임을 벗어나거나 너무 커져서일 수도)이 진짜 원인인지, 아니면
exp71 헤드 자체가 애초에 이런 flicker가 있는 입력 분포로 학습이 안 돼서(V5 학습
데이터는 PG2 기반이라 detection이 훨씬 안정적이었을 가능성) 이렇게 반응하는 건지
의견 부탁드립니다.

## 참고: soda 쪽에서 검토 중인 완화 방향 (아직 미적용, 의견 원함)
1. `STOP_MODE=proximity`로 전환 — 지금은 `learned` 모드라 헤드가 직접 STOP을
   예측해야만 멈추는데, 이 모드는 학습된 헤드의 STOP 예측 자체가 안 나와서 무력함.
2. **sticky bbox** — `has_bbox=False`일 때 매번 fallback(cx=0.5, area=0.06)으로
   리셋하지 말고, 마지막 실제 검출값을 N프레임 정도 유지해서 window 입력이
   flicker 때마다 완전히 리셋되는 것을 막는 방향. 이게 근본 문제(조건 무관 패턴
   수렴)를 더 직접적으로 완화할 것 같아서 이쪽이 우선순위 높다고 보는데 의견 주세요.

---

*관련: DEPLOY_20260703_OWLV2_AB.md, VERIFY_20260706_OWLV2_DEPLOY_SODA.md, NOTE_20260706_two_owlv2_sessions.md*
