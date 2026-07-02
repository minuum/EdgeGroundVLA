# 런타임 모드 가이드 & 변경 이력

> Tab 4 "⚙️ 런타임 모드" 버튼 설명 + 최근 반영 사항 기록
> 서버 재시작 없이 `/config`로 즉시 반영됨

---

## 런타임 모드 항목 설명

파이프라인 흐름:
```
카메라 → PG2-448 grounding → bbox(cx,cy,area,has_bbox) → Stage2 Transformer → 액션
              ↑ skip_n / P2 필터              ↑ has_bbox=False 시 Preview / hint_cx
```

| 항목 | 버튼 | 켜면(ON) | 끄면(OFF) | 관여 지점 | 트레이드오프 |
|------|------|----------|-----------|-----------|-------------|
| **Preview 격리회전** | 🟢/⚫ Preview | 미검출 시 회전 재탐색 후 주행 시작 | 미검출이어도 즉시 추론 | 미검출 대응 | 켜면 안전 / 끄면 엉뚱 출발 위험 |
| **hint_cx 회전** | 🟢/⚫ hint_cx | FILTER된 탐지 cx 방향으로 회전 | 고정방향(ROT_R)만 회전 | 미검출 대응(Preview ON 시) | 켜면 방향 스마트 / 검증 필요, origin엔 없음 |
| **grounding skip_n** | 📦 skip_n | (1) 매프레임 fresh grounding | (3) 캐시 재사용 | grounding 품질 | 1=정확·느림 / 3=빠름·stale bbox |
| **cx 급변 필터 (P2)** | 🟢/⚫ P2필터 | 직전 대비 cx 급점프 탐지 기각 | 모든 탐지 수용 | grounding 품질 | 오탐 방어 / 진짜 이동 놓칠 위험 |
| **jump 임계값** | 🎚 jump>N | — 순환 0.20→0.30→0.40→0.50 | — | P2 민감도 | 낮음=민감 / 높음=둔감 |

**추천 시작 조합(재현 주행):** Preview ON · hint_cx OFF · skip_n=3 · P2 OFF
(= 현재 서버 기본값, origin 공식 라인에 가장 가까움)

**⚠️ 주의:** 100143 t=7 같은 "연속 중앙 오탐"은 점프가 없어 P2로 못 잡음. P2는 "정상→갑자기 튐" 유형 전용.

---

## 변경 이력 (반영 페이스)

### 2026-07-02

**서버 (`stage2_v2_inference_server.py`)**
- ✅ **Fix A**: preview 반환 dict에 `source` 키 추가 → preview ROT 시 `/predict` 500(KeyError) 해결
- ✅ **Fix B-1**: PG2 grounding 해상도 224→448 native (`resize_for_vlm` 제거). 학습 annotation과 정합
  - ⚠️ 실측: minum ★★★(주변부 중앙 오탐)은 448로도 미해결 → 해상도 아닌 PG448 학습 편향
- ✅ **런타임 `/config` 확장**: `preview_enabled` / `preview_hint_cx` / `grounding_skip_n` / `cx_jump_filter` / `cx_jump_thresh`
- ✅ **P2 필터 구현**: fresh grounding 시 직전 대비 cx 급변하면 기각+캐시 유지 (기본 OFF)
- ✅ `/health`에 preview·skip_n·cx_jump·inference_count 리포트 추가

**대시보드 (`gradio_inference_dashboard.py`)**
- ✅ Tab 4 "⚙️ 런타임 모드" — 균등 폭 토글 버튼 5종 (누르면 즉시 반영)
- ✅ 상단 배너 옆 수집 세션/Grounding/최근스텝 블록
- ✅ "📷 수집 모니터" 탭 (2초 갱신)
- ✅ post-action 프레임 동기 수집 (`{step}p`)
- ✅ ROS spin `ExternalShutdownException` 복구

**분석 (minum ↔ soda 교차)**
- ✅ minum 커밋 4af532f7: grounding center bias 분석 + FIX_GUIDE + 시각화 7종
- ✅ minum ★★★ 시각 확인(100143 t=7): 바스켓 우측(cx≈0.84)인데 PG2가 빈 바닥 중앙(cx=0.52) 오탐
- ✅ Fix B가 ★★★ 미해결 실측 규명 → 근본원인 = PG448 학습분포 편향(cx>0.7이 7.8%)

**미결정 / TODO**
- ⬜ 재현 주행 (Tab4 조합별 obj_center) → episode_log
- ⬜ post-action 처리: minum P1(bbox 상속) vs 재-grounding vs 분석 제외
- ⬜ 코드 정합: preview 포크(로컬 hint_cx ↔ origin 2-path) origin 기준 재정합
- ⬜ P3(중기): 주변부 cx 0.65~0.85 학습 데이터 보강 7.8%→15%

**첫 프레임 인식 실패 규명 (2026-07-02)**
- 증상: 어제/오늘 12세션 첫 프레임 중 `gray basket`으로 2/12만 검출. 나머지는 full-frame 환각(area≈0.99)→필터.
- 원인: PG2-448이 "회색 바스켓 vs 흰벽/회색바닥" 저대비+과노출 장면에서 불안정. 바스켓이 정중앙 명확히 보여도 첫 프레임 실패(육안 확인 084101·084555·100143). 448 해상도 문제 아님.
- 프롬프트별로 잡는 프레임 다름: 100143="laundry basket", 084101="gray plastic bin", 212540="gray basket".
- 멀티프롬프트 순차 5종 시도 → **6/12 복구(3배)**. 남은 6개는 5프롬프트 전부 실패(PG2 근본 취약).
- **다음**: ① 멀티프롬프트 fallback을 grounder에 구현(첫 탐지 실패 시만 대체 프롬프트) ② 중기 PG2 fine-tune.

**물리 드리프트 교정 (나중에 — 좋은 방법으로 재설계)**
- 문제: FWD 명령해도 완벽 직선 안 감(바퀴/모터 편향 좌우 쏠림). CL(모델)엔 드리프트 교정 로직 없음.
- P2는 인지필터라 드리프트 직접교정 불가. 오히려 낮은 임계값은 cx 보정신호를 얼려 드리프트 고착 → 드리프트 원하면 P2 OFF/높게(0.5).
- 진짜 레버: ① 모델 폐루프(grounding cx 정직할 때만) ② **FWD az 트림 상수** ③ CX_RULE 기하보정(현재 OFF)
- **다음 작업**: FWD 1회당 좌/우 쏠림 실측 → az 트림 계산. 측정 탭이 필요할 수 있음. (시간 없어 보류, 나중에 좋은 방법으로)
- 참고 캘리브레이션: W 1회 ≈ 12~13cm, ROT az=±0.25 (2026-06-26 실측)
