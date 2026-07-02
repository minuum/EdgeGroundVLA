# Grounding Center Bias — 수정 가이드

> 분석 근거: `grounding_center_bias_analysis.md`  
> 검증 일자: 2026-07-02  
> 실제 수정 적용은 사용자가 직접 진행

---

## Fix 1 (P2) — Post-action 프레임 cx=0.5 주입 버그

**파일:** `soda:~/MoNaVLA/scripts/gradio_inference_dashboard.py`  
**줄:** 1577~1583  

**현재 코드:**
```python
# 액션 완료 직후 캡처한 프레임 즉시 기록
_post = state.get("stable_frame")
if _post is not None:
    logger_instance.log_step(
        f"{current_step}p",
        result["action"],
        0,
        image=_post,
    )
```

**수정 코드:**
```python
# 액션 완료 직후 캡처한 프레임 즉시 기록
_post = state.get("stable_frame")
if _post is not None:
    logger_instance.log_step(
        f"{current_step}p",
        result["action"],
        0,
        image=_post,
        bbox=result.get("bbox"),                     # 추가: 직전 bbox 상속
        grounding_cached=result.get("grounding_cached"),  # 추가
        grounding_latency_ms=0.0,                    # 추가: post-action은 0ms
    )
```

**효과:** post-action 프레임(NONE)이 cx=0.5 기본값 대신 직전 live/cache bbox를 상속.  
**기대 개선:** 100143 세션 기준 52.6% NONE 프레임의 cx=0.5 오염 제거.

---

## Fix 2 (P1) — PG448 주변부 오탐 필터

**파일:** `soda:~/MoNaVLA/robovlm_nav/serve/stage2_v2_inference_server.py`  
**위치:** `predict()` 내부, grounding 호출 직후 (약 line 778~780 영역)

**추가할 코드:**
```python
# bbox 급변 필터: 이전 탐지 대비 cx 0.3 이상 점프하면 직전 캐시 유지
if (self._grounding_cache is not None
        and bbox.get("has_bbox")
        and self._grounding_cache.get("has_bbox")
        and abs(bbox["cx"] - self._grounding_cache["cx"]) > 0.30):
    logger.info("[FILTER] cx jump %.3f→%.3f rejected, keep cache",
                self._grounding_cache["cx"], bbox["cx"])
    bbox = self._grounding_cache
    use_cache = True
```

**효과:** 바스켓이 cx=0.48에서 cx=0.51로 "점프"하는 False Positive 필터링.  
**주의:** 임계값 0.30은 보수적 설정. 실주행 후 조정 필요.

---

## Fix 3 (P1 보완) — 주변부 학습 데이터 보강

수정 코드 없음. 데이터 수집 작업.

- 현재: V5 학습 데이터 cx>0.7 케이스 **7.8%** (200/2567 프레임)
- 목표: cx 0.65~0.85 구간 추가 수집 → **15% 이상**
- 방법: 바스켓을 우측/좌측 극단에 배치하고 right_left 경로 추가 에피소드 수집

---

## 검증 결과 요약

| 항목 | 결과 |
|------|------|
| P1 오탐 재현성 | **확정** — t7/t13 재탐지 시 완전히 동일 cx 반환 (결정론적) |
| skip_n=1 vs 3 | **영향 없음** — 9스텝 전부 동일 예측, CH49 재확인 |
| Padding 불일치 우려 | **기각** — 학습·추론 공식 동일 (line 115 ≡ 700) |
| P2 버그 구조 | **확정** — `log_step` bbox 미전달 → 직렬화 시 cx=0.5 |

---

*작성: 2026-07-02*
