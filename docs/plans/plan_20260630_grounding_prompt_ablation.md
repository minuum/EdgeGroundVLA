# Plan: Grounding Prompt Ablation (CH58)
> 작성: 2026-06-30 | 상태: 검토 대기

---

## 목표

Kosmos-2 + OWL-v2(PG2) 두 모델 모두 텍스트 쿼리 변형별 박스 검출 품질 어블레이션.  
현재 세션 46% 박스 일치율 → 쿼리만 바꿔서 얼마나 개선되는지 측정.  
두 모델을 동시 실행해 최선 조합(모델 × 쿼리) 도출.

---

## 배경 (리서치 요약)

### 현재 방식 (inference_server.py:1341)
```python
prompt = f"<grounding>The basket is at"
```
- **비공식 completion 방식** — Kosmos-2가 이어서 위치 토큰 생성
- `post_process_generation()`이 `<object>...<patch_index>...</object>` 파싱에 최적화돼있음
- entity 검출 실패 시 caption 텍스트 fallback (cx 신뢰도 더 낮음)

### Kosmos-2 공식 refexp 형식 (processing_kosmos2.py:102)
```python
# 공식: <phrase>target</phrase><object> → 모델이 patch_index 생성
prompt = "<grounding><phrase>basket</phrase><object>"
```
- processor가 `<phrase>...<object>...<patch_index_XXXX><patch_index_YYYY></object>` 파싱에 최적화
- 명시적 reference expression → 더 정확한 bbox 기대

### 현재 박스 품질 (수동 레이블 39세션)
- ✅ FULL: 8개 (20.5%)
- 🟡 PART_IN: 10개 (25.6%)
- ⚠️ PART_OUT: 7개 (17.9%)
- ❌ WRONG: 14개 (35.9%)
- 점이 바스켓 안: **46.2%** ← 이걸 개선 목표

---

## 어블레이션 변형

### Kosmos-2 (6가지) — Pure `.vlms/kosmos-2-patch14-224`

| ID | 프롬프트 | 방식 | 기대 효과 |
|----|---------|------|-----------|
| `K_current` | `<grounding>The basket is at` | completion | 기준선 (재현) |
| `K_refexp` | `<grounding><phrase>basket</phrase><object>` | refexp | 공식 형식 |
| `K_refexp_gray` | `<grounding><phrase>gray basket</phrase><object>` | refexp | 색상 힌트 추가 |
| `K_refexp_laundry` | `<grounding><phrase>gray laundry basket</phrase><object>` | refexp | 최대 구체성 |
| `K_locate` | `<grounding>The gray basket is located at` | completion | 더 구체적 completion |
| `K_nav` | `<grounding>An image of a robot navigating toward the basket` | completion | 학습 프롬프트 유사 |

### OWL-v2 / PG2 (5가지) — `google/owlv2-base-patch16-ensemble`

OWL-v2는 text query → threshold 기반 탐지. 프롬프트 = 쿼리 문자열.

| ID | 쿼리 | 기대 효과 |
|----|------|-----------|
| `O_current` | `"gray basket"` | 기준선 (현재 PHRASE) |
| `O_basket` | `"basket"` | 단순 쿼리 |
| `O_laundry` | `"gray laundry basket"` | 최대 구체성 |
| `O_container` | `"gray container"` | 외형 기술 |
| `O_multi` | `["basket", "laundry basket", "gray container"]` | 다중 쿼리 OR |

---

## 구현 계획

### 파일: `scripts/ablate_grounding_prompt.py` (신규)

```
구조:
1. 세션 이미지 로드 (39세션, frame 1)
   + 수동 레이블 로드 (pos: L/C/R, box: FULL/PART_IN/PART_OUT/WRONG)
2. [Kosmos-2 블록] Pure Kosmos-2 로드 → K_* 6가지 실행
3. [OWL-v2 블록] owlv2-base 로드 → O_* 5가지 실행
   (두 블록은 순차 실행 — GPU 메모리 절약)
4. 각 변형마다:
   - det_rate: bbox 검출 성공률
   - cx_mean / cx_std: 분포 확인
   - dir_vs_manual_14: L/R 레이블 14개 기준 방향 일치율
   - fallback_rate: caption 폴백 비율 (Kosmos 전용)
5. 결과 → docs/v5/ablate_grounding_prompt.json
```

### refexp 형식 핵심 코드 차이
```python
# completion (현재)
prompt = "<grounding>The basket is at"
inputs = proc(text=prompt, images=img, return_tensors="pt")
generated = model.generate(**inputs, max_new_tokens=64)
caption, entities = proc.post_process_generation(raw)

# refexp (신규)
# processor에 bboxes=[[None]]을 주면 <object> placeholder 삽입
inputs = proc(text="<grounding><phrase>basket</phrase>",
              images=img, bboxes=[[None]], return_tensors="pt")
# 또는 직접: prompt + <object> 토큰으로 조건화
```

> ⚠️ refexp에서 bboxes=[[None]] vs 프롬프트 직접 구성 방식 중 어느 게 맞는지
> → 실제 실행에서 빠르게 실험으로 확인할 것

### 메트릭 계산
```python
# 수동 레이블 매핑 (위치)
manual = {"316.h5": "C", "644.h5": "R", "013.h5": "L", ...}  # 23개

# dir_vs_manual (L/R 기준, CENTER 제외)
for key, pos in manual.items():
    if pos not in ("L","R"): continue
    pred_side = "R" if pred_cx[key] > 0.5 else "L"
    dir_ok += (pred_side == pos)
```

### 출력: `docs/v5/ablate_grounding_prompt.json`
```json
{
  "A_current":   {"det_rate": 0.xx, "cx_mean": 0.xx, "dir_vs_manual_14": 0.xx, "fallback_rate": 0.xx},
  "B_refexp":    {...},
  ...
}
```

---

## 문서화 계획

### CH58 (새 챕터)
- `research_story.html`에 CH58 섹션 추가
- `grounding_hub.html`에 §L 섹션 추가 (프롬프트 어블레이션)
- 결과 표 + 박스 비교 시각화 (최선 변형 vs 현재)

---

## 완료 체크리스트

- [ ] `scripts/ablate_grounding_prompt.py` 작성
- [ ] Kosmos-2 K_* 6가지 실행 (39세션)
- [ ] OWL-v2 O_* 5가지 실행 (39세션)
- [ ] 결과 JSON 저장
- [ ] 최선 변형 박스 시각화 (수동 레이블된 세션 몇 개)
- [ ] CH58 섹션 작성
- [ ] grounding_hub §L 추가

---

## 트레이드오프 / 리스크

1. **refexp 형식이 더 나쁠 수 있음** — Google-robot post-training 이후 Kosmos-2 generate() 자체가 망가짐. refexp도 generate() 필요 → 같은 문제일 수 있음. `Pure Kosmos-2`로만 테스트.
2. **bboxes=[[None]] 형식 불확실** — 실제 코드 실험으로 확인 필요.
3. **39세션뿐** — 샘플이 작아 통계 노이즈 있음. 방향성만 판단.
