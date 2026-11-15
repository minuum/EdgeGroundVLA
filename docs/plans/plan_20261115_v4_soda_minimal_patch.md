# V4/V4R soda 최소 패치 계획 (승인 대기 — 코드 수정 없음)

작성 2026-11-15. 선행: `plan_20261115_v4_serving.md`(설계 승인됨), soda 환경 확인 결과 `_received_1113_env_check/FINDINGS_1113_summary.md`.
이 문서는 **soda 쪽에 넣을 변경을 최소화하는 방법**만 다룬다. 승인 전에는 soda에도, 이 저장소에도 코드를 바꾸지 않는다.

## 1. 확인된 사실 (soda 수신 파일 기준)

| 항목 | soda 실제 | 이 저장소 |
|:--|:--|:--|
| HEAD / 브랜치 | `c74b8687` / `monavla-driving`, 미커밋 수정 11개 파일(402KB) | `inference-integration` |
| `/predict` egvla2 분기 | `_PHRASE_FROM_REQUEST` 켜짐이고 지시어가 있고 `"basket"`이 아니면 `service.predict(image, 문구)`, 아니면 `service.predict(image)` | 지시어 무시 |
| `predict_rgb` 인자 | `phrase` → `extractor(image, phrase)` (phrase=None이면 1인자 호출) | `instruction` → `set_phrase` |
| `run_batch` | `phrase=None` 인자 추가(박스 문구로 사용) | 없음 |
| 스위치 `phrase_from_request` | 런타임 전용, `switch_family`·재시작 시 False로 복귀, 현재 true | 없음 |
| 서버 응답 키 | `florence_phrase`(Florence에 들어간 문구) | `instruction_used`, `variant` |

핵심: soda의 egvla2 서비스는 이미 "지시어를 선택 인자로 서비스까지 전달"하는 구조다. V4용 경로는 그 인자를 **박스 문구가 아니라 문구 특징**으로 쓰기만 하면 된다.

## 2. 접근: soda의 기존 파일을 한 줄도 덮어쓰지 않는다

soda는 미커밋 수정이 많아서 브랜치 pull이나 rsync로 덮으면 그 수정이 사라진다. 따라서 **새 파일 1개 추가 + 기존 파일 1곳에 분기 삽입**만 한다.

### 변경 A — 새 파일 추가: `robovlm_nav/serve/egvla2_text_policy.py` (soda에 없는 파일, 충돌 없음)
이 저장소 버전에서 아래만 바꿔 전달한다.

```python
# FusedF2TextExtractor: 상태(set_phrase) 방식 → 무상태 인자 방식으로 변경
@torch.no_grad()
def __call__(self, image_rgb, phrase=None):
    phrase = norm_phrase(phrase)          # None/빈 값 → "gray basket"
    ...                                   # 박스는 P.PHRASE("gray basket") 고정 — 기존과 동일
    pe = self.model.get_input_embeddings()(self._phrase_ids(phrase))
    ...
```
- soda의 `predict_rgb`가 이미 `extractor(image_rgb)` 또는 `extractor(image_rgb, phrase)`로 호출하므로 `predict_rgb`·`predict`·`/predict`는 **수정 불필요**.
- `set_phrase` 상태를 없애 요청 간 문구가 새어 들어가는 위험(직전 요청 문구가 남는 문제)도 같이 제거된다.
- 응답의 `florence_phrase`는 text 경로에서 "박스 문구"가 아니라 "지시어"가 찍힌다. 로그 해석 주의사항으로 기록(응답 키 이름은 바꾸지 않음).

### 변경 B — soda의 `egvla2_inference_server.py` `load_service_from_env()`에 분기 7줄 삽입
이 저장소 커밋 `a7b3822d`의 분기와 동일한 내용(`VLA_EGVLA2_TEXT=1`일 때만). 기본 꺼짐이라 A3 경로는 그대로.
삽입 위치는 soda 파일 기준으로 `ctrl` 생성 직후이며, 삽입 후 soda에서 `git diff`로 7줄만 늘었는지 확인한다.

### 변경 C — 없음
- `stage2_v2_inference_server.py`, `mona_dashboard.py`, `egvla2_fm_policy.py`, `precompute_v6_florence2_fused.py`는 **건드리지 않는다**.
- 체크포인트 2개(`textcond/EG4-04_V4_full_s1.pt`, `EG4-07_V4R_full_s1.pt`)는 가이드대로 `runs/egvla2_fm/deploy/` 옆에 복사한다(2.8MB급, sha256 대조).

## 3. 기동·시험 절차 (단계마다 승인)

1. **오프라인 검증(이 서버, 승인 후 바로 가능):** soda의 두 파일(`egvla2_fm_policy.py`, `egvla2_inference_server.py`)을 스크래치에 복사해 변경 A·B를 적용하고, soda가 쓰는 호출 방식(`predict(image)`, `predict(image, 문구)`)으로 동등성 검사를 다시 돌린다. 기존 기준 그대로(문구 cos ≥0.999, 각도 중앙값 ≤1°, 정지 일치 ≥95%).
2. **soda 전달:** 파일 1개 + 7줄 삽입 + 체크포인트 2개. soda에서 모델은 로드하지 않고 `git diff --stat`·`python -m py_compile`까지만.
3. **API만 응답 확인(운영서버 위에서 모델 추가 로드 금지, 서버 재시작 필요 여부는 동환 판단):** 전환 순서
   `switch_family(egvla2)` → 환경변수 `VLA_EGVLA2_TEXT=1`과 V4 체크포인트로 서비스 로드 → `/config {"phrase_from_request": true}` → `/health`에서 `phrase_from_request: true` 확인.
   - **`phrase_from_request`를 꼭 켜야 한다.** 꺼져 있으면 모든 요청의 문구가 "gray basket"이 되어 V4가 항상 바구니 지시로 동작한다(시험이 무효).
   - `switch_family`·재시작 시 이 스위치가 자동으로 꺼지므로 매번 다시 켜고 `/health`로 확인한다.
4. **구동 시험:** 바구니 → 4물체 × G2/G3/G4, V4와 V4R 시드 1. 로봇이 움직이므로 단계별 승인과 동환 입회가 필요하다.

## 4. 위험과 대응

| 위험 | 대응 |
|:--|:--|
| 운영서버 위에서 모델을 하나 더 올리면 Jetson 리부팅 전례 | 운영서버 정지 후 한 개만 로드하는 절차를 동환이 결정. 이 쪽에서는 로드하지 않음 |
| 지시어가 `"basket"`(단독)일 때 soda가 문구를 무시 | 시험 지시어는 "go to the gray basket" 등 문구형으로 고정 |
| 기존 A3 경로 회귀 | `VLA_EGVLA2_TEXT` 미설정 시 변경 B 분기를 안 탐. 전/후 `git diff`와 A3 응답 한 번 비교(API만) |
| soda 미커밋 수정이 이 저장소에 없음 | **이 저장소에도 반영해야 한다**(별도 항목): soda의 `phrase_from_request` 스위치와 `run_batch(phrase=)`. 반영 여부는 승인 필요 |
| 후처리 규칙(`proximity_override`, `near_area=0.25`, `lost_frames=5`)이 V4 출력에도 적용됨 | 재생과 실기를 직접 비교하지 말 것. 시험 결과 해석에 명시 |
| 지연 시간 미측정(T5-b) | 3단계에서 응답의 `latency_ms`만 기록 |

## 5. 하지 않는 것
soda 서버 재시작·모델 로드(이쪽에서), 로봇 구동, A3 서빙 코드 수정, V4·V4R 재학습, LiDAR 정지, push.

## 6. 승인 요청 항목
1. 변경 A(무상태 인자 방식)와 변경 B(7줄 분기)로 간다 — 예/아니오
2. 3단계의 서버 기동 방식은 동환이 정한다(제안: 운영서버 정지 후 V4 하나만 로드) — 예/아니오
3. soda의 미커밋 `phrase_from_request` 수정을 이 저장소에도 반영한다 — 예/아니오/보류

---
## 7. 진행 상태 (2026-11-15) — 승인: 변경 A·B 진행(항목 1). 항목 2·3은 아직 승인 없음
- [x] 변경 A: `egvla2_text_policy.py`의 추출기에 `phrase` 인자 추가(저장소). `set_phrase` 경로는 이 저장소 서버 호환용으로 남김(계획의 "상태 제거"에서 벗어난 점) — 인자 경로는 `_phrase`를 바꾸지 않음을 확인.
- [x] 변경 B: soda `egvla2_inference_server.py` 복사본에 분기 삽입. 실제 추가는 **6줄**(계획의 7줄은 오기). 패치 파일 `docs/plans/patches/soda_changeB_egvla2_inference_server.diff`.
- [x] 1단계 오프라인 검증(soda 파일 복사본 + soda 호출 방식 `predict_rgb(img, phrase)`/`predict_rgb(img)`): 결과 `docs/v5/egvla2_fm/soda_overlay_equiv_fp32.json`, 스크립트 `scripts/check_soda_overlay_equiv.py`.
  - V4: 문구 코사인 최저 0.99989, 각도 중앙값 0.079°, 정지 일치 100%. V4R: 각도 중앙값 0.068°, 정지 일치 100%. 저장소 서버 결과와 동일.
  - 상태 비누출: 다른 지시어 호출을 끼워도 인자 없는 호출 결과 동일(cos 1.0, 최대 절대차 0).
  - **미해명:** 박스 중심오차 최대 9.0(기준 0.02, 중앙값 0.0011) — 일부 프레임에서 박스 유무가 캐시와 다름. 원인 미조사.
- [ ] 2단계 soda 전달 / 3단계 API 확인 / 4단계 구동 시험: 대기(전달·서버 기동 방식 승인 필요, soda 미접촉).
