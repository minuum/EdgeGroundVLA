# OWL-v2 그라운더 A/B 추가 — 로봇(soda) 배포 계획

> 작성일: 2026-07-03
> 상태: 리서치 완료, 사용자 승인 대기 (코드 미착수)
> 배경: [[REPLY_20260703_OWLV2_ALTERNATIVE]] 논의에서 OWL-v2를 Phase 3 A/B 후보로 넣기로
> 합의. 실제 코드 작성 후 soda(monavla-driving)에 push, 서버 반영/재시작은 soda 쪽에서 수행.

---

## 1. 현재 구조 (리서치)

`robovlm_nav/serve/stage2_v2_inference_server.py`:
- `PG2Grounder` 클래스(478행)가 `run(image_rgb, phrase) -> {"cx","cy","area","has_bbox","x1","y1","x2","y2","filter_reason"?}` 인터페이스로 동작
- `Stage2V2Model.__init__`(629행)에서 `self.grounder = PG2Grounder(...)`로 고정 생성
- `self.grounder.run(...)`이 예측 루프(`predict`, `_ground_multi`, `preview_align`) 전역에서 호출됨
- `_log_pg2_decision()`이 매 호출마다 `logs/grounding_decisions.jsonl`에 판정 근거 기록 (Fix4)

## 2. 제안 — `OwlV2Grounder` 클래스 추가 + 환경변수로 A/B 스위치

### 2-1. 신규 클래스 (PG2Grounder와 동일 인터페이스)

```python
class OwlV2Grounder:
    """OWL-v2 zero-shot detector 기반 bbox grounder — PG2Grounder와 동일 인터페이스."""

    def __init__(self, device: torch.device):
        self._device = device
        self._proc = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            logger.info("OwlV2Grounder: loading google/owlv2-base-patch16-ensemble")
            self._proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
            self._model = Owlv2ForObjectDetection.from_pretrained(
                "google/owlv2-base-patch16-ensemble").to(self._device).eval()

    def run(self, image_rgb: np.ndarray, _unused_path=None, return_raw=False,
            phrase: str = "gray basket", return_hidden: bool = False) -> dict:
        _t0 = time.time()
        self._ensure_loaded()
        pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
        W, H = pil.width, pil.height
        query = phrase if "laundry" in phrase or "basket" in phrase else f"{phrase}"
        inp = self._proc(text=[[f"gray {query}" if "gray" not in query else query]],
                          images=pil, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**inp)
        res = self._proc.post_process_object_detection(
            out, threshold=0.1, target_sizes=[(H, W)])[0]
        boxes = res["boxes"]
        if len(boxes) == 0:
            result = {"cx": 0.5, "cy": 0.6, "area": 0.06, "has_bbox": False,
                      "x1": None, "y1": None, "x2": None, "y2": None,
                      "filter_reason": "no-locs"}
        else:
            best = int(res["scores"].argmax())
            x1, y1, x2, y2 = [v / (W if i % 2 == 0 else H)
                               for i, v in enumerate(boxes[best].cpu().tolist())]
            area = (x2 - x1) * (y2 - y1)
            cx_val, cy_val = (x1 + x2) / 2, (y1 + y2) / 2
            filters = get_ground_filters(phrase)   # PG2와 동일 필터 재사용
            if area > 0.9:
                result = {"cx": 0.5, "cy": 0.6, "area": 0.06, "has_bbox": False, "filter_reason": "full-frame"}
            elif area < filters["min_area"]:
                result = {"cx": 0.5, "cy": 0.6, "area": 0.06, "has_bbox": False, "hint_cx": cx_val, "filter_reason": "tiny"}
            else:
                result = {"cx": cx_val, "cy": cy_val, "area": area, "has_bbox": True,
                           "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        if return_raw:
            result["raw_output"] = f"owlv2 boxes={len(boxes)}"
        _log_pg2_decision(phrase=phrase, raw=result.get("raw_output", ""), locs=[],
                          result=result, latency_ms=(time.time() - _t0) * 1000.0,
                          model="owlv2")   # Fix4 로그에 model 필드 추가 (아래 3항)
        return result
```

**주의**: `return_hidden=True` 요청은 OWL-v2가 PG2 전용 hidden-state 파이프라인(hub 프로젝션)과
호환되지 않으므로 무시하고 `hidden_state=None` 반환 — 이 부분은 A/B 중 hidden-state 의존 head를
쓰는 실험(exp71/72 계열)과는 병행 불가, 순수 bbox 경로(MLP/Transformer head)에서만 A/B 유효.

### 2-2. 생성 스위치 (환경변수)

```python
# Stage2V2Model.__init__, 기존 self.grounder = PG2Grounder(...) 부분 교체
_grounder_kind = os.getenv("VLA_GROUNDER", "pg2").lower()
if _grounder_kind == "owlv2":
    self.grounder: Any = OwlV2Grounder(device)
    logger.info("[A/B] Grounder = OWL-v2")
else:
    self.grounder: Any = PG2Grounder(_pg2, device)
```

### 2-3. Fix4 로그에 `model` 필드 추가 (`_log_pg2_decision`)

기존 `grounding_decisions.jsonl`에 `model: "pg2"` 기본값 필드를 추가해서, OWL-v2로 A/B 돌릴 때
같은 로그 파일에서 두 모델 판정을 구분 집계할 수 있게 한다. 기존 PG2 로그는 하위호환을 위해
`model` 필드가 없으면 분석 스크립트에서 `"pg2"`로 간주.

## 3. 배포 방법

1. 로컬(minum)에서 위 클래스+스위치 구현 → `monavla-driving`에 push (임시 worktree, 로컬 브랜치는
   `inference-integration` 유지)
2. soda가 pull 후 `pip install` 확인(owlv2 처리를 위한 추가 의존성 없음 — 이미 transformers에 포함)
3. `VLA_GROUNDER=owlv2` 환경변수로 서버 재시작 → A/B 세션 수집
4. 비교 시 `grounding_decisions.jsonl`의 `model` 필드로 PG2/OWL-v2 판정 근거·검출률·latency 비교

## 4. 롤백

`VLA_GROUNDER` 미설정(또는 `pg2`) 시 기존 동작 완전 동일 — 리스크 없음.

## 5. 완료 기준 (DoD)

- [ ] `OwlV2Grounder` 클래스 추가, `PG2Grounder`와 동일 반환 스키마 검증
- [ ] `VLA_GROUNDER` 환경변수 스위치
- [ ] `_log_pg2_decision`에 `model` 필드 추가 (하위호환 유지)
- [ ] 로컬에서 `OwlV2Grounder.run()` 스모크 테스트 (fallback 샘플 프레임 몇 개로 반환 스키마 확인)
- [ ] monavla-driving push, soda에 배포 안내 문서 첨부
