# PG2 stopping-criteria 세미콜론 병합 버그 수정

> 작성일: 2026-07-04

## 버그

`StopOnTokenCriteria`가 세미콜론을 단일 토큰 id(`235289`, `";"` 단독)로만 체크했음.
그런데 PaliGemma2 토크나이저는 `" ;"`(공백+세미콜론)를 **다른 id(`2161`)**로 병합함:

```python
tok.encode(";", add_special_tokens=False)        # [235289]
tok.encode(" ;", add_special_tokens=False)        # [2161]  <- 다른 토큰!
```

모델이 `"...bin ;"`처럼 공백을 붙여 세미콜론을 생성하면 stopping criteria가 걸리지 않고,
원래 막으려던 다중 탐지(`;`로 구분된 두 번째 detect) 중복 생성이 이 경로에서 재발함.

## 수정

토큰 id 하나만 비교하지 않고, 마지막 생성 토큰을 디코드해서 문자열에 `;`가 포함되는지 체크.
병합 변형(`;`, ` ;`, 그 외 어떤 조합이든) 전부 커버됨.

```python
class StopOnTokenCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if input_ids.shape[1] == 0:
            return False
        last_str = self._tokenizer.decode([input_ids[0, -1].item()], skip_special_tokens=False)
        return ";" in last_str
```

`PG2Grounder._ensure_loaded()`에서 하던 `semicolon_token_id` 사전 계산 로직 제거,
`run()`에서 `StopOnTokenCriteria(self._proc.tokenizer)`로 직접 생성.

## 검증

로컬에서 fake input_ids로 3가지 케이스 확인:
- 병합 토큰(`2161`, `" ;"`) → trigger `True` (기존 버그였다면 `False`였을 것)
- 단독 토큰(`235289`, `";"`) → trigger `True` (기존과 동일하게 정상)
- 무관한 토큰 → trigger `False` (오탐 없음)

실제 fallback 프레임(`session_20260701_220400` frame0)에 `"gray basket"` / `"gray plastic bin"`
두 phrase로 generate 재실행 — 두 경우 모두 `<eos>`로 정상 종료 확인, 회귀 없음.

## 영향

- 순수 버그 수정, 인터페이스/반환 스키마 변경 없음
- `VLA_GROUNDER=pg2`(기본값)에서만 영향, OWL-v2 A/B 경로는 무관
- 롤백 필요 없음 — 이전 동작은 이 버그를 포함한 상태였으므로 되돌아갈 이유 없음
