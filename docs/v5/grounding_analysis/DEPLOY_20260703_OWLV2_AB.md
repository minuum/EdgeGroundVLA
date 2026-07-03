# OWL-v2 A/B 배포 안내 (soda)

**관련**: `docs/plans/plan_20260703_owlv2_ab_grounder.md`, `REPLY_20260703_OWLV2_ALTERNATIVE.md`

## 무엇이 바뀌었나

`robovlm_nav/serve/stage2_v2_inference_server.py`에 `OwlV2Grounder` 클래스 추가.
`PG2Grounder`와 완전히 동일한 반환 스키마(`cx/cy/area/has_bbox/filter_reason`)라
나머지 파이프라인(필터, `_grounding_cache`, action head 입력)은 코드 변경 없이 그대로 동작.

로컬 스모크 테스트 완료 (`session_20260701_220400.h5` frame0): `has_bbox=True, cx=0.513, area=0.029` — 정상.

## 활성화 방법

```bash
export VLA_GROUNDER=owlv2   # 기본값 "pg2" — 설정 안 하면 기존 동작 100% 동일
# 서버 재시작
```

## 확인할 것

1. `/health`에서 `git_commit`이 이번 커밋으로 잡히는지 (Fix3 핸드셰이크)
2. 서버 로그에 `[A/B] Grounder: OWL-v2` 출력되는지
3. `logs/grounding_decisions.jsonl`에 `"model": "owlv2"` 필드 찍히는지 — 기존 PG2 로그는 `"model": "pg2"`
4. OWL-v2는 첫 호출 시 `google/owlv2-base-patch16-ensemble` HF 다운로드 필요 (약 900MB) — 인터넷 안 되면
   실패하니 사전에 `python3 -c "from transformers import Owlv2Processor; Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')"`로 캐시 확인 권장

## 주의

- `return_hidden=True`(hidden-state 의존 head, exp71/72)일 때 OWL-v2는 `hidden_state=None` 반환 —
  이 모드로 A/B 돌리면 head가 오작동할 수 있음. **순수 bbox 경로(MLP/Transformer head)에서만 A/B 진행**
- 롤백: `VLA_GROUNDER` 미설정 또는 `unset VLA_GROUNDER` 후 재시작하면 기존 PG2 경로로 완전 복귀
