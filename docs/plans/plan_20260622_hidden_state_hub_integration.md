# Plan — hidden-state action head를 그라운딩 허브에서 반복 테스트 가능하게

> 작성: 2026-06-22 · 상태: **승인됨 — GB10 로컬 구현·검증부터 진행**
> 동기: CH40에서 PM은 좋아졌지만(75.9%→89%) closed-loop SR은 안 따라왔다(96.6%→93~97%, val 29개뿐). 사용자가 "실제 그라운딩 허브에서 반복 테스트되게 해달라"고 요청 — 오프라인 JSON 비교 말고, 같은 이미지를 baseline/add/replace 모드로 번갈아 눌러보며 직접 비교하고 싶다는 뜻으로 이해.

---

## 0. 리서치 결과 — 현재 구조

| 항목 | 사실 |
|---|---|
| "허브"(7860)의 실체 | 9개 서비스 상태 모니터/원클릭 접속 대시보드일 뿐, 자체 추론 없음 |
| 실제 그라운딩+VLA 예측 UI | `scripts/gradio_grounding_demo.py`(포트 7863) "VLA Inference" 탭 — `run_vla_predict()`(L738) → `_vla_call()`(L719)가 `http://localhost:8001/predict`(운영 추론 서버)로 HTTP POST |
| 운영 서버 predict 흐름 | `stage2_v2_inference_server.py: Stage2V2Model.predict()`(L498~600) — ① `PG2Grounder.run()`으로 bbox ② `Stage1Encoder.encode_image()`로 image feat(256) ③ `self.head(x)`로 8-class 로짓. **hidden state는 현재 어디서도 추출/보관 안 함** |
| `InferenceRequest` 스키마 | `image`, `instruction` 두 필드뿐(L647) — 데모가 보내는 `vlm_model`은 현재 스키마에 없어 pydantic이 조용히 버리는 것으로 추정(별도 이슈, 이번 plan 범위 밖) |
| CH40 체크포인트 | `runs/v5_nav/mlp/exp_hidden_state/stage2_v2/stage2_hidden_{add,replace}.pt` — 학습은 끝났지만 **추론 서버에 아직 미연결** |

---

## 1. 설계 — "같은 이미지로 모드를 바꿔가며 반복 클릭"이 되게

### 1-1. 서버 측: 3개 head를 동시에 들고 있다가 요청마다 선택
`Stage2V2Model.__init__`에서 baseline + add + replace 헤드를 **전부 로드**(작은 MLP라 메모리 부담 거의 없음, PG2/Kosmos-2는 1벌만 유지). `InferenceRequest`에 `head_mode: str = "baseline"` 필드 추가 → `predict()`가 모드에 따라 다른 head/입력조합 사용.

```python
class InferenceRequest(BaseModel):
    image: str
    instruction: str = "basket"
    head_mode: str = "baseline"   # "baseline" | "add" | "replace"
```

### 1-2. hidden state 추출 — 캐시가 아니라 실시간 forward에서
학습 때는 미리 추출한 npz 캐시를 썼지만, 실시간 추론은 **그 자리에서 들어오는 새 이미지**라 캐시에 없다. `PG2Grounder.run()`이 이미 PG2로 forward를 하고 있으므로(bbox 뽑는 호출), **그 호출에 `output_hidden_states=True`를 얹어 같은 forward에서 hidden state도 같이 받아온다** — 추가 forward 비용 없음(이전 세션 리서치로 이미 확인된 부분).

`head_mode == "baseline"`이면 hidden state 추출 자체를 스킵(기존 latency 그대로 유지) — **기본값을 안 건드리는 게 최우선**.

### 1-3. 데모(7863) UI — 모드 선택 드롭다운 추가
"VLA Inference" 탭에 이미 있는 `vla_vlm_dd` 같은 `gr.Dropdown` 패턴을 그대로 따라 `head_mode_dd = gr.Dropdown(["baseline","add","replace"], value="baseline")` 추가. `run_vla_predict()`/`_vla_call()`에 파라미터 한 줄 추가해서 그대로 전달.

→ 결과: **같은 이미지를 올려놓고 드롭다운만 바꿔가며 버튼을 반복해서 누르면, 3개 모드의 예측(action label + bbox)을 바로 비교**할 수 있게 됨 — 사용자가 원한 "반복되게" 테스트 흐름.

---

## 2. 변경 파일

| 파일 | 변경 |
|---|---|
| `robovlm_nav/serve/stage2_v2_inference_server.py` | `InferenceRequest.head_mode` 필드 추가, `Stage2V2Model`에 3-head 로드, `PG2Grounder.run()`에 `return_hidden: bool=False` 옵션 추가, `predict()`가 `head_mode`로 분기 |
| `scripts/gradio_grounding_demo.py` | VLA Inference 탭에 `head_mode_dd` 드롭다운 추가, `run_vla_predict`/`_vla_call`에 파라미터 전달 |

**기존 호출(head_mode 안 보내는 기존 클라이언트)은 전부 `"baseline"` 기본값으로 동작 — 하위호환, 운영 동작 변경 없음.**

---

## 3. 테스트 순서 (soda 먼저 건드리지 않음)

1. **GB10 로컬에서 먼저** — 로컬에 `stage2_v2_inference_server.py` 인스턴스를 띄우고(`--port 8011`, soda 운영과 충돌 없음), baseline 모드가 기존과 동일한 예측을 내는지(회귀 테스트) 먼저 확인.
2. add/replace 모드로 같은 이미지 반복 호출 → 모드별 예측 비교, latency 측정(hidden state 추출 추가로 얼마나 느려지는지).
3. **여기까지 통과하면** soda 배포 여부를 별도로 물어봄.

### 실행 결과 (2026-06-22, GB10 로컬)

- 버그 발견·수정: `bbox` 딕셔너리에 `hidden_state`(numpy array)가 들어간 채로 FastAPI 응답에 직렬화되어 `add`/`replace` 모드가 500 에러 — `predict()`에서 `bbox.pop("hidden_state")`로 응답 직전에 제거(내부 feature 계산용 변수로는 따로 보관)하도록 수정.
- 3개 모드 모두 200 OK, 같은 이미지(`green apple`)에 대해 정상 동작:

| head_mode | label | latency(warm, 3회 평균) |
|---|---|---|
| baseline | FORWARD | ~590ms |
| add | FORWARD | ~567ms |
| replace | FORWARD | ~565ms |

- **hidden state 추출이 추가 latency를 거의 만들지 않음** — 플랜 §1-2에서 예상한 대로 `generate(output_hidden_states=True, return_dict_in_generate=True)`로 같은 forward에서 받아오는 게 실측으로도 확인됨(3개 모드 간 차이 25ms 이내, PG2 generate 자체의 변동 범위 내).
- 첫 호출만 PG2 lazy-load(`_ensure_loaded()`) 비용으로 ~43초 걸림 — 기존부터 있던 동작(hidden state와 무관), 워밍업 후에는 위 표대로 정상.
- baseline 모드 예측이 기존 동작과 동일(회귀 없음) 확인.
- 로컬 테스트 서버는 검증 후 종료, soda는 건드리지 않음.

**다음 결정 필요**: soda에 배포할지 — 배포 시 [[soda-pg2-concurrent-load-crash]] 메모리대로 `go.sh --stop` 후 git pull → 단독 검증 → `go.sh --server` 재기동 순서를 따름.

---

## 4. "다음 단계로 늘어나야 한다"는 부분 — CH40에서 미해결로 남긴 검증 확장

CH40 결론(PM은 좋아졌는데 SR은 안 따라옴, val 29개·천장효과 추정)에 대한 후속 검증도 이번 plan에 포함:

- **확대 검증**: val split을 키우거나(5-seed 평균처럼, ablate_stop_proximity.py 패턴 재사용), 어려운 path_type(좌/우 회전류)만 따로 떼어 SR 변별력을 높여 재평가 — 새 데이터 수집 없음, 기존 150개로 seed만 여러 개.
- 위 §1~3(허브 통합)과 별개로 진행 가능 — 순서는 사용자 선택(허브 통합 먼저 vs 검증 확대 먼저).

---

## 5. 결론적으로 뭐가 달라지는가 (이 plan 승인·구현 후)

| | 지금(CH40 직후) | 이 plan 구현 후 |
|---|---|---|
| hidden state 결과 확인 방법 | JSON 파일(`hidden_state_comparison.json`) 읽기만 가능 | **그라운딩 데모에서 이미지 올리고 드롭다운으로 baseline/add/replace 반복 비교** |
| 운영 서버 동작 | 변화 없음(아직 hidden state 미연결) | 기본값(`head_mode=baseline`)은 그대로, 명시적으로 모드를 바꿔야만 새 head 사용 — **운영 리스크 없음** |
| SR 미해결 문제 | val 29개로는 답이 안 나옴 | §4 확대 검증으로 표본 늘려 재확인(이 plan 또는 후속 plan) |

---

## 6. 위험도

- 서버 코드 수정이지만 **신규 필드는 기본값으로 하위호환**, 기존 클라이언트/soda 동작 불변.
- hidden state 추출이 PG2 forward에 끼어들 때 latency가 얼마나 늘어나는지는 실측 전엔 모름 — §3-2에서 측정 후 보고.
- soda 배포는 이 plan 안에서 자동으로 하지 않음 — GB10 검증 통과 후 별도 확인.
