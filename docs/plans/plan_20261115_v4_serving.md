# V4·V4R 실기 서빙 경로 추가 (T6) — 계획서 (승인 대기)
작성 2026-11-15. **승인 전까지 구현하지 않는다.** (CLAUDE.md 5단계 중 2단계) 로봇을 움직이는 코드가 포함되므로 단계별 승인을 받는다.
참고: `docs/plans/STANDARD_PLAN_TEMPLATE.md`가 없어 `plan_20261111_v4r_bbox_aug.md` 형식을 따랐다.

## 1. 왜 하나 (리서치로 확인한 사실)
| 사실 | 근거 |
|:--|:--|
| V4(EG4-04)·V4R(EG4-07) 체크포인트는 이 서버(minum) `runs/egvla2_fm/textcond/`에만 있고, 배포 폴더 `deploy/MANIFEST.json`에는 V4·EG4 항목이 0건이다 | 파일 확인 |
| 체크포인트 구조가 다르다: V4는 `{state, variant, bdim=12, fdim=1536, ...}`이고 `window`·`arch` 키가 없다. 서빙 정책 `EgVla2FMPolicy`는 `ck["window"]`·`ck["arch"]`를 읽으므로 V4를 로드하지 못한다 | `egvla2_fm_policy.py`, V4 ckpt 키 |
| V4 입력 = 박스 창(3프레임×4) + 이미지 특징 768 + **문구 특징 768**. 서빙 추출기는 768만 만든다 | `t7_compare_seeds.py`, `FusedF2Extractor` |
| 문구 특징 = 융합 인코더에 (이미지 토큰 + 문구 토큰)을 넣고 `last_hidden_state[:, 이미지토큰수:].mean(1)` | `precompute_text_vocab.py`, `t7_extract_session_feats.py` |
| 서버는 지시어를 무시하고 인식 문구를 `PHRASE="gray basket"`으로 고정한다. `InferenceRequest.instruction`은 "호환용(무시)" | `egvla2_inference_server.py:143`, `precompute_v6_florence2_fused.py:36`. 세션에 기록된 커밋 5ff4a598과 `monavla-driving` 브랜치도 같다 |
| V4·V4R 학습 때 박스는 **항상 gray basket 문구로 찾은 박스**였고, 없는 문구(NEG)에서는 박스를 남긴 채(KEEP 0.5) 정지 라벨로 학습했다 | `eval_text_cond_t4.py`, `t7_train_v4r.py` |
| 로봇 서버(soda)가 이 저장소와 다른 수정본으로 도는지 **확인하지 못했다** | 11/13 세션에 의자·화분 지시어가 H5에 기록돼 있으나 코드는 무시하게 돼 있음 |

## 2. 설계 (사용자가 승인한 방향)
**인식 문구는 gray basket으로 고정하고, 사용자 지시어는 문구 특징으로만 넣는다.** 학습 분포와 같은 구조다.
- 박스: 기존과 동일하게 `<CAPTION_TO_PHRASE_GROUNDING>gray basket`으로 생성.
- 지시어: 요청의 `instruction`을 융합 인코더에 넣어 문구 특징 768을 계산 → 이미지 특징 768과 이어 붙여 1536 입력.
- 결과: 지시어가 gray basket 계열이면 주행, 다른 물체면 정지(V4·V4R의 학습된 동작). **이 설계에서는 새 물체(의자·책상)로 가는 동작이 나오지 않는 것이 정상이다.** 새 물체로 가려면 T3(두 물체 수집)과 문구 설계가 별도로 필요하며 이 계획에 포함하지 않는다.
- 기본값은 **꺼짐**: 환경변수 `VLA_EGVLA2_TEXT`가 없으면 현재 A3 경로가 그대로 동작한다(A3 경로 코드는 건드리지 않음).

## 3. 수정·추가할 파일
| 파일 | 작업 |
|:--|:--|
| `robovlm_nav/serve/egvla2_text_policy.py` (신규) | `FusedF2TextExtractor`(기존 `FusedF2Extractor` 확장: 이미지 인코더 1회 → 박스 생성 + 문구 특징 1회 인코더 통과), `TextCondPolicy`(`PiZeroFM(12, 1536)` 로드, 창 3, `zero_on_miss=True`, `sample_chunk`) |
| `robovlm_nav/serve/egvla2_inference_server.py` (수정) | `load_service_from_env`에 `VLA_EGVLA2_TEXT=1` 분기, `EgVla2Service.predict_rgb(image, instruction=None)`, `/predict`가 `request.instruction`을 전달, 응답 `meta`에 `variant`·`phrase` 기록 |
| `scripts/check_text_serving_equiv.py` (신규) | 오프라인 동등성 점검(§4) |
| `tests/test_egvla2_text_policy.py` (신규) | 모의 추출기로 정책·서비스 단위 테스트 |
| `docs/plans/plan_20261115_v4_serving.md` | 완료 표시 |

`egvla2_fm_policy.py`, `egvla2_only_logic.py`(컨트롤러), 학습 코드, 기존 체크포인트는 수정하지 않는다.

### 코드 스니펫 (신규 파일 안에서만)
```python
class FusedF2TextExtractor(FusedF2Extractor):
    def __call__(self, image_rgb, phrase):
        pil = Image.fromarray(np.ascontiguousarray(image_rgb).astype(np.uint8)).convert("RGB")
        P = self._P
        inp = self.proc(text=[P.TASK + P.PHRASE], images=[pil], return_tensors="pt")   # 박스 문구는 gray basket 고정
        pv = inp["pixel_values"].to(self.device, self.dtype)
        img = self.model._encode_image(pv)                                            # 1회
        bbox = ...  # 기존 run_batch의 생성·파싱과 동일
        ids = self._phrase_ids(phrase)                                                # 문구별 토큰 캐시
        emb = self.model.get_input_embeddings()(ids)
        emb, att = self.model._merge_input_ids_with_image_features(img, emb)
        enc = self.model.language_model.get_encoder()(inputs_embeds=emb, attention_mask=att).last_hidden_state.float()
        txt = enc[:, img.shape[1]:].mean(1)[0]
        return bbox, np.concatenate([img.float().mean(1)[0].cpu().numpy(), txt.cpu().numpy()])   # 1536
```
`TextCondPolicy.step`은 `EgVla2FMPolicy.step`과 같은 창·샘플링·`to_drive_command`를 쓰되 입력 특징이 1536이다. 구현 때 기존 `run_batch`와 박스 생성이 비트 동일한지 먼저 확인한다.

## 4. 검증 (구현 후, 로봇을 움직이기 전에)
**오프라인 동등성 (이 서버, 로봇 없음)**: 기록된 프레임(11/13 의자·바구니·화분 각 일부 + 11/05 일부, 총 약 60프레임)을 서빙 경로(JPEG q95 왕복 포함)에 넣어, 학습·재생에 쓴 캐시(`t13_obj_cache.npz`, `t7_session_cache.npz`)와 비교한다.
| 기준 | 값 |
|:--|:--|
| 문구 특징 코사인 유사도 | ≥ 0.999 (fp16/fp32 허용) |
| 박스 | 캐시 박스와 중심 오차 ≤ 0.02 |
| 명령 방향 각도 차(중앙값) | ≤ 1° |
| 정지 판정 일치율 | ≥ 95% |
- 기준은 구현 전에 정해 둔다. 하나라도 어기면 로봇 단계로 넘어가지 않는다.
- 지연(latency)은 이 서버에서 측정하되, 로봇 서버 측정(T5-b)은 별도다. 문구 특징 계산이 프레임당 인코더 1회를 추가하므로 지연이 늘 수 있다(**미측정**).
- 컨트롤러가 V4의 "작은 명령"을 실제 정지로 바꾸는지는 **미확인**이라 이 단계에서 같이 본다. 11/13 의자 가운데에서 A3 명령은 정확히 0.0이었고 로봇은 움직이지 않았다.

## 5. 배포·실기 단계 (각 단계마다 사용자 승인)
| 단계 | 내용 | 로봇 |
|:--|:--|:--|
| 0 | 위 오프라인 동등성 통과 | 없음 |
| 1 | 체크포인트(각 3.3MB)와 코드를 soda로 전달. 전달 경로(rsync 또는 브랜치)는 승인 때 결정 | 없음 |
| 2 | soda 서버에서 기록 프레임을 API로 보내 응답만 확인(구동 끔). **운영 서버가 떠 있을 때 다른 모델을 별도로 로드하지 않는다**(Jetson 재부팅 사례, 메모리 기록) — API 호출로만 테스트 | 정지 |
| 3 | gray basket 문구로 A3와 같은 바구니 시나리오 몇 건(비상정지 대기) | 구동 |
| 4 | 물체 4종 × G2·G3·G4, 지시어 4종, V4와 V4R 각각. 건수는 승인 때 결정 | 구동 |
- 3~4단계의 성공/실패 판정 기준은 **사전에 문서로 정한다**(지금까지는 operator 수동 판정, 기준 없음). 제안: 바구니 문구에서는 "바구니 도착", 다른 물체 문구에서는 "출발하지 않고 정지가 정답"을 별도 표로 기록. 이 판정표는 동환과 합의가 필요하다.
- 세션 H5에 지시어와 서버의 `variant`가 모두 기록되게 한다(11/13의 "문구는 메타데이터일 뿐" 문제 재발 방지).

## 6. 체크포인트 선택 (결정 필요)
V4 시드 0(`EG4-04_V4_full.pt`)은 10시드 중 **반전 좌우 일치가 42%로 가장 약한 시드**다. 파일명만 보고 기본 체크포인트를 올리면 최약 시드가 배포된다.
| 시드 | V4 정상 각도 / 반전 일치 / 동의문구 정지 | V4R 정상 각도 / 반전 일치 / 동의문구 정지 |
|:--|:--|:--|
| 0 | 2.79° / **42%** / 6% | 2.71° / 96% / 3% |
| 1 | 2.77° / 93% / 8% | 2.63° / 95% / 8% |
- 제안: **V4와 V4R 모두 시드 1을 쓴다**(10시드 평균에 가까운 시드, 사전 규칙으로 고정). 어떤 시드를 고르든 선택에 따라 결과가 달라질 수 있으므로, 실기 결과를 보고할 때 10시드 분포를 같이 적는다.

## 7. 트레이드오프·위험
- **새 물체 목표와 반대 방향:** V4·V4R은 gray basket이 아닌 문구에서 정지하도록 설계됐다. 이번 실기는 "정지 동작 + 바구니 성능" 검증이지 "새 물체 주행" 개선이 아니다.
- **soda 코드 불일치:** soda에 수정본이 있으면 이 저장소 기준으로 만든 코드가 그대로 안 붙을 수 있다. 1단계 전에 soda의 서버 코드(해당 파일 해시나 diff)를 동환이 확인해 알려야 한다.
- **문구 분포 밖:** 의자·책상·화분 지시어 중 화분은 학습 때 NEG 문구라 정지가 설계된 동작이고, 의자·책상은 학습 문구에 없다. "brown wooden chair" 같은 문구에서의 정지는 재생에서 확인됐으나(V4R 0~23% 움직임), 열린고리 재생이라 실기와 다를 수 있다.
- **지연 증가:** 인코더 1회 추가. 측정 전이라 크기를 모른다.
- **A3 유사도 기준:** V4·V4R의 우열은 A3 명령과의 유사도로만 비교했다. 실기 검증이 없다.

## 8. 예상 시간 (추정, 보장 아님)
구현 1~2시간, 오프라인 동등성 30분. 3~4단계는 동환 일정에 따른다.

## 9. 이 계획에 포함하지 않는 것
T3 데이터 수집, V4·V4R 재학습, 새 물체로 가는 모델, LiDAR 정지(교수님 지시로 "알아보기만"이며 별도), A3 서빙 코드 수정, 커밋과 push.

## 10. 승인 때 정해 주세요
1. 이 설계(박스 문구 고정 + 지시어는 문구 특징)로 진행 (제안대로 / 수정)
2. V4와 V4R 둘 다 시드 1로 (제안대로 / 다른 시드·한쪽만)
3. 오프라인 동등성 기준 (제안대로 / 수정)
4. soda 전달 방식 (rsync / 브랜치 커밋 후 pull / 동환이 결정)
5. 4단계 실기 건수와 판정표는 동환과 합의 후 별도 확정 (이번 승인 범위 아님)

---
## 11. 구현·검증 결과 (2026-11-15, 승인된 §10 항목 1~3을 제안대로 구현, 4는 미정)
### 완료
- [x] `robovlm_nav/serve/egvla2_text_policy.py` 신규(`FusedF2TextExtractor`, `TextCondPolicy`)
- [x] `egvla2_inference_server.py`: `VLA_EGVLA2_TEXT=1` 분기(기본 꺼짐), `predict(image, instruction)`, 응답에 `instruction_used`·`variant`
- [x] `tests/test_egvla2_text_policy.py` 3건 통과(pytest 미설치라 직접 실행). 기존 `test_egvla2_inference_server.py`는 6건 통과·1건 실패(`test_lost_search_rotation_in_response`)인데 **내 수정 전 원본에서도 같은 1건이 실패**한다(이번 변경과 무관, 별도 확인 필요)
- [x] `scripts/check_text_serving_equiv.py`: 시드 1 체크포인트(V4·V4R), 기록 세션 8개 109프레임, JPEG q95 왕복, fp32·fp16 둘 다

### 동등성 결과 (기준은 사전 고정)
| 기준 | 결과 (V4·V4R, fp32; fp16도 거의 같음) | 판정 |
|:--|:--|:--|
| 문구 특징 코사인 ≥0.999 | 최소 0.99989, 평균 0.99999 | 통과 |
| 이미지 특징 코사인 | 최소 0.99993 | 통과 |
| 명령 방향 각도 차 중앙값 ≤1° | V4 0.08°, V4R 0.07° (비교 가능 프레임 44~45개) | 통과 |
| 정지 판정 일치 ≥95% | 100% | 통과 |
| 박스 중심 오차 ≤0.02 | **바구니 세션·11/05 세션은 전부 일치.** 의자·화분·책상 세션은 프레임 일부가 불일치(아래) | 바구니 기준 통과, 나머지는 원인 확인됨 |

### 새로 확인한 것 (중요)
65세션 전체를 gray basket 고정 문구로 다시 추출해 기록된 박스와 비교했다.
| 물체 | 박스 불일치 프레임 비율(평균) |
|:--|:--|
| 바구니 15건 | **0%** (전부 일치) |
| 의자 | 가운데 100%, 옆 80% |
| 작은 책상 | 가운데 100%, 옆 94% |
| 화분 | 가운데 76%, 옆 100% |
- 바구니가 0%로 일치하므로 서빙 경로의 박스 재현은 맞다. 나머지 물체에서 다른 것은 **11/13에 로봇 서버가 지시어로 찾은 박스를 썼다**는 뜻이다. 이 저장소의 서버 코드(지시어 무시)와 다르다. 앞서 "지시어를 무시할 가능성이 크다"고 쓴 §1의 해석은 11/13 실제 실행에는 맞지 않는다(코드는 그렇게 돼 있으나 soda가 다른 코드로 돎).
- 따라서 **soda 서버의 실제 코드 확인이 1단계의 선결 조건**이다. 이 저장소 기준으로 만든 경로를 soda에 붙이면 soda의 지시어 전달 방식과 충돌할 수 있다.
- 서빙 설계(박스 gray basket 고정, 지시어는 문구 특징) 그대로 11/13 65건을 재생한 V4R·V4 움직임 비율: 바구니 98~99%, 의자 0%, 책상 13~24%, 화분 5~15%. 기록 박스로 재생한 결과와 같은 경향이다(열린고리·실기 아님).

### 남은 것 (이 계획의 단계 1~4, 승인 대기·로봇 필요)
- [ ] 단계 1: soda 서버 코드 확인(동환) 후 체크포인트·코드 전달 방식 결정 (§10-4)
- [ ] 단계 2~4: API 응답 확인, 바구니 시나리오, 물체 4종×G2·G3·G4 (판정표는 동환과 합의)
- [ ] 지연 측정(로봇 서버, T5-b)은 미실시
