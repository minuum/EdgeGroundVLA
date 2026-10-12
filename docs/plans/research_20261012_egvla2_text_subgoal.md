# 리서치: EGVLA2 — 서브골 인코더에 텍스트 결합 (FiLM 확장)

## 팀 미팅 합의 사항 (2026-10-12, 동환님과 공동 작업)

### EGVLA2 개요
1. **Backbone 변경**: Two models(G,V): OWLv2 + Kosmos-2 → **One model(G,V): Florence-2** (그라운딩+비전을 단일 백본으로 통합)
2. **액션 헤드 변경**: action head → MLP → **FM(FiLM)** — 본 문서에서 검증 중인 서브골 FiLM 조향 방향과 일치

### TODO (팀 합의, 미완료 항목 추적용)
- [ ] 리서치
  - [ ] EdgeGround-VLA 자체 리서치 (담당: 동환님)
  - [ ] 최신 VLA 모델 적용 사례 리서치 (담당: 이민우)
- [ ] 플래닝 — 스프린트 계획표(데이터베이스) 작성
- [ ] 데이터셋 — 양을 500~1000개 수준으로 확대
- [ ] 학습 — 기존 Stage 2 방식(decomposition: bbox+image MLP) 고수 여부 결정
- [ ] 추론
  - [ ] Florence-2 단일 모델 vs OWLv2+Kosmos-2 조합의 로봇 실측 리소스(지연/메모리/전력) 비교 — 효율성 검증
  - [ ] 그라운딩과 비전 인코딩을 Florence-2에서 동시에 할지 분리할지 결론
- [ ] 시각화 — train/val 그래프, 혼동행렬 (논문 작성 단계 이후로 보류)

### 본 리서치 문서와의 연결점
- 위 "Backbone 변경(Florence-2 단일화)"은 아래 §5 "flo2 출력 위치토큰 → bbox 스키마 정합" 항목과 직결됨 — Florence-2를 그라운딩+비전 동시 백본으로 쓰게 되면, 위치토큰 디코딩 후 bbox 스키마를 맞추는 작업이 "선택적 개선"이 아니라 **EGVLA2 필수 경로**가 됨.
- "액션 헤드 MLP→FM"은 이미 본 세션에서 검증한 `SubgoalSteeredMLP`(FiLM) 구조와 동일한 방향 — 팀 합의가 현재 실험 방향을 그대로 지지함.
- 텍스트 서브골 결합(§본문 전체)은 이 팀 TODO에는 명시 안 돼 있지만, "Florence-2 단일 백본"이 애초에 텍스트+이미지+그라운딩을 한 모델에서 처리하는 구조이므로 자연스럽게 합류 가능 — plan.md에서 "Florence-2 전환"과 "텍스트 서브골 결합"을 **하나의 로드맵(먼저 Florence-2 전환 → 그 위에 텍스트 FiLM 추가)**으로 묶을지, 독립적으로 검증할지 결정 필요.

---

## 배경
현재 서브골 조향(`SubgoalSteeredMLP`, `scripts/eval_subgoal_steering_extended_v6.py:156`)은 bbox(cx,cy,area,has_bbox)만 FiLM 조건으로 씀. EGVLA1(Exp11, Kosmos-2+LoRA end-to-end)은 텍스트 경로를 갖고 있지만 **text attention = 0%로 구조적 사망** 상태([[project_text_path_dead]], Exp15 head-only 재확인). 사용자 제안: EGVLA2 = bbox + 텍스트를 함께 서브골 인코더에 넣어 FiLM 생성 → "진짜 VLA"에 더 가까운 구조.

## 체크한 트레이드오프

### 1. 텍스트 인코더 선택지
| 옵션 | 가용성 확인 | 비고 |
|---|---|---|
| Kosmos-2 자체 텍스트 경로 | 이미 사망(attn 0%) | EGVLA1 그대로 쓰면 의미 없음 — 반드시 **별도 경로** 필요 |
| HF CLIPTextModel/Tokenizer | `.venv`에 `transformers==4.49.0` 설치 확인됨 | 바로 사용 가능, 가장 간단 |
| open_clip (ViT-B-32 텍스트타워) | `.venv`에 `open_clip==3.3.0` 설치 확인, `weights/clip/ViT-B-32.pt` 로컬 가중치 존재 | third_party/RoboVLMs의 `ClipTextFeatureEncoder`가 이미 이 패턴을 구현해둠(`third_party/RoboVLMs/robovlms/model/text_encoder/clip_text_encoder.py`) — 단, third_party 코드는 수정 금지 규칙이라 **참고만 하고 우리 쪽에 별도 소형 래퍼로 재구현** 필요 |
| `.venv_lerobot` (지금 eval 스크립트들이 쓰는 환경) | `transformers`/`clip`/`open_clip` 전부 미설치 | 텍스트 인코더 쓰려면 이 가상환경 바꾸거나 패키지 추가 설치 필요 — **환경 분리 이슈** |

**결론**: CLIP text encoder(HF CLIPTextModel, frozen) 사용이 제일 현실적. 단 현재 서브골 eval 스크립트들은 `.venv_lerobot`(패키지 없음)으로 돌아가므로, 텍스트 인코더 추가 시 실행 환경을 `.venv`로 바꾸거나 필요 패키지를 `.venv_lerobot`에 설치해야 함.

### 2. 텍스트 instruction 데이터 가용성
- `bbox_dataset_v6_owl.json`에는 **자유 텍스트 instruction 필드 없음** (path_type, direction만 존재).
- h5 raw 데이터(`ROS_action/mobile_vla_dataset_v5/*.h5`)에도 attrs에 `cx_path`, `cx_position`만 있고 `scenario`는 빈 문자열, 자유 텍스트 없음.
- 그러나 `robovlm_nav/datasets/nav_h5_dataset_impl.py:21` `PATH_TYPE_INSTRUCTIONS` 딕셔너리에 **left/right/straight 3종 템플릿 문장**(영어+한국어 변형, 각 8~10개) 존재 — 학습 시 `random.choice`로 매 스텝 다르게 샘플링해서 사용 중.
- V6는 5-direction(weak/strong left/right + center) × 3 curve인데, 기존 템플릿은 3종(left/right/straight)뿐이라 **weak/strong 구분이 텍스트에 반영 안 됨** → EGVLA2용으로는 direction 세분화 반영한 신규 템플릿 작성이 필요함(기존 재사용 시 weak_left와 strong_left가 같은 문장을 받게 됨, 정보 손실).

### 3. 구조적 설계 포인트
- FiLM 입력을 어떻게 합칠지 선택지 2가지:
  - (a) bbox_feat와 text_feat를 concat한 뒤 하나의 `subgoal_enc`에 통과 → 단일 (γ,β)
  - (b) bbox용 FiLM과 text용 FiLM을 따로 만들어 순차 적용(2-stage FiLM, CLIPort 등에서 쓰는 방식)
  - (a)가 구현 단순, 파라미터 적음 — 먼저 (a)로 베이스라인 잡는 게 합리적
- 텍스트 인코더는 **frozen**으로 두는 게 안전(현재 모든 실험이 소규모 MLP 비교라 텍스트 인코더까지 학습시키면 비교축이 흔들림, GPU/시간도 커짐)
- CLIP text 임베딩 차원(ViT-B-32 기준 512-d)이 현재 bbox_feat(4~6-d)보다 훨씬 커서, 그대로 concat하면 텍스트가 FiLM을 지배할 위험 — 축소 projection(512→32 등) 필요

### 4. "진짜 VLA로서 쓰려면" 추가로 필요한 것
단순히 서브골 인코더에 텍스트를 추가하는 것과, EGVLA 전체를 "진짜 VLA"라고 부를 수 있으려면 격차가 있음. 체크리스트:
1. **텍스트가 실제로 행동을 바꾸는지 검증** — ablation으로 같은 bbox, 다른 text(예: "left" vs "right" instruction)일 때 예측 action이 달라지는지 확인해야 함. 지금 데이터가 코스(direction)와 텍스트가 1:1로 고정돼 있어서, 이 ablation을 하려면 **동일 bbox 상황에 다른 text를 주입하는 synthetic probe**가 필요(Exp54 masking 검증과 같은 패턴, [[project_exp54_5track_validation]]).
2. **자유형 언어 지시 데이터 부족** — 현재 전부 템플릿/고정 어휘라 "VLA가 임의 자연어 지시를 따른다"는 주장은 못 함. 템플릿 수준의 text-conditioning 검증 정도로 scope를 명확히 해야 함(논문에서 overclaim 방지).
3. **closed-loop(실제로는 offline replay)에서의 text 조건 평가** — 지금 쓰는 `evaluate_closed_loop_v5.py`류는 teacher-forced라서 text가 실제 경로를 바꾸는지 못 봄. 진짜 영향력 검증은 PM(frame-level) + 위 1번 probe로 충분.
4. **EGVLA1과의 공정 비교** — EGVLA1은 전체 VLM 파인튜닝, EGVLA2(text+bbox FiLM MLP)는 소형 decomposition 모델이라 파라미터 규모 차이가 큼 → PM 비교 시 "경량 decomposition이 그라운딩+텍스트만으로 대형 VLM end-to-end보다 낫다"는 식으로 프레이밍하면 이전 concat/steered 비교와 일관성 유지됨.

### 5. Florence-2(flo2) 출력 포맷 정합 — bbox 스키마 통일 필요
- Florence-2(`<CAPTION_TO_PHRASE_GROUNDING>` 등)는 원시 출력이 **위치 토큰(`<loc_xxx>` quantized bin)**이고, OWLv2처럼 바로 (cx,cy,area) 실수값을 주지 않음.
- 선례 확인: `scripts/l1_florence2_grounder.py:103-110`에서 이미 `proc.post_process_generation(...)`으로 위치 토큰을 픽셀 박스(x1,y1,x2,y2)로 디코딩한 뒤, `cx=(x1+x2)/2/W, cy=(y1+y2)/2/H, area=(x2-x1)*(y2-y1)/(W*H)` 형태로 **OWLv2(`cx_det,cy_det,area_det`)와 동일한 정규화 스키마**로 변환해서 쓰고 있음 — 동일 변환 패턴을 그대로 재사용 가능.
- EGVLA2 설계에 반영할 점: 서브골 인코더(`subgoal_enc`)의 입력 `BBOX_DIM` 스키마(cx,cy,area,has_bbox 등)를 **그라운딩 백엔드 불문(OWLv2든 Florence-2든) 동일 포맷으로 고정**해야, 나중에 그라운더를 OWLv2→Florence-2로 교체하거나 두 백엔드를 섞어 써도 같은 `SubgoalSteeredMLP`/FiLM 레이어를 재학습 없이(혹은 최소 변경으로) 그대로 쓸 수 있음.
  - 즉 **"Florence-2 위치토큰 → (cx,cy,area,has_bbox) 디코딩"을 전처리 단계(Stage 0/1)에서 끝내고, MLP 입력단에는 백엔드 종류와 무관한 단일 bbox 텐서만 들어가게** 한다 — 지금 V6 파이프라인(OWLv2 기준 `cx_det/cy_det/area_det`)과 동일한 계약(contract)을 Florence-2 경로에도 강제.
  - `has_bbox`/`detected` 플래그도 Florence-2 쪽 "박스 없음"(boxes 리스트 empty) 케이스를 동일하게 False로 매핑해야 `steered_reg`의 identity-regularization 로직이 백엔드 교체 후에도 그대로 작동함.
  - 주의사항(CH59 이력 참고): Florence-2 OVD 자체 디코딩 결함(pred cx가 0.559 근처서 막히는 현상)이 과거 있었고, 67-5에서 "비전 피처로 쓰면 결함 사라짐" 확인된 바 있음 — flo2를 그라운딩 백엔드로 실제 채택할 경우 이 결함 재발 여부를 cx 분포로 먼저 확인한 뒤 bbox 스키마 변환에 들어가야 함.

## 리서치 결론 (구현 전 요약)
- 기술적으로 실행 가능: CLIP text encoder(frozen, HF `transformers` 또는 `open_clip`) + 기존 FiLM 구조 확장.
- 선행 조건 3가지: (1) `.venv_lerobot`에 텍스트 인코더 패키지 설치 또는 실행 환경 전환, (2) V6 5-direction을 반영한 신규 텍스트 템플릿 작성, (3) "텍스트가 실제로 영향을 주는지" 검증할 ablation 프로브 설계.
- 이 네 가지(텍스트 인코더/환경, V6 방향별 템플릿, text-influence 검증, **Florence-2 위치토큰→bbox 스키마 정합**)를 반영한 plan.md를 다음 단계로 작성 예정 — **아직 코드 작성 안 함**, 사용자 승인 대기.
