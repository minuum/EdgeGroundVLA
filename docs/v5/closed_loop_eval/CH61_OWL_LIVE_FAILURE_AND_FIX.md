# CH61 — 실로봇 OWL-v2 배포 첫날: 방향 편향 원인 규명 + vis_feat 정규화 버그 발견

> 작성일: 2026-07-07
> 배경: 7/6 OWL-v2(th=0.25) 실배포 첫 실로봇 테스트에서 obj_left/right 실패 다수 관측
> 선행: docs/v5/grounding_benchmark/CONCLUSION.md, docs/v5/closed_loop_eval/CH60_OWL_TEXT_CLOSED_LOOP.md

## 요약

실로봇 obj_right 테스트가 반복 실패(SR 0/16, 세션 로그 기준)한 원인을 추적한 결과,
**그라운더(OWL-v2) 문제가 아니라 두 가지 별개 이슈**로 좁혀졌다:
1. **exp71 헤드가 실전 flicker 분포(has_bbox 40~60% 결측)를 학습 때 거의 못 봄**
   (학습 데이터는 has_bbox 95.9~97.8%) — 재학습으로 개선 가능성 확인
2. **재현/검증 파이프라인 자체에 있던 vis_feat L2-정규화 누락 버그** 발견 — 오늘 만든
   "진짜 exp71 레시피" 실험 다수가 이 버그 상태로 진행됐다가 발견 후 재검증

## 1. 실로봇 실패 관측 (obj_left/right, 32 에피소드 누적)

| 경로 | n | 성공 | 특이사항 |
|---|---|---|---|
| obj_left (타겟 좌측) | 8 | 3 | top액션이 오히려 우측계열(ROT_R/FWD+R) 6/8 |
| obj_right (타겟 우측) | 16 | **0** | 방향 맞아도(7/16) 전부 실패 |

7/6 실측 OWL-v2 세션(171922 preview=True, 172030 preview=False) 직접 h5 분석:
- 172030: cx가 0.75→0.94로 실제 우측 드리프트가 있었는데도 **14프레임 전부 FORWARD 고정**
- 171922: preview 스캔 전체(10프레임)에서 탐지율 0%

## 2. 그라운더 vs 헤드 — 어느 쪽 문제인가

- **PG2 시절 세션 재확인**: "방향으로 간" 것처럼 보였던 세션들은 cx가 0.50 근처로 거의
  안 움직이는데도 FWD+L→RIGHT→FORWARD 블록으로 방향이 바뀜 → **grounding cx가 아니라
  raw image 자체로 모델이 판단한 것**으로 추정 (feature ablation: image_only 75.6% >
  bbox_only 67.4%와 일치)
- **preview 로직 코드 확인**: `inference_count==0`일 때만, 최대 5회, ROT_L/ROT_R만 반환
  가능 — FWD+L/RIGHT 같은 다중 블록은 preview가 만들 수 없는 값 → **PG2 시절 sweep은
  preview가 아니라 본 모델의 자체 판단이었다** (최초 가설 정정)
- **45ep 경량 프록시 실험**: exp71과 전혀 다른 아키텍처도 동일 cx 궤적에서 100% 동일하게
  FORWARD로 수렴 → 그라운더/헤드 특정 조합이 아니라 **데이터 분포 자체의 문제**

## 3. soda 관측: OWL-v2 flicker + 조건 무관 액션 수렴 (FINDING_20260706)

- 세션당 has_bbox=False 40~60%, 근접 직후(area 임계값 통과 직후) 검출이 끊기는 패턴
- 학습 데이터 has_bbox=95.9%(bbox_dataset_base_pg2_cx.json) / 97.8%(exp71 실제 소스
  bbox_dataset_pg448_cx.json) — 실전 flicker율과 **큰 분포 괴리**

## 4. 재학습 실험 (1차: 45ep 프록시 → 2차: 실제 exp71 레시피)

### 45ep 프록시 (decomposition MLP, IID flicker)
baseline이 랜덤 dropout에 이미 강건(78.5%→77.2%) — 그러나 IID flicker는 실전(상관형)과
다르므로 결론 보류.

### 상관형 flicker(근접 직후 집중) + 실제 exp71 레시피 (150ep, FrozenCLIPV2+Transformer)

| 변형 | val_acc | 진동율 |
|---|---|---|
| baseline (w6) | 98.4% | 3.0% |
| dropout_aug | 97.1% | 5.2%(최악) |
| sticky_aug | 98.1% | 1.9% |
| **window3** | **98.7%** | **1.9%** |

조합(sticky+window3, 확률 스윕)은 단일 window3보다 낫지 않음 → dropout류 증강은 폐기,
window3 단순 축소가 최선 후보.

**그러나 진짜 성공기준(FPE/SR/TLD, rollout_core 리플레이)으로 재확인하면 window3와
window6(운영)이 사실상 동률**(SR 97.7% 동일, FPE 0.087 vs 0.091m) — 진동율 지표로 낸
"window3 우세" 결론은 철회. 리플레이 자체가 카메라 피드백이 없는 근본적 한계
(CH60-c 기존 지적)라 더 이상 이 방식으론 판별 불가.

## 5. 치명적 버그 발견: vis_feat L2-정규화 누락

운영 서버(`Stage1Encoder.encode_image`, `stage2_v2_inference_server.py:381`)는
`F.normalize()`로 이미지 feature를 L2 정규화하는데, 연구용 재현 스크립트
(`train_exp71_stage2_transformer.py`의 `FrozenCLIPV2.encode_batch`)는 정규화를 안 함.

**soda 실제 7세션(72~78) 실측 bbox로 검증**:

| 세션 | 정규화 없이 재현 일치율 | 정규화 후 재현 일치율 |
|---|---|---|
| 231153 | 80.0% | 93.3% |
| 233327 | 25.0% | **91.7%** |
| 233424 | 11.1% | **88.9%** |
| 233159 | 46.7% | 86.7% |

**영향 범위**: 이번 세션에서 "진짜 exp71 레시피"라고 진행한 실험 다수(flicker
robustness, 헤드 구조 비교, truth_mini 검증, window3 vs window6 라이브 비교)가 전부
이 버그 상태로 이뤄짐 — 이후 실험(§4 후반부, §6)은 정규화 수정 후 재검증한 결과.

## 6. 셀프라벨링 데이터(bbox_truth_mini) 활용 — 오염 발견 및 정정

- 1차 시도: truth_mini(72프레임/18ep) clean-bbox 검증에서 **3개 헤드 전부 100%** —
  그러나 18ep 중 **15개가 이미 학습셋에 포함**돼 있어 오염(암기) 의심
- **격리 재학습**(truth_mini 18ep 완전 제외, 132ep 재분할) 후 재검증:

| | val_acc | truth_mini 진짜 held-out acc |
|---|---|---|
| baseline_w6 | 97.0% | 95.8% |
| window3 | 94.8% | **98.6%** |

진짜 held-out에서도 95.8~98.6%로 여전히 높음 — **헤드가 clean bbox에 대해 실제로
일반화한다는 것 자체는 신뢰성 있게 확인됨.** (다만 정적 분류 정확도라 폐루프 진동은
검증 불가 — §4 한계와 동일)

## 7. VLA 사다리 ② (언어 조건화) 재검증 — 버그 수정 후, 결론이 더 강하게 부정적으로 바뀜

| 비교군 | PM (43ep 프록시, 버그 상태) | PM (150ep 실제 레시피, 버그 수정) |
|---|---|---|
| no_text | 78.4% | 87.5% |
| with_text(real) | 81.8% (+3.4%p) | **85.0% (오히려 하락)** |
| shuffled_text | 77.5% | 80.6% |

- Permutation: 두 실험 모두 큰 하락폭(−14.6%p / **+16.1%p**) — 텍스트를 실제로 참조함
- **Counterfactual 변화율: 43ep 실험 20.3% → 150ep 실제 레시피 정확히 0.0%** —
  왼쪽/오른쪽 지시를 강제로 바꿔도 예측이 단 하나도 안 바뀜. 버그를 고치니 오히려
  "텍스트는 경로 맥락(prior)일 뿐 명령이 아니다"라는 결론이 더 명확해짐

## 8. Vision encoder 비교 — PG2(SigLIP) vs Kosmos-2

exp71 계열이 지금까지 전부 Kosmos-2 vision encoder(`FrozenCLIPV2`, 실제로는 CLIP이
아님)를 썼는데, PG2(PaliGemma2-448)의 SigLIP vision tower(1152d)로 교체해도 동등한지
확인 (`scripts/train_exp71_pg2vision_head.py`).

| Vision 소스 | val_acc |
|---|---|
| Kosmos-2 (256d, 정규화 수정판) | 97.0~98.4% |
| **PG2/SigLIP (1152d→256 학습된 projection)** | **96.2%** |

1차 시도(raw 1152d를 그대로 Transformer에 투입)는 val_acc 73.4%로 다수클래스(FORWARD
72.2%) 붕괴와 정확히 일치 — 학습 실패였음. Kosmos-2 버전처럼 gradient가 정상 흐르는
학습형 projection을 헤드 내부에 넣어 재학습하니 96.2%로 정상화, **Kosmos-2와 사실상
대등**. → 어떤 vision encoder를 쓰든 큰 차이 없음, 병목은 다른 곳(§3, §7)에 있음이
재확인됨.

## 9. 조이스틱 이질 지시 — 합성 근사 테스트 (실물 수집 전 사전 확인)

실물 조이스틱 데이터 없이, 같은 프레임에 상충되는 두 합성 지시("curve left
decisively"/"curve right decisively")를 강제로 붙이고 각각 다른 정답 클래스
(FWD+L / FWD+R)로 라벨링한 합성 쌍을 훈련 데이터에 섞어서, "같은 이미지에 다른
지시 → 다른 정답"이 실제로 존재할 때 counterfactual 반응이 살아나는지 확인.
(`scripts/train_exp71_synthetic_obedience.py`)

**결론**: 조이스틱 이질 지시 데이터(장면-지시 상관이 깨진 데이터) 없이는 언어 조건화가
헤드 구조/레시피를 어떻게 고쳐도 명령 순응으로 이어지지 않는다는 게 이번 재검증으로
더 확고해짐.

## 8. 프리뷰 재설계 검토 (docs/plans/plan_20260706_preview_redesign.md)

옵션 A(제거)/B(양방향 탐색)/C(threshold 조정)/D(로깅 강화) 중 **D → A 순서 추천**.
다음 로봇 테스트(obj_right preview=False 5개 추가 수집)가 그대로 A의 실측 데이터가 됨.

## 10. obj_left/right/center 테스트 — 학습 분포 밖(OOD) 스트레스 테스트였다

> **정정 (path_type 명명)**: `mobile_vla_data_collector.py` 원본 확인 결과, path_type의
> 첫 단어는 "로봇 시작 위치"가 아니라 **목표(바구니) 위치**(좌/중/우), 두 번째 단어는
> **접근 경로의 곡선 방향**(좌/직진/우)이다. 예: `center_left` = 목표는 중앙, 접근은
> 왼쪽 곡선. 교수님 3/27 프로토콜의 "Step3 33/33/33(left/straight/right)"은 이 중
> 두 번째 축(경로 곡선 방향)만 3분류로 묶은 커리큘럼 배합비이며, 목표 위치 축과는 별개.
> 아래 cx 커버리지 수치 자체(실측값)는 이 정정과 무관하게 그대로 유효.

`right_left`(exp66 시절, 6/26) vs `obj_*`(6/30~, 신규 테스트) 성공률 비교:

| 경로 | 성공률 | 비고 |
|---|---|---|
| **right_left** | **34/34 (100%)** | V5 9종 path_type 중 하나 — in-distribution |
| obj_left | 3/8 (37.5%) | 9종 path_type에 없음 — 새 테스트 프로토콜 |
| obj_center | 3/8 (37.5%) | 〃 |
| obj_right | **0/26 (0%)** | 〃 |

`right_left`는 학습 때 본 9종 path_type 중 하나라 100% 성공은 당연한 결과 — 그라운더/
프리뷰 교체와 무관. `obj_*`는 6/30부터 시작된 별개의(임의 배치) 스트레스 테스트.

**`CX_RULE_THRESHOLDS`(서버 방향 판정 5구간) 기준 학습 데이터 커버리지**:

| 구간 | cx 범위 | 학습 프레임 비율 |
|---|---|---|
| ROT_L (강한좌) | <0.25 | 1.8% |
| FWD_L | 0.25~0.40 | 15.9% |
| FORWARD (중앙) | 0.40~0.60 | **68.0%** |
| FWD_R | 0.60~0.75 | 12.9% |
| ROT_R (강한우) | >0.75 | 1.4% |

강한좌+강한우 합쳐 **3.2%**뿐. `center_left`/`center_right`는 이 구간을 **단 한 프레임도**
겪은 적 없음(right_right가 그나마 8.3%로 최다). obj_right가 요구하는 cx 0.9대 영역은
학습 데이터가 거의 커버 못 하는 지대 — 오늘 관측된 "cx 0.9에서도 FORWARD 고정"의
근본 원인과 정합적. class-level 통계(ROT_L/R gt_class 각 0.63%)와 cx-bucket 통계(3.2%)가
서로 다른 각도에서 같은 결론을 가리킴.

**시사점**: 재학습으로 이 문제를 풀려면 헤드 구조나 그라운더가 아니라 **강한좌/강한우
구간(cx<0.25 또는 >0.75) 프레임 자체를 데이터에 훨씬 더 많이 확보**해야 함 — 조이스틱
이질 지시 데이터 수집 시 이 구간도 같이 충분히 커버하도록 설계 필요.

## 종합 결론 및 다음 단계

1. **그라운더(OWL-v2)는 무죄에 가까움** — clean bbox 검증(§6)에서 헤드 자체 일반화力 확인
2. **진짜 병목은 (a) flicker 분포 불일치, (b) 언어 미조건화, (c) obj_* 테스트 자체가
   학습 분포 밖(강한좌/우 구간 3.2%뿐)** 세 가지
3. **오프라인 리플레이 방법론은 한계에 도달** — 카메라 피드백 부재로 더 이상 판별 불가,
   실로봇 A/B(window3 재학습 헤드 vs 기존 window6)가 사실상 유일하게 남은 확정적 검증
4. **다음 우선순위**: (a) 실로봇 window3 vs window6 A/B, (b) 조이스틱 이질 지시 +
   강한좌/우 구간 데이터 수집 설계 착수, (c) preview 옵션 D(로깅 강화)부터

## 산출물

- `scripts/train_owl_flicker_robustness.py`, `scripts/eval_correlated_flicker_oscillation.py` — 45ep 프록시
- `scripts/train_exp71_flicker_robustness.py`, `scripts/train_exp71_multihead_truthmini.py` — 실제 exp71 레시피(버그 있는 상태)
- `scripts/eval_truthmini_holdout_and_cl.py` — truth_mini 격리 재학습 + FPE/SR/TLD 리플레이
- `scripts/eval_live_sessions_owl_flicker_windows.py` — soda 실제 세션 재현 + 정규화 버그 발견
- `scripts/train_exp71_instr_conditioning.py` — VLA 사다리 ② 정규화-수정 재검증
- `docs/plans/plan_20260706_preview_redesign.md` — 프리뷰 재설계 플랜

## 11. 좌/우 비대칭 발견 + bbox_scale 대응 + 배포

**발견**: FWD+L recall 88.5%인데 FWD+R recall 67.6%(21.6%가 FORWARD로 오분류) — 극단
cx가 아닌 정상범위에서도 좌/우 비대칭 존재. cx 0.6~0.75 구간의 실제 정답이 FWD+L이
더 많은 경우도 확인(경로 곡선 형태 때문 — cx 단독으로는 정답이 안 정해짐, §10 정정 참고).

**원인 후보 검증**: class-weight(`diag_mult`)만으로는 단일시드 표준편차가 24~27%p로
극도로 불안정(같은 설정 재현 시 89.8%↔72.4% 널뛰기 확인). **`bbox_scale`(bbox 4dim을
이미지feature 대비 상대적으로 키우는 것)이 훨씬 효과적** — 멀티시드(5) 비교:

| 조합 | val_acc | FWD+L recall | FWD+R recall |
|---|---|---|---|
| diag1.0+bbox1x(기존) | 72.3%±5.1% | 43.8%±27.5% | 47.6%±24.9% |
| diag2.0+bbox1x | 69.0%±5.2% | 80.0%±7.6% | 68.6%±6.1% |
| **diag1.0+bbox3x** | **84.6%±2.9%** | 88.1%±3.9% | 70.8%±4.0% |
| diag2.0+bbox3x | 80.2%±3.7% | 91.5%±4.0% | **77.8%±4.3%** |

bbox_scale=3x가 정확도 향상뿐 아니라 **학습 안정성(표준편차 27%p→4%p)을 크게 개선**.

**배포**: `bbox_scale`을 학습에만 적용하면 정규화 버그(§5)와 같은 종류의 학습/추론
불일치가 재발하므로, 서버 코드(`stage2_v2_inference_server.py`)에 `ckpt.get("bbox_scale",
1.0)`을 읽어 `_build_flat/seq/seq_trans_feature`에 일괄 적용하는 하위호환 지원을 먼저
추가(기본값 1.0, 기존 체크포인트 무영향). 이후 전체 150ep로 **window=3 + bbox_scale=3.0**
최종 체크포인트를 5-seed 학습(`scripts/train_exp71_window3_bboxscale_final.py`) —
val_acc 80.7%±4.3%(최고 84.4%, seed=4) 선정.

**전달 완료 (2026-07-07)**:
- 서버 코드: `monavla-driving` 커밋 `51adce65`(preview 옵션D 로깅) + `604f2661`(bbox_scale
  지원) — soda 서버에 `git pull`로 fast-forward 반영 확인
- 체크포인트: `runs/v5_nav/mlp/exp71_window3_bboxscale3/action_transformer.pt`(4.4MB)
  rsync로 `soda@100.85.118.58:~/MoNaVLA/`에 전송 완료
- **아직 안 한 것**: checkpoint_path 전환 + 서버 재시작(운영 중인 로봇 서비스라 soda
  확인 후 진행 필요) — 이게 되면 실로봇 window3+bbox_scale3 vs 기존 window6 A/B 가능

## 12. soda 제기 BGR/RGB 의심 건 — 실물 대조로 기각 (2026-07-08)

soda가 `mobile_vla_data_collector.py`의 `cv_bridge.compressed_imgmsg_to_cv2(...,"bgr8")`로
받은 배열이 JPEG 인코딩 없이 H5 raw로 저장되는데, 학습 로더(`nav_h5_dataset_impl.py`)는
이를 RGB로 가정하고 읽는다는 의심을 제기(241/241 에피소드 전부 raw 저장 확인).
색 채널을 반전(BGR→RGB)해서 vis_feat 캐시 재생성 + window6+bbox_scale3 재학습까지
진행했으나(5-seed 79.6%±3.7%, 원본 84.6%±2.9%와 큰 차이 없음), **사용자가 실제 촬영
공간을 직접 눈으로 확인한 결과 현재 로더(반전 없음) 쪽이 실물 색에 더 가까움**을 확인.

코드 추적(명시적 `"bgr8"` 요청, 중간 변환 없음)은 여전히 이론상 스왑 가능성을 가리키나,
실물 대조라는 더 강한 증거 앞에서 기각 — 카메라 원본 인코딩이 이미 bgr8이라 무변환
(no-op)이었을 가능성. **재학습 불필요, 어제 배포한 window6+bbox_scale3(원본) 그대로
유지.** 색 반전판 체크포인트(`exp71_window6_bboxscale3_colorfixed`)는 참고용으로만 보존.

## 13. FWD+L이 obj_right에서 반복된 이유 — 세 번째 confound (2026-07-09)

`bbox_dataset_pg448_cx.json`의 cx>0.6 프레임 gt_class 분포: `FORWARD:259, FWD+L:67,
RIGHT:29, ROT_L:20, FWD+R:19, ROT_R:3`. FWD+L(67개) 대부분이 `path_type="right_left"`
(목표=우측, 접근경로=좌곡선) 한 종류에 집중(cx>0.6 구간의 68%, 55/81건). 원인: 이
path_type은 바구니가 화면 오른쪽에 보여도 접근 경로 자체가 좌곡선이라 정답 라벨이
FWD+L로 기록됨 — 모델은 이 패턴을 정확히 학습했을 뿐. `CX_RULE_THRESHOLDS` 룰 오버라이드는
기본 꺼져있어(`VLA_CX_RULE=0`) 관여 안 함, 순수 학습된 헤드의 정직한 예측. ①극단cx
부족, ②텍스트-장면 confound에 이은 **세 번째 confound**: "화면상 위치(cx)"와
"경로 곡선 방향"이 path_type 설계 단계에서 뒤섞여 있어 cx만으로 목표 방향을 안정적으로
추론 불가.

## 14. 수집/학습 Hz 정합 + action chunking 검토 (2026-07-09)

실추론 레이턴시 450~600ms(~2Hz) vs 수집 4~7Hz(이벤트 트리거) — window(3/6프레임)가
프레임개수 기준이라 조작 밀도에 따라 체감 시간폭이 최대 3배 이상 들쭉날쭉, 추론 시
실제 시간폭과 불일치. 타 VLA 비교: RT-1/RT-2 ~1~3Hz, ViNT/NoMaD ~4Hz(우리 수집레이트와
근접), π0/SmolVLA는 action chunking으로 "느린 추론+빠른 실행"을 구조적으로 분리.
결론: 수집 Hz는 유지, 학습 window 구성 시 ~2Hz(500ms) 리샘플링으로 추론 cadence 정합
(값싼 절충), 근본적으로는 action chunking 도입을 별도 트랙으로 검토.

## 15. 수집 플랜 트랙 분리 — 도달성능(A) vs 언어조건화(B) (2026-07-09)

목표가 gray basket 도달뿐이라면 텍스트-장면 confound(②)는 실패 원인이 아님 — obj_*
실패의 직접 원인은 ①극단cx 부족과 ③cx-경로곡선 confound뿐. 수집 플랜을 트랙 A(핵심,
180ep, 극단cx 4곳×경로다양성, 지시문 불필요, 도달성능 직결)와 트랙 B(선택, 60ep,
지시-경로 디커플링, 언어조건화 로드맵용 후순위)로 분리. soda 쪽도 이미 같은 방향
(4-position 극단배치 수집 UI, `930a6180`/`984b0ae4`)으로 대시보드 구현 중.

## 16. 재수집 전 마지막 완화책 3종 실측 — 전부 무효 확인 (2026-07-10)

재수집 없이 현재 150ep로 시도 가능한 마지막 학습레벨 완화책을 window6+bbox_scale3
기준 5-seed로 비교(`scripts/train_exp71_confound_mitigation.py`):

| 설정 | val_acc | FWD+L recall | cx>0.75 acc(n=2) | cx<0.25 acc(n=10) |
|---|---|---|---|---|
| A. 현재 배포(baseline) | 76.0%±4.5% | 0.78 | 1.00 | 0.44 |
| B. confound reweight(right_left 모순라벨 downweight) | 75.2%±5.1% | 0.63↓ | 1.00 | 0.42 |
| C. hybrid cx-rule 오버레이(극단cx만 기하규칙 덮어쓰기) | 74.2%±4.4% | 0.78 | **0.00** | **0.00** |

B: FWD+L 과확신은 줄였지만(0.78→0.63) 전체 정확도만 깎이고 극단cx 개선 없음 — 그
구간엔 "맞는 방향" 대안 신호 자체가 없어 downweight해도 대체할 게 없음. C: 오히려
완전 악화(0%) — 서버 `CX_RULE_THRESHOLDS`가 가정하는 "cx>0.75→ROT_R"이 현재 데이터
라벨과 실제로 안 맞음(그런 기하학적 가정 없이 수집됐기 때문) — 극단cx 라벨 자체가
기하학적으로 일관되지 않다는 직접 증거. (cx 서브셋 n=2/n=10은 표본 작아 절대수치
신뢰 낮지만 방향성은 결론 뒷받침 충분)

**결론**: "재수집 없이는 안 된다"가 이론이 아니라 실측으로 재확인됨. 배포된
window6+bbox_scale3(baseline) 그대로 유지 — 어떤 학습레벨 트릭도 도움 안 됨.

## 17. window3+bbox_scale3 배포 후 obj_right 실측 — 병목이 confound에서 그라운딩으로 이동 (2026-07-10/11)

2026-07-10 soda 로봇에서 window3+bbox_scale3(val_acc 84.4%) 체크포인트로 obj_right
30회 실주행(`episode_log.csv` row 79~108, session 20260710_130942~160656) 결과:

| 구분 | 성공/전체 | 성공률 |
|---|---|---|
| 전체 | 17/30 | 56.7% |
| **OWL-v2 그라운딩 성공** (실제 bbox 검출) | **16/18** | **88.9%** |
| **그라운딩 실패** (폴백 cx=0.5, area=0.06) | 1/12 | 8.3% |

이전 세션들(§7~15, window6 checkpoint 기준)에서는 obj_right 성공률이 0%였고 실패
원인이 주로 FWD+L 반복(confound) 자체였다. 이번엔 그 confound가 사실상 해소된
채로(top액션이 거의 전부 FORWARD로 수렴, FWD+L 반복 재현 안 됨) **그라운딩이
성공한 케이스에서는 88.9%라는 매우 높은 성공률**이 나왔고, 실패는 대부분(13건 중
10건) "OWL-v2가 바구니를 못 찾아 중심좌표(0.5,0.5)로 폴백" 케이스였다. 나머지
소수 실패(2~3건)는 물리적 바퀴 드리프트 등 그라운딩과 무관한 별도 원인.

**의미**: `[[project_focus_grounding_for_direction]]`(실주행 핵심은 그라운딩/인식
개선이라는 메모리 가설)이 실측으로 뒷받침됨. 트랙A(극단cx) 재수집과 별개로,
OWL-v2 그라운딩 실패율 자체를 낮추는 것(임계값 튜닝, multi_prompt 폴백, 재시도
전략 등)이 지금 시점에서 성공률을 가장 빠르게 올릴 수 있는 지렛대로 보임.

세션 뷰어(`mona_dashboard.py` Session History 탭)에 이 분석을 세션별로 바로 볼 수
있도록 `episode_log.csv` 조인(실주행 결과/메모/FPE 배지)을 추가함 — 기존엔 H5
attrs 원본 텍스트만 노출되어 오퍼레이터가 남긴 성공/실패 메모가 묻혀 있었음.
