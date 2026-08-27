---
# ══ EdgeGround-VLA Research Status — Single Source of Truth ═══════════════
# 이 파일만 편집하면 README / docs/index.html / menemory 전부 자동 갱신됨
#   python3 scripts/utils/sync_research.py         # 실제 적용
#   python3 scripts/utils/sync_research.py --diff  # 변경사항 미리보기
#   python3 scripts/utils/sync_research.py --dry-run --validate
#
# ⚠️ 2026-08-16 현행화: 2026-06-16(Exp66/CLIP 계열)에 멈춰 있던 내용을
#    현재 배포 구성(exp73 / OWL-v2 + Kosmos-2 vision + image_proj + MLP)으로 전면 갱신.
#    그 사이 index.html은 직접 수정되어 이 파일보다 최신이었음 — 이제 양쪽 정합됨.
#
updated: "2026-08-16"

# ── SOTA 모델 (현재 실기 배포 구성) ────────────────────────────────────────
sota_exp: "exp73"
sota_label: "Ours (exp73) ★"
sota_arch: "OWL-v2 + Kosmos-2 vision + image_proj + MLP"
sota_cl: "95.0%"
sota_fpe: "—"
sota_val_acc: "74.1%"
sota_ckpt_s1: "runs/v5_nav/mlp/stage1_v3_5cls/stage1_v3_5cls_owl_projs.pt"
sota_ckpt_s2: "runs/v5_nav/mlp/exp73_stage1v3/exp73_owl_stage1v3_v6_mlp.pt"
sota_note: "95/100 · 5 target positions × 20 trials · 2026-08-07"
sota_metric_label: "real-robot success"

# ── 보조 결과 ─────────────────────────────────────────────────────────────
best_fpe: "—"
best_fpe_exp: "—"
best_fpe_note: "exp73 계열은 FPE 대신 실기 성공률(95/100)로 평가"

pipeline_gap: "—"
pipeline_gap_note: "CLIP 계열(Exp65b/66) 지표는 아래 legacy 표 참조"

e2e_exp: "Exp11"
e2e_cl: "0%"
e2e_fpe: "1.454 m"
e2e_note: "Kosmos-2 + LoRA, text attn 0%"

decomp_v1_exp: "Exp14"
decomp_v1_cl: "66.7%"
decomp_v1_fpe: "0.555 m"
decomp_v1_note: "CLIP + BBox MLP, first decomp baseline"

ablation_simple_exp: "Exp65b"
ablation_simple_cl: "10.3%"
ablation_simple_note: "CLIP + plain MLP, no L2/aug"

# ── Hero 수치 (index.html Metrics Bar) ────────────────────────────────────
hero_metric1_value: "95%"
hero_metric1_label: "실기 성공률"
hero_metric1_note: "95/100 · 5위치 × 20회 (2026-08-07 재검증)"
hero_metric1_color: "#16a34a"

hero_metric2_value: "90.5%"
hero_metric2_label: "그라운딩 성공률"
hero_metric2_note: "985/1088 프레임 · 같은 100세션 실측"
hero_metric2_color: "#2563eb"

hero_metric3_value: "1902 ms"
hero_metric3_label: "검출 지연 (병목)"
hero_metric3_note: "OWL-v2 fp32 · 전체 지연의 97%"
hero_metric3_color: "#d97706"

hero_metric4_value: "0.46B"
hero_metric4_label: "실연산 파라미터"
hero_metric4_note: "학습 파라미터는 1.128M · 언어 디코더 미사용"
hero_metric4_color: "#7c3aed"
---

<!-- ══ 이 아래는 사람이 읽기 위한 영역 (동기화에는 아래 섹션 블록이 사용됨) ══ -->

## 프로젝트 현황 요약

**현재 배포 구성**: exp73 — OWL-v2(frozen) + Kosmos-2 vision_model(frozen) + image_proj + MLP action head, **실기 95/100(95.0%)**
**실험 단계**: 실기 100건 재검증 완료(2026-08-07), 논문(KCI) 초안 작성 중
**다음 목표**: 검출기 경량화(양자화/증류) — 지연의 97%가 OWL-v2

---

<!-- BEGIN:results_table -->
## 핵심 결과 표

| Method | Architecture | 실기 성공률 ↑ | val_acc | Note |
|---|---|---|---|---|
| E2E VLA (Exp11) | Kosmos-2 + LoRA | 0.0% (CL) | 58.6% PM | Text attn 0%, structural failure |
| Decomp v1 (Exp14) | CLIP + BBox MLP | 66.7% (CL) | 75.9% PM | First decomposition baseline |
| Simple MLP (Exp65b) | CLIP + plain MLP | 10.3% (CL) | — | No L2-norm, no aug → pipeline ablation |
| Ours (Exp66, legacy) | CLIP + L2-norm + aug | 96.6% (CL, 시뮬) | 93.5% | 구 SOTA · 시뮬 closed-loop 기준 |
| **Ours (exp73) ★** | **OWL-v2 + Kosmos-2 vision + image_proj + MLP** | **95.0% (실기 95/100)** | **74.1%** | 현재 배포 구성 |

> ⚠️ CL(시뮬 closed-loop)과 실기 성공률은 **측정 축이 다르므로 직접 비교 불가**.
> Exp66의 96.6%는 오프라인 시뮬 재생 기준, exp73의 95.0%는 실로봇 100회 주행 기준.
<!-- END:results_table -->

---

<!-- BEGIN:checkpoints -->
## 체크포인트

| 모델 | 경로 | 비고 |
|---|---|---|
| Stage 1 (image_proj, 5-class) ★ | `runs/v5_nav/mlp/stage1_v3_5cls/stage1_v3_5cls_owl_projs.pt` | val_acc 94.09% · 225ep · OWL-v2 라벨 |
| Stage 2 (MLP action head) ★ | `runs/v5_nav/mlp/exp73_stage1v3/exp73_owl_stage1v3_v6_mlp.pt` | val_acc 74.13% · window=6 · bbox_scale=3.0 |
| 비전 특징 캐시 | `docs/v5/closed_loop_eval/exp73_v6_vis_cache_stage1v3.pt` | 225ep 재인코딩 (Stage1 완료 후 생성) |
| Stage 1 v2 (legacy, CLIP) | `runs/v5_nav/mlp/shared/stage1_v2_projs.pt` | Exp66 계열 |
| Stage 2 MLP w=4 (legacy, Exp66) | `runs/v5_nav/mlp/exp66/action_mlp.pt` | Exp66 계열 |
<!-- END:checkpoints -->

---

<!-- BEGIN:key_findings -->
## 핵심 발견 (확정)

1. **Text attention = 0%** — Google-robot post-trained Kosmos-2의 구조적 사망. LoRA/head-only 모두 복구 불가. E2E 실패의 근본 원인.
2. **병목은 정책이 아니라 인지** — 세션 내 검출 성공률이 80% 이상이면 주행 성공률 98.8%(79/80), 미만이면 51.2%. (2026-07-31 159세션 배치 기준)
3. **지연의 97%가 검출기** — OWL-v2 그라운딩 1901.7ms(fp32) / 962.1ms(fp16) vs Kosmos-2 비전 53.7ms, MLP 헤드 <1ms. 경량화의 유일한 표적.
4. **학습 파라미터는 1.128M뿐** — image_proj 0.262M + MLP 헤드 0.866M. 두 백본(0.458B)은 전부 frozen.
5. **bbox_scale은 현재 파이프라인에서 무영향** — 1.0/2.0/3.0 재검증 결과 val acc 73.8/74.2/73.8%(3-seed, 오차범위 내 동일). 옛 exp71(PG2+Transformer)에서 보고된 72%→85% 효과는 현 구성에서 재현되지 않음.
6. **val 지표와 실기 성능은 직결되지 않음** — val 74.1%인 헤드가 실기 95/100 달성. 특히 ROT_L/ROT_R은 val 표본이 1.15%(28/2431)뿐이라 val 정확도가 낮게 나오지만, 실기에서는 6.71%(73/1088) 빈도로 쓰이면서도 성공률 유지.
<!-- END:key_findings -->

---

<!-- BEGIN:experiment_history -->
## 실험 이력 (append-only)

| Exp | 날짜 | Architecture | 성능 | val_acc | 특이사항 |
|-----|------|--------------|------|---------|---------|
| Exp11 | 2026-04-16 | Kosmos-2 LoRA (E2E) | CL 0% / FPE 1.454 m | 58.6% PM | Text attn 0%, E2E baseline |
| Exp14 | 2026-04-20 | CLIP + BBox MLP (Decomp v1) | CL 66.7% / FPE 0.555 m | 75.9% PM | First decomp |
| Exp25 | 2026-04-22 | Kosmos-2 balanced (E2E) | CL 55.6% / FPE 0.382 m | 52.4% PM | Best E2E (retired) |
| Exp39 | 2026-05-02 | Exp25 + last-4 LoRA | — | 21.7% PM | FORWARD collapse |
| Exp40 | 2026-05-04 | Exp39 + grounding_aux | — | 0% (all FWD) | grounding 67% 생존, action head 붕괴 |
| Exp53 | 2026-05-15 | CLIP LoRA 16-24 frozen LM | — | — | 설계 후 grounding 개선용 |
| Exp54 | 2026-05-20 | Stage2 v2 FrozenCLIP + MLP | CL 96.6% / FPE 0.094 m | 93.5% | Stage2 v2 첫 SOTA |
| Exp65b | 2026-06-08 | Stage2 v2 plain MLP (no L2) | CL 10.3% | — | Pipeline ablation control |
| Exp66 | 2026-06-10 | Stage2 v2 MLP w=4 + L2 + aug | CL 96.6% / FPE 0.094 m | 93.5% | 구 SOTA (base PG2) |
| Exp66-LSTM | 2026-06-10 | Stage2 v2 LSTM w=16 + L2 + aug | CL 96.6% / FPE 0.080 m | — | Best FPE |
| Exp67 | 2026-06-11 | Stage2 v2 MLP w=4 (HSV cx) | CL 96.6% | — | grounding source irrelevant 확정 |
| exp71 | 2026-07-07 | window6 + bbox_scale 3.0 (PG2, Transformer) | — | 84.6%±2.9%p | bbox_scale 효과 최초 보고(현 구성에선 재현 안 됨) |
| **exp73** | **2026-08-07** | **OWL-v2 + Kosmos-2 vision + image_proj + MLP** | **실기 95/100 (95.0%)** | **74.13%** | **현재 배포 구성** |
| exp74 | 2026-08-19 | OWL-v2 + Florence-2 vision(대체) + image_proj + MLP | 실기 미검증 (판정 재검토 중) | 75.24% (3-seed 75.15%±0.09%p) | val은 exp73 대비 +1.29%p 우수. Jetson 지연 fp16 167.2ms(+113ms/frame)는 확인됐으나, 실제 배포 cadence(~1.3Hz, 그라운딩이 97% 지배) 대비로 재계산하면 저하폭은 15~20% 수준 — 상세: 아래 §Florence-2 백본 검정 |
<!-- END:experiment_history -->

---

## 실기 재검증 상세 (2026-08-07, 100세션)

**목표 위치별 성공률** — 전체 95/100(95.0%), 평균 11.64스텝

| 목표 위치 | 성공률 | 성공(회) | 평균 스텝 |
|---|---|---|---|
| 중앙 | 100% | 20/20 | 10.00 |
| 강좌 | 95% | 19/20 | 13.35 |
| 약좌 | 95% | 19/20 | 11.55 |
| 강우 | 95% | 19/20 | 12.75 |
| 약우 | 90% | 18/20 | 10.55 |

**세션 원본 실측 집계** (`inference_sessions_recv/20260807/h5/` 100개, 총 1,088개 결정)

- 그라운딩 성공 프레임: **985/1088 = 90.5%**
- 세션별 검출률: 평균 88.7%, 중앙값 100.0% (gnd%≥80 세션 82/100, gnd%=100 세션 59/100)
- 그라운딩 캐시 재사용: 553/1088 = 50.8% (`grounding_skip_n=3`)
- OWL-v2 threshold: 100/100 세션 모두 `0.2` 확인
- 액션 분포: STOP 85(7.81%) · FORWARD 893(82.08%) · LEFT 0 · RIGHT 12(1.10%) · FWD+L 15(1.38%) · FWD+R 10(0.92%) · ROT_L 35(3.22%) · ROT_R 38(3.49%)

> ⚠️ **배치 구분 주의**: 위 100세션(2026-08-07)에는 세션별 성공/실패 라벨이 H5에 없어
> (`status`가 전량 `manual_stop`), "gnd%≥80 → 98.8%" 상관관계는 **이 배치로 재계산 불가**.
> 해당 지표는 별도의 **2026-07-31 159세션 배치(79/80)** 기준이며, 성공률 89/100도 그 배치 것이다.
> 두 배치를 같은 근거로 섞어 인용하지 말 것.

---

## Florence-2 백본 검정 (2026-08-16 ~ 2026-08-19, 진행 중 — 지연 판정 정정됨)

교수님 제안(Florence-2)에 따라 Kosmos-2 vision_model → Florence-2-base로 백본만 교체하는
검정을 진행 중.

| 단계 | 지표 | Kosmos-2(기존) | Florence-2(신규) | 판정 |
|---|---|---|---|---|
| 그라운딩 cx MAE (GB10) | `docs/v5/detector/florence2_backbone.json` | 0.0020 | 0.00152 (-24%) | ✅ 통과 |
| Stage1 5-class val_acc | `docs/v5/detector/stage1_florence2_5cls.json` | 94.09% | 94.92% (+0.83%p) | ✅ 통과 |
| Stage2 exp74 val_acc (3-seed) | `docs/v5/closed_loop_eval/exp74_florence2_stage2.json` | 73.87%±0.20%p | 75.15%±0.09%p (+1.29%p) | ✅ 통과 (단 RIGHT 클래스 -8.5%p 회귀) |
| Jetson Orin NX 절대 지연 (fp16) | `docs/v5/detector/florence2_backbone_jetson_latency.json` | 53.7ms | 167.2ms (+113ms/frame) | 사실 확인됨 (더 느림) |
| **실 배포 cadence 영향 (재계산)** | 평균 프레임당 지연(그라운딩 캐시 재사용률 50.8% 반영) | ~528ms (~1.3Hz 실측과 근사) | ~641ms 추정 (~1.1Hz) | ⚠️ **저하 15~20%, "10Hz 예산 초과"는 오판정 — 재검토 중** |

- **정정 이력**: 처음엔 "fp16 167.2ms가 10Hz(100ms) 예산을 초과해 구조적으로 불가능"으로
  판정했으나, 이는 실제 배포 시스템의 진짜 cadence를 확인하지 않은 오류였다. 이 문서
  §핵심 발견 3번에 이미 있듯 **지연의 97%는 OWL-v2 그라운딩**(fp16 962.1ms)이고 비전
  백본은 3%뿐이라, 실제 cadence는 애초에 10Hz가 아니라 **~1.3Hz**(그라운딩 캐시
  50.8% 재사용 포함 평균)다. 이 기준으로 재계산하면 Florence-2 교체 시 cadence는
  ~1.3Hz→~1.1Hz로 15~20% 저하되는 정도이지, "10배 이상 예산 초과"는 아니다.
- 이 cadence 저하가 실기 성공률에 실제로 영향을 주는지는 val 지표로 판정 불가(핵심
  발견 6번과 동일 원칙) — **소규모 실기 A/B 검증이 필요**하며, 아직 채택/기각 어느
  쪽도 확정되지 않았다. soda에게 이 재계산 크로스체크 요청함.
- **검출기 재현율 — 실 배포 배치(2026-08-19)로 재확인, 이전 추정보다 더 나쁨.**
  `docs/v5/detector/florence2_grounding_0807_fullbatch.json` — OWL-v2와 **완전히 같은
  2026-08-07 100세션**에서 직접 측정. 원래 헤드라인 985/1088=90.5%는 `actions`/`bbox`
  배열(결정 단위) 기준이며 그대로 정확하다. 다만 세션 1개(`session_20260807_045839.h5`)는
  마지막 결정에 대응하는 카메라 프레임이 저장되지 않아(actions 16개 vs images 15개,
  그 마지막 결정도 OWL 성공이었음) **이미지가 있어야 하는 Florence-2 비교에서는
  1087프레임만 평가 가능**(OWL 성공 984/1087=90.5%, 비율은 동일). 아래 수치는 이
  1087프레임 기준:

  | 방식 | 커버리지(뭐라도 고른 비율) | OWL 정답 대비 일치율 | **실질 재현율(OWL 성공 984 기준)** |
  |---|---|---|---|
  | `<OD>` | 23.3% (253/1087) | 75.5% | **19.4%** |
  | `<DENSE_REGION_CAPTION>` | 37.5% (408/1087) | 68.4% | **28.4%** |

  이전에 학습용 val 주석(n=80, 큐레이션된 에피소드)에서 잰 52.5%는 **실제 배포
  환경(다양한 각도·거리·조명의 실기 세션)보다 낙관적인 추정치**였음이 확인됨 —
  실제로는 19.4~28.4%로 OWL-v2(90.5%) 대비 격차가 60%p 이상.

- **재현율 개선 시도(beam5 + 키워드 확장 + 합집합, 2026-08-19)** —
  `docs/v5/detector/florence2_grounding_0807_variants.json`, 같은 1087프레임:

  | 방식 | 재현율 |
  |---|---|
  | OD(beam3) | 19.4% |
  | DENSE(beam3) | 28.4% |
  | OD(beam5) | 20.7% (+1.3p) |
  | DENSE(beam5) | 31.9% (+3.5p) |
  | OD∪DENSE(beam3) | 30.2% |
  | **OD∪DENSE(beam5) — 최선** | **34.7%** |
  | 키워드 목록 확장(laundry/garbage/bucket 등 추가) | **효과 없음(모든 조합 0.0%p)** |

  키워드 확장이 전혀 효과가 없었다는 건 병목이 "매칭 규칙"이 아니라 **Florence-2
  자체가 해당 프레임에서 물체를 아예 인식/서술하지 못한다**는 뜻 — 단, 이건
  **"열린 질문"(뭐가 있는지 다 말해봐) 방식으로 물었을 때만 그렇다는 게 바로
  아래에서 뒤집힌다.**

- **🔄 판정 뒤집힘 — 명시적 phrase 지정 `<CAPTION_TO_PHRASE_GROUNDING>` 재현율 84.96%
  (2026-08-20)**. `docs/v5/detector/florence2_phrase_grounding_0807.json`, 같은
  100세션 1087프레임. 지금까지 전부 "열린 질문"(`<OD>`/`<DENSE_REGION_CAPTION>` —
  "이 방에 뭐가 있는지 다 말해봐")으로만 테스트했는데, **OWL-v2처럼 타겟 문구를
  직접 지정**(`<CAPTION_TO_PHRASE_GROUNDING>` + phrase="gray basket")하는 방식은
  한 번도 안 해봤었다:

  | 지표 | 값 |
  |---|---|
  | 재현율(OWL 정답 대비) | **84.96%** (OWL 90.5% 대비 -5.5%p) |
  | cx MAE | 0.0228 |
  | cx median AE | **0.0021** (대부분 거의 완벽히 일치, 평균은 소수 큰 오차에 끌림) |
  | coverage | 100% (거부 모드 없음) |

  이전 시도들(19.4~34.7%)과는 격이 다른 수치 — **OWL-v2와 5.5%p 차이까지 좁혀짐**.
  다만 **coverage 100%가 owl_success=0인 103프레임까지 포함한 값**이라는 점이
  핵심 제약: 타겟이 화면에 없어도 Florence-2는 "없다"고 답 못 하고 무조건 어딘가를
  짚는다(OWL-v2는 threshold 기반으로 거부 가능). 즉 **정밀도(오탐률)를 아직
  측정 안 했고, 실전 배치 시 이게 얼마나 문제될지는 별도 검증 필요** — 지금까지의
  "검출기 교체 기각" 판정을 최종 확정 짓기 전에 이 경로를 더 파봐야 한다.
  스크립트: `scripts/florence2_phrase_grounding_test.py`. 인터랙티브 비교 도구
  (`scripts/label/serve_florence2_owl_compare.py`, localhost:7795 `/live`)의
  `<CAPTION_TO_PHRASE_GROUNDING>` 옵션에서 직접 테스트 가능.

- **그라운더 스왑 Stage2 재학습(2026-08-19)** — exp73과 완전히 동일한 조건
  (Kosmos-2 vision, MLP 헤드, window=6, bbox_scale=3.0, 3-seed)에서 **bbox 주석만
  OWL-v2→Florence-2(DENSE 우선+OD 폴백, V6 전체 225ep 재주석)로 교체**:

  | | OWL bbox(exp73) | Florence-2 bbox | 차이 |
  |---|---|---|---|
  | val_acc mean | 73.87%±0.20%p | 73.26%±0.29%p | -0.61%p |
  | val_acc best | 74.13% | 73.59% | -0.54%p |

  전체 평균은 소폭 하락에 그쳤지만 클래스별로는 방향성 회귀가 뚜렷함:
  **L 66.7%→52.8%(-13.8%p), ROT_L 50.0%→33.3%(-16.7%p)** — 나머지 클래스는
  ±5%p 이내(FL/ROT_R은 오히려 소폭 상승). Florence-2의 낮은 재현율이 특정 방향
  프레임에 편중되어 나타난 것으로 보이며, 실기에서 이미 취약한 좌측 방향
  (CH66 좌우 비대칭)과 겹쳐 주의가 필요.
  체크포인트: `runs/v5_nav/mlp/exp73_stage1v3/exp73_florence2_grounder_v6_mlp.pt`,
  주석: `docs/v5/bbox_nav_florence2/bbox_dataset_v6_florence2.json`.

- **exp75 — 완전 통합(그라운더+비전인코더+프로젝션 전부 Florence-2, 2026-08-19)**.
  `docs/v5/closed_loop_eval/exp75_florence2_full_stage2.json` — 지금까지 각각 따로 검증한
  두 변경(exp74 비전 단독, 그라운더 스왑 단독)을 합친 버전:

  | 실험 | 그라운더 | 비전인코더 | val_acc mean | val_acc best |
  |---|---|---|---|---|
  | exp73(베이스라인) | OWL-v2 | Kosmos-2 | 73.87%±0.20%p | 74.13% |
  | exp74(비전만) | OWL-v2 | Florence-2 | 75.15%±0.09%p | 75.24% |
  | 그라운더 스왑(bbox만) | Florence-2 | Kosmos-2 | 73.26%±0.29%p | 73.59% |
  | **exp75(완전 통합)** | Florence-2 | Florence-2 | **73.52%±0.25%p** | **73.84%** |

  베이스라인 대비 -0.35%p로 거의 동일. 클래스별로 보면 **ROT_L 33.3%→66.7%,
  ROT_R 36.4%→68.2%로 회복**(비전 교체의 회전 클래스 개선 효과가 그라운더를
  같이 바꿔도 유지됨)했지만, **L은 52.8%→53.7%로 거의 회복 안 됨**(그라운더 교체로
  인한 손상은 비전 교체로 상쇄되지 않음) — 두 변경이 서로 다른 메커니즘으로
  클래스별 성능에 영향을 준다는 뜻. 텍스트 프로젝션/인코더는 세 실험 모두 Kosmos-2
  그대로(배포 시 미사용 컴포넌트, 요인 통제 목적으로 안 건드림).
  실제 이미지 성공/실패 케이스 갤러리: `docs/v5/ch64_figs/fig_florence2_case_gallery.png`
  (`docs/v5/research_story.html#ch67` 67-7 카드).

- **🏆 exp76/exp77 — phrase 그라운딩 방식(CH69)으로 재학습, 역대 최고 성적 (2026-08-21)**.
  CH69에서 발견한 명시적 phrase 그라운딩(`<CAPTION_TO_PHRASE_GROUNDING>`+"gray basket",
  0807 재현율 84.96%·V6 사람검증 100%)으로 V6 전체를 재주석(`gen_v6_florence2_phrase_annotation.py`,
  라이브 샘플 5752/5752=100% 검출)하고 그라운더 스왑·완전 통합을 다시 학습:

  | 실험 | 그라운더 | 비전인코더 | val_acc mean | best |
  |---|---|---|---|---|
  | exp73(베이스라인) | OWL-v2 | Kosmos-2 | 73.87%±0.20%p | 74.13% |
  | exp74(비전만) | OWL-v2 | Florence-2 | 75.15%±0.09%p | 75.24% |
  | 그라운더 스왑(구, 열린질문) | Florence-2(열린질문) | Kosmos-2 | 73.26%±0.29%p | 73.59% |
  | exp75(구 완전통합, 열린질문) | Florence-2(열린질문) | Florence-2 | 73.52%±0.25%p | 73.84% |
  | exp76(신 그라운더 스왑, phrase) | Florence-2(phrase) | Kosmos-2 | 73.76%±0.25%p | 74.04% |
  | **exp77(신 완전통합, phrase) ★역대 최고** | Florence-2(phrase) | Florence-2 | **75.58%±0.07%p** | **75.65%** |

  exp77 클래스별(vs exp73): STOP +3.4p, F +0.4p, **L +8.9p**(66.7%→75.6%, 구방식의
  -13.8p 회귀 완전 해소), R -4.6p(유일한 하락), FL +0.9p, FR +0.8p, ROT_L +16.7p,
  **ROT_R +54.6p**(31.8%→86.4%). exp76(그라운더만 교체)도 베이스라인과 거의 동일
  (73.76% vs 73.87%)해서, 구 그라운더 스왑의 손상(73.26%)이 순수히 그라운딩 품질
  문제였음이 재확인됨. 분산도 exp75(±0.25%p)보다 안정적(±0.07%p).
  ⚠️ 확정 발견 6번 그대로 유효 — 전부 오프라인 val 지표이며 실기 검증은 미실시.
  체크포인트: `runs/v5_nav/mlp/exp77_florence2_phrase_full/exp77_florence2_phrase_full_v6_mlp.pt`.
  상세: `docs/v5/research_story.html#ch69` 69-5 카드.

- **⚠️ 실기 전 오프라인 보강 검증 3종 (2026-08-21) — leave-one-direction-out에서 낙관 편향 발견**.
  가장 중요한 결과: 목표(direction) 하나를 통째로 학습에서 빼고 재학습하면
  (`scripts/eval_leave_one_direction_out.py`) 무작위 split val_acc(75.65%)가
  **평균 54.0%로 -21.65%p 낙관 편향**돼 있었음이 드러남. 좌/우 비대칭도 뚜렷:
  약좌 68.5%·강좌 60.5% vs **약우 41.7%·강우 33.3%** — 오른쪽 방향을 처음 보면
  성능이 반토막. CH66 좌우 비대칭이 이전 생각보다 훨씬 심각할 수 있음을 시사.
  나머지 2종은 방향 확인용: 궤적 재생 근사(exp71/72와 동일 방법론, `eval_exp77_closed_loop_sim.py`)
  — exp73 success 24.2%→exp77 30.3%(방향 일치, 절대값은 실기와 거리 있음);
  bbox_scale 재검증(phrase 그라운더 기준, `eval_bbox_scale_phrase_grounder.py`)
  — 1.0=75.62%/2.0=75.55%/3.0=75.58%, 여전히 무영향 재확인.
  confusion matrix 재분석 결과 원인도 특정됨 — 오른쪽 방향 제외 시 진짜 F(전진)
  프레임이 FR(전진+우회전)로 오판되는 비율이 폭증(center 11.6% vs weak_right 53.3%
  vs strong_right 79.9%) — 폐루프나 좌우 라벨 정의 문제가 아니라 해당 방향의
  "아직 전진할 때 vs 이제 꺾을 때" 학습 커버리지 부족.
  상세: `docs/v5/research_story.html#ch69` 69-6 카드.

- **🔴 exp73(배포중) apples-to-apples 비교 — exp77이 일반화에서는 오히려 진다 (2026-08-21)**.
  위 leave-one-direction-out을 exp73에도 동일하게 실행(`eval_leave_one_direction_out_exp73.py`):

  | held-out | exp73 | exp77 | exp73 R | exp77 R |
  |---|---|---|---|---|
  | center | 61.5% | 66.1% | 2.0% | 33.0% |
  | weak_left | 64.6% | 68.5% | 12.5% | 50.0% |
  | weak_right | **62.4%** | 41.7% | 46.3% | 41.4% |
  | strong_left | 58.7% | 60.5% | 4.3% | — |
  | strong_right | **57.1%** | 33.3% | 15.9% | 20.4% |
  | **평균** | **60.85%** | **54.00%** | ~16.2% | ~36.2% |

  무작위 split에서는 exp77(75.65%)이 exp73(74.13%)을 이겼지만, **완전히 처음 보는
  방향 조건에서는 exp73(60.85%)이 exp77(54.00%)보다 낫다** — 특히 오른쪽 방향에서
  exp77이 크게 뒤집힘. 단 R클래스만 보면 정반대(exp77 일반화가 exp73보다 뚜렷이
  나음) — exp77은 R 개념 자체는 더 견고히 배웠지만 F↔FR 경계에서 새 방향에 유독
  취약해 전체 점수가 깎이는 구조.
  **판단: 현재 증거로는 exp77을 exp73 대신 그대로 배포하자고 권하기 어렵다.**
  폐기할 이유도 없음 — R 일반화 개선은 실재하고 문제가 국소적(F↔FR, 우측)이라
  원인이 이미 밝혀져 있음. 다음 후보: 우측 방향 데이터 증강, 또는 exp73/exp77
  앙상블. 상세: `docs/v5/research_story.html#ch69` 69-7 카드.

- **🏆 V6 사람 검증 최종 재검증(329개, 클릭 기반 독립 GT) — Florence-2 100%, OWLv2(가중) 88.5% (2026-08-26)**.
  초기 97개 표본(2026-08-21)을 succ 스팟체크 확대 + fail 칸 다중 프레임으로 288개로
  늘리고, OWLv2/Florence-2 어느 좌표도 참고 안 하는 클릭 기반 독립 GT(`true_cx`)를
  추가해 순환논리를 차단했다. 이전 SAMPLE 재구성 과정에서 화면에 안 보이던 41개
  기라벨 프레임(과거 표본 잔재)까지 표본 끝에 편입해 최종 329개 전수를 "가운데
  정렬, 가장자리 포함만으로는 불일치"라는 더 엄격한 기준으로 재검토:
  **Florence-2 329/329(100%, 가중 100.0%) vs OWLv2 126/329(38.3%, 가중 88.5%)**.
  69-4의 초기 가중치(92.8%)보다 소폭 낮아진 건 표본 확대(288→329)와 엄격 기준
  적용 때문 — Florence-2 압도 우위 결론은 그대로 유지.
  **cx 정확도와 액션헤드 성능은 별개 축**: 그라운딩 좌표 정확도(34.7%→100%)는
  검출기 자체의 문제이고, Stage2 val_acc 개선폭이 작은 이유는 feature ablation에서
  이미 확인된 대로 MLP 헤드가 vis(이미지 임베딩)를 주 신호로, bbox(cx)를 보조
  신호로만 쓰기 때문(bbox_only 67.4%±9.8% / image_only 75.6%±0.8% /
  bbox+image 76.7%±1.3%) — 그라운딩 개선은 폐루프 실기·백본 교체 근거로는
  유효하지만 val_acc 상승분 자체를 크게 설명하지는 못한다.
  도구: `scripts/label/serve_v6_phrase_grounding_verify.py`(:7796). 라벨:
  `docs/v5/detector/v6_phrase_grounding_human_labels.json`(329개). 상세:
  `docs/v5/research_story.html#ch69` 69-11 카드.

- **🟠 cx 강조 헤드 3종 신규 실험(CH70) — val_acc는 올랐지만 R클래스는 나빠졌다 (2026-08-26)**.
  CH69에서 concat 방식(cxgeom·hybrid·bbox_scale)은 이미 다 해봤고 효과가 미미했음을
  확인 후, 곱셈적 결합(FiLM)·시간변화율(Δcx)·보조손실(cxaux) 3종을 새로 구현해
  exp77 캐시로 mlp 베이스라인부터 apples-to-apples 비교:

  | 헤드 | val_acc mean | best | R클래스 | FR클래스 |
  |---|---|---|---|---|
  | mlp(기존) | 75.58%±0.07%p | 75.65% | 66.2% | 73.6% |
  | film | 75.87%±0.26%p | 76.14% | **50.8%**(-15.4p) | 72.8% |
  | **deltacx ★최고** | **76.25%±0.14%p** | **76.39%** | 60.8%(-5.4p) | 76.7%(+3.1p) |
  | cxaux | 75.52%±0.49%p | 76.06% | 61.5%(-4.7p) | 76.5%(+2.9p) |

  deltacx가 val_acc·FR클래스 둘 다 최고지만 **R클래스가 5.4%p 하락** — F→FR 오분류는
  줄었지만 R→FR 오분류가 늘었을 가능성.

  **후속 검증(2026-08-26~27) — deltacx leave-one-direction-out 재검증했더니 무작위
  split 개선(+0.67%p)이 완전히 소멸**(LOO 평균 mlp 54.00% vs deltacx 53.83%,
  69-7 패턴 재현). 이어서 actionquery(경량 cross-attention, self-attn 없음)까지
  포함해 6개 헤드 구조를 전부 시도했지만 **LOO 일반화를 실제로 개선시킨 구조는
  하나도 없었다** — 헤드 구조 축 자체가 한계에 도달했다는 결론.

  대신 **손실함수를 손댄 ordinal soft label(D)이 CH70에서 유일하게 성과를 냄**:
  `mona_dashboard.py`의 THRESHOLD=0.50 하드컷을 sigmoid로 완화한 소프트 8-class
  타겟 + soft CE로 학습(`soft_class_targets()`), epoch도 300→200+조기종료로 축소
  (학습곡선 분석에서 150~220이면 val 수렴, 뒤는 과적합만 심화됨을 확인).

  | 헤드 | LOO 평균 hard | soft(D) | Δ |
  |---|---|---|---|
  | mlp | 50.63% | **54.62%** | **+3.99%p** |
  | deltacx | 50.99% | **54.71%** | **+3.71%p** |

  특히 최악 방향 strong_right는 **+14.52%p**(32.68%→47.20%), R클래스도 23.9%→35.1%로
  동시 개선 — 지금까지 반복된 "전체 개선 vs R클래스 희생" 트레이드오프가 이 방향에서는
  안 나타남. 부수적으로 **honest checkpoint selection 검증에서 val 표본이 작은
  leave-one-direction-out 조건은 체크포인트 선택 자체가 최대 +6.64%p 낙관편향을
  만든다는 것도 확인**(이후 모든 실험에 소급 적용 필요). 아직 실기 검증 전 —
  다음은 D+actionquery 결합, 궤적 재생 근사 재평가, 소규모 실기 A/B 순서.
  상세: `docs/v5/research_story.html#ch70`(70-3~70-7 카드). 계획:
  `docs/plans/plan_20260826_cx_emphasis_head.md`.

- fp16+TensorRT 변환 등 추가 경량화는 별건(`plan_20260801_specialized_detector.md`).
- 상세: `docs/plans/plan_20260816_stt_florence2_flow.md` (§2, §6 2''단계), `docs/DATASET_V6_STATUS.md` (2026-08-19 항목), `docs/v5/research_story.html#ch69`(2026-08-21 phrase 그라운딩 대발견).

---

## 데이터셋 (V6)

| 항목 | 값 |
|---|---|
| 에피소드 | 225개 (트랙A 180 + 트랙F 45) |
| 총 프레임 | 16,599 |
| train / val | 192 / 33 (VAL_RATIO=0.15, SPLIT_SEED=42) |
| bbox 주석 | `docs/v5/bbox_nav_owl/bbox_dataset_v6_owl.json` (OWL-v2 기반) |
| 액션 분포 | STOP 8.75% · FORWARD 46.03% · LEFT 4.31% · RIGHT 5.17% · FWD+L 15.25% · FWD+R 18.78% · ROT_L 0.82% · ROT_R 0.89% |

---

<!-- BEGIN:menemory_proposal -->
## [menemory 업데이트 제안] — `scripts/utils/sync_research.py --propose-menemory` 실행 시 출력

master_memory.md의 "현재 최선 모델" 섹션을 아래로 교체 권장:

```markdown
## 현재 최선 모델 (2026-08-16 기준)

### 배포 구성 (실로봇 운영 중)
- **exp73** — OWL-v2(frozen) + Kosmos-2 vision_model(frozen) + image_proj + MLP action head
  - 실기: **95/100(95.0%)**, 평균 11.64스텝 (2026-08-07 재검증)
  - val_acc: 74.13% (stride=1 채점, n=2431)
  - ckpt S1: `runs/v5_nav/mlp/stage1_v3_5cls/stage1_v3_5cls_owl_projs.pt` (val 94.09%)
  - ckpt S2: `runs/v5_nav/mlp/exp73_stage1v3/exp73_owl_stage1v3_v6_mlp.pt`
  - 규격: window=6, bbox_scale=3.0, 8-class, OWL threshold 0.20(운영)/0.25(코드 기본)
  - 추론 서버: `robovlm_nav/serve/stage2_v2_inference_server.py --port 8001`

### End-to-end (폐기)
- **Exp11** — Kosmos-2 + LoRA: CL 0%, text attn 구조적 사망 → E2E 경로 완전 종료

### 핵심 음성 결과
- bbox_scale은 현 구성에서 무영향(1.0/2.0/3.0 → 73.8/74.2/73.8%, 3-seed)
- val 지표와 실기 성능 직결 안 됨 (val 74.1% 헤드가 실기 95/100)
- 지연의 97%가 OWL-v2 검출기 → 경량화의 유일 표적
```
<!-- END:menemory_proposal -->
