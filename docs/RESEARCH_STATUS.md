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
