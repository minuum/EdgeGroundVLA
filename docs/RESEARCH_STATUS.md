---
# ══ MoNaVLA Research Status — Single Source of Truth ══════════════════════
# 이 파일만 편집하면 README / docs/index.html / menemory 전부 자동 갱신됨
#   python3 scripts/utils/sync_research.py         # 실제 적용
#   python3 scripts/utils/sync_research.py --diff  # 변경사항 미리보기
#   python3 scripts/utils/sync_research.py --dry-run --validate
#
updated: "2026-06-16"

# ── SOTA 모델 ─────────────────────────────────────────────────────────────
sota_exp: "Exp66"
sota_label: "Ours (Exp66) ★"
sota_arch: "CLIP + L2-norm + aug"
sota_cl: "96.6%"
sota_fpe: "0.094 m"
sota_val_acc: "93.5%"
sota_ckpt_s1: "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
sota_ckpt_s2: "runs/v5_nav/mlp/exp66/action_mlp.pt"
sota_note: "SOTA · MLP w=4 · base PG2 grounding"

# ── 보조 결과 ─────────────────────────────────────────────────────────────
best_fpe: "0.080 m"
best_fpe_exp: "Exp66 LSTM"
best_fpe_note: "LSTM w=16"

pipeline_gap: "×9.4"
pipeline_gap_note: "vs Simple MLP (Exp65b) 10.3%"

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
hero_metric1_value: "96.6%"
hero_metric1_label: "Closed-Loop (SOTA)"
hero_metric1_note: "Exp66 · CLIP + L2-aug"
hero_metric1_color: "#16a34a"

hero_metric2_value: "0%"
hero_metric2_label: "E2E VLA Baseline"
hero_metric2_note: "Exp11 · Kosmos-2 LoRA"
hero_metric2_color: "#dc2626"

hero_metric3_value: "0.080 m"
hero_metric3_label: "Best FPE (↓)"
hero_metric3_note: "LSTM w=16 · MLP w=4: 0.094 m"
hero_metric3_color: "#2563eb"

hero_metric4_value: "×9.4"
hero_metric4_label: "Pipeline Gap"
hero_metric4_note: "vs simple MLP 10.3%"
hero_metric4_color: "#d97706"
---

<!-- ══ 이 아래는 사람이 읽기 위한 영역 (동기화에는 아래 섹션 블록이 사용됨) ══ -->

## 프로젝트 현황 요약

**현재 SOTA**: Exp66 — CLIP + L2-norm + aug, **CL 96.6%**, FPE 0.094 m  
**실험 단계**: Stage2 v2 완성, 실로봇 테스트 진행 중 (2026-06-16/17)  
**다음 목표**: 실로봇 96%+ 재현 → 논문 제출

---

<!-- BEGIN:results_table -->
## 핵심 결과 표

| Method | Architecture | CL ↑ | FPE ↓ | Note |
|---|---|---|---|---|
| E2E VLA (Exp11) | Kosmos-2 + LoRA | 0.0% | 1.454 m | Text attn 0%, structural failure |
| Decomp v1 (Exp14) | CLIP + BBox MLP | 66.7% | 0.555 m | First decomposition baseline |
| Simple MLP (Exp65b) | CLIP + plain MLP | 10.3% | — | No L2-norm, no aug → pipeline ablation |
| **Ours (Exp66) ★** | **CLIP + L2-norm + aug** | **96.6%** | **0.094 m** | SOTA · MLP w=4 |
| Ours (Exp66 LSTM) | CLIP + L2-norm + aug | 96.6% | 0.080 m | Best FPE · LSTM w=16 |
<!-- END:results_table -->

---

<!-- BEGIN:checkpoints -->
## 체크포인트

| 모델 | 경로 | 크기 |
|---|---|---|
| Stage 1 v2 (encoder) | `runs/v5_nav/mlp/shared/stage1_v2_projs.pt` | 3.1 MB |
| Stage 2 MLP w=4 ★ (Exp66) | `runs/v5_nav/mlp/exp66/action_mlp.pt` | 456 KB |
| Stage 2 LSTM w=16 (Exp66) | `runs/v5_nav/mlp/exp66/action_mlp_lstm.pt` | — |
<!-- END:checkpoints -->

---

<!-- BEGIN:key_findings -->
## 핵심 발견 (확정)

1. **Text attention = 0%** — Google-robot post-trained Kosmos-2의 구조적 사망. LoRA/head-only 모두 복구 불가. E2E 실패의 근본 원인.
2. **Pipeline이 유일 결정 변수** — L2-norm + bbox augmentation이 성능의 전부. Grounding 소스 무관 (HSV = base PG2 = LoRA = 96.6%).
3. **Basket localization 이중 증명** — Zero-shot probe 96.6% + masking 9/9 flip. CLIP encoder가 basket 픽셀을 독립적으로 인식.
4. **Image가 핵심, BBox는 보조** — image_only 75.6% vs bbox_only 67.4%. BBox cx/cy 정보는 보조적.
5. **Decomposition이 E2E 압도** — CL 96.6% vs 0%. FPE 0.094 vs 1.454m.
<!-- END:key_findings -->

---

<!-- BEGIN:experiment_history -->
## 실험 이력 (append-only)

| Exp | 날짜 | Architecture | CL% | FPE | val_acc | 특이사항 |
|-----|------|--------------|-----|-----|---------|---------|
| Exp11 | 2026-04-16 | Kosmos-2 LoRA (E2E) | 0% | 1.454 m | 58.6% PM | Text attn 0%, E2E baseline |
| Exp14 | 2026-04-20 | CLIP + BBox MLP (Decomp v1) | 66.7% | 0.555 m | 75.9% PM | First decomp |
| Exp25 | 2026-04-22 | Kosmos-2 balanced (E2E) | 55.6% | 0.382 m | 52.4% PM | Best E2E (retired) |
| Exp39 | 2026-05-02 | Exp25 + last-4 LoRA | — | — | 21.7% PM | FORWARD collapse |
| Exp40 | 2026-05-04 | Exp39 + grounding_aux | — | — | 0% (all FWD) | grounding 67% 생존, action head 붕괴 |
| Exp53 | 2026-05-15 | CLIP LoRA 16-24 frozen LM | — | — | — | 설계 후 grounding 개선용 |
| Exp54 | 2026-05-20 | Stage2 v2 FrozenCLIP + MLP | 96.6% | 0.094 m | 93.5% | Stage2 v2 첫 SOTA |
| Exp65b | 2026-06-08 | Stage2 v2 plain MLP (no L2) | 10.3% | — | — | Pipeline ablation control |
| Exp66 | 2026-06-10 | Stage2 v2 MLP w=4 + L2 + aug | **96.6%** | **0.094 m** | **93.5%** | **현재 SOTA (base PG2)** |
| Exp66-LSTM | 2026-06-10 | Stage2 v2 LSTM w=16 + L2 + aug | 96.6% | 0.080 m | — | Best FPE |
| Exp67 | 2026-06-11 | Stage2 v2 MLP w=4 (HSV cx) | 96.6% | — | — | grounding source irrelevant 확정 |
<!-- END:experiment_history -->

---

<!-- BEGIN:menemory_proposal -->
## [menemory 업데이트 제안] — `scripts/utils/sync_research.py --propose-menemory` 실행 시 출력

master_memory.md의 "현재 최선 모델" 섹션을 아래로 교체 권장:

```markdown
## 현재 최선 모델 (2026-06-16 기준)

### Decomposition SOTA (실로봇 배포 중)
- **Exp66** — Stage2 v2 FrozenCLIP + MLP + L2-norm + aug
  - closed-loop: **96.6%**, FPE: 0.094m, val_acc: 93.5%
  - ckpt S1: `runs/v5_nav/mlp/shared/stage1_v2_projs.pt`
  - ckpt S2: `runs/v5_nav/mlp/exp66/action_mlp.pt`
  - 추론 서버: `robovlm_nav/serve/stage2_v2_inference_server.py --port 8001`

### End-to-end (폐기)
- **Exp11** — Kosmos-2 + LoRA: CL 0%, text attn 구조적 사망 → E2E 경로 완전 종료

### 핵심 음성 결과
- Grounding source (HSV/PG2/LoRA) irrelevant: 모두 96.6% → LoRA grounding 연구의 action 기여 = 0
- Pipeline (L2+aug)이 유일 결정 변수
```
<!-- END:menemory_proposal -->
