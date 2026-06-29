# Master Core Memory

이 파일은 Menemory 프롬프트에 항상 포함되는 핵심 메모리입니다.

**마지막 업데이트: 2026-06-17**

---

## 프로젝트 장기 목표

교수님 테스트 프로토콜 3단계 완전 통과:
- Step 1: 곡선만 학습 → 직선 이미지를 줘도 곡선으로 가는가? ✅ (Exp11, PM 58.6%)
- Step 2: 50/50 비율 → 동작하는가? ❌ (Exp16 완전 collapse, Exp25 55.6% CL로 우회)
- Step 3: 33/33/33 (left/straight/right) 전방향 자율 내비게이션 ⬜
- 최종: 실로봇에서 자율 경로 추종 (실패 시 TICVLA / MobilityVLA 대안 검토)

---

## 아키텍처 원칙

- Backbone: Kosmos-2 (frozen) + LoRA — `third_party/RoboVLMs/` 절대 수정 금지
- Google-robot pretrained backbone: `kosmos_ph_google-robot-post-train.pt`
- Pure HF Kosmos-2: `.vlms/kosmos-2-patch14-224` — text generation 정상, grounding 가능
- 액션 공간: V5 **8-class** discrete (STOP/FWD/LEFT/RIGHT/FWD+L/FWD+R/ROT_L/ROT_R)
- 데이터: `ROS_action/mobile_vla_dataset_v5/` — 150개 H5 에피소드
  - straight 3종 × 20 = 60개 / non-straight 6종 × 15 = 90개
- V4 `basket_dataset_v2/` (528 ep)는 현재 학습 미사용

---

## 현재 최선 모델 (2026-06-17 기준)

### Decomposition SOTA ★ (실로봇 배포 중)
- **Exp66** — Stage2 v2 FrozenCLIP + MLP + L2-norm + bbox aug
  - closed-loop: **96.6%**, FPE: **0.094 m**, val_acc: 93.5%
  - ckpt S1: `runs/v5_nav/mlp/shared/stage1_v2_projs.pt` (3.1 MB)
  - ckpt S2: `runs/v5_nav/mlp/exp66/action_mlp.pt` (456 KB)
  - 추론 서버: `robovlm_nav/serve/stage2_v2_inference_server.py --port 8001`
  - 대시보드: `scripts/gradio_inference_dashboard.py` (포트 7865)

- **Exp66 LSTM w=16** — 동일 파이프라인, LSTM head
  - closed-loop: 96.6%, FPE: **0.080 m** (최저 FPE)

### End-to-end (폐기 — 연구 기록용)
- **Exp11** — Kosmos-2 + LoRA: CL **0%**, FPE 1.454m
  - text attention 구조적 사망 (Google-robot backbone 기인) → E2E 경로 완전 종료
- **Exp25** — Pure HF Kosmos-2 balanced: CL 55.6% — 이전 practical baseline, 현재 폐기

### 로봇 서버 현재 배포 (soda@100.85.118.58)
- Primary: **Exp66** Stage2 v2 (포트 8001)
- 실행: `mona-up` (go.sh — 서버+대시보드+허브 동시 시작)
- API: `robovlm_nav/serve/stage2_v2_inference_server.py`

---

## 핵심 발견 (확정된 사실)

1. **Text attention = 0%**: Google-robot post-training이 text 경로 완전 붕괴
   - Pure HF Kosmos-2: text 22.7% / image 77.3% (정상)
   - Google-robot 학습 후: text 0.000% / image 91.7% (붕괴)
   - LoRA/head-only 모두 복구 불가 — backbone 기인
   - 측정: `scripts/measure_attention.py`

2. **Image가 핵심, BBox는 보조**
   - bbox_only: 67.4%±9.8% / image_only: 75.6%±0.8% / bbox+image: 76.7%±1.3%
   - Pure Kosmos-2 grounding의 cx,cy,area < raw 16×16 image 정보량

3. **Pipeline이 유일 결정 변수** — L2-norm + bbox augmentation
   - Simple MLP(Exp65b) 10.3% → L2+aug(Exp66) 96.6% (×9.4)
   - Grounding 소스 무관: HSV = base PG2 = LoRA cx → 모두 96.6%

4. **Basket localization 이중 증명**
   - Zero-shot linear probe: 96.6% (frozen CLIP, 학습 없이)
   - Masking ablation: 9/9 프레임 행동 반전 (Exp66, base PG2)
   - 증거: `docs/v5/masking_ablation_proof.html`

5. **Offline PM vs Closed-loop 괴리**
   - Exp26: PM 70.2%, CL 0% (offline 강함 ≠ rollout 강함)
   - PM 높아도 누적 방향 오류 → rollout 실패 가능

---

## 실험 이력 요약 (V5 전체)

| 실험 | 특이사항 | PM | CL |
|------|---------|----|----|
| Exp01~03 | V4 기반, FORWARD collapse | — | — |
| Exp04 | Google-robot 첫 도입, val 0.776 | 0% | — |
| Exp10 | BBox grounding (IoU 0.87) | — | — |
| Exp11 | Google-robot 8-class baseline | 58.6% | 0% |
| Exp12~13 | instruction cond 시도, 폐기 | — | — |
| Exp14 Step2 | BBox+Image MLP decomposition | 75.9% | 66.7% |
| Exp15 | head-only ablation | 37.5% | — |
| Exp16 | all-path 150ep, center_straight 포함 | 0% | — |
| Exp17 | step3 balanced (로봇 서버 primary) | — | 11.1% |
| Exp18 | VLA text fusion (로봇 서버 fallback) | — | 11.1% |
| Exp19 | BBox proxy MLP, Exp14 기반 | 76.6% | 55.6% |
| Exp21~24 | Pure HF controlled ablation | — | — |
| Exp25 | Pure HF balanced (구 baseline, 폐기) | 52.4% | 55.6% |
| Exp39~45 | LoRA/grounding 시도, 전부 collapse | — | — |
| Exp54 | Stage2 v2 첫 SOTA | — | 96.6% |
| Exp65b | Simple MLP (pipeline ablation) | — | 10.3% |
| **Exp66** | **Stage2 v2 SOTA (base PG2)** | **93.5% val** | **96.6%** |
| Exp66-LSTM | LSTM w=16, best FPE 0.080m | — | 96.6% |
| Exp67 | HSV cx (grounding source test) | — | 96.6% |

---

## 현재 상태 / 다음 단계 (2026-06-28 갱신)

1. **CH55 Preview Model Ablation 완료** (2026-06-27~28)
   - CLIP / OWL-v2 / Kosmos-2 / Florence-2 × ZS/probe/LoRA/last-layer-FT
   - **핵심 결론**: Stage 0 워밍업(CH54)이 frame 0 cold-start 해결 → preview model 필요성 낮음
   - PG2 warm=1440ms, det≈100% → 가장 신뢰도 높음 (대체 모델 불필요)
   - 최선 대안: OWL-v2 ZS (432ms, sess dir=50%), Kosmos-2 ZS (711ms, det=100%, sess dir=36%)
   - FT 방법: 140개 학습 데이터로는 오버피팅 심함. ft_last1이 LoRA보다 나음 (sess 51.7%)
   - 스크립트: `scripts/ablate_preview_ft_v2.py`, `scripts/ablate_last_layers.py`
   - 결과: `docs/v5/ablate_preview_ft_v2.json`, `docs/v5/ablate_last_layers.json`

2. **실로봇 테스트** (2026-06-16~17)
   - Exp66 Stage2 v2 soda 배포 완료, 대시보드 http://100.85.118.58:7865
   - 체크리스트: `docs/v5/REAL_ROBOT_CHECKLIST_20260616.md`

3. **논문 제출 결정** (6/12 미팅)
   - Table 1 초안: `docs/v5/TABLE1_PAPER_DRAFT.md`
   - 실로봇 결과 나오면 `mona-sync --add-exp` 로 이력 추가

4. **카메라 서비스 미연결** (soda)
   - `ros2 run` / entry_point 메타데이터 오류 (importlib.metadata)

---

## 금지 규칙

- `third_party/RoboVLMs/` 수정 금지
- inference_server.py의 9-class 공간과 학습의 8-class 공간 혼용 금지
- Google-robot backbone으로 `generate()` 호출 금지 (텍스트 생성 망가짐 — "Tin Tin..." 반복)
- `master_memory.md`는 Claude가 사용자 요청 없이 직접 수정하지 않음

---

## 메모리 시스템 통합 조회

**참조**:
- `.menemory/core/memory_systems_integration.md`
- `docs/MEMORY_SYNC_MAP.md`
- `.agent/skills/memory-sync-hub/SKILL.md`

세 개의 메모리 시스템 (Claude Code, Codex IDE, AntiGravity-Server)을 통합 관리하는 맵.
- Claude memory: 프로젝트 격리, MEMORY.md 인덱스
- Codex memory: 로컬 IDE, SQLite 로그, history
- AntiGravity: 시스템 런타임, 서버 로그

세션 시작 시 `docs/AGENT_ENTRYPOINT.md` → `docs/MEMORY_SYNC_MAP.md` →
`MEMORY.md` → `memory_systems_integration.md` 순으로 읽는다.

주의:
- Antigravity 복구 원문은 `~/.gemini/antigravity/brain/<uuid>/` 에 있다.
- `conversations/*.pb` 는 인덱스일 뿐이다.
