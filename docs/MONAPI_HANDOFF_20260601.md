# MoNa-Pi ↔ MoNaVLA 핸드오프 가이드
작성: 2026-06-01 (minum 서버 기준)

---

## 1. 현재 상황 요약

MoNaVLA (minum 서버)에서 **HSV → PG2(PaliGemma2) 그라운딩** 으로 파이프라인을 전환하는 실험을 완료했습니다.

### 최신 실험 결과

| 실험 | val_acc | CL | 비고 |
|------|---------|-----|------|
| Exp54 HSV (기준) | 92.6% | 96.67% | 색상 필터 기반 |
| Exp57 PaliGemma LoRA | 100% | — | "gray basket" grounding |
| Exp59 Hard Negative | TP=95%, FP=0% | — | 텍스트 조건부 분리 |
| Exp60 PG2 cx MLP | 97.0% | 60% | 현재 최선 |

### CL 60% 도달한 구성

```
PG2 그라운딩 (Exp59 LoRA) → cx, cy → Stage2 MLP (flip aug + center×3)
```

center_straight 경로가 0%로 병목 중. 나머지 right/left 계열은 80~100%.

---

## 2. MoNa-Pi 데이터 통합 (6/1 진행 중)

### 신규 73개 에피소드 발견

MoNa-Pi의 `mobile_vla_dataset_V5_add_free`에 minum에 없는 에피소드 73개가 있습니다.

```
/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_V5_add_free/
```

| 경로 타입 | 신규 | 효과 |
|---------|------|------|
| right_left | 18개 | 기존 15 → 33개 |
| right_right | 16개 | 기존 15 → 31개 |
| right_straight | 14개 | 기존 20 → 34개 |
| left_left | 11개 | 기존 15 → 26개 |
| left_right | 8개 | 기존 15 → 23개 |
| left_straight | 5개 | 기존 20 → 25개 |
| center_straight | 1개 | 기존 20 → 21개 |

현재 minum에서 PG2 재주석 + MLP 재학습 진행 중 (Exp61).

---

## 3. MoNa-Pi에서 해야 할 것

### A. 실로봇 Exp59 배포 (가장 급함)

```bash
cd /home/soda/MoNaVLA
git pull origin inference-integration

# Exp59 LoRA adapter는 이미 rsync로 전송됨
ls runs/v5_nav/grounding/exp59/adapter_model.safetensors  # 확인

# PaliGemma2 base model 다운 필요 (약 5GB, 아직 없음)
huggingface-cli download google/paligemma2-3b-mix-224 \
    --local-dir ~/.cache/huggingface/hub/models--google--paligemma2-3b-mix-224

# 실시간 그라운딩 데모 실행
python3 scripts/run_grounding_realtime.py EPISODE.h5 \
    --adapter exp59 --phrases "gray basket" "red ball" "person"
```

### B. 실로봇 Closed-Loop 테스트

현재 Exp60 MLP (CL 60%)를 실로봇에서 테스트하면 시뮬레이터보다 좋을 가능성 있음.

```bash
# 필요한 파일들
runs/v5_nav/mlp/exp60/stage2_pg2cx_flip_mlp.pt  # Stage2 MLP
runs/v5_nav/mlp/exp54/stage1_v2/stage1_v2_projs.pt  # CLIP LoRA
runs/v5_nav/grounding/exp59/  # PG2 LoRA
```

### C. center_straight 추가 데이터 수집

현재 center_straight가 CL에서 0%. 20개 에피소드가 있지만 부족.
- 10~15개 추가 수집 권장
- 다양한 시작 위치에서 basket 정중앙 접근

---

## 4. 핵심 파일 경로 (minum 서버)

```
모델 체크포인트:
  runs/v5_nav/grounding/exp57/        # PaliGemma (1-class)
  runs/v5_nav/grounding/exp59/        # PaliGemma2 (hard negative)
  runs/v5_nav/mlp/exp60/stage2_pg2cx_flip_mlp.pt  # 현재 최선 MLP

어노테이션:
  docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json  # PG2 cx 주석

학습 스크립트:
  scripts/train_exp60_flip_aug.py    # Stage2 MLP (flip aug)
  scripts/eval_exp59_closedloop.py   # CL 평가

데이터:
  ROS_action/mobile_vla_dataset_v5/  # 244개 에피소드 (심링크 포함)
```

---

## 5. 파이프라인 구조

```
실시간 추론 파이프라인:
  카메라 이미지
    ↓
  PaliGemma2 Exp59 LoRA ("detect gray basket")
    → cx, cy, area (신경망 그라운딩)
    ↓
  Stage1 v2 CLIP LoRA
    → visual feature 256dim
    ↓
  Stage2 MLP (288dim = 32bbox_hist + 256visual)
    → 8-class action (FORWARD/LEFT/RIGHT/FWD+L/FWD+R/ROT_L/ROT_R/STOP)
```

**HSV를 완전히 제거한 순수 신경망 파이프라인**입니다.

---

## 6. 교수님 반박 대응 현황

| 반박 | 답변 | 근거 |
|------|------|------|
| "basket을 보는가?" | ✅ 증명 | Exp57: 100% / red ball 0% |
| "LoRA 기여?" | ✅ | left +6.2%p 방향별 균등 |
| "다른 물체 입력?" | ✅ | Exp59: TP=95%, FP=0% |
| "텍스트로 목표 변경?" | ✅ | "gray basket" vs "red ball" 완전 분리 |
| "실로봇 확인?" | 🔄 | CL 60% 달성, 실로봇 테스트 준비 중 |

---

## 7. 브랜치

- **메인 작업 브랜치**: `inference-integration`
- docs 서버: `http://100.101.73.21:9000/v5/research_story.html`

```bash
git pull origin inference-integration
```
