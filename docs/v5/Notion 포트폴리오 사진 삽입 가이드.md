# Notion 포트폴리오 사진 삽입 가이드

현재 사용 가능한 이미지와 직접 만든 에셋 목록입니다.

---

## ✅ 바로 쓸 수 있는 이미지 (실물·연구 결과)

| 순서 | 이미지 | 로컬 경로 | Notion 위치 | 용도 |
|:---:|:---|:---|:---|:---|
| **1** | 로봇 근접 실물 사진 | `docs/SCR-20260609-oxdv.png` | **페이지 커버 or 최상단** | "실제 하드웨어" 즉시 각인 |
| **2** | 로봇+트랙 전경 사진 | `docs/SCR-20260609-oxhg.png` | 커버 대안 또는 로봇 섹션 위 | 실험 환경 규모 보여주기 |
| **3** | BBox Grounding 인식 화면 | `docs/v5/portfolio/grounding_center.jpg` | 기술 설명 바로 아래 | "목표물을 직접 인식한다" 직관 증명 |
| **4** | Masking Ablation 결과 (Exp66 SOTA) | `docs/v5/portfolio/masking_comparison.png` | "내부 분석" 항목 아래 | Exp66 Stage2 v2 · Center 8/8 (100%) 행동 반전 · 이미지 경로가 basket을 본다는 증명 |
| **5** | Zero-shot Probe (96.6%) | `docs/v5/portfolio/linear_probe_results.png` | 성능 수치 근처 | frozen CLIP이 이미 바스켓 위치 인식 증명 |

> ⚠️ `hero_robot.png` — AI 생성 이미지. 절대 사용 금지.  
> ⚠️ `exp_progression.png` — Exp46~51 구버전. 최신 수치와 다름. 사용 금지.

---

## 🆕 직접 생성한 에셋

### 아키텍처 다이어그램 (SVG)
- **경로:** `docs/v5/portfolio/architecture_diagram.svg`
- **GitHub Pages URL:** `https://minuum.github.io/MoNaVLA/v5/portfolio/architecture_diagram.svg`
- 다크모드 배경, 전체 파이프라인 시각화
  - Kosmos-2 Vision Encoder (frozen) → 256-dim L2-norm
  - BBox History ×8 → Concatenate 288-dim
  - ActionMLP 3-layer → 8 actions
  - Proximity STOP Override 표시

### 성능 비교표 (Notion 직접 입력용)
아래 텍스트를 Notion Table 블록에 그대로 복붙.

---

## 📋 Notion 성능 비교표 (복붙용)

```
Method          | Architecture             | CL ↑   | FPE ↓
E2E VLA (Exp11) | Kosmos-2 + LoRA          |  0.0%  | 1.454 m
Decomp v1       | CLIP + BBox MLP          | 66.7%  | 0.555 m
Ours (Exp66) ★  | CLIP + L2-norm + aug     | 96.6%  | 0.102 m
```

**보충 수치 (Ablation):**
```
Pipeline (단순 MLP)   → CL 10.3%  (파이프라인 바꾸면 ×9.4)
Head: Linear          → CL 69.0%
Head: FCHead          → CL 93.1%
Head: LSTM = MLP      → CL 96.6%  (동등)
Window w=4 포화       → CL 96.6%, FPE 0.094 m
Window LSTM w=16      → CL 96.6%, FPE 0.080 m (전체 최저)
```

---

## 📐 Notion 아키텍처 텍스트 (이미지 없을 때 대안)

```
[RGB Frame 224×224]
       ↓
[Kosmos-2 Vision Encoder] — frozen
       ↓ 1024-dim
[image_proj → 256-dim, L2-normalize]
       ↓
[Concatenate] ← BBox History (cx,cy,area,has_bbox × 8frames) = 32-dim
       ↓ 288-dim
[ActionMLP: 256→128→64→8]
       ↓
[8 Actions: STOP / FORWARD / LEFT / RIGHT / FWD+L / FWD+R / ROT_L / ROT_R]
       ↑
[Proximity Override: area≥0.50 AND |cx-0.5|≤0.30, 2 consecutive frames → STOP]
```

---

## 📌 삽입 우선순위 요약

1. **커버:** `SCR-20260609-oxdv.png` (실물 로봇 근접)
2. **성능표:** 위 복붙용 Table 직접 입력
3. **아키텍처:** `architecture_diagram.svg` embed or 텍스트 대안
4. **증거:** `masking_comparison.png` or `linear_probe_results.png` 중 1장
5. **데모:** `grounding_center.jpg` (BBox 인식 화면)
