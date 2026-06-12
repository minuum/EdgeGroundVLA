# Stage2 v2 체크포인트 설명 (2026-06-12)

> 추론 서버: `robovlm_nav/serve/stage2_v2_inference_server.py --port 8001`
> 공통 파이프라인: Kosmos-2 vision encoder → image_proj (256-dim, L2-norm) + bbox history → ActionHead

---

## Stage1 (공통 — 모든 Stage2에 필요)

| 파일 | 설명 |
|------|------|
| `runs/v5_nav/mlp/exp54/stage1_v2/stage1_v2_projs.pt` | Kosmos-2 vision_model + image_proj(1024→256, L2-norm). val_acc 98.1%. 모든 Stage2 모델이 이걸 공유. |

---

## Stage2 — 운영 모델

### ⭐ stage2_v2_mlp_base_pg2_aug.pt — **기본값 (권장)**
- **실험**: Exp66 (Stage2 v2, cx 소스 = base PaliGemma2, bbox aug)
- **구조**: ActionMLP, window=8, d_in=288 (8×4 + 256)
- **성능**: val_acc **93.5%**, CL **96.6%**, FPE **0.102m**
- **grounding**: base PaliGemma2 zero-shot (LoRA 없음) — 가장 안정적
- **환경변수**: `VLA_S2V2_STAGE2=runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp_base_pg2_aug.pt`

### stage2_v2_mlp.pt — HSV cx 소스 (대안)
- **실험**: Exp54 (Stage2 v2, cx 소스 = HSV 색상 필터)
- **구조**: ActionMLP, window=8, d_in=288
- **성능**: val_acc **92.6%**, CL **96.6%**, FPE **0.110m**
- **grounding**: HSV 색상 필터 (VLM 불필요, 빠름)
- **참고**: CL 성능은 동일하나 실환경 변동(조명·배경)에 취약할 수 있음
- **환경변수**: `VLA_S2V2_STAGE2=runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt`

---

## Stage2 — Ablation 모델 (참고용, 배포 비권장)

| 파일 | 실험 | Head | val_acc | CL | FPE | 비고 |
|------|------|------|---------|-----|-----|------|
| `stage2_v2_linear_base_pg2_aug_linear.pt` | Exp68 | Linear (1-layer) | 76.8% | 69.0% | 0.377m | 성능 부족 |
| `stage2_v2_fc_base_pg2_aug_fc.pt` | Exp69 | FCHead (deep MLP) | 95.3% | 93.1% | 0.109m | temporal 없음 |
| `stage2_v2_lstm_base_pg2_aug_lstm.pt` | Exp70 | LSTMHead (RoboVLMs) | 95.7% | 96.6% | 0.112m | LSTM, MLP와 동등 |
| `stage2_v2_mlp_w2_w2.pt` | Window w=2 | MLP | 94.9% | 93.1% | 0.145m | window 부족 |
| `stage2_v2_mlp_w4_w4.pt` | Window w=4 | MLP | 92.9% | 96.6% | 0.094m | FPE 최저 (MLP) |
| `stage2_v2_mlp_w16_w16.pt` | Window w=16 | MLP | 89.6% | 96.6% | 0.102m | val_acc 희석 |
| `stage2_v2_lstm_w4_w4.pt` | Window w=4 | LSTM | 95.1% | 96.6% | 0.123m | |
| `stage2_v2_lstm_w16_w16.pt` | Window w=16 | LSTM | 96.9% | 96.6% | 0.080m | FPE 전체 최저 |

---

## 추론 서버 실행

```bash
cd ~/MoNaVLA

# best 모델 (기본값, 환경변수 없어도 됨)
python3 robovlm_nav/serve/stage2_v2_inference_server.py --port 8001

# HSV 모델로 교체
VLA_S2V2_STAGE2=runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt \
  python3 robovlm_nav/serve/stage2_v2_inference_server.py --port 8001
```

## 헬스체크

```bash
curl http://localhost:8001/health
curl http://localhost:8001/model/info
```

## grounding skip 조절 (/config)

```bash
# 매 3프레임마다 grounding (중간은 캐시) — 속도 향상
curl -X POST http://localhost:8001/config \
  -H "Content-Type: application/json" \
  -d '{"grounding_skip_n": 3}'
```
