# VLM Text Attention = 0% 공식 결론

**작성일:** 2026-05-11  
**결론:** Google-robot post-training backbone에서 text pathway는 구조적으로 붕괴됨. LoRA로 복구 불가능.

---

## 증거 요약

### 1. Attention 직접 측정 (measure_attention.py)

| 실험 | 접근 방식 | text_ratio 평균 |
|------|-----------|----------------|
| Exp41b (resume Exp40 PTA) | pretrained backbone | **0.000000%** |
| Exp41c (scratch PTA) | scratch 학습 | **0.000000%** |
| Exp42 (counterfactual PTA) | counterfactual loss | **0.000000%** |
| Exp43 (cross-attn text) | cross-attn 구조 변경 | **0.000000%** |
| Exp15 (head-only) | VLM 완전 frozen | **0.000000%** (별도 측정) |

모든 실험에서 action token의 text token attention = 0%.  
이미지 token attention이 46~100% 범위에서 작동 중.

### 2. val_loss 동일성 (Exp45 vs Exp48)

```
Exp45 (vision-only LoRA):  epoch=00 → 10.626, epoch=01 → 10.271, epoch=02 → 10.119
Exp48 (+ synthetic instr): epoch=00 → 10.626, epoch=01 → 10.271  ← 소수점 3자리까지 동일
```

Exp48은 Exp45에 instruction conditioning만 추가한 실험.  
val_loss가 epoch=0부터 완전히 동일 → **instruction이 loss gradient에 전혀 기여하지 않음**.  
즉, text 경로가 완전히 차단된 상태에서 학습.

### 3. 실험 스펙트럼

text attention 복구를 시도한 Exp17~Exp48 전체에 걸쳐:
- LoRA target 다양화 (text decoder, vision, projection layer)
- 학습 방식 변경 (scratch, resume, PTA, counterfactual)
- Instruction 다양화 (단일 → 9종 synthetic)
- Architecture 변경 (cross-attn, head-only)

**모두 text_ratio = 0% 또는 val_loss 동일 → text 영향 없음.**

---

## 근본 원인

Google-robot post-training (`kosmos_ph_google-robot-post-train.pt`)이 text generation 경로를 붕괴시킴.  
이는 robot action prediction 특화 fine-tuning 과정에서 text decoder 가중치가 image-only 예측에 최적화된 결과로 추정.

- Pure HF Kosmos-2 (`.vlms/kosmos-2-patch14-224`): text generation 정상, BBox grounding 가능
- Google-robot backbone: `generate()` 완전 망가짐, text attention = 0%

---

## 결론 및 방향

### 공식 종결 사항
- **Exp17~Exp48 LoRA text attention 복구 실험 계열 → 종결**
- 추가 LoRA 시도는 의미 없음 (구조적 문제, 가중치 문제가 아님)

### 채택 방향: Decomposition (Exp47 MLP)
| 지표 | Exp11 (e2e) | Exp47 MLP (decomp) |
|------|-------------|-------------------|
| val_loss | 1.010 | — |
| PM | 58.6% | 99.2% |
| closed-loop success | 0% (FPE 1.45m) | 100% (FPE 0.013m) |

MLP decomposition이 실질적으로 작동하는 유일한 경로.

### 남은 과제 (Exp47 계열)
1. 실제 로봇 + 카메라 테스트 (grounding 품질 측정)
2. NO_BBOX 구간 핸들링 (`coarse_direction_clf.pt` 학습)
3. 교수님 프로토콜 Step 2 (Exp16): e2e 관점 별도 트랙

---

## 관련 파일

- 측정 스크립트: `scripts/measure_attention.py`
- attention 분석 결과: `docs/v5/attention_analysis/summary.json`
- Exp45 config: `configs/mobile_vla_v5_exp45_vision_lora_only.json`
- Exp48 config: `configs/mobile_vla_v5_exp48_synthetic_instr.json`
- 현재 최선 모델: `docs/v5/bbox_nav_exp47/exp47_mlp.pt`
