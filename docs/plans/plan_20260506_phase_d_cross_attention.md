# Plan — Track 3 / Phase D: Cross-Attention Text Head (Exp43)

작성: 2026-05-06 (Exp42 학습 중 사전 작성)
브랜치: `inference-integration`
조건부 trigger: Exp42 Phase A FAIL 시 진입
선행 plan: `plan_20260506_hybrid_C_three_track.md` §4

## 0. 한 줄 요약

`NavRoboKosMos.forward_continuous`의 `_instr_emb_cache = ... .detach()` 로 text gradient가 끊겨 있는 게 Exp13 additive-bias 실패의 진짜 원인 가능성을 5/6 코드 리딩에서 확인. 이를 해결하기 위해 (1) detach 제거 + (2) mean pool → full token sequence + (3) LSTM hidden에 cross-attention하는 head를 새로 추가.

---

## 1. 근거 (research summary, 5/6 코드 리딩)

### 1.1 현재 text 흐름 — Exp13 additive bias

**`robovlm_nav/models/nav_robokosmos.py:19-37`**:
```python
def forward_continuous(self, vision_x, lang_x, attention_mask=None, text_embedding=None, **kwargs):
    if text_embedding is not None:
        self._instr_emb_cache = text_embedding
    elif lang_x is not None:
        instr_embeds = self.word_embedding(lang_x)              # (B, T, D)
        # ⚠️ DETACH — text gradient path 끊김
        self._instr_emb_cache = instr_embeds.mean(dim=1).detach()  # (B, D)
    else:
        self._instr_emb_cache = None
    return super().forward_continuous(...)
```

**`robovlm_nav/models/policy_head/nav_policy_impl.py:318-354`** (`MobileVLAClassificationDecoder`):
```python
# __init__:
if instr_in_features is not None:
    self.instr_proj = nn.Linear(instr_in_features, in_features * latent)
else:
    self.instr_proj = None

# forward:
if instruction_emb is not None and self.instr_proj is not None:
    instr_feat = self.instr_proj(instruction_emb.to(tok_seq.dtype))  # (B, in_features)
    instr_feat = instr_feat.unsqueeze(1).expand_as(tok_seq)
    tok_seq = tok_seq + instr_feat   # additive bias on every timestep
```

### 1.2 문제 진단

이 구조의 결정적 실패 원인 3개:

| # | 문제 | 결과 |
|---|------|------|
| **A** | `.detach()`로 gradient가 word embedding → instr_proj 통과만 가능 | LSTM의 어떤 weight도 text를 학습할 incentive 없음. instr_proj가 zero에 수렴해도 무방 |
| **B** | Mean pool — 다중 단어 정보 1개 vector로 압축 | "Move left"와 "Move right"의 차이는 한두 토큰 — mean에 쉽게 묻힘 |
| **C** | Additive bias — vision feature scale 압도 | Pure HF Kosmos-2의 image attn 91.7%로 우세. 단순 더하기로는 수치적으로 묻힘 |

(A)는 Exp13/Exp25/Exp41C/Exp42 모두 동일하게 적용된 코드 경로 — text attn 0% 결과의 충분 조건.

### 1.3 검증 가능한 가설

H1. `.detach()` 제거만으로 text attn 비-0이 됨 (코스트 거의 0)
H2. (H1) 만으로는 부족하고 cross-attention까지 필요
H3. (H1+H2) 만으로 부족하고 backbone(Kosmos-2 LM head)까지 unfreeze 필요

→ Exp43은 H2 검증 (H1까지 포함). H3는 Phase E 후보.

---

## 2. 디자인 (Exp43 — Cross-Attention Text Conditioning)

### 2.1 아키텍처

```
                                    ┌──────────────────────────┐
                                    │   Token-level text feats │
                                    │   (B, T_text, D_txt)     │  ← detach 제거
                                    └────────────┬─────────────┘
                                                 │
            vision tokens                        │
            (B, T_vis, D_vis)                    │
                  │                              │
                  ▼                              │
            ┌──────────────┐                     │
            │ Vision pool  │                     │
            └──────┬───────┘                     │
                   ▼                             │
              tok_seq (B, ws, D)                 │
                   │                             │
                   ▼                             │
              ┌──────────┐                       │
              │   LSTM   │                       │
              └────┬─────┘                       │
                   ▼ hidden (B, ws, H)           │
                                                 │
                ┌──────────────────┐    Q ◄──────┘
                │ MultiheadAttn    │ ──┐
                │  (cross-attn)    │   │ K, V
                └──────────────────┘ ◄─┘
                         │
                         ▼ attended (B, ws, H)
                         │
                         + α · attended  (residual + learned scalar)
                         │
                         ▼
                   ┌──────────┐
                   │ Linear   │ → action logits (B, ws, n × C)
                   └──────────┘
```

**핵심 변경 vs 현재:**
- text feature: `mean.detach()` → **token-level, no detach**
- 적용 위치: `tok_seq + instr_feat` (input bias) → **LSTM hidden + cross_attn(text)** (output transform)
- 새 모듈: `MultiheadAttention(num_heads=4, embed_dim=H)` + scalar gate `α`

### 2.2 코드 변경 위치

| 파일 | 변경 |
|---|---|
| `robovlm_nav/models/nav_robokosmos.py` | (1) `.detach()` 제거 (2) mean pool 제거 — full sequence 캐시 |
| `robovlm_nav/models/policy_head/nav_policy_impl.py` | (3) `MobileVLAClassificationDecoder`에 cross-attn block 추가 (4) forward에서 hidden output에 적용 |

**없어지는 것:** `instr_proj` (additive bias) — Exp43에서는 사용 안 함. 기존 ckpt 호환성 위해 init은 유지하되 forward에서 cross-attn 우선 적용.

### 2.3 변경 의사 코드

#### 2.3.1 `nav_robokosmos.py`

```python
def forward_continuous(self, vision_x, lang_x, attention_mask=None, text_embedding=None, **kwargs):
    if text_embedding is not None:
        # (B, D) 또는 (B, T, D)
        self._text_seq_cache = text_embedding
        self._text_mask_cache = None
    elif lang_x is not None:
        # ⚠️ no detach. token sequence 그대로
        self._text_seq_cache = self.word_embedding(lang_x)        # (B, T, D)
        self._text_mask_cache = attention_mask                     # (B, T)
    else:
        self._text_seq_cache = None
        self._text_mask_cache = None

    # 기존 cache 키도 backward-compat 위해 유지 (mean pool, but detached 아님)
    if self._text_seq_cache is not None and self._text_seq_cache.dim() == 3:
        self._instr_emb_cache = self._text_seq_cache.mean(dim=1)   # ⚠️ no detach
    else:
        self._instr_emb_cache = self._text_seq_cache

    return super().forward_continuous(vision_x, lang_x,
                                      attention_mask=attention_mask, **kwargs)


def _forward_action_head(self, action_tokens, action_labels, action_mask, mode="train", **kwargs):
    # Phase D: token-level text features
    if self._text_seq_cache is not None:
        kwargs["text_features"] = self._text_seq_cache
        kwargs["text_mask"] = self._text_mask_cache
    # backward-compat: instruction_emb은 그대로 (구식 instr_proj 경로)
    if self._instr_emb_cache is not None:
        kwargs["instruction_emb"] = self._instr_emb_cache
    return super()._forward_action_head(action_tokens, action_labels, action_mask, mode=mode, **kwargs)
```

#### 2.3.2 `nav_policy_impl.py` — `MobileVLAClassificationDecoder`

```python
# __init__ 추가:
text_in_features = kwargs.get("text_in_features", None)   # 예: 2048 (Kosmos-2 D)
self.text_cross_attn_enabled = text_in_features is not None
if self.text_cross_attn_enabled:
    self.text_proj = nn.Linear(text_in_features, hidden_size)  # match LSTM hidden
    self.text_cross_attn = nn.MultiheadAttention(
        embed_dim=hidden_size,
        num_heads=4,
        batch_first=True,
        dropout=0.0,
    )
    self.text_gate = nn.Parameter(torch.tensor(0.1))   # α, learned

# forward 끝부분 (logits 직전):
if self.text_cross_attn_enabled:
    text_feats = kwargs.get("text_features", None)   # (B, T, D_text)
    text_mask = kwargs.get("text_mask", None)         # (B, T)
    if text_feats is not None:
        K = self.text_proj(text_feats.to(x.dtype))    # (B, T, H)
        if text_mask is not None:
            key_padding_mask = ~text_mask.bool()      # invert (True=padding)
        else:
            key_padding_mask = None
        attended, _ = self.text_cross_attn(
            query=x, key=K, value=K,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.text_gate * attended

logits = self.logits(x)
```

### 2.4 Config (Exp43)

`configs/mobile_vla_v5_exp43_cross_attn_text.json` (신규):

```json
{
    "_comment": "V5 Exp43: Cross-attention text head (Phase D). Exp42 base + .detach() 제거 + token-level cross-attn. text_in_features=2048 (Kosmos-2 hidden). Path D 첫 시도.",
    "parent": "configs/mobile_vla_v5_exp42_counterfactual_pta.json",
    "exp_name": "v5-exp43-cross-attn-text",
    "task_name": "mobile_vla_v5_exp43",

    "act_head": {
        "text_in_features": 2048
    },

    "trainer": {"max_epochs": 8, "check_val_every_n_epoch": 1}
}
```

> Note: `text_in_features=2048`은 Kosmos-2 hidden_size 가정. 실제 값은 `model.config.hidden_size`로 검증 필요. 코드 작성 시 첫 단계.

### 2.5 Resume vs Scratch

- **Scratch (추천)**: `instr_proj`만 있던 구식 ckpt와 새 `text_cross_attn` 가중치는 호환 안 됨. 8 epoch scratch.
- Resume from Exp42 vision/LSTM weights: 가능하지만 strict=False 로딩 + cross_attn 새 init. 복잡도 ↑, 효과 검증 모호. 보류.

---

## 3. 검증 (Phase A 동일 + 추가)

### 3.1 Phase A 4기준

| 기준 | 목표 | 비고 |
|---|---|---|
| text attention (24L 평균) | ≥ 5% | Exp42에서 0% 확정 시 핵심 |
| action L1 diff (left↔right) | ≥ 1e-2 | |
| pred change / 30 frames | > 0 | |
| PM | ≥ 50% | Exp25 baseline 회귀 없음 |

### 3.2 추가 — text gate 학습 흔적

```python
# 학습 후 ckpt에서 text_gate 파라미터 확인
gate = state_dict["model.policy_head.text_gate"]
# 초기값 0.1, 학습 후 |gate| ≥ 0.05 면 conditioning 진짜 활용
```

### 3.3 실패 시 분기

- **3-기준 통과 + PM 회귀**: Exp43b — gate를 sigmoid로 bound, lr scaling 조정
- **text attn 비-0인데 action diff 작음**: Exp43c — gate 초기값 0.5, head dim ↑ (8)
- **모두 0**: Phase E (TICVLA / MobilityVLA, Kosmos-2 backbone 자체 교체) — 별도 plan

---

## 4. 트레이드오프

### 장점
- backbone 무수정. 새 head ~1.5M params (4-head, hidden=1024).
- detach 제거가 단독 효과 가능 (low-cost subset of changes)
- Exp42 결과 보고 들어가는 거라 Phase A 데이터 측 전부 검증된 후 시작

### 단점
- 새 head — 학습 안정성 미보장. early instability 가능
- Exp43 학습 시간 (~10h) — Exp42와 같은 페이스
- text_in_features 값 검증 필요 (config 추측 → 실제 모델에서 확인)

### 위험
- Cross-attn에 너무 큰 gate 초기값 → vision feature를 덮어 PM 깎을 수 있음. 그래서 0.1 시작.
- Multi-head 4 < 8: 표현력은 작지만 안정성 ↑. 실패 시 ↑

---

## 5. 일정 및 단계

전제: Exp42 Phase A FAIL 결과 받은 시점

1. **Step 1 (10분)**: `text_in_features` 실제 값 확인 — Pure HF Kosmos-2 hidden_size 출력
2. **Step 2 (30분)**: code 변경 (nav_robokosmos.py + nav_policy_impl.py)
3. **Step 3 (5분)**: Exp43 config 작성
4. **Step 4 (5분)**: smoke 학습 1 step — shape mismatch 등 즉시 발견
5. **Step 5 (~10h)**: 8 epoch 학습 (백그라운드)
6. **Step 6 (~30분)**: `run_exp43_pipeline.sh` (Exp42 pipeline 복제) 자동 평가
7. **Step 7**: verdict 보고 분기 결정

총 학습 외 ~1.5 시간.

---

## 6. 수정 파일 요약 (구현 시)

| 파일 | 변경 |
|---|---|
| `robovlm_nav/models/nav_robokosmos.py` | detach 제거, token sequence 캐시 추가 (~10줄) |
| `robovlm_nav/models/policy_head/nav_policy_impl.py` | cross_attn block 추가 (~25줄) |
| `configs/mobile_vla_v5_exp43_cross_attn_text.json` | 신규 |
| `scripts/measure_attention.py` | exp43 등록 |
| `scripts/run_exp43_pipeline.sh` | 신규 (run_exp42_pipeline.sh 복제) |

---

## 7. 사용자 결정 필요 (Exp42 결과 후)

1. **Exp42 PASS면 Phase D 폐기**: Exp42 자체를 deployment 후보로
2. **PARTIAL이면 Phase D 진행 vs counterfactual 비율 조정 재시도**
3. **FAIL이면 Phase D 즉시 진입**

또한 진입 시:
- A. detach만 제거하고 결과 보기 (single-variable test)
- B. detach + cross-attn 동시 변경 (Exp43 본안)

추천: **B 본안**. detach 단독 변경은 ablation으로 부속 실험 가능 (Exp43a).

---

## 8. CLAUDE.md 준수

이 plan은 Exp42 결과 도착 시 사용자 검토/주석 → 승인 후에만 §6 파일들을 수정한다. **승인 전 코드 변경 0줄.**
