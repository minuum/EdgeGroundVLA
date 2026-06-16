# Plan — Hybrid C: 3-Track Recovery (실용 + 데이터 + 구조)

작성: 2026-05-06
브랜치: `inference-integration`
선행 진단: 2026-05-06 세션 리캡 (Phase A 두 trial 모두 실패 — text attn 0%, action L1 ~1e-5)
관련 plan: `plan_20260501_bbox_proxy_deploy.md`, `plan_20260505_exp41_path_type_aware.md`, `plan_20260502_grounding_comparison.md`

## 0. 한 줄 요약

End-to-end 텍스트 conditioning 학습 신호가 죽었음을 5/6 확정. 이를 3개 트랙으로 동시 공략:

- **T1 (실용)** — 검증된 자산(Exp25 + BBox proxy)을 로봇에 즉시 배포해 운영 CL 11% → 56~67% 회복
- **T2 (데이터)** — 이미 코드에 있는 `counterfactual_steer_prob`를 활성화한 Exp42 학습 (구조 변경 없이 마지막 데이터-side 시도)
- **T3 (구조)** — T2도 실패 시 LSTM decoder에 text token cross-attention 추가 (Phase D)

T1은 즉시 시작, T2는 T1과 병렬 학습, T3는 T2 결과 본 뒤 진입 결정.

---

## 1. 근거 (research summary)

### 1.1 현재까지 확정된 것

| 자산 | 지표 | 상태 |
|------|------|------|
| Exp25 ckpt | PM 52.4% / CL 55.6% / FPE 0.382 | **billy에 있음, minum에 없음** |
| Exp19 BBox+Image MLP | PM 75.9% / CL 66.7% / FPE 0.55m | 가중치 미학습 (proxy 첫 실행 시 자동) |
| `bbox_dataset_full.json` | 2626 frames, 150ep | ✅ 이미 추출됨 |
| Pure HF Kosmos-2 grounding | IoU 0.679 (Exp40 검증) | ✅ 정상 |
| 로봇 서버 (soda@100.85.118.58) | Exp17/18 CL 11.1% | ⏳ Exp25 미배포 |

### 1.2 코드 상 핵심 발견

**`nav_h5_dataset_impl.py`에 counterfactual이 이미 구현되어 있음** (line 84-85, 569-616):

```python
counterfactual_stop_prob=0.0,    # __init__ default
counterfactual_steer_prob=0.0,
```

활성화 시 동작:
- `counterfactual_stop_prob`: 무작위로 액션을 0으로 만들고 instruction을 "Stop", "정지해" 등으로 교체
- `counterfactual_steer_prob`: 무작위로 액션을 left/right strafe 또는 turn-in-place로 만들고 instruction을 "Move left", "Rotate right" 등으로 교체

**그러나 Exp17/Exp25/Exp41B/Exp41C 어느 config에서도 활성화 안 됨** (grep 0 hits). 즉 Phase A 실패는 데이터-측 마지막 카드를 안 써본 것일 수 있다 — Track 2의 핵심 이유.

### 1.3 미커밋 변경

- `proxy_inference_server.py` +137 (3-tier integration + LoRA off + fix 필요한 path resolver)
- `nav_trainer.py` +4 (chunking bug fix — `arm_action_chunck` 자리 수정, Exp40 이후 학습에 영향)
- `test_v5_pm_dm.py` +35 (eval 개선)
- `measure_attention.py` +14 (Exp41 등록)

T1 진행 전에 chunking fix와 path resolver는 커밋해야 재현성 확보.

---

## 2. Track 1 — 실용 배포 (CL 11% → 56~67%)

### 2.1 사전 작업 (블로커)

#### Step 0a. Exp25 ckpt를 minum으로 가져오기

**문제**: `runs/v5_nav/kosmos/mobile_vla_v5_exp25/...epoch_02-val_loss=10.117.ckpt`는 billy에만 있음. 기존 sync 스크립트는 billy→minum 전송용이라 billy에서 실행해야 함.

**선택지**:
- (a) billy 머신에서 `bash scripts/sync/push_exp_to_minum.sh exp25` 실행 — 작성자가 billy 접근 가능 시
- (b) minum에서 pull: `rsync -avzP billy:/home/billy/25-1kp/MoNaVLA/runs/v5_nav/kosmos/mobile_vla_v5_exp25/ ~/26CS/MoNaVLA/runs/v5_nav/kosmos/mobile_vla_v5_exp25/`
- (c) 직접 재학습 — 비용 큼 (Exp25는 3 epoch, ~30분)

→ **추천 b**: 사용자가 billy ssh config 가지고 있으면 한 줄. 못 하면 (c)로 fallback.

#### Step 0b. 미커밋 변경 정리 (커밋)

- `nav_trainer.py` chunking fix → 별도 커밋 ("fix: nav_trainer chunking bug — use action_chunck not arm_action")
- `proxy_inference_server.py` 3-tier + LoRA off + path resolver → 커밋
- `measure_attention.py` Exp41 등록 → 커밋
- `test_v5_pm_dm.py` → 커밋
- 신규 untracked 스크립트는 T1/T2/T3가 사용하는 것만 골라서 add

### 2.2 실행 단계

#### Step T1.1 — fastapi 환경 설치 (minum)

```bash
cd /home/minum/26CS/MoNaVLA
source .venv/bin/activate
uv pip install fastapi 'uvicorn[standard]' python-multipart
```

#### Step T1.2 — proxy_inference_server.py path resolver 마무리

`proxy_inference_server.py:56` 부근에 `_resolve_data_dir()` 추가 (Plan 5/1 §3.2 그대로):

```python
_DATA_PATH_CANDIDATES = [
    Path("/home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v5"),
    Path("/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5"),
    ROOT / "ROS_action" / "mobile_vla_dataset_v5",
]

def _resolve_data_dir() -> Path:
    if (override := os.getenv("VLA_PROXY_DATA_DIR")):
        return Path(override)
    for cand in _DATA_PATH_CANDIDATES:
        if cand.exists():
            return cand
    return _DATA_PATH_CANDIDATES[-1]

DATA_DIR = _resolve_data_dir()
```

확인: 현재 미커밋 변경(+137줄)에 이미 들어가 있는지 grep로 검사 후 없으면 추가.

#### Step T1.3 — BBox proxy MLP 학습 (full 150ep)

```bash
export VLA_PROXY_DATASET_FILE="$PWD/docs/v5/bbox_nav_step1/bbox_dataset_full.json"
export VLA_PROXY_FORCE_RETRAIN=true
python3 robovlm_nav/serve/proxy_inference_server.py --port 8001 &
# 학습 로그 확인 후 test_acc >= 0.70 검증, 가중치 저장 위치 확인
# pkill 후 정상 운영 모드로 재기동
```

산출물: `docs/v5/bbox_nav_exp19_proxy/exp19_proxy_mlp.pt`

#### Step T1.4 — 두 서버 동시 운영 + 헬스 체크

```bash
# 8000 (Exp25 end-to-end)
EXP25_CKPT="runs/v5_nav/kosmos/mobile_vla_v5_exp25/2026-04-22/v5-exp25-step3-balanced-objective/epoch_epoch=epoch=02-val_loss=val_loss=10.117.ckpt"
python3 robovlm_nav/serve/inference_server.py --port 8000 --ckpt "$EXP25_CKPT" &

# 8001 (BBox proxy)
python3 robovlm_nav/serve/proxy_inference_server.py --port 8001 &

curl http://localhost:8000/health
curl http://localhost:8001/health
```

#### Step T1.5 — soda 로봇 서버에서 호출 검증

기존 client config를 두 endpoint로 분기 가능하게 수정 (별도 client 코드는 본 plan 범위 밖, robot-local-api-handoff 스킬 활용).

#### Step T1.6 — closed-loop 실측 (smoke 1~2 ep)

`scripts/sim/evaluate_closed_loop_v5.py --model exp25` (가능 시 추가 옵션 `--proxy http://localhost:8001`).

#### Step T1.7 — 문서 + Hero 링크

- `docs/v5/bbox_nav_exp19_proxy/index.html` 배포 섹션 갱신 (Plan 5/1 §3.6)
- `docs/index.html` Hero 영역에 "Exp25 + BBox Proxy 듀얼 배포" 버튼 추가 (CLAUDE.md 규칙)

### 2.3 T1 검증 기준

- [ ] Exp25 inference_server `/health` 응답 OK
- [ ] BBox proxy `/health` 응답 OK + `proxy_info.test_acc >= 0.70`
- [ ] 두 서버 동일 이미지 5장에 대해 응답 비교 — 한 클래스 collapse 아님 확인
- [ ] (가능 시) 로봇 closed-loop smoke 1~2 ep — Exp25 단독 CL ≥ 30% 또는 BBox proxy CL ≥ 50%

---

## 3. Track 2 — Counterfactual Steer (Exp42)

### 3.1 가설

Phase A 실패의 원인 후보 두 개:

1. **데이터 측**: 학습 데이터의 instruction이 const라 텍스트가 학습 신호로 안 들어감 → path_type_aware preset만으론 부족. 실제 *액션 라벨 자체*를 prompt에 종속시켜야 함.
2. **구조 측**: LSTM decoder가 text feature를 무시하도록 수렴

T2는 **(1)이 진짜 원인인지 마지막으로 확인**. 코드에 이미 있는 counterfactual을 켜는 한 줄 변경이라 비용 ≈ 0.

### 3.2 Exp42 config

`configs/mobile_vla_v5_exp42_counterfactual_pta.json` (신규):

```json
{
    "_comment": "Exp42: Pure HF Kosmos-2 + path_type_aware + counterfactual_steer/stop. Exp41C가 path_type_aware만으로 실패 → 액션-instruction 강제 결합으로 마지막 데이터-side 시도. 구조 미수정.",
    "parent": "configs/mobile_vla_v5_exp41c_scratch_pta.json",
    "exp_name": "v5-exp42-counterfactual-pta",
    "task_name": "mobile_vla_v5_exp42",

    "trainer": {"max_epochs": 8, "check_val_every_n_epoch": 1},

    "train_dataset": {
        "instruction_preset": "path_type_aware",
        "counterfactual_steer_prob": 0.30,
        "counterfactual_stop_prob": 0.10
    },
    "val_dataset": {
        "instruction_preset": "path_type_aware",
        "counterfactual_steer_prob": 0.0,
        "counterfactual_stop_prob": 0.0
    }
}
```

검증 단계에서는 counterfactual 끔 (학습 시에만 신호 강화).

### 3.3 실행

```bash
source .venv/bin/activate
python3 robovlm_nav/train.py configs/mobile_vla_v5_exp42_counterfactual_pta.json
# 예상 ~40분 (Exp41C와 같은 8 epoch)
```

### 3.4 평가 (Phase A 동일 기준)

```bash
# attention
python3 scripts/measure_attention.py  # exp42 등록 후

# prompt sensitivity
python3 scripts/eval_prompt_sensitivity.py \
    --ckpt runs/v5_nav/kosmos/mobile_vla_v5_exp42/.../epoch_*.ckpt \
    --config configs/mobile_vla_v5_exp42_counterfactual_pta.json \
    --n-frames 30 \
    --output_json docs/v5/exp41_prompt_lockin/exp42_sensitivity.json

# PM
python3 scripts/test_v5_pm_dm.py \
    --config configs/mobile_vla_v5_exp42_counterfactual_pta.json \
    --instruction_preset path_type_aware \
    --eval_split val --eval_t 0 \
    --output_json docs/v5/pm_eval/exp42_results.json
```

### 3.5 T2 검증 기준 (Phase A re-run)

| 기준 | 목표 | 처리 |
|------|------|------|
| text attention (24L 평균) | ≥ 5% | 통과 → T3 불필요, Exp42 곧 배포 후보 |
| action L1 diff (left↔right) | ≥ 1e-2 | |
| pred change / 30 frames | > 0 | |
| PM | ≥ 50% (Exp25 baseline) | |

**모두 통과**: T3 진행 안 함, Exp42를 추가 배포 후보로.
**1~3개만 통과**: T2 변형 (counterfactual 비율 ↑) 1회 더 시도.
**모두 실패**: T3 진입.

---

## 4. Track 3 — Phase D 구조 변경 (조건부)

T2 결과가 명확히 실패할 때만 진입. 본 plan 안에 디자인까지 포함하되, 학습/구현은 T2 결과 본 뒤 사용자 승인 받고 시작.

### 4.1 디자인 후보

#### 후보 A — Cross-attention text head (추천)

`robovlm_nav/models/policy_head/nav_policy_impl.py`의 `MobileVLALSTMDecoder`에 multi-head cross-attention 추가:

```python
# 현재 (text 무시):
hidden = self.lstm(vision_pooled)
logits = self.classifier(hidden)

# 변경:
text_kv = self.text_proj(text_features)        # (B, T_text, D)
hidden = self.lstm(vision_pooled)              # (B, D)
attended = self.cross_attn(hidden.unsqueeze(1), text_kv, text_kv)  # (B, 1, D)
fused = hidden + self.gate * attended.squeeze(1)
logits = self.classifier(fused)
```

장점: text가 명시적 학습 경로로 들어감. backbone 무수정.
단점: 새 파라미터 (~200K), 학습 곡선 변동 가능.

#### 후보 B — Multiplicative gating (가벼움)

```python
text_pooled = text_features.mean(dim=1)          # (B, D)
gate = torch.sigmoid(self.text_gate(text_pooled))  # (B, D)
fused = vision_pooled * (1 + self.alpha * gate)   # element-wise
```

장점: 거의 무비용 (~D 파라미터)
단점: 정보량 작음 — text가 vision 강도만 조절

#### 후보 C — Text token concat (간단)

LSTM input에 text feature를 concat:

```python
text_pooled = text_features.mean(dim=1)
combined = torch.cat([vision_pooled, text_pooled], dim=-1)  # (B, 2D)
hidden = self.lstm(combined)
```

장점: 1줄 변경
단점: LSTM 첫 layer dim 변경 → resume 불가 (scratch 필요)

→ **추천: A**. 효과/비용 balance 가장 좋음. 실패 시 B fallback.

### 4.2 Exp43 config (예시)

T3 진행 결정 시 실제 작성. 파일명 후보: `configs/mobile_vla_v5_exp43_cross_attn_text.json`.

### 4.3 T3 검증 기준

T2와 동일 (Phase A) + 추가:
- val_loss가 Exp25 baseline 대비 회귀하지 않음 (Exp40처럼 액션 head 깨지지 않음)

T3 모두 실패 시: TICVLA / MobilityVLA 대안 검토 (별도 plan).

---

## 5. 트랙 간 의존성

```
T1.0a (Exp25 sync) ─┐
T1.0b (커밋 정리)   ─┼─► T1.1~T1.7 (배포)  ─┐
                     │                      ├─► 운영 CL 회복
T2.1 (config)       ─┴─► T2.2 (학습)       ─┤    (T1, T2 독립)
                            │
                            └─► T2.3~T2.5 (평가) ─┐
                                                   ├─► T3 진입 결정
                                              실패 시
                                                   └─► T3 (구조 변경)
```

T1과 T2는 GPU 한 대라면 직렬, 두 대(billy + minum) 있으면 병렬.

---

## 6. 수정 파일 요약 (구현 시)

| 파일 | 변경 | 단계 |
|------|------|------|
| `runs/v5_nav/kosmos/mobile_vla_v5_exp25/...` | sync from billy (~20GB) | T1.0a |
| `robovlm_nav/serve/proxy_inference_server.py` | path resolver 확정 + 커밋 | T1.0b, T1.2 |
| `robovlm_nav/trainer/nav_trainer.py` | chunking fix 커밋 | T1.0b |
| `scripts/measure_attention.py` | Exp42 등록 | T2.4 |
| `configs/mobile_vla_v5_exp42_counterfactual_pta.json` | 신규 (~15줄) | T2.1 |
| `docs/v5/bbox_nav_exp19_proxy/index.html` | 배포 섹션 갱신 | T1.7 |
| `docs/index.html` | Hero 링크 추가 | T1.7 |
| `docs/v5/exp41_prompt_lockin/exp42_*.json` | 평가 결과 | T2.4 |
| `robovlm_nav/models/policy_head/nav_policy_impl.py` | cross-attn 추가 | **T3 (조건부)** |
| `configs/mobile_vla_v5_exp43_cross_attn_text.json` | 신규 | **T3 (조건부)** |

기존 학습 코드 (`train.py`, `nav_h5_dataset_impl.py`) 수정 없음. 단 `nav_trainer.py`는 chunking fix 1줄만.

---

## 7. 일정 추정

| 단계 | 시간 |
|------|------|
| T1.0a (sync 6.7GB) | 5~30분 (네트워크 의존) |
| T1.0b (커밋 정리) | 15분 |
| T1.1~T1.4 (환경 + proxy 학습 + 서버 기동) | 1시간 |
| T1.5~T1.7 (검증 + 문서) | 1시간 |
| **T1 합계** | **~3시간** |
| T2.1 (config) | 5분 |
| T2.2 (학습 8 epoch) | ~40분 |
| T2.3~T2.5 (평가 + 결과 정리) | 30분 |
| **T2 합계** | **~1.5시간** (T1과 병렬 가능) |
| T3 (조건부) | 진입 시 별도 plan |

전체 (T3 제외): **반나절 ~ 1일**

---

## 8. 트레이드오프

### T1
- **장점**: 효과 즉시 (CL 11% → 56~67%), 검증된 자산만 사용
- **단점**: 6.7GB sync 시간, billy ssh 접근 필요
- **위험**: Exp25 ckpt가 billy에서 옮겨졌거나 손상됐으면 재학습 (~30분)

### T2
- **장점**: 1개 config 파일로 시도. 코드 무수정. 실패해도 T3 결정에 명확한 정보 제공
- **단점**: 두 번째 데이터-측 시도 (path_type_aware는 이미 실패) — 또 실패할 가능성
- **위험**: counterfactual 비율이 너무 높으면 PM이 깎일 수 있음 (학습 데이터 분포가 왜곡)

### T3
- **장점**: 구조 측 마지막 카드. 진짜 원인 분리 가능
- **단점**: 새 파라미터 + 학습 시간. PR 사이즈 큼
- **위험**: 후보 A의 cross-attn이 학습 불안정 유발 시 후보 B/C로 fallback 비용

---

## 9. 사용자 결정 필요 (구현 전 답해주세요)

1. **Exp25 ckpt sync 방법**: (a) 사용자가 billy에서 직접 push / (b) 본 세션에서 minum→billy ssh로 pull / (c) 재학습. 추천: (a)
2. **T1과 T2 동시 진행 가능한가?** GPU 사용량/머신 상황. T1은 추론만 — T2 학습이 T1 평가 GPU와 충돌 없으면 병렬 OK.
3. **T2 counterfactual 비율**: 추천 `steer=0.30, stop=0.10`. 더 공격적 / 더 보수적으로?
4. **T3 미리 디자인 코드까지 작성할지** vs T2 결과 보고 결정할지. 추천: T2 결과 후 결정 (지금 작성하면 dead code 위험)
5. **미커밋 변경 커밋 단위**: 단일 커밋 vs 기능별 분리? 추천: nav_trainer fix / proxy server / measure_attention 별도 커밋
6. **로봇 client 변경 범위**: T1.5의 endpoint 분기 — robot-local-api-handoff 스킬을 별도 세션에서 다룰지?

---

## 10. 완료 정의 (Definition of Done)

### T1 완료
- [ ] Exp25 ckpt가 minum의 `runs/v5_nav/kosmos/mobile_vla_v5_exp25/...` 에 존재
- [ ] 미커밋 변경 4개 파일 커밋됨
- [ ] proxy MLP 가중치 `exp19_proxy_mlp.pt` 존재 + test_acc 기록
- [ ] 두 서버 health 응답 OK
- [ ] `docs/index.html` Hero 영역에 신규 버튼 추가
- [ ] (선택) 로봇 closed-loop smoke 1 ep 결과 기록

### T2 완료
- [ ] `configs/mobile_vla_v5_exp42_counterfactual_pta.json` 작성
- [ ] Exp42 8 epoch 학습 완료, ckpt 존재
- [ ] `docs/v5/exp41_prompt_lockin/exp42_sensitivity.json` 생성
- [ ] `docs/v5/pm_eval/exp42_results.json` 생성
- [ ] Phase A 4 기준에 대한 verdict (PASS/PARTIAL/FAIL) 기록
- [ ] master_memory에 결과 한 줄 갱신 제안 (직접 수정 금지 규칙)

### T3 완료 (조건부)
- 별도 plan 작성 후 정의

---

## 11. 다음 단계

이 plan에 사용자 메모/주석 추가 → 반복 수정 → 승인 후에만 구현 시작.

CLAUDE.md 규칙: **승인 전까지 코드 작성 금지.** §9의 결정사항에 답을 받기 전엔 어떤 파일도 수정하지 않는다.
