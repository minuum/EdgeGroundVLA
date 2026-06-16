# Plan — BBox Proxy 배포 (Exp19 → 로봇 서버)

작성: 2026-05-01
브랜치: inference-integration

## 0. 목표

Decomposition baseline (Exp19, PM 75.9% / CL 66.7%)을 로봇 서버에서 실제 호출 가능하게 만들기. 현재 로봇은 Exp17/18 (CL 11%)로 가동 중 → BBox proxy 경로 활성화 시 약 0% → 67% 점프 기대.

end-to-end 학습은 collapse 미해결(Exp35/36/38 PM 53.5% 100% FORWARD 확정), 그래서 디코mpos 경로를 메인 운영선으로 격상.

## 1. 현재 상태 (research summary)

| 자산 | 위치 | 상태 |
|---|---|---|
| Proxy 서버 코드 | `robovlm_nav/serve/proxy_inference_server.py` | **660줄, FastAPI + Grounding + ProxyMLP + history 다 있음**. 별도 앱. |
| 메인 서버 | `robovlm_nav/serve/inference_server.py` | 1722줄. 2D action 전용 (`InferenceRequest.strategy`만 chunk_reuse/receding_horizon). |
| 학습 가중치 | `docs/v5/bbox_nav_exp19_proxy/exp19_proxy_mlp.pt` | **존재 안 함**. 첫 실행 시 220 epoch 자동 학습. |
| BBox 데이터 (45ep) | `docs/v5/bbox_nav_step1/bbox_dataset.json` | 794 frames, MLP PM 75.9% |
| BBox 데이터 (150ep full) | `docs/v5/bbox_nav_step1/bbox_dataset_full.json` | 2626 frames, MLP PM 74.5% (오늘 새로 추출) |
| 데이터 hardcoded | `proxy_inference_server.py:56` `ROS_action/mobile_vla_dataset_v5` | billy 경로. minum에서 동작 안 함 |
| Python 의존성 | `.venv` (minum) | torch/lightning은 OK. **fastapi/uvicorn 미설치** |

## 2. 결정 사항 (사용자 검토용)

### Q1. 별도 서버 vs 메인 통합

| 옵션 | 변경량 | 클라이언트 | 운영 |
|---|---|---|---|
| **A. 메인 inference_server.py에 strategy="p1roxy_bbox" 추가** | 大 (1722줄 파일에 ProxyMLP/Grounding 두 번째 모델 로드 + 라우팅) | 한 endpoint, 단순 | 단일 서버, 두 모델 GPU 동시 거주 (메모리↑) |
| **B. proxy_inference_server.py 그대로 별도 포트** | 小 (path resolver만 추가) | 두 endpoint 중 선택 | 두 서버 동시 가동 또는 교체 |
| **C. proxy_inference_server.py를 메인으로 격상, inference_server.py는 deprecate** | 中 (deprecate 표시 + README 업데이트) | proxy로 일원화 | end-to-end 경로 잠금. 향후 end-to-end 재시도 시 복구 비용 |

**추천: B**. 메인 서버는 그대로 두고 proxy를 별 포트로 운영. 로봇 클라이언트 코드에서 endpoint URL만 바꿔 전환. 변경 최소, 양 모델 비교 가능.

### Q2. 데이터셋 (45ep vs 150ep full)

| 옵션 | PM | 첫 학습 시간 | 일반화 |
|---|---|---|---|
| **a. 45ep (`bbox_dataset.json`)** | 75.9% | ~30초 (220ep × 794 frames) | 학습 데이터 좁음 |
| **b. 150ep full (`bbox_dataset_full.json`)** | 74.5% | ~2분 추정 | 일반화↑ (3배 frames) |

**추천: b (full)**. PM 차이 1.4%p로 작고, full이 unseen 환경에서 generalization 더 좋을 가능성.

### Q3. billy ↔ minum 호환

평가 스크립트(`test_v5_pm_dm.py`)에 했던 `_resolve_data_dir` 로직을 proxy server에도 적용. 환경변수 `VLA_PROXY_DATA_DIR` 추가도 함께.

## 3. 구현 단계

### Step 1. 환경 추가 (minum)

```bash
source .venv/bin/activate
uv pip install fastapi uvicorn[standard] python-multipart
```

### Step 2. proxy_inference_server.py path resolver

`DATA_DIR` 결정을 환경변수 + billy/minum 자동 변환으로:

```python
# 변경: line 56
_DATA_PATH_CANDIDATES = [
    Path("/home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v5"),
    Path("/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5"),
    ROOT / "ROS_action" / "mobile_vla_dataset_v5",
]

def _resolve_data_dir() -> Path:
    override = os.getenv("VLA_PROXY_DATA_DIR")
    if override:
        return Path(override)
    for cand in _DATA_PATH_CANDIDATES:
        if cand.exists():
            return cand
    return _DATA_PATH_CANDIDATES[-1]

DATA_DIR = _resolve_data_dir()
```

### Step 3. 데이터셋 파일 선택 (env 변수)

이미 `VLA_PROXY_DATASET_FILE` 환경변수 지원하므로 추가 코드 0. 실행 시:

```bash
export VLA_PROXY_DATASET_FILE="$PWD/docs/v5/bbox_nav_step1/bbox_dataset_full.json"
```

### Step 4. 가중치 학습 (한 번)

서버 첫 실행 시 자동으로 학습되고 저장됨. 명시적 사전학습:

```bash
source .venv/bin/activate
export VLA_PROXY_DATASET_FILE="$PWD/docs/v5/bbox_nav_step1/bbox_dataset_full.json"
export VLA_PROXY_FORCE_RETRAIN=true
python3 robovlm_nav/serve/proxy_inference_server.py --port 8001 &
# 모델 학습 + 저장 후 로그에서 test_acc 확인 후 kill
```

### Step 5. 로봇 클라이언트에서 호출 검증

기존 `inference_server.py`(8000)와 새 proxy(8001)를 둘 다 띄우고 동일 이미지로 호출, 응답 비교.

```bash
# Health check
curl http://minum-host:8001/health

# Predict (b64 이미지)
curl -X POST http://minum-host:8001/predict \
  -H "X-API-Key: $VLA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"image":"<b64>","instruction":"navigate to gray basket"}'
```

### Step 6. 문서화 + Hero 링크

- `docs/v5/bbox_nav_exp19_proxy/index.html`에 “로봇 서버 배포 방법” 섹션 추가
- `docs/index.html` Hero 영역에 “Exp19 Proxy 서버 배포 가이드” 버튼 추가 (CLAUDE.md 규칙)

## 4. 수정 파일

| 파일 | 변경 |
|---|---|
| `robovlm_nav/serve/proxy_inference_server.py` | DATA_DIR resolver 추가 (~10줄) |
| `.venv` (minum) | `uv pip install fastapi uvicorn[standard] python-multipart` |
| `docs/v5/bbox_nav_exp19_proxy/exp19_proxy_mlp.pt` | 신규 생성 (학습 결과물) |
| `docs/v5/bbox_nav_exp19_proxy/index.html` | 배포 섹션 추가 |
| `docs/index.html` | Hero 버튼 추가 |

`inference_server.py`는 **건드리지 않음** (옵션 B).

## 5. 검증 기준

1. `curl /health` → `model_loaded: true`, `proxy_info.test_acc >= 0.70` (full dataset 학습 후)
2. `curl /predict` → `predicted_label`이 STOP 외의 클래스로 다양하게 나오는지 (한 클래스 collapse 아닌지)
3. 동일 이미지 5장에 대해 메인 서버(Exp17/18)와 응답 비교 → proxy가 left/right scene에서 LEFT/RIGHT 예측 잘 하는지
4. 로봇 실로봇 closed-loop 1~2 에피소드 시도 (별도 후속 작업)

## 6. 트레이드오프

- **장점**: end-to-end collapse 우회, PM 75% / CL 67% 검증된 baseline 활성화. 변경 최소.
- **단점**: Grounding(Pure HF Kosmos-2 generate) latency 추가 — 첫 inference에서 200~500ms 추정. proxy MLP 자체는 1ms 수준.
- **미커버**: full 150ep로 학습한 proxy MLP의 closed-loop 평가는 plan 범위 밖 (별도 후속).
- **위험**: proxy_inference_server.py 작성자가 가정한 데이터 포맷이 production 이미지와 다를 가능성 — Step 5에서 검증.

## 7. 일정 추정

- Step 1~4: 30분 (환경 설치 + 코드 10줄 + 학습 2분)
- Step 5: 15분 (smoke test)
- Step 6: 20분 (문서)
- 총 ~1시간 + 사용자 검증

## 8. 미해결 / 사용자 확인 요청

1. **옵션 A/B/C 중 어느 것** (추천 B)
2. **데이터셋 a/b 중 어느 것** (추천 b, 150ep full)
3. **로봇 서버에 띄울 곳**: minum vs billy? (현재 GPU 점유율 96%인 곳이 minum)
4. **포트**: 메인 8000과 별도면 8001? 다른 숫자?
