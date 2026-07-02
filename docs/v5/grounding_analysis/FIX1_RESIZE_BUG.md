# Fix 1: PG2Grounder `resize_for_vlm` 이중 리사이즈 버그

**우선순위:** ★★★ (그라운딩 튀는 현상 주 원인)  
**작업 위치:** `soda:~/MoNaVLA/robovlm_nav/serve/stage2_v2_inference_server.py`  
**작업 난이도:** 1줄 제거 + 1줄 제거 (총 2줄)  
**검증:** 서버 재시작 → `/ground` 엔드포인트 테스트

---

## 버그 요약

PG2Grounder가 이미지를 PaliGemmaProcessor에 넘기기 전에 `resize_for_vlm(224)`를 먼저 적용함.  
PaliGemmaProcessor는 이미 내부에서 448×448로 리사이즈하므로 결과적으로:

```
실주행 추론: 720×1280 → 224×224 (다운) → 448×448 (업스케일, 흐림) ← 버그
학습 annotation: 720×1280 →                  448×448 (다운스케일, 선명) ← 정상
```

`resize_for_vlm`은 Kosmos-2(224 입력 전용) 용으로 만들어진 함수이며,  
PG448에는 적용해서는 안 됨.

**근거:**
```json
// ~/.cache/huggingface/hub/.../paligemma2-3b-mix-448/preprocessor_config.json
{
  "do_resize": true,
  "size": {"height": 448, "width": 448},
  "resample": 3
}
```
PG448 processor는 입력 해상도와 무관하게 항상 448×448로 리사이즈함.

---

## 수정 대상 파일

`robovlm_nav/serve/stage2_v2_inference_server.py`

파일 내 `resize_for_vlm` 호출 위치 4곳:

| 줄 | 위치 | 처리 |
|----|------|------|
| 322 | `Stage1Encoder.encode_image()` | **유지** — Kosmos-2용, 224 필요 |
| 365 | `Grounder.run()` (Kosmos-2 fallback) | **유지** — Kosmos-2용, 224 필요 |
| **443** | **`PG2Grounder.run()`** | **제거** ← 버그 위치 1 |
| **1201** | **`/ground` debug endpoint** | **제거** ← 버그 위치 2 |

---

## 수정 내용

### 수정 1: `PG2Grounder.run()` (line 443 근처)

```python
# BEFORE (버그)
    def run(self, image_rgb: np.ndarray, _unused_path: Optional[Path] = None,
            return_raw: bool = False, phrase: str = "gray basket",
            return_hidden: bool = False) -> dict[str, Any]:
        ...
        self._ensure_loaded()
        pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
        pil = resize_for_vlm(pil)          # ← 이 줄 제거
        inp = self._proc(text=f"detect {phrase}", images=pil, return_tensors="pt").to(self._device)
```

```python
# AFTER (수정)
    def run(self, image_rgb: np.ndarray, _unused_path: Optional[Path] = None,
            return_raw: bool = False, phrase: str = "gray basket",
            return_hidden: bool = False) -> dict[str, Any]:
        ...
        self._ensure_loaded()
        pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
        # resize_for_vlm 호출 제거 — PG448 processor가 내부에서 448×448로 처리
        inp = self._proc(text=f"detect {phrase}", images=pil, return_tensors="pt").to(self._device)
```

### 수정 2: `/ground` debug endpoint (line 1201 근처)

```python
# BEFORE (버그)
    image_rgb = m._decode_image(request.image)
    pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    pil = resize_for_vlm(pil)          # ← 이 줄 제거
    prompt = request.prompt or "detect gray basket"
    inp = grounder._proc(text=prompt, images=pil, return_tensors="pt").to(grounder._device)
```

```python
# AFTER (수정)
    image_rgb = m._decode_image(request.image)
    pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    # resize_for_vlm 호출 제거 — PG448 processor가 내부에서 448×448로 처리
    prompt = request.prompt or "detect gray basket"
    inp = grounder._proc(text=prompt, images=pil, return_tensors="pt").to(grounder._device)
```

---

## 서버 반영 방법

```bash
# soda 서버에서
cd ~/MoNaVLA

# 현재 서버 종료
bash scripts/run/stop.sh   # 또는 kill $(lsof -t -i:8001)

# 파일 수정 (위 내용 참고)
nano robovlm_nav/serve/stage2_v2_inference_server.py

# 서버 재시작
bash scripts/run/go.sh
```

---

## 검증 방법

서버 재시작 후 `/ground` 엔드포인트 테스트:

```bash
# 바스켓이 우측에 있는 실제 이미지로 테스트
# 수정 전: cx ≈ 0.51 (center false positive)
# 수정 후: cx ≈ 0.80+ (실제 바스켓 위치)

python3 - <<'EOF'
import base64, requests, numpy as np
from PIL import Image
import io

# 테스트용 이미지 — 실 카메라 프레임 or session H5에서 추출
img = Image.open("inference_sessions_recv/20260702/session_20260702_100143_t7.jpg").convert("RGB")
buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

r = requests.post("http://localhost:8001/ground",
    json={"image": b64, "prompt": "detect gray basket"},
    headers={"X-API-Key": "vla-secret-key-2025"}, timeout=30)
print(r.json())
# 기대: cx > 0.70 (바스켓이 실제로 우측에 있는 프레임)
EOF
```

또는 Gradio 대시보드에서:
- 바스켓을 카메라 우측 끝에 배치
- 수정 전: bbox가 중앙에 그려짐
- 수정 후: bbox가 실제 바스켓 위치에 그려짐

---

## 부수 효과 없음 확인

- `Stage1Encoder.encode_image()` (Kosmos-2): 수정 없음 → 기존 동작 유지
- `Grounder.run()` (Kosmos-2 fallback): 수정 없음 → 기존 동작 유지
- `PG2Grounder._ensure_loaded()`: 수정 없음
- grounding 필터 (area, cy, x-width): 수정 없음
- skip_n 캐시 정책: 수정 없음
- Stage2 action head: 수정 없음

---

## 관련 분석 문서

- `docs/v5/grounding_analysis/grounding_center_bias_analysis.md` — 전체 center bias 원인 분석
- `docs/v5/grounding_analysis/FIX_GUIDE.md` — P1/P2/P3 전체 수정 가이드
- `inference_sessions_recv/20260702/` — 분석에 사용된 실로봇 세션 8개

---

*작성: 2026-07-02 | 근거: gen_pg448_annotation.py vs PG2Grounder.run() 전수 비교*
