# Plan: 추론 플로우 / 서버 grounding 정합성 수정 (2026-07-02)

> 상태: **Fix A + B 구현·검증 완료 (2026-07-02). Fix C 보류(재현 주행 후 결정), Fix D 모니터링만**
> 작성 근거: 코드 대조(train_exp71 vs inference_server) + 라이브 서버 로그/리소스 실측

---

## 배경 / 발견된 문제 4종

학습(`train_exp71_stage2_transformer.py`)과 추론(`stage2_v2_inference_server.py`)을
코드 레벨로 대조하고, 라이브 서버 로그·GPU를 실측한 결과 4가지 문제 확인.

| # | 심각도 | 문제 | 근거 |
|---|--------|------|------|
| **A** | 🔴 치명 (라이브) | `/predict`가 preview ROT마다 **500 KeyError 'source'** | `server.py:1092` reads `result["source"]`, preview dict(`750-768`)엔 없음. 로그 다수 재현 |
| **B** | 🟠 높음 | 추론 grounding이 **224로 다운스케일** 후 PG2-448 입력 (학습은 네이티브 448) | `PG2Grounder.run` → `resize_for_vlm(pil)` (`VLM_INPUT_SIZE=224`) vs `gen_pg448_annotation.py:39` 무리사이즈 |
| **C** | 🟡 중간 | skip_n=3 → W=6 윈도우에 **stale bbox 3연속 복제** (학습은 프레임마다 fresh) | `server.py:777-792` vs `train_exp71:114-119` |
| **D** | 🟢 정보 | Orin RAM 13.7/15.6GB(87%), swap 982MB — 빡빡하나 OOM 아님 | `tegrastats` 실측 |

- Padding(edge-replication)은 train/infer **동일** → 문제 아님 (검증 완료).
- vis_feat는 추론에서도 매 프레임 fresh → temporal 주축은 온전.

---

## 수정 방침 (우선순위 순)

### Fix A — preview 응답 `source` 키 누락 (즉시)
**문제:** 바스켓 미검출 시 preview ROT 반환 dict에 `source` 없음 → 500.
**변경:** `server.py` preview 반환 dict(라인 750 근처)에 `"source": "stage2_v2"` 추가.
```python
# server.py ~750, preview 반환 dict
return {
    "action": ACTION_2D[preview_rot],
    ...
    "buffer_status": {...},
    "source": "stage2_v2",   # ← 추가
}
```
**대안:** 엔드포인트에서 `result.get("source", "stage2_v2")`로 방어 (line 1092).
→ 둘 다 적용 권장 (근본 + 방어).
**리스크:** 없음. 트레이드오프 없음.
**검증:** preview 강제 유발(바스켓 치우고 START) → 500 안 뜨고 ROT 정상 반환.

---

### Fix B — grounding 입력 해상도 224 → 448 정합
**문제:** 추론만 224로 줄여 PG2-448에 넣음 → 검출 품질 저하(full-frame 환각).
**핵심 결정 (사용자 판단 필요):**
- **B-1 (권장):** `PG2Grounder.run`에서 `resize_for_vlm(pil)` 호출 **제거** →
  PaliGemmaProcessor가 원본을 네이티브 448로 처리 (학습 annotation과 동일 경로).
  ```python
  # server.py PG2Grounder.run
  pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
  # pil = resize_for_vlm(pil)   # ← 제거 (Processor가 448 자동 처리)
  inp = self._proc(text=f"detect {phrase}", images=pil, ...)
  ```
- **B-2:** `resize_for_vlm(pil, size=448)` 명시 (Kosmos-2 encoder 경로와 분리 관리).

**주의:** Kosmos-2 vision encoder(`enc.encode_image`)는 **별개**로 224 유지해야 함
(Stage1은 224로 학습됨). B 수정은 **PG2 grounding 경로에만** 적용, Stage1 건드리지 말 것.
**리스크:** PG2 448 native는 224 대비 연산량↑ → 레이턴시 소폭 증가 예상. Orin 메모리(C) 확인 필요.
**검증:** 동일 프레임으로 224 vs 448 grounding cx/area 비교, full-frame 환각 감소 확인.

---

### Fix C — skip_n stale bbox mismatch
**문제:** 학습은 프레임마다 fresh bbox, 추론은 3연속 복제.
**옵션 (사용자 선택):**
- **C-1 (가장 단순):** `VLA_GROUNDING_SKIP_N=1` — 매 스텝 fresh grounding.
  분포 완전 일치. 대가: 스텝당 grounding 레이턴시(웜 ~350ms). go.sh env만 변경.
- **C-2:** skip_n 유지하되 캐시 대신 **직전 fresh bbox + 현재 action으로 cx 보간** —
  얼어붙는 대신 이동 방향만큼 bbox를 밀어줌. 복잡. 정확도 이득 불확실.
- **C-3 (근본, minum 서버):** skip_n=3 stale 패턴을 학습 augmentation에 주입해
  모델을 robust하게 재학습. 로봇 서버 범위 밖 → minum 분석 후 결정.

**참고:** `server.py:519` CH49 주석 "skip_n=3 SR/FPE 변화 없음 확정" —
이미 성능 무영향 검증됨. Fix B로 검출 품질 올라가면 C 필요성 재평가.
**권장 순서:** Fix A+B 먼저 → 재현 주행 → 그래도 mismatch 영향 보이면 C-1.

---

### Fix D — Orin 리소스 (모니터링만)
- 현재 OOM 아님. 조치 불필요.
- Fix B로 448 native 시 메모리↑ 가능 → 적용 후 `tegrastats` 재확인.
- 필요 시 PG2 bf16 유지(현행), dashboard/hub 동시 실행 부담 점검.

---

## 수정 대상 파일

| 파일 | Fix | 변경 |
|------|-----|------|
| `robovlm_nav/serve/stage2_v2_inference_server.py` | A | preview dict에 `source` 추가 + 엔드포인트 `.get` 방어 |
| `robovlm_nav/serve/stage2_v2_inference_server.py` | B | `PG2Grounder.run` resize 경로 수정 (Stage1 미변경) |
| `scripts/run/go.sh` | C-1 | `VLA_GROUNDING_SKIP_N` env (선택 시) |

---

## 실행 순서 (승인 후)

1. **Fix A** 적용 → 서버 재시작 → preview 500 사라짐 확인 (5분)
2. **Fix B-1** 적용 → 224 vs 448 grounding 비교 스크립트 → full-frame 환각 감소 확인
3. 재현 주행 1~2회 (obj_center) → episode_log 기록
4. 여전히 mismatch 영향 시 **Fix C-1** (skip_n=1) 토글 후 재비교
5. 결과를 minum 전송 데이터에 첨부 (C-3 재학습 판단용)

---

## 미결정 (사용자 판단 필요)

- [ ] Fix B: B-1(resize 제거) vs B-2(size=448 명시) — 어느 쪽?
- [ ] Fix C: 지금 skip_n=1 갈지, Fix A+B 먼저 보고 결정할지
- [ ] Fix B 후 레이턴시 증가 허용 범위 (스텝당 목표 ms)
