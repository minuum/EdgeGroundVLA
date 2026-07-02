# soda 작업 요청: flicker 원인 규명용 로깅 구현 + 재주행 (2026-07-02)

**보낸 쪽:** minum
**목적:** PG2 has_bbox flicker(동일 구도인데 탐지 성공/실패 반복)의 원인을
"생성 실패(no-locs)" vs "필터 걸림" 으로 확정하기 위한 데이터 확보.

---

## 배경 요약 (minum 분석 결과)

- session_20260702_212648.h5의 t9/t12/t15는 **거의 동일한 장면**인데
  t12만 has=True, t9/t15는 has=False로 기록됨.
- minum 로컬(GB10, torch 2.11/transformers 4.49)에서 **같은 H5 프레임을 재실행하면
  셋 다 정상 탐지 + 필터 전부 통과** → 필터 로직 버그 아님.
- soda(Orin, torch 2.3.0/transformers 4.45.2)의 실제 생성 결과가 로컬과 달랐다는
  뜻인데, 해당 시각 s2v2_server.log가 로테이션으로 소실되어 확인 불가.
- grounding latency가 2.1s(단일 호출)와 8s(멀티프롬프트 4회) 2단 분포 —
  **2.1s인데 has=False인 케이스**가 핵심 미스터리.

## 작업 순서

### 1. Fix3 구현 — 서버 버전 핸드셰이크
`docs/v5/grounding_analysis/FIX3_SERVER_VERSION_HANDSHAKE.md` 참조.
`/health` 응답에 `git_commit` / `process_started_at` / `code_mtime` 3필드 추가.

### 2. Fix4 구현 — PG2 판정 영구 로그
`docs/v5/grounding_analysis/FIX4_PG2_DECISION_LOG.md` 참조 (오늘 확장판 반영).
핵심 포인트:
- `PG2Grounder.run()` 반환 직전에 `logs/grounding_decisions.jsonl` append
- **filter_reason 필드 필수**: no-locs / tiny / top / full-frame / x-full / None(통과)
- **호출 1회당 latency_ms** (predict 레벨 합산 말고)
- raw locs 좌표도 필터 전 값으로 보존

### 3. 서버 재시작 (필수)
```bash
bash scripts/run/stop.sh && bash scripts/run/go.sh
curl -s localhost:8001/health -H "X-API-Key: vla-secret-key-2025" | python3 -m json.tool
# git_commit / process_started_at / code_mtime 3필드 확인 후 진행
```

### 4. 재주행 — obj_left / obj_center / obj_right 각 2회 이상
- 이전 실패 조건 재현: 바스켓을 좌/중/우로 배치
- 특히 **로봇 정지 상태에서 같은 장면을 여러 스텝 보는 구간**이 있으면 좋음
  (flicker가 정지 상태에서도 나는지 = 모션 무관 확인)

### 5. sync-inference-session으로 전송
기존 패키지 + **`grounding_decisions.jsonl`의 오늘 구간** 동봉:
```bash
grep "\"ts\": \"$(date +%Y-%m-%d)" ~/MoNaVLA/logs/grounding_decisions.jsonl \
  > /tmp/grounding_decisions_$(date +%Y%m%d).jsonl
```

## minum이 받은 뒤 할 분석

1. has=False 프레임의 filter_reason 분포 → 원인 확정
2. no-locs 케이스의 H5 프레임을 로컬 PG2로 재실행 → raw_output 문자열 diff
   (Jetson 수치 불안정 가설 최종 검증)
3. 결과에 따라 Phase 3 분기: 필터 완화 / 버전 업그레이드 실험 / temporal smoothing 강화

---

*관련: FIX1_RESIZE_BUG.md, FIX3_SERVER_VERSION_HANDSHAKE.md, FIX4_PG2_DECISION_LOG.md*
