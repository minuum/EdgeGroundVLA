---
name: receive-inference-session
description: soda의 sync-inference-session 스킬이 전송한 추론 세션 패키지를 수신·분석한다. "방금 세션 분석해줘", "받은 세션 확인해줘", "그라운딩 현황 어때" 같은 요청에 사용. soda→minum 페어링 스킬.
---

# receive-inference-session

soda `sync-inference-session` 스킬의 minum 수신 측 짝.  
`inference_sessions_recv/YYYYMMDD/` 패키지를 읽어 grounding 품질·cx 분포·에피소드 성공률을
자동 분석하고 이전 기준점과 비교한다.

## 트리거

- "방금 세션 분석해줘 / 받은 거 확인해줘 / 그라운딩 현황"
- soda에서 `sync-inference-session` 실행 완료 메시지가 왔을 때
- Fix 적용 전후 비교가 필요할 때

---

## Step 0. 수신 패키지 확인

```bash
RECV=/home/minum/MoNaVLA/inference_sessions_recv
ls -lt $RECV/                    # 가장 최근 날짜 디렉토리 확인
DATE=$(ls $RECV | tail -1)       # ex) 20260702
ls $RECV/$DATE/
```

패키지 구성:
| 파일 | 내용 |
|------|------|
| `session_*.h5` | 추론 세션 (images/actions/grounding) |
| `server_health.json` | 전송 시점 서버 설정 스냅샷 |
| `episode_log.csv` | 실주행 에피소드 로그 |
| `models/*.pt` | 사용된 액션 헤드 체크포인트 |
| `README.txt` | 전송 매니페스트 |

---

## Step 1. 서버 설정 스냅샷 확인

```bash
cat $RECV/$DATE/server_health.json | python3 -c "
import json, sys
h = json.load(sys.stdin)
print('head:', h['head'], 'window:', h['window'])
print('multi_prompt:', h.get('multi_prompt'), 'fallback:', h.get('fallback_prompts'))
print('cx_jump_filter:', h.get('cx_jump_filter'), 'fix1_applied:', h.get('fix1_applied', False))
print('grounder input_px:', h['grounder']['input_px'])

# Fix 3: 코드-프로세스 버전 불일치 자동 감지
git_commit  = h.get('git_commit', 'unknown')
code_mtime  = h.get('code_mtime')
started_at  = h.get('process_started_at')
print(f'git_commit: {git_commit}  process_started_at: {started_at}  code_mtime: {code_mtime}')
if code_mtime and started_at and code_mtime > started_at:
    print('⚠️ 경고: 코드 파일이 프로세스 시작 이후 수정됨 → 이 세션은 구코드(버그 미수정)로 수집됐을 가능성 있음. soda 서버 재시작 필요.')
elif not code_mtime or not started_at:
    print('⚠️ 참고: git_commit/process_started_at/code_mtime 필드 없음 — soda가 FIX3_SERVER_VERSION_HANDSHAKE.md 미반영 상태. 버전 검증 불가.')
"
```

**핵심 플래그:**
- `grounder.input_px` — 448이면 Fix1 미적용(processor가 224→448 업스케일), 없으면 확인 불가
- `multi_prompt` — True면 fallback_prompts 순환 탐지 중
- `cx_jump_filter` — True면 급변 필터 활성
- `code_mtime > process_started_at` — **코드는 고쳐졌지만 서버가 재시작 안 된 상태**. 이 세션들은 구 코드로 수집됨(2026-07-02 Fix1 사고와 동일 패턴). 관련: `FIX3_SERVER_VERSION_HANDSHAKE.md`

---

## Step 2. 세션별 grounding 통계

```python
import h5py, numpy as np, json
from pathlib import Path

RECV = Path("/home/minum/MoNaVLA/inference_sessions_recv")
DATE = sorted(RECV.iterdir())[-1]   # 가장 최근 날짜

sessions = sorted(DATE.glob("session_*.h5"))
print(f"{'세션':38s} {'N':>3} {'LIVE':>5} {'CACHE':>5} {'NONE':>5} {'has%':>5} {'live_cx':>7} {'live_cx_std':>11}")

for f in sessions:
    with h5py.File(str(f), 'r') as h:
        n = h['observations/images'].shape[0]
        if n < 3: print(f"{f.name:38s} {n:3d}  (skip)"); continue
        cx      = h['grounding/bbox'][:, 0]
        hasbbox = h['grounding/bbox'][:, 3]
        cached  = h['grounding/cached'][:]
        live_cx = cx[cached == 0.0]
        print(f"{f.name:38s} {n:3d} {int((cached==0).sum()):5d} {int((cached==1).sum()):5d} "
              f"{int((cached==-1).sum()):5d} {hasbbox.mean()*100:5.1f} "
              f"{live_cx.mean():7.3f} {live_cx.std():11.4f}")
```

**읽는 법:**
- `has%` → 탐지율. < 50%이면 그라운딩 불량
- `live_cx` → LIVE 탐지 평균. 0.50±0.02 범위 고착이면 center anchoring (Fix1 미적용 특징)
- `live_cx_std` → **핵심 지표.** Fix1 미적용 시 ≈ 0.000~0.020, 적용 후 ≥ 0.050 기대

---

## Step 3. PG448 학습 분포 vs 실주행 LIVE cx 비교

```python
import json, numpy as np
from pathlib import Path

ann = json.load(open("/home/minum/26CS/MoNaVLA/docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json"))
cx_train = [fr["cx_det"] for ep in ann for fr in ep["frames"] if fr.get("has_bbox")]
cx_t = np.array(cx_train)

print(f"[학습 PG448 annotation] mean={cx_t.mean():.3f} std={cx_t.std():.3f}")
print(f"  cx>0.70: {(cx_t>0.70).mean()*100:.1f}%  cx<0.30: {(cx_t<0.30).mean()*100:.1f}%")

# 실주행 LIVE cx (위 Step2에서 추출)
# live_all = [ 세션별 live_cx 값들 ]
# print(f"[실주행 LIVE cx] mean={np.mean(live_all):.3f} std={np.std(live_all):.4f}")
# print(f"  std 격차: {cx_t.std()/np.std(live_all):.1f}배 → 클수록 center anchoring 심각")
```

**Fix1 효과 판단 기준:**
| 지표 | Fix1 미적용 | Fix1 적용 후 기대 |
|------|------------|----------------|
| live_cx std (세션 평균) | 0.016 | ≥ 0.050 |
| obj_* 성공률 | 33.3% | 목표 50%+ |
| LIVE area mean | 0.08~0.11 | 증가 (더 정확한 bbox) |

---

## Step 4. episode_log 성공률 집계

```python
import csv
from pathlib import Path

rows = list(csv.DictReader(open(PATH / "episode_log.csv")))
for prefix in ["right_left", "center_straight", "obj_left", "obj_center", "obj_right"]:
    eps = [r for r in rows if r["경로"].startswith(prefix)]
    if not eps: continue
    ok = sum(1 for r in eps if r["결과"] == "성공")
    print(f"{prefix:18s}: {ok}/{len(eps)} = {ok/len(eps)*100:.1f}%  "
          f"area=0 비율: {sum(1 for r in eps if float(r['area'])==0.0)/len(eps)*100:.0f}%")
```

**area=0 비율 → 그라운딩 완전 실패(has_bbox=False) 에피소드 비율**

---

## Step 5. Fix 적용 전후 비교 기준점 관리

`docs/v5/grounding_analysis/fix_comparison_baseline.json` 에 현재 수치 저장:
```json
{
  "timestamp": "YYYY-MM-DD",
  "fix1_applied": false,
  "live_cx_std_mean": 0.016,
  "obj_success_rate": 0.333,
  "has_bbox_pct_mean": 61.2,
  "sessions": "20260702 9개"
}
```

Fix1 적용 후 재측정 → 동일 파일에 `fix1_applied: true` 항목 추가해 비교.

---

## 관련 문서

- `docs/v5/grounding_analysis/grounding_center_bias_analysis.md` — P1/P2/P3 원인 분석
- `docs/v5/grounding_analysis/FIX1_RESIZE_BUG.md` — Fix1 수정 가이드 (soda 적용 대상)
- `docs/v5/grounding_analysis/FIX_GUIDE.md` — Fix2/3 가이드
- `docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json` — 학습 annotation cx 분포
