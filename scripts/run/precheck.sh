#!/usr/bin/env bash
# mona-precheck — 주행 세트 시작 전 ~10초 점검 (plan_20260702_healthcheck_routine.md §2)
#
# 사용법: bash scripts/run/precheck.sh
# 종료코드: 0=모두 통과, 1=하나라도 실패
#
# 점검 항목:
#   ① 서비스 응답 (추론 8001 / 대시보드 7800) + 프로세스 신선도(PID/가동시간)
#   ② 카메라 신선도 (camera_ok + frame_age_s)
#   ③ API 키 일치 (서버 vs 대시보드 — 403 연쇄의 전조)
#   ④ 런타임 설정 현재값 (의도 확인용 출력)
#   ⑤ 리소스 (메모리/디스크)

set -uo pipefail
cd "${VLA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

FAIL=0
ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
bad()  { printf "  \033[31m✗\033[0m %s\n" "$1"; FAIL=1; }
info() { printf "    %s\n" "$1"; }

echo "━━ mona-precheck ($(date '+%H:%M:%S')) ━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── ① 서비스 응답 ─────────────────────────────────────────────────────
INFER_H=$(curl -sf -m 3 http://localhost:8001/health 2>/dev/null)
if [[ -n "$INFER_H" ]]; then
    ok "추론 서버 (8001)"
else
    bad "추론 서버 (8001) 응답 없음 — go.sh --server"
fi

DASH_H=$(curl -sf -m 3 http://localhost:7800/health 2>/dev/null)
if [[ -n "$DASH_H" ]]; then
    PID=$(echo "$DASH_H" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('pid',''))" 2>/dev/null)
    UP=$(echo "$DASH_H"  | python3 -c "import sys,json;d=json.load(sys.stdin);print(round(d.get('uptime_s',0)))" 2>/dev/null)
    ok "대시보드 (7800)  PID=$PID  uptime=${UP}s"
    # 좀비 감시: 같은 스크립트 프로세스가 2개 이상이면 경고
    NPROC=$(pgrep -fc "robovlm_nav/serve/mona_dashboard.py" 2>/dev/null || echo 0)
    if [[ "$NPROC" -gt 1 ]]; then
        bad "대시보드 프로세스 ${NPROC}개 감지 — 좀비 의심: pgrep -f mona_dashboard.py"
    fi
else
    bad "대시보드 (7800) 응답 없음 — go.sh --mona-dash"
fi

# ── ② 카메라 신선도 ───────────────────────────────────────────────────
if [[ -n "$DASH_H" ]]; then
    CAM=$(echo "$DASH_H" | python3 -c "
import sys,json; d=json.load(sys.stdin)
ok = d.get('camera_ok'); age = d.get('frame_age_s')
print(f'{ok}|{age}')" 2>/dev/null)
    CAM_OK="${CAM%%|*}"; CAM_AGE="${CAM##*|}"
    if [[ "$CAM_OK" == "True" ]]; then
        ok "카메라 (frame_age=${CAM_AGE}s)"
    else
        bad "카메라 멈춤/미수신 (camera_ok=$CAM_OK, age=${CAM_AGE}s) — 대시보드 '카메라 프로세스 ▶시작' 버튼 또는 /camera_proc/start"
    fi
fi

# ── ③ API 키 일치 ─────────────────────────────────────────────────────
KEYS=()
for pat in "stage2_v2_inference_server" "robovlm_nav/serve/mona_dashboard.py"; do
    P=$(pgrep -f "$pat" | head -1)
    if [[ -n "$P" ]]; then
        K=$(tr '\0' '\n' < "/proc/$P/environ" 2>/dev/null | grep "^VLA_API_KEY=" | cut -d= -f2)
        KEYS+=("${K:-<없음>}")
    fi
done
if [[ ${#KEYS[@]} -eq 2 ]]; then
    if [[ "${KEYS[0]}" == "${KEYS[1]}" ]]; then
        ok "API 키 일치 (${KEYS[0]:0:12}...)"
    else
        bad "API 키 불일치! 서버=${KEYS[0]:0:12}... 대시보드=${KEYS[1]:0:12}... → 403 연쇄로 STOP만 나감. 같은 셸에서 두 서비스 재시작 필요"
    fi
fi

# ── ④ 런타임 설정 현재값 ──────────────────────────────────────────────
if [[ -n "$INFER_H" ]]; then
    echo "  ─ 런타임 설정 (의도와 맞는지 눈으로 확인) ─"
    echo "$INFER_H" | python3 -c "
import sys,json; d=json.load(sys.stdin)
p = d.get('preview') or {}
print(f\"    preview={p.get('enabled')} hint_cx={p.get('hint_cx')} \"
      f\"skip_n={d.get('grounding_skip_n')} multi_prompt={d.get('multi_prompt')} \"
      f\"cx_jump={d.get('cx_jump_filter')}({d.get('cx_jump_thresh')}) \"
      f\"stop_mode={d.get('stop_mode')} head={d.get('head')}\")" 2>/dev/null
fi

# ── ⑤ 리소스 ─────────────────────────────────────────────────────────
MEM_AVAIL=$(free -m | awk '/^Mem:/{print $7}')
DISK_PCT=$(df /home --output=pcent | tail -1 | tr -d ' %')
if [[ "$MEM_AVAIL" -lt 800 ]]; then
    bad "가용 메모리 ${MEM_AVAIL}MB (<800MB) — OOM 위험"
else
    ok "메모리 가용 ${MEM_AVAIL}MB"
fi
if [[ "$DISK_PCT" -gt 90 ]]; then
    bad "디스크 ${DISK_PCT}% 사용 — 세션/로그 정리 필요 (주간 루틴 §4)"
else
    ok "디스크 ${DISK_PCT}% 사용"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ "$FAIL" -eq 0 ]]; then
    echo "  ✅ 전부 통과 — 주행 시작 가능"
else
    echo "  ⛔ 실패 항목 있음 — 위 ✗ 항목 해결 후 주행"
fi
exit "$FAIL"
