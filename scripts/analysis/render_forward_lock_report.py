import json
from pathlib import Path

ROOT = Path("/home/minum/26CS/MoNaVLA")
OUT_DIR = ROOT / "docs/v5/analysis_reports"
data = json.loads((OUT_DIR / "forward_lock_data_20260712.json").read_text(encoding="utf-8"))

success = [r for r in data if r["result"] == "성공"]
fail = [r for r in data if r["result"] == "실패"]


def card(r):
    badge = "OK" if r["result"] == "성공" else "FAIL"
    color = "#34d399" if r["result"] == "성공" else "#f87171"
    det_img = f'<img src="frames_20260712/{r["det_fn"]}" loading="lazy">' if r["det_fn"] else '<div class="noimg">미탐지</div>'
    nonfwd = ", ".join(str(x) for x in r["nonfwd_frames"]) or "없음"
    return f"""
    <div class="card" style="border-color:{color}">
      <div class="card-head">
        <span class="sid">{r["sid"]}</span>
        <span class="badge" style="background:{color}">{badge}</span>
      </div>
      <div class="imgs">
        <div><img src="frames_20260712/{r["f0_fn"]}" loading="lazy"><div class="cap">frame 0 (시작)</div></div>
        <div>{det_img}<div class="cap">첫 탐지 (frame {r["first_detect_frame"]}, cx={r["first_cx"]})</div></div>
        <div><img src="frames_20260712/{r["last_fn"]}" loading="lazy"><div class="cap">마지막 프레임 (cx={r["last_cx"]}, has_bbox={r["last_has_bbox"]})</div></div>
      </div>
      <div class="meta">
        <div>회전 액션 낸 프레임: <b>{nonfwd}</b></div>
        <div>총 프레임: {r["n_frames"]}</div>
        <div class="note">{r["note"] or "—"}</div>
      </div>
    </div>
    """


cards_success = "\n".join(card(r) for r in sorted(success, key=lambda x: x["sid"]))
cards_fail = "\n".join(card(r) for r in sorted(fail, key=lambda x: x["sid"]))

html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>CH61 §19 — FWD 고착 원인 분석 (2026-07-12)</title>
<style>
  :root {{ color-scheme: dark light; }}
  body {{ font-family: -apple-system, "Pretendard", sans-serif; background:#0b1220; color:#e5e7eb; margin:0; padding:24px; }}
  h1 {{ font-size:20px; margin-bottom:4px; }}
  .sub {{ color:#9ca3af; font-size:13px; margin-bottom:20px; }}
  .summary {{ background:#101726; border:1px solid #263041; border-radius:10px; padding:16px; margin-bottom:24px; font-size:14px; line-height:1.7; }}
  .summary b {{ color:#fbbf24; }}
  .summary .ok {{ color:#34d399; }}
  .summary .bad {{ color:#f87171; }}
  h2 {{ font-size:16px; margin:28px 0 12px; border-left:4px solid #6366f1; padding-left:10px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:14px; }}
  .card {{ background:#101726; border:1px solid #263041; border-left:4px solid; border-radius:10px; padding:12px; }}
  .card-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
  .sid {{ font-family:monospace; font-size:13px; color:#93c5fd; }}
  .badge {{ font-size:11px; font-weight:700; color:#0b1220; padding:2px 8px; border-radius:6px; }}
  .imgs {{ display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }}
  .imgs img {{ width:100%; border-radius:6px; display:block; background:#000; }}
  .noimg {{ width:100%; aspect-ratio:4/3; display:flex; align-items:center; justify-content:center; background:#1f2937; border-radius:6px; color:#6b7280; font-size:12px; }}
  .cap {{ font-size:11px; color:#9ca3af; margin-top:3px; text-align:center; }}
  .meta {{ margin-top:10px; font-size:12px; color:#d1d5db; line-height:1.6; }}
  .note {{ color:#fbbf24; margin-top:4px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; margin:12px 0; }}
  th, td {{ border:1px solid #263041; padding:6px 10px; text-align:left; }}
  th {{ background:#1a2436; }}
</style>
</head>
<body>

<h1>CH61 §19 — window3+bboxscale3 FWD 고착 원인 분석</h1>
<div class="sub">2026-07-11 obj_right 세션 11개(유효) · frame0 / 첫 탐지 / 마지막 프레임 이미지 비교 · 생성 2026-07-12</div>

<div class="summary">
  <b>결정적 발견</b>: 성공 세션은 전부 <span class="ok">first_detect_frame = 0</span> (첫 프레임부터 즉시 grounding 탐지),
  실패 세션은 전부 <span class="bad">first_detect_frame ≥ 3</span> 또는 미탐지.<br>
  <b>회전 액션(action[1] ≠ 0)은 오직 에피소드 초반 0~4프레임 구간에서만 발생</b> — 이후로는 grounding이
  나중에 정확히 탐지(cx 0.7~0.8)해도 액션은 FORWARD로 고정.<br>
  213650 / 220142는 grounding이 늦게라도 성공(첫 탐지 후 cx 0.73~0.76 연속 3회 유지)했지만
  <span class="bad">회전 액션이 단 한 프레임도 없었음</span> — grounding 실패가 아니라
  액션 트랜스포머(window=3)가 "초반 window"에만 반응하고 이후 grounding 업데이트를 무시하는 구조적 문제로 추정.
</div>

<h2>✅ 성공 ({len(success)}개) — 전부 frame 0 즉시 탐지</h2>
<div class="grid">
{cards_success}
</div>

<h2>❌ 실패 ({len(fail)}개) — 전부 초반 탐지 지연/실패</h2>
<div class="grid">
{cards_fail}
</div>

</body>
</html>
"""

out = OUT_DIR / "ch61_forward_lock_20260712.html"
out.write_text(html, encoding="utf-8")
print("wrote", out)
