#!/usr/bin/env python3
"""research_story.html에 CH78(EGVLA2 실기 전 단계: 로드맵·서빙·점검·시사점·논문 방향·타임라인)을 (재)생성한다. 멱등.
실행: .venv/bin/python3 -I scripts/gen_ch78.py   (CH77 레거시 부분을 재사용하므로 gen_ch77.py 이후 실행)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import ch_common as CC  # noqa: E402
import gen_ch77 as G77  # noqa: E402

HTML, tbl, H, st, R = G77.HTML, G77.tbl, G77.H, G77.st, G77.R


def deliverables():
    return (H("⑦ 산출물·핸드오프 (실기 전 단계)") + tbl(["종류", "경로 / 내용"], [
        ["핸드오프", "<code>HANDOFF_20261020_ROBOT_QUICKSTART_egvla2.md</code>(한 장), <code>HANDOFF_20261013_continuous_collection_for_robot_session.md</code> §15(체크포인트·조건)·§16(드라이런 전환)"],
        ["plan", "<code>plan_20261019_egvla2_inference_server.md</code>(서버 단계 A/B, 응답 버그, 젯슨 결과), <code>plan_20261015_egvla2_300ep_pipeline.md</code> §11~§15, <code>plan_20261016_egvla2_serving_path.md</code>"],
        ["체크포인트", "<code>runs/egvla2_fm/deploy/C_pi0_adaln_drop30_e120.pt</code>(권장), <code>C_pi0_adaln_e120.pt</code>(대조), <code>MANIFEST.json</code>(sha256)"],
        ["서버 코드", "<code>stage2_v2_inference_server.py</code>(단계 A, 응답 필드 수정), <code>egvla2_inference_server.py</code>·<code>egvla2_only_logic.py</code>(단계 B 스켈레톤), <code>egvla2_fm_policy.py</code>, <code>continuous_speed.py</code>, <code>mona_dashboard.py</code>(drive·JPEG 옵션)"],
        ["점검 스크립트", "<code>check_serving_equiv.py</code>, <code>robot_check_server_vs_local.py</code>, <code>replay_episodes_through_server.py</code>, <code>analyze_startpos_replay.py</code>"],
        ["결과", "<code>serving_equiv*.json</code>, <code>jetson_check_20261020.json</code>, (대기) <code>startpos_replay.json</code>"]])
        + f'{st("한계", R)} 실기 성공률·시작 위치 일반화는 아직 측정되지 않았다. 서버 단계 B의 정지·소실 규칙은 드라이런 로그 확정 전 기본값이다.')


TABS78 = [("t1", "🧭 로드맵", [("html", "ROADMAP")]), ("t2", "🤖 EGVLA1 vs 2", ["⑥-2", "⑥-7"]), ("t3", "🔧 서빙·실기 준비", ["⑥", "⑥-10", "⑧"]),
          ("t4", "📊 시사점·문제점·논문 방향", [("html", "ANALYSIS")]), ("t5", "⏱ 시도 타임라인", [("html", "TIMELINE")]), ("t6", "🧰 산출물·핸드오프", [("html", "DELIV")])]


def main():
    s = HTML.read_text(encoding="utf-8")
    head, parts = G77.body()
    import ch78_analysis, ch78_roadmap, ch78_timeline
    raw = {"ROADMAP": ch78_roadmap.block(tbl, H, st), "ANALYSIS": ch78_analysis.block(tbl, H, st), "TIMELINE": ch78_timeline.block(tbl, H, st), "DELIV": deliverables()}
    tabs = [(t, lab, [("html", raw[it[1]]) if isinstance(it, tuple) else it for it in items]) for t, lab, items in TABS78]
    inner = CC.tabs_html("ch78", tabs, "", parts)
    blk = CC.chapter_block(78, "#fb923c", "#fdba74", "EGVLA2 실기 전 단계 — 로드맵·서빙·점검·시사점·논문 방향·시도 타임라인",
                           "배포 체크포인트·서버·젯슨 점검 결과와 구동 전 로드맵(R0~R8), 시사점·문제점·논문 방향, 10/12~10/20 시도 타임라인. 학습 분야는 CH77.", "2026-10-20", inner)
    row = CC.summary_row(78, "EGVLA2 실기 전 단계", "로드맵 R0~R8 중 R0~R3 완료(젯슨 서버 경로 pass, 왕복 892ms=Florence-2 821+FM 54), R4 pos2·3 재생 대기, 시사점·문제점·논문 방향(A 경량·연속 제어+B 정직한 분석), 시도 타임라인", "#fb923c")
    s = CC.insert_chapter(s, 78, blk, "<!--/CH77-->", "EGVLA2 실기 전 단계 — 로드맵·점검·시사점·타임라인", "🧭 CH78 EGVLA2 실기 전 로드맵", "신규분기", "#fb923c", "#fdba74", "#fb923c", "#2a1306", row, 77)
    s = CC.replace_toc_label(s, 78, "EGVLA2 실기 전 단계 — 로드맵·점검·시사점·타임라인")
    s = CC.replace_summary_row(s, 78, "EGVLA2 실기 전 단계", "로드맵 R0~R8 중 R0~R3 완료(젯슨 서버 경로 pass, 왕복 892ms=Florence-2 821+FM 54), R4 pos2·3 재생 대기, 시사점·문제점·논문 방향(A 경량·연속 제어+B 정직한 분석), 시도 타임라인", "#fb923c")
    HTML.write_text(s, encoding="utf-8")
    for t, a, b in CC.balance(s):
        print(t, a, b)


if __name__ == "__main__":
    main()
