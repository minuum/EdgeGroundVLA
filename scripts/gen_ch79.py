#!/usr/bin/env python3
"""research_story.html에 CH79(EGVLA2 분석·시사점: 시사점·문제점·논문 방향, 시도 타임라인)를 (재)생성한다. 멱등. 실행 순서: gen_ch77 → gen_ch78 → gen_ch79"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import ch_common as CC  # noqa: E402
import gen_ch77 as G77  # noqa: E402

HTML, tbl, H, st = G77.HTML, G77.tbl, G77.H, G77.st


def main():
    s = HTML.read_text(encoding="utf-8")
    import ch78_analysis, ch78_timeline, ch79_egvla3, ch79_index, ch79_discussion
    tabs = [("t0", "🗓 작업 정리·색인", [("html", ch79_index.block(tbl, H, st))]), ("t4", "💬 토의 기록·도전 과제", [("html", ch79_discussion.block(tbl, H, st))]), ("t1", "📊 시사점·문제점·논문 방향", [("html", ch78_analysis.block(tbl, H, st))]), ("t2", "🧭 EGVLA3 구상·기술 시사점", [("html", ch79_egvla3.block(tbl, H, st))]), ("t3", "⏱ 시도 타임라인·정정", [("html", ch78_timeline.block(tbl, H, st))])]
    inner = CC.tabs_html("ch79", tabs, "", {})
    blk = CC.chapter_block(79, "#a78bfa", "#c4b5fd", "EGVLA2 분석·시사점 — 시사점·문제점·논문 방향과 10/12~10/20 시도 타임라인",
                           "CH77(학습)·CH78(실기 전 단계)까지의 결과를 종합한 분석. 무엇을 말할 수 있고, 무엇이 막혀 있고, 논문을 어느 방향으로 쓸지, 그리고 시도와 정정의 시간순 기록.", "2026-10-20", inner)
    text = "작업 정리 색인(10/20~21), 시사점 8개(연속 학습≈사람 변동성, 지연 2.41×, FM≈MLP, 서브골 ≤0.4°, 드롭아웃 견고성), 문제점 8개(실기 미측정·pos1 단일 학습), 논문 방향(경량·연속 제어 + 정직한 분석), EGVLA3 구상(지시문 조건·2층), 시도 타임라인과 정정"
    row = CC.summary_row(79, "EGVLA2 분석·시사점", text, "#a78bfa")
    s = CC.insert_chapter(s, 79, blk, "<!--/CH78-->", "EGVLA2 분석·시사점 — 시사점·문제점·논문 방향·시도 타임라인", "📊 CH79 EGVLA2 분석·시사점", "신규분기", "#a78bfa", "#c4b5fd", "#a78bfa", "#1a1030", row, 78)
    s = CC.replace_toc_label(s, 79, "EGVLA2 분석·시사점 — 시사점·문제점·논문 방향·시도 타임라인")
    s = CC.replace_summary_row(s, 79, "EGVLA2 분석·시사점", text, "#a78bfa")
    s = re.sub(r"현재 상태 \(CH1→CH\d+ · [0-9-]+\)", "현재 상태 (CH1→CH79 · 2026-10-21)", s, count=1)
    HTML.write_text(s, encoding="utf-8")
    for t, a, b in CC.balance(s):
        print(t, a, b)


if __name__ == "__main__":
    main()
