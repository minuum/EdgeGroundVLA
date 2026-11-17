#!/usr/bin/env python3
"""research_story.html에 CH80(텍스트 조건화 E2E: T0~T5·C·T7)을 (재)생성한다. 멱등. 텍스트 조건화 탭은 CH79에서 이 챕터로 옮겼다.
실행 순서: gen_ch79 → gen_ch80 (gen_ch79가 '현재 상태' 줄을 덮어쓰므로)"""
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
    import ch79_textcond, ch80_t7
    tabs = [("t1", "🔤 지시문 텍스트 조건화 T0~T5·C", [("html", ch79_textcond.block(tbl, H, st))]),
            ("t2", "▶ T7 실기 세션 재생(A3 대비)", [("html", ch80_t7.block(tbl, H, st))])]
    inner = CC.tabs_html("ch80", tabs, "", {})
    title = "텍스트 조건화 E2E — 지시문·부정 지시·박스 훼손·실기 세션 재생"
    blk = CC.chapter_block(80, "#38bdf8", "#7dd3fc", title,
                           "교수님 시험(지시문 변경·물체 교체)에서 출발한 텍스트 조건화 E2E(V0~V5). 오프라인 평가(T0~T5·C)와 11/09 실기 세션 재생(T7, A3 대비). CH79에서 분리.", "2026-11-05~11-09", inner)
    text = "V0~V5 텍스트 조건화. T4 어휘 확장 V4(미학습 부정 정지 99.5%), T5 토큰 어텐션 V5(V4 대비 이득 없음·불안정), C 박스 훼손, T7 A3 세션 452개 재생(V4가 V0보다 A3에 가깝고 박스 반전에 강하나 시드 편차 큼, 실기 성공 예측은 불가)"
    row = CC.summary_row(80, "텍스트 조건화 E2E", text, "#38bdf8")
    s = CC.insert_chapter(s, 80, blk, "<!--/CH79-->", title, "🔤 CH80 텍스트 조건화 E2E", "신규분기", "#38bdf8", "#7dd3fc", "#38bdf8", "#0a1830", row, 79)
    s = CC.replace_toc_label(s, 80, title)
    s = CC.replace_summary_row(s, 80, "텍스트 조건화 E2E", text, "#38bdf8")
    s = re.sub(r"현재 상태 \(CH1→CH\d+ · [0-9-]+\)", "현재 상태 (CH1→CH80 · 2026-11-10)", s, count=1)
    HTML.write_text(s, encoding="utf-8")
    __import__("story_publish").publish()   # 원본 갱신 후 발행(분리·이미지 파일화)
    for t, a, b in CC.balance(s):
        print(t, a, b)


if __name__ == "__main__":
    main()
