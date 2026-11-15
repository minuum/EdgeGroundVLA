#!/usr/bin/env python3
"""research_story.html에 CH81(종합 브리핑 2026-11-15)을 (재)생성한다. 멱등.
실행 순서: gen_ch79 → gen_ch80 → gen_ch81 (앞 둘이 '현재 상태' 줄을 덮어쓰므로 마지막에 실행)"""
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
    import ch81_brief, ch81_vis, ch81_a3
    ext = [("t5", "🖼 실제 화면(11/13)", ch81_vis.images_tab(H, st)), ("t6", "🔀 구조도", ch81_vis.diagrams_tab(H)), ("t7", "⚖ A3 대비·시뮬 대조·오차", ch81_a3.tab(tbl, H, st))]
    tabs = [(t, lab, [("html", re.sub(r"`([^`]+)`", r"<code>\1</code>", "".join(body)))]) for t, lab, body in ch81_brief.block(tbl, H, st) + ext]
    inner = CC.tabs_html("ch81", tabs, "", {})
    title = "종합 브리핑 — V4R·11/13 물체 실험·V4/V4R 서빙·soda 환경·LiDAR"
    blk = CC.chapter_block(81, "#34d399", "#6ee7b7", title,
                           "11/13 실기 물체 실험(A3, 65건) 분석, V4R 판정, V4/V4R 서빙 경로 구현·검증, soda 실행 환경 확인, LiDAR 검토와 남은 결정을 한 곳에 정리. 실기 미검증 항목은 표에 명시.", "2026-11-15", inner)
    text = ("11/13 실기 65건 49/65(옆 39/40·가운데 10/24, 의자·화분 집중 실패, 문구 형태는 물체·위치와 교락돼 분리 불가), V4R 기준 ①~④ 충족, V4/V4R 서빙 경로 구현·오프라인 동등성 통과(실기 전), "
            "soda는 지시어로 박스를 찾았음을 코드로 확인(저장소 코드와 다름), 로봇에 LiDAR 없음, 실기 명령 크기는 서버가 1.15로 고정(모델 원출력은 세션 JSON), 가운데 실패 원인은 이미지 특징(드롭30 단독 원인은 시드 비교로 기각, 배포 A3 시드 0의 편차 가능성), A3 개선 시도 3종은 서빙 후보 없음")
    row = CC.summary_row(81, "종합 브리핑(11/15)", text, "#34d399")
    s = CC.insert_chapter(s, 81, blk, "<!--/CH80-->", title, "🧾 CH81 종합 브리핑", "종합", "#34d399", "#6ee7b7", "#34d399", "#06241a", row, 80)
    s = CC.replace_toc_label(s, 81, title)
    s = CC.replace_summary_row(s, 81, "종합 브리핑(11/15)", text, "#34d399")
    s = re.sub(r"현재 상태 \(CH1→CH\d+ · [0-9-]+\)", "현재 상태 (CH1→CH81 · 2026-11-15)", s, count=1)
    HTML.write_text(s, encoding="utf-8")
    for t, a, b in CC.balance(s):
        print(t, a, b)


if __name__ == "__main__":
    main()
