#!/usr/bin/env python3
"""research_story.html에 CH82(a3 vs a3-s2 실기·시뮬레이션 대조 브리핑, 2026-11-16)를 (재)생성한다. 멱등.
실행 순서: gen_ch79 → gen_ch80 → gen_ch81 → gen_ch82 (앞 챕터 생성기가 '현재 상태' 줄을 덮어쓰므로 마지막에 실행)"""
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
    import ch82
    tabs = [(t, lab, [("html", re.sub(r"`([^`]+)`", r"<code>\1</code>", "".join(body)))]) for t, lab, body in ch82.tabs(tbl, H, st)]
    inner = CC.tabs_html("ch82", tabs, "", {})
    title = "a3 vs a3-s2 — 실기(11/13·11/16)와 시뮬레이션 대조 브리핑"
    blk = CC.chapter_block(82, "#fb923c", "#fdba74", title,
                           "가운데 새 물체 출발은 해결(5/5)되었으나 바구니·의자에서 더 가까이 가서 끝남(접촉 보고 6건). 실기·재생·선택지를 한 곳에 정리하고 판단 항목을 분리했다. 실기는 운영자 수동 판정.", "2026-11-16", inner)
    text = ("a3-s2 실기 45/45(11/16): 의자·가운데 5/5·책상·가운데 5/5(a3 11/13 의자 0/5), 그러나 바구니 종료 박스 면적 0.30~0.45(a3 0.23)·바구니 옆 접촉 보고 6건. "
            "11/16 실기 프레임 588장 재생(박스 재현 검증 통과)에서 면적 0.25~0.40 구간 정지 판단 a3 41~63% vs a3-s2 12~22%: a3 헤드가 더 일찍 멈춰 접촉 증가의 유력한 원인(인과 확정 아님), 시드 평균·a3-s1도 a3 수준에 못 미침. "
            "로봇 쪽 'near_area 설계 탓' 결론은 지지되지 않음. 선택지 A~C-3과 판단 전 결정 항목(성공 기준·맹검·접촉 기록) 정리")
    row = CC.summary_row(82, "a3 vs a3-s2 실기·시뮬 대조(11/16)", text, "#fb923c")
    s = CC.insert_chapter(s, 82, blk, "<!--/CH81-->", title, "🧭 CH82 a3 vs a3-s2", "비교", "#fb923c", "#fdba74", "#fb923c", "#2a1405", row, 81)
    s = CC.replace_toc_label(s, 82, title)
    s = CC.replace_summary_row(s, 82, "a3 vs a3-s2 실기·시뮬 대조(11/16)", text, "#fb923c")
    s = re.sub(r"현재 상태 \(CH1→CH\d+ · [0-9-]+\)", "현재 상태 (CH1→CH82 · 2026-11-16)", s, count=1)
    HTML.write_text(s, encoding="utf-8")
    __import__("story_publish").publish()   # 원본 갱신 후 발행(분리·이미지 파일화)
    for t, a, b in CC.balance(s):
        print(t, a, b)


if __name__ == "__main__":
    main()
