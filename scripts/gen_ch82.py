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
    title = "a3 → a3-s2 → MOD-DR-Safe (a3-ag23) — 실기(11/13·11/16·11/25) 대조 및 45/45 무충돌 해결"
    blk = CC.chapter_block(82, "#fb923c", "#fdba74", title,
                           "11/16 a3-s2에서 드러난 근접 접촉(6건) 문제를 Area-Grounded 정지 학습(a3-ag23) + beam1_fp16 + S2 연속 감속(MOD-DR-Safe)으로 해결해 11/25 실기 45/45(100%, 접촉 0건, 397ms)을 달성했다. 최신 도착 사진 6장 및 AG28/Y2 후속 개선 포함.", "2026-11-16 ~ 11-28", inner)
    text = ("a3-s2 실기 45/45(11/16, 접촉 보고 6건) 원인을 11/16 588프레임 재생으로 규명(정지 판단 12~22% vs a3 41~63%)한 뒤, "
            "a3-ag23(area≥0.23 기하 정합 + 50% 시각 교체) + beam=1 fp16 + S2 연속 감속(MOD-DR-Safe)으로 11/25 실기 45/45(바구니 30 + 의자 15, 접촉 0건, 종단 397.1ms) 완전 해결. "
            "의자 원거리 정지(y2=0.82) 해소를 위한 AG28 및 바닥선 A3C_Y2_s0 학습·soda 전달 완료")
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
