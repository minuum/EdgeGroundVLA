#!/usr/bin/env python3
"""LLM wiki 3단계 — 주제별 markdown 위키 페이지 생성 (Karpathy LLM-wiki 방식).

docs/plans/plan_20260828_llm_wiki.md 3단계. topic_index.json 기준으로 관련
챕터의 finding-card를 시간순(챕터 번호순)으로 모아 markdown 페이지를 만든다.
원본(research_story.html)은 안 건드리고, 이 위키는 그 위에 얹는 읽기 전용
재구성 레이어 — 각 항목은 `research_story.html#chXX`로 백링크.

이번 실행은 "기계적 재구성"(주제별로 원문 카드를 시간순 재배열)까지만 한다.
Karpathy 방식의 핵심인 "LLM이 압축한 요약"은 각 페이지 맨 위 개요 섹션에
사람이 이어서 채우도록 플레이스홀더를 남겨둔다(이번 스크립트가 자동 생성하는
건 원문 발췌 나열이지 요약이 아님 — 그래서 파일마다 "## 압축 요약 (TODO)" 헤더를
넣어 다음 반복에서 실제 압축 작업을 하도록 표시).

출력: docs/wiki/<topic-slug>.md × N + docs/wiki/index.md
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
WIKI_DIR = ROOT / "docs/wiki"
CHAPTERS = json.loads((WIKI_DIR / "data/chapters.json").read_text())
TOPICS = json.loads((WIKI_DIR / "data/topic_index.json").read_text())

CHAPTER_BY_ID = {c["id"]: c for c in CHAPTERS}


def chapter_order_key(cid):
    """ch1, ch2, ... 순서로 정렬하되 숫자 없는 특수 챕터(meeting-* 등)는 뒤로."""
    m = re.match(r"ch(\d+)$", cid)
    if m:
        return (0, int(m.group(1)))
    return (1, cid)


def format_card(card):
    lines = []
    if card["title"]:
        lines.append(f"**{card['title']}**")
    lines.append(card["text"])
    return "\n\n".join(lines)


def build_topic_page(slug, topic):
    lines = []
    lines.append(f"# {topic['title']}")
    lines.append("")
    lines.append(f"> {topic['summary']}")
    lines.append("")
    lines.append("## 압축 요약 (TODO — 다음 반복에서 채울 것)")
    lines.append("")
    lines.append("*이 섹션은 아직 자동 생성되지 않았다. 아래 원문 발췌를 실제로 읽고,*")
    lines.append("*Karpathy LLM-wiki 방식대로 \"지금 이 주제에 대해 확정적으로 아는 것\"을*")
    lines.append("*3~10문장으로 압축해서 채워야 한다. 지금은 챕터별 원문을 시간순으로*")
    lines.append("*재배열한 것까지만 되어 있다.*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 챕터별 원문 발췌 (시간순)")
    lines.append("")

    ordered_ids = sorted(topic["chapter_ids"], key=chapter_order_key)
    missing = [cid for cid in ordered_ids if cid not in CHAPTER_BY_ID]
    if missing:
        print(f"  ⚠️ [{slug}] 존재하지 않는 챕터 id: {missing}")

    for cid in ordered_ids:
        ch = CHAPTER_BY_ID.get(cid)
        if not ch:
            continue
        num = ch["num"] or cid.upper()
        title = ch["title"] or "(제목없음)"
        lines.append(f"### {num} — {title}")
        if ch["subtitle"]:
            lines.append(f"*{ch['subtitle']}*")
        lines.append("")
        for card in ch["cards"]:
            lines.append(format_card(card))
            lines.append("")
        lines.append(f"[→ 원문 전체 보기(research_story.html#{cid})](../v5/research_story.html#{cid})")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def build_index():
    lines = []
    lines.append("# MoNaVLA 연구 위키 — 주제별 색인")
    lines.append("")
    lines.append("`docs/v5/research_story.html`(시간순 연구일지, 70+챕터)을 주제별로")
    lines.append("재구성한 위키다. Karpathy LLM-wiki 방식(raw/wiki/index 3계층, 벡터DB 없음) —")
    lines.append("미래 세션은 이 index.md만 먼저 읽고, 필요한 주제 파일만 열어보면 된다.")
    lines.append("")
    lines.append("원본(research_story.html)은 이 위키가 절대 대체하지 않는다 — 각 항목은")
    lines.append("원문 챕터로 백링크되어 있으니, 세부 근거·수치·이미지가 필요하면 원문을 본다.")
    lines.append("")
    lines.append("## 주제 목록")
    lines.append("")
    for slug, topic in TOPICS.items():
        n = len(topic["chapter_ids"])
        lines.append(f"- **[{topic['title']}]({slug}.md)** ({n}개 챕터) — {topic['summary']}")
    lines.append("")
    lines.append("## 메타")
    lines.append("")
    lines.append(f"- 원본 챕터 수: {len(CHAPTERS)}개 (`docs/v5/research_story.html`)")
    lines.append(f"- 위키 주제 수: {len(TOPICS)}개")
    lines.append("- 생성 스크립트: `scripts/wiki/parse_research_story.py`, `scripts/wiki/build_wiki_pages.py`")
    lines.append("- 최신 상태 요약(시간순, 별도 문서): `docs/RESEARCH_STATUS.md`")
    return "\n".join(lines)


def main():
    for slug, topic in TOPICS.items():
        page = build_topic_page(slug, topic)
        out = WIKI_DIR / f"{slug}.md"
        out.write_text(page)
        print(f"  {slug}.md ({len(topic['chapter_ids'])}챕터, {len(page)}자)")

    index = build_index()
    (WIKI_DIR / "index.md").write_text(index)
    print(f"\n저장 → {WIKI_DIR}/*.md ({len(TOPICS)}개 주제 + index.md)")


if __name__ == "__main__":
    main()
