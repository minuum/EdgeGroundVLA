#!/usr/bin/env python3
"""LLM wiki 3단계 — 주제별 markdown 위키 페이지 생성 (Karpathy LLM-wiki 방식).

docs/plans/plan_20260828_llm_wiki.md 3단계. topic_index.json 기준으로 관련
챕터의 finding-card를 시간순(챕터 번호순)으로 모아 markdown 페이지를 만든다.
원본(research_story.html)은 안 건드리고, 이 위키는 그 위에 얹는 읽기 전용
재구성 레이어 — 각 항목은 `research_story.html#chXX`로 백링크.

**증분 안전성(2026-08-29 추가, wiki-sync 스킬용)**: 이 스크립트는 몇 번을 다시
돌려도 이미 사람/LLM이 채운 "## 압축 요약" 섹션을 절대 지우지 않는다 — 기존
파일이 있으면 그 요약 텍스트를 그대로 보존하고 "챕터별 원문 발췌"만 최신
chapters.json 기준으로 재생성한다. 새 챕터가 topic에 추가돼 기존 요약이
낡아졌으면(파일에 아직 없던 챕터 id가 topic_index.json에 나타나면) 요약
섹션 위에 "⚠️ 새 챕터 추가됨 — 요약 갱신 필요" 배지를 자동으로 붙인다.

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


PLACEHOLDER_SUMMARY = """*이 섹션은 아직 자동 생성되지 않았다. 아래 원문 발췌를 실제로 읽고,*
*Karpathy LLM-wiki 방식대로 "지금 이 주제에 대해 확정적으로 아는 것"을*
*3~10문장으로 압축해서 채워야 한다. 지금은 챕터별 원문을 시간순으로*
*재배열한 것까지만 되어 있다.*"""


def extract_existing_summary_and_chapters(slug):
    """기존 파일이 있으면 (압축요약 텍스트, 이미 렌더링됐던 챕터id set)을 반환.
    파일이 없으면 (None, set())."""
    path = WIKI_DIR / f"{slug}.md"
    if not path.exists():
        return None, set()
    text = path.read_text()
    m = re.search(r"## 압축 요약[^\n]*\n\n(.*?)\n\n---\n", text, re.DOTALL)
    summary = m.group(1).strip() if m else None
    if summary:
        # 이전 실행이 붙인 "새 챕터 추가됨" 배지는 스크립트가 매번 새로 판단해서
        # 붙이는 것이지 저장된 요약의 일부가 아니다 — 여기서 벗겨내지 않으면
        # 배지 문구가 요약 텍스트에 영구히 눌어붙는다.
        summary = re.sub(r"^⚠️ \*\*새 챕터 추가됨.*?\*\*\n\n", "", summary, flags=re.DOTALL)
    if summary and summary.startswith("*이 섹션은 아직 자동 생성되지"):
        summary = None  # 플레이스홀더 그대로였던 경우는 "아직 안 채워짐"과 동일 취급
    existing_ids = set(re.findall(r"research_story\.html#([a-z0-9_-]+)\)", text))
    return summary, existing_ids


def build_topic_page(slug, topic):
    prev_summary, prev_ids = extract_existing_summary_and_chapters(slug)
    ordered_ids = sorted(topic["chapter_ids"], key=chapter_order_key)
    new_ids = [cid for cid in ordered_ids if cid not in prev_ids]
    stale = bool(prev_summary and prev_ids and new_ids)

    lines = []
    lines.append(f"# {topic['title']}")
    lines.append("")
    lines.append(f"> {topic['summary']}")
    lines.append("")
    lines.append("## 압축 요약")
    lines.append("")
    if stale:
        lines.append(f"⚠️ **새 챕터 추가됨({', '.join(new_ids)}) — 아래 요약이 이 챕터들을 "
                      "반영 못 했을 수 있음, 재압축 필요**")
        lines.append("")
    lines.append(prev_summary if prev_summary else PLACEHOLDER_SUMMARY)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 챕터별 원문 발췌 (시간순)")
    lines.append("")

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
    lines.append("## 부록")
    lines.append("")
    lines.append("- **[아카이브 색인(archive-index.md)](archive-index.md)** — `docs/*.md` 357개 스냅샷 파일")
    lines.append("  (2025-12~2026-04, research_story.html 이전 시기의 별개 프로젝트 단계, 대부분 폐기된")
    lines.append("  방향의 죽은 기록). 압축 없이 제목/날짜/한줄요약만 모은 찾아가기용 색인.")
    lines.append("")
    lines.append("## 메타")
    lines.append("")
    lines.append(f"- 원본 챕터 수: {len(CHAPTERS)}개 (`docs/v5/research_story.html`)")
    lines.append(f"- 위키 주제 수: {len(TOPICS)}개")
    lines.append("- 생성 스크립트: `scripts/wiki/parse_research_story.py`, `scripts/wiki/build_wiki_pages.py`,")
    lines.append("  `scripts/wiki/build_archive_index.py`, `scripts/wiki/render_wiki_html.py`")
    lines.append("- **새 챕터 추가 시 자동 갱신**: `wiki-sync` 스킬(`.claude/skills/wiki-sync/SKILL.md`)")
    lines.append("  또는 `scripts/wiki/sync_wiki.py` 직접 실행 — 기존 압축 요약은 보존되고, 새로")
    lines.append("  추가된 챕터가 걸린 주제만 재압축 대상으로 표시됨")
    lines.append("- 위키 재생성 의존성: HTML 렌더링(`render_wiki_html.py`)만 "
                 "`pip install -r scripts/wiki/requirements.txt` 필요(Markdown 패키지) — "
                 "나머지 스크립트와 위키 페이지 열람 자체는 의존성 없음")
    lines.append("- 최신 상태 요약(시간순, 별도 문서): `docs/RESEARCH_STATUS.md`")
    return "\n".join(lines)


def find_unassigned_chapters():
    """chapters.json엔 있지만 어떤 topic에도 안 걸린 챕터id (vis처럼 의도적 제외는
    수동으로 알고 있어야 함 — 이 함수는 그냥 목록만 보여줌)."""
    assigned = set()
    for topic in TOPICS.values():
        assigned |= set(topic["chapter_ids"])
    return [c["id"] for c in CHAPTERS if c["id"] not in assigned]


def main():
    stale_slugs = []
    for slug, topic in TOPICS.items():
        _, prev_ids = extract_existing_summary_and_chapters(slug)
        new_ids = [cid for cid in topic["chapter_ids"] if cid not in prev_ids]
        if prev_ids and new_ids:
            stale_slugs.append((slug, new_ids))
        page = build_topic_page(slug, topic)
        out = WIKI_DIR / f"{slug}.md"
        out.write_text(page)
        print(f"  {slug}.md ({len(topic['chapter_ids'])}챕터, {len(page)}자)")

    index = build_index()
    (WIKI_DIR / "index.md").write_text(index)
    print(f"\n저장 → {WIKI_DIR}/*.md ({len(TOPICS)}개 주제 + index.md)")

    if stale_slugs:
        print("\n⚠️ 요약 재압축 필요(새 챕터 추가됨):")
        for slug, new_ids in stale_slugs:
            print(f"  - {slug}: {new_ids}")

    unassigned = find_unassigned_chapters()
    if unassigned:
        print(f"\n⚠️ 어떤 주제에도 안 걸린 챕터(topic_index.json에 수동 배정 필요): {unassigned}")


if __name__ == "__main__":
    main()
