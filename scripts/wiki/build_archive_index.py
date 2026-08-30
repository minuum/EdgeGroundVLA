#!/usr/bin/env python3
"""LLM wiki 부록 — docs/*.md 357개 스냅샷 파일 카탈로그 (압축 없음).

docs/plans/plan_20260828_llm_wiki.md 후속 결정(2026-08-28): 이 파일들은
research_story.html(2026-05~)보다 이전 시기(2025-12~2026-04)의 별개 프로젝트
단계(양자화/서빙 인프라, manipulation vs navigation 비교 등) 기록이고, 대부분
폐기된 방향의 죽은 기록으로 판단됨 — 원문을 읽어 압축하지 않고 "제목/날짜/
첫 줄 요약"만 뽑아 아카이브 색인만 만든다(빠르고 안전, 환각 리스크 없음).

원본 파일은 절대 옮기거나 지우지 않는다 — docs/*.md 그대로 둔 채 색인만 생성.

출력: docs/wiki/archive-index.md
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = ROOT / "docs"
OUT = ROOT / "docs/wiki/archive-index.md"

DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})|(\d{8})")


def extract_date(text, filename):
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = DATE_RE.search(text[:500])
    if m:
        s = m.group(0)
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s
    return ""


def extract_title_and_desc(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    title = ""
    desc = ""
    for l in lines:
        if l.startswith("#"):
            title = l.lstrip("#").strip()
            break
    if not title and lines:
        title = lines[0][:80]
    # 제목 다음의 첫 "설명형" 줄(볼드 메타 줄 **날짜**: 등은 건너뜀)
    seen_title = False
    for l in lines:
        if l.startswith("#") and not seen_title:
            seen_title = True
            continue
        if not seen_title:
            continue
        if re.match(r"^\*\*.*\*\*[:：]", l) or l.startswith("---") or l.startswith("#"):
            continue
        if len(l) > 10:
            desc = l[:120]
            break
    return title, desc


def main():
    files = sorted(DOCS_DIR.glob("*.md"))
    rows = []
    for f in files:
        text = f.read_text(errors="ignore")
        date = extract_date(text, f.name)
        title, desc = extract_title_and_desc(text)
        rows.append(dict(name=f.name, date=date, title=title, desc=desc))

    # 날짜순 정렬(날짜 없는 건 뒤로)
    rows.sort(key=lambda r: (r["date"] == "", r["date"], r["name"]))

    lines = []
    lines.append("# 아카이브 색인 — docs/*.md 스냅샷 357개 (압축 없음, 원문 그대로)")
    lines.append("")
    lines.append("> 2025-12~2026-04 시기(research_story.html이 다루는 2026-05~ 이전)의 별개")
    lines.append("> 프로젝트 단계 기록 — 양자화/서빙 인프라, manipulation vs navigation 비교 등.")
    lines.append("> 2026-08-28 검토 결과 대부분 폐기된 방향의 죽은 기록으로 판단되어, 원문을")
    lines.append("> 읽어 압축하지 않고 **제목/날짜/한줄요약만** 모았다. 실제로 필요한 파일이")
    lines.append("> 있으면 `docs/<파일명>`으로 직접 열어서 확인한다 — 이 색인은 찾아가기용.")
    lines.append("")
    lines.append(f"총 {len(rows)}개 파일.")
    lines.append("")
    lines.append("| 파일 | 날짜 | 제목 | 한줄요약 |")
    lines.append("|---|---|---|---|")
    for r in rows:
        desc = r["desc"].replace("|", "\\|")
        title = r["title"].replace("|", "\\|")
        lines.append(f"| [`{r['name']}`](../{r['name']}) | {r['date']} | {title} | {desc} |")

    OUT.write_text("\n".join(lines))
    print(f"저장 → {OUT} ({len(rows)}행)")


if __name__ == "__main__":
    main()
