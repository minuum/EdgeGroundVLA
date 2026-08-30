#!/usr/bin/env python3
"""LLM wiki 4단계 — markdown 위키를 GitHub Pages용 정적 HTML로 렌더링.

docs/plans/plan_20260828_llm_wiki.md 4단계. .venv의 `markdown` 패키지로
docs/wiki/*.md를 변환, research_story.html과 통일된 다크 테마로 감싼다.
JS 의존성 없음(클라이언트 markdown 파서 대신 빌드 시점에 정적 HTML 생성 —
이 프로젝트의 다른 docs/*.html들과 같은 방식).

출력: docs/wiki/<slug>.html × N + docs/wiki/index.html
"""
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent.parent
WIKI_DIR = ROOT / "docs/wiki"

CSS = """
body{background:#0a0f1a;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     max-width:900px;margin:0 auto;padding:32px 20px 80px;line-height:1.8}
h1{font-size:1.7rem;color:#4ade80;border-bottom:2px solid #1e293b;padding-bottom:12px}
h2{font-size:1.25rem;color:#38bdf8;margin-top:36px}
h3{font-size:1.05rem;color:#fbbf24;margin-top:28px;border-left:3px solid #fbbf24;padding-left:10px}
p{color:#94a3b8;font-size:0.92rem}
a{color:#7dd3fc}
blockquote{border-left:3px solid #4ade80;padding:8px 16px;margin:12px 0;background:rgba(74,222,128,0.06);
           color:#86efac;font-size:0.9rem}
hr{border:none;border-top:1px solid #1e293b;margin:24px 0}
code{background:#111827;padding:2px 6px;border-radius:4px;font-size:0.85rem;color:#fbbf24}
ul{color:#94a3b8}
strong{color:#e2e8f0}
em{color:#64748b}
.nav-bar{margin-bottom:24px;font-size:0.82rem}
.nav-bar a{color:#64748b;text-decoration:none;margin-right:12px}
.nav-bar a:hover{color:#7dd3fc}
"""

PAGE_TMPL = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head><body>
<div class="nav-bar"><a href="index.html">← 위키 색인</a><a href="../v5/research_story.html">연구일지 원본(research_story.html)</a></div>
{body}
</body></html>
"""


def render_file(md_path):
    text = md_path.read_text()
    # [xxx](slug.md) 형태 내부 위키 링크를 slug.html로 교체(정적 HTML 간 이동용)
    text = re.sub(r"\]\(([a-z0-9-]+)\.md\)", r"](\1.html)", text)
    body_html = markdown.markdown(text, extensions=["tables", "fenced_code"])
    title_m = re.search(r"<h1[^>]*>(.*?)</h1>", body_html)
    title = re.sub(r"<[^>]+>", "", title_m.group(1)) if title_m else md_path.stem
    html = PAGE_TMPL.format(title=title, css=CSS, body=body_html)
    out_path = md_path.with_suffix(".html")
    out_path.write_text(html)
    return out_path


def main():
    md_files = sorted(WIKI_DIR.glob("*.md"))
    for md_path in md_files:
        out = render_file(md_path)
        print(f"  {md_path.name} → {out.name}")
    print(f"\n렌더링 완료: {len(md_files)}개")


if __name__ == "__main__":
    main()
