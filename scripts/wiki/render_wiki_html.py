#!/usr/bin/env python3
"""LLM wiki 4단계 — markdown 위키를 GitHub Pages용 정적 HTML로 렌더링.

docs/plans/plan_20260828_llm_wiki.md 4단계, 2026-08-30 디자인 개편(카드형 블록 +
사이드바 내비게이션 — 이전 버전은 텍스트 나열이라 가시성이 나쁘다는 피드백 반영).
.venv의 `markdown` 패키지로 docs/wiki/*.md를 변환한다. build_wiki_pages.py가
markdown 안에 raw HTML 블록(<div class="card" markdown="1">...)을 심어두므로
`md_in_html` 확장으로 그 안의 markdown(볼드/코드 등)도 같이 처리한다.
JS는 아카이브 색인 페이지의 표 필터링(357행 스크롤 문제) 한 곳에만 최소한으로
쓰고, 나머지는 빌드 시점 정적 HTML — 이 프로젝트의 다른 docs/*.html들과 같은 방식.

출력: docs/wiki/<slug>.html × N + docs/wiki/index.html
"""
import json
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent.parent
WIKI_DIR = ROOT / "docs/wiki"
TOPICS = json.loads((WIKI_DIR / "data/topic_index.json").read_text())

CSS = """
:root{
  --bg:#0a0f1a; --bg-raised:#0d1420; --line:#1e293b; --line-soft:#16202f;
  --ink:#e2e8f0; --ink-soft:#94a3b8; --ink-dim:#64748b;
  --accent-a:#4ade80; --accent-b:#38bdf8; --accent-c:#fbbf24; --accent-d:#f472b6; --accent-e:#a78bfa;
  --accent-archive:#94a3b8;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     margin:0;line-height:1.75;font-size:15px}
.layout{display:flex;max-width:1180px;margin:0 auto;align-items:flex-start}
.sidebar{position:sticky;top:0;height:100vh;overflow-y:auto;flex:0 0 240px;
         padding:28px 16px;border-right:1px solid var(--line);font-size:0.82rem}
.sidebar-brand{font-weight:800;color:var(--ink);font-size:0.95rem;margin-bottom:4px}
.sidebar-brand a{color:inherit;text-decoration:none}
.sidebar-sub{color:var(--ink-dim);font-size:0.72rem;margin-bottom:18px}
.sidebar-section{color:var(--ink-dim);font-size:0.68rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.06em;margin:18px 0 8px}
.sidebar a.side-link{display:block;color:var(--ink-soft);text-decoration:none;padding:6px 8px;
                      border-radius:6px;margin:1px 0}
.sidebar a.side-link:hover{background:var(--bg-raised);color:var(--ink)}
.sidebar a.side-link.current{background:var(--bg-raised);color:var(--ink);font-weight:700;
                              box-shadow:inset 3px 0 0 var(--accent-b)}
.sidebar a.side-link .n{color:var(--ink-dim);font-size:0.7rem;float:right}
.main{flex:1 1 auto;min-width:0;padding:36px 40px 100px;max-width:900px}
h1{font-size:1.6rem;color:var(--ink);margin:0 0 8px;letter-spacing:-0.01em}
h2{font-size:1.05rem;color:var(--ink-soft);font-weight:700;text-transform:uppercase;
   letter-spacing:0.04em;margin-top:40px;margin-bottom:16px}
p.tagline{color:var(--ink-soft);font-size:0.95rem;margin:0 0 28px;max-width:70ch}
a{color:#7dd3fc}
code{background:#111827;padding:2px 6px;border-radius:4px;font-size:0.85rem;color:var(--accent-c)}
strong{color:var(--ink)}
em{color:var(--ink-dim)}
ul{color:var(--ink-soft)}

/* 요약/메타 강조 박스 */
.summary-box{background:var(--bg-raised);border:1px solid var(--line);border-left:3px solid var(--accent-a);
             border-radius:10px;padding:18px 22px;margin:0 0 32px}
.summary-box>p:first-child strong:first-child{color:var(--accent-a);font-size:0.8rem;
             text-transform:uppercase;letter-spacing:0.05em}
.summary-box p{color:var(--ink-soft);font-size:0.9rem;margin:10px 0}

/* 챕터 블록(카드) */
.chapter-block{background:var(--bg-raised);border:1px solid var(--line);border-radius:12px;
                padding:20px 24px;margin-bottom:20px}
.chapter-block-head{font-size:1.0rem;font-weight:700;color:var(--ink);margin-bottom:4px}
.chapter-badge{display:inline-block;background:var(--line-soft);color:var(--ink-soft);
                font-size:0.7rem;font-weight:700;padding:2px 8px;border-radius:5px;margin-right:8px}
p.chapter-subtitle-line{color:var(--ink-dim);font-size:0.8rem;font-style:italic;margin:0 0 14px}
.accent-a{border-left:3px solid var(--accent-a)} .accent-a .chapter-badge{background:rgba(74,222,128,0.15);color:var(--accent-a)}
.accent-b{border-left:3px solid var(--accent-b)} .accent-b .chapter-badge{background:rgba(56,189,248,0.15);color:var(--accent-b)}
.accent-c{border-left:3px solid var(--accent-c)} .accent-c .chapter-badge{background:rgba(251,191,36,0.15);color:var(--accent-c)}
.accent-d{border-left:3px solid var(--accent-d)} .accent-d .chapter-badge{background:rgba(244,114,182,0.15);color:var(--accent-d)}
.accent-e{border-left:3px solid var(--accent-e)} .accent-e .chapter-badge{background:rgba(167,139,250,0.15);color:var(--accent-e)}

/* 카드 안의 finding-card */
.card{background:var(--bg);border:1px solid var(--line-soft);border-radius:8px;
      padding:12px 16px;margin:10px 0;font-size:0.86rem}
.card p{margin:6px 0;color:var(--ink-soft)}
.card>p:first-child strong:only-child{color:var(--ink)}
a.src-link{display:inline-block;margin-top:8px;font-size:0.8rem;color:var(--ink-dim);text-decoration:none}
a.src-link:hover{color:#7dd3fc}
.card img{max-width:100%;height:auto;border-radius:8px;border:1px solid var(--line);
          display:block;margin:10px 0}

table{border-collapse:collapse;width:100%;font-size:0.8rem;margin:12px 0;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;color:var(--ink-soft)}
th{color:var(--ink);background:var(--bg-raised);position:sticky;top:0}

/* 홈 주제 카드 그리드 */
.topic-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0 32px}
@media(max-width:700px){.topic-grid{grid-template-columns:1fr}}
a.topic-card{display:flex;flex-direction:column;gap:6px;background:var(--bg-raised);
             border:1px solid var(--line);border-radius:12px;padding:16px 18px;
             text-decoration:none;transition:transform .12s,border-color .12s}
a.topic-card:hover{transform:translateY(-2px);border-color:var(--ink-dim)}
.topic-card-count{font-size:0.68rem;color:var(--ink-dim);text-transform:uppercase;letter-spacing:0.05em}
.topic-card-title{font-size:1.0rem;font-weight:700;color:var(--ink)}
.topic-card-summary{font-size:0.8rem;color:var(--ink-soft);line-height:1.5}
.topic-card.accent-a{border-top:3px solid var(--accent-a)}
.topic-card.accent-b{border-top:3px solid var(--accent-b)}
.topic-card.accent-c{border-top:3px solid var(--accent-c)}
.topic-card.accent-d{border-top:3px solid var(--accent-d)}
.topic-card.accent-e{border-top:3px solid var(--accent-e)}
.topic-card.accent-archive{border-top:3px solid var(--accent-archive)}

/* 아카이브 색인 검색창 */
.archive-search{width:100%;padding:10px 14px;border-radius:8px;border:1px solid var(--line);
                 background:var(--bg-raised);color:var(--ink);font-size:0.9rem;margin-bottom:14px}
.archive-search:focus{outline:2px solid var(--accent-b);outline-offset:1px}
.archive-count{color:var(--ink-dim);font-size:0.78rem;margin-bottom:10px}

@media(max-width:820px){
  .layout{display:block}
  .sidebar{position:static;height:auto;border-right:none;border-bottom:1px solid var(--line)}
  .main{padding:24px 20px 80px;max-width:100%}
}
"""

PAGE_TMPL = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head><body>
<div class="layout">
<nav class="sidebar">{sidebar}</nav>
<main class="main">
{body}
</main>
</div>
{extra_script}
</body></html>
"""

ARCHIVE_SEARCH_SCRIPT = """
<script>
(function(){
  var input = document.getElementById('archive-filter');
  var table = document.querySelector('.main table');
  if(!input || !table) return;
  var rows = Array.prototype.slice.call(table.querySelectorAll('tbody tr'));
  var countEl = document.getElementById('archive-count');
  function apply(){
    var q = input.value.trim().toLowerCase();
    var shown = 0;
    rows.forEach(function(r){
      var match = !q || r.textContent.toLowerCase().indexOf(q) !== -1;
      r.style.display = match ? '' : 'none';
      if(match) shown++;
    });
    if(countEl) countEl.textContent = shown + ' / ' + rows.length + '개 표시 중';
  }
  input.addEventListener('input', apply);
  apply();
})();
</script>
"""


def build_sidebar(current_slug):
    items = []
    items.append('<div class="sidebar-brand"><a href="index.html">MoNaVLA 연구 위키</a></div>')
    items.append('<div class="sidebar-sub">주제별 재구성 · research_story.html 기반</div>')
    items.append('<div class="sidebar-section">주제</div>')
    for slug, topic in TOPICS.items():
        n = len(topic["chapter_ids"])
        cls = "side-link current" if slug == current_slug else "side-link"
        items.append(f'<a class="{cls}" href="{slug}.html">{topic["title"]}<span class="n">{n}</span></a>')
    items.append('<div class="sidebar-section">부록</div>')
    cls = "side-link current" if current_slug == "archive-index" else "side-link"
    items.append(f'<a class="{cls}" href="archive-index.html">아카이브 색인<span class="n">357</span></a>')
    items.append('<div class="sidebar-section">원본</div>')
    items.append('<a class="side-link" href="../v5/research_story.html">research_story.html ↗</a>')
    return "\n".join(items)


def render_file(md_path):
    text = md_path.read_text()
    # [xxx](slug.md) 형태 내부 위키 링크를 slug.html로 교체(정적 HTML 간 이동용)
    text = re.sub(r"\]\(([a-z0-9-]+)\.md\)", r"](\1.html)", text)
    body_html = markdown.markdown(text, extensions=["tables", "fenced_code", "md_in_html"])
    title_m = re.search(r"<h1[^>]*>(.*?)</h1>", body_html)
    title = re.sub(r"<[^>]+>", "", title_m.group(1)) if title_m else md_path.stem

    slug = md_path.stem
    extra_script = ""
    if slug == "archive-index":
        # 357행 표를 스크롤만으로 훑기 힘드니 검색창 + 카운터 삽입
        search_html = ('<input id="archive-filter" class="archive-search" type="text" '
                       'placeholder="파일명/제목/요약 검색…">'
                       '<div id="archive-count" class="archive-count"></div>')
        body_html = re.sub(r"(<table)", search_html + r"\1", body_html, count=1)
        extra_script = ARCHIVE_SEARCH_SCRIPT

    html = PAGE_TMPL.format(title=title, css=CSS, body=body_html,
                             sidebar=build_sidebar(slug), extra_script=extra_script)
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
