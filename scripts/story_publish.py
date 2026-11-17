"""research_story 발행 스크립트 (멱등).

원본(source of truth) `docs/v5/research_story_src.html` 은 gen_ch*.py 가 챕터를 끼워 넣는 전체 문서다.
이 스크립트는 원본을 읽어
  1) base64 인라인 이미지를 `docs/v5/figures/inline/<해시>.<확장자>` 파일로 빼고(원본도 같이 가벼워진다),
     `<img>` 에 loading="lazy" decoding="async" 를 붙이고,
  2) CH74~ (EGVLA2) 를 `research_story_egvla2.html` 로 분리해
     `research_story.html`(CH1~73 + 요약) 과 두 쪽을 발행한다.
두 쪽 사이의 `#chNN` 링크는 자동으로 상대 쪽으로 바꾸고, 옛 주소(`research_story.html#ch82`)는 JS로 새 쪽으로 넘긴다.

사용: python3 scripts/story_publish.py   (gen_ch*.py 끝에서 자동 호출됨)
"""
from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "docs/v5"
SRC = V5 / "research_story_src.html"
MAIN = V5 / "research_story.html"
EG2 = V5 / "research_story_egvla2.html"
IMG_DIR = V5 / "figures/inline"
SPLIT_FROM = 74                      # 이 번호부터 EGVLA2 쪽

EXT = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "webp": "webp", "gif": "gif", "svg+xml": "svg"}
DATA_RE = re.compile(r"data:image/([a-z+]+);base64,([A-Za-z0-9+/=\s]+)")   # 줄바꿈·공백이 섞인 것도 있다


def externalize(s: str) -> tuple[str, int]:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    n = 0

    def rep(m: re.Match) -> str:
        nonlocal n
        ext = EXT.get(m.group(1))
        if ext is None:
            return m.group(0)
        try:
            raw = base64.b64decode(re.sub(r'\s+', '', m.group(2)), validate=True)
        except Exception:
            return m.group(0)                 # 예시 문자열·잘린 데이터는 그대로 둔다
        name = hashlib.sha1(raw).hexdigest()[:16] + "." + ext
        p = IMG_DIR / name
        if not p.exists():
            p.write_bytes(raw)
        n += 1
        return f"figures/inline/{name}"

    s = DATA_RE.sub(rep, s)

    def lazy(m: re.Match) -> str:
        tag = m.group(0)
        if "loading=" in tag:
            return tag
        end = "/>" if tag.endswith("/>") else ">"
        return tag[:-len(end)].rstrip() + ' loading="lazy" decoding="async"' + end

    s = re.sub(r"<img\b[^>]*>", lazy, s)
    return s, n


def _ids(html: str) -> set[str]:
    return set(re.findall(r'\bid=["\']([^"\']+)["\']', html))


def _retarget(html: str, other_page: str, own_ids: set[str]) -> str:
    """이 쪽에 없는 #id 로 가는 링크를 다른 쪽 페이지로 바꾼다."""
    def rep(m: re.Match) -> str:
        q, frag = m.group(1), m.group(2)
        if frag in own_ids or frag == "":
            return m.group(0)
        return f'href={q}{other_page}#{frag}{q}'
    return re.sub(r"href=([\"'])#([^\"'#]*)\1", rep, html)


def _toc_item(label_html: str) -> str:
    return label_html


def publish() -> None:
    s = SRC.read_text(encoding="utf-8")
    s, n = externalize(s)
    SRC.write_text(s, encoding="utf-8")

    a = s.index(f'<div class="chapter" id="ch{SPLIT_FROM}"')
    k = s.rfind('<hr class="chapter-divider">', 0, a)
    if k != -1 and a - k < 60:
        a = k
    k = s.rfind("<!--CH%d-->" % SPLIT_FROM, 0, a + 1)
    if k != -1 and a - k < 80:
        a = k
    b = re.search(r'<[a-z]+[^>]*\bid="summary"', s).start()      # CH 블록은 요약 섹션 앞에서 끝난다
    chapters = s[a:b]
    body_main = s[:a] + s[b:]

    head_end = s.index("<body>") + len("<body>")
    head = s[:head_end]
    toc_start = s.index('<div id="toc-panel">')
    toc_end = s.index("</nav>", toc_start)
    toc_panel = s[toc_start:toc_end]
    ctab = s[s.rindex("<script>", 0, s.index("function ctabShow")): s.index("</script>", s.index("function ctabInit")) + 9]

    # EGVLA2 쪽 TOC: CH74+ 항목만
    eg_items = [m.group(0) for m in re.finditer(r'<a class="toc-item" href="#(ch\d+)" data-target="\1">.*?</a>', toc_panel, flags=re.S)
                if int(m.group(1)[2:]) >= SPLIT_FROM]
    eg_ids = [m for m in re.findall(r'<div class="chapter" id="(ch\d+)"', chapters)]
    eg_set = set(eg_ids)

    # 메인: TOC 항목 중 EGVLA2 쪽 항목은 새 페이지로 연결(스크롤 스파이 대상에서 제외)
    def main_toc(m: re.Match) -> str:
        num = int(m.group(1)[2:])
        if num < SPLIT_FROM:
            return m.group(0)
        t = m.group(0).replace(f'href="#{m.group(1)}"', f'href="research_story_egvla2.html#{m.group(1)}"', 1)
        return t.replace(f' data-target="{m.group(1)}"', "", 1)

    main_html = re.sub(r'<a class="toc-item" href="#(ch\d+)" data-target="\1">.*?</a>', main_toc, body_main, flags=re.S)
    main_html = _retarget(main_html, "research_story_egvla2.html", _ids(main_html))
    shim = ("<script>(function(){var R=/^#(ch(?:7[4-9]|[89]\\d|\\d{3,}))(?![\\w-])/;"
            "function go(){var m=R.exec(location.hash);if(m&&!document.getElementById(m[1]))location.replace('research_story_egvla2.html'+location.hash)}"
            "go();window.addEventListener('hashchange',go)})()</script>\n")
    main_html = main_html.replace("<body>", "<body>\n" + shim, 1)
    main_html = re.sub(r"research_story\.html#(ch(?:7[4-9]|[89]\d))", r"research_story_egvla2.html#\1", main_html)
    MAIN.write_text(main_html, encoding="utf-8")

    # EGVLA2 쪽
    eg_head = re.sub(r"<title>.*?</title>", "<title>MoNaVLA — EGVLA2 연구 기록 (CH74~)</title>", head, count=1, flags=re.S)
    toc = ('<nav id="toc">\n  <button id="toc-toggle" onclick="document.getElementById(\'toc-panel\').classList.toggle(\'open\')">목차</button>\n'
           '  <div id="toc-panel">\n'
           '    <a class="toc-item" href="research_story.html"><span class="toc-badge" style="background:rgba(148,163,184,0.25);color:#cbd5e1;font-weight:900">←</span><span class="toc-label">CH1~73 전체 연구 여정</span></a>\n    '
           + "\n    ".join(eg_items) + "\n  </div>\n</nav>\n")
    banner = ('<div class="hero" style="padding-bottom:20px"><h1>EGVLA2 연구 기록 (CH74~)</h1>'
              '<p class="subtitle">Florence-2 단일화 이후의 학습·배포·실기·모델 비교 기록입니다. 이전 챕터(CH1~73)는 '
              '<a href="research_story.html" style="color:#60a5fa">전체 연구 여정</a>에 있습니다.</p></div>\n')
    spy = ("<script>\n(function(){const S=" + str(eg_ids).replace("'", '"') + ";const items=document.querySelectorAll('.toc-item[data-target]');"
           "function upd(){const mid=window.scrollY+window.innerHeight*0.35;let a=S[0];for(const id of S){const el=document.getElementById(id);"
           "if(el&&el.getBoundingClientRect().top+window.scrollY<=mid)a=id}items.forEach(i=>i.classList.toggle('active',i.dataset.target===a))}"
           "window.addEventListener('scroll',upd,{passive:true});upd();"
           "items.forEach(i=>i.addEventListener('click',()=>document.getElementById('toc-panel').classList.remove('open')))})();\n</script>\n")
    eg_body = ctab + "\n" + chapters + "\n" + spy
    eg_html = eg_head + "\n" + toc + banner + eg_body + "\n</body>\n</html>\n"
    eg_html = _retarget(eg_html, "research_story.html", _ids(eg_html))
    eg_html = re.sub(r"research_story\.html#(ch(?:7[4-9]|[89]\d))", r"#\1", eg_html)
    EG2.write_text(eg_html, encoding="utf-8")
    print(f"story_publish: 이미지 {n}개 파일화 | src {len(s)/1e6:.2f}MB → main {len(main_html)/1e6:.2f}MB + egvla2 {len(eg_html)/1e6:.2f}MB | EGVLA2 챕터 {len(eg_ids)}개")


if __name__ == "__main__":
    publish()
