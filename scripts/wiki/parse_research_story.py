#!/usr/bin/env python3
"""LLM wiki 1단계 — research_story.html(70챕터)을 파싱해 챕터별 구조화 데이터 생성.

docs/plans/plan_20260828_llm_wiki.md 1단계.

**설계 변경 이력**: 처음엔 div 중첩 깊이를 정확히 추적(정규식 스택, 그다음
html.parser.HTMLParser 트리)해서 챕터/카드 블록을 잡으려 했으나, CH6~CH66 구간
어딘가에 실제로 <div>/</div> 짝이 안 맞는 부분이 있어(19,123줄짜리 파일을 여러
세션에 걸쳐 수작업으로 편집한 히스토리 때문 — 파일 전체 합계는 우연히 맞아떨어짐)
CH7이 CH66까지 통째로 삼켜버리는 오류가 반복 재현됨. **정확한 HTML 트리가 필요
없다는 점에 착안**해 접근을 바꿈: 각 챕터/카드는 "다음 같은 종류 마커가 나오기
전까지의 텍스트 구간"으로 정의 — 짝이 안 맞는 div가 있어도 마커 자체(챕터
시작 태그, finding-card/callout 시작 태그)의 선형 순서만 있으면 안전하게 파싱됨.

출력: docs/wiki/data/chapters.json
"""
import base64
import hashlib
import html as html_module
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "docs/v5/research_story.html"
OUT = ROOT / "docs/wiki/data/chapters.json"
BASE_DIR = SRC.parent               # docs/v5 — <img> 상대경로는 이 폴더 기준
WIKI_DIR = ROOT / "docs/wiki"
ASSETS_DIR = WIKI_DIR / "assets"    # base64 임베드 이미지를 파일로 추출해 저장

CHAPTER_START_RE = re.compile(r'<div class="chapter" id="([a-z0-9_-]+)"[^>]*>')
CARD_START_RE = re.compile(r'<div class="(finding-card|callout)\b[^"]*"[^>]*>')
TITLE_RE = re.compile(r'class="finding-title"[^>]*>(.*?)</div>', re.DOTALL)
HEADER_NUM_RE = re.compile(r'class="chapter-num[^"]*"[^>]*>(.*?)</(span|div)>', re.DOTALL)
HEADER_TITLE_RE = re.compile(r'class="chapter-title"[^>]*>(.*?)</(h2|div)>', re.DOTALL)
HEADER_SUB_RE = re.compile(r'class="chapter-subtitle"[^>]*>(.*?)</(p|div)>', re.DOTALL)
HEADER_END_RE = re.compile(r'class="chapter-header"[^>]*>.{0,3000}?</p>', re.DOTALL)

IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
SRC_ATTR_RE = re.compile(r'src="([^"]*)"')
ALT_ATTR_RE = re.compile(r'alt="([^"]*)"')
BASE64_SRC_RE = re.compile(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
BLOCK_CLOSE_RE = re.compile(r"</(p|div|li|tr|td|h[1-6])>", re.IGNORECASE)
LI_OPEN_RE = re.compile(r"<li\b[^>]*>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t]+")
NL_RE = re.compile(r"\n{3,}")

_IMG_SRC_CACHE = {}  # 원본 src -> docs/wiki/ 기준 상대경로(또는 None=미해결)


def resolve_image_src(src):
    """<img src="...">를 docs/wiki/*.md에서 쓸 상대경로로 변환.
    - base64 임베드: 디코드해서 docs/wiki/assets/<sha1>.<ext>로 저장, 그 상대경로 반환.
    - 상대경로: research_story.html 기준(docs/v5/)으로 실제 존재하는 파일을 찾아
      docs/wiki/ 기준 상대경로로 재계산. 원본 HTML 자체에 깨진 경로가 있으면
      (예: "../inference_viz/..."가 실제로는 docs/v5/inference_viz/에 있는 경우)
      파일이 없으므로 None(스킵) — 위키에서 깨진 이미지 아이콘을 보여주지 않기 위함."""
    if src in _IMG_SRC_CACHE:
        return _IMG_SRC_CACHE[src]

    result = None
    m = BASE64_SRC_RE.match(src)
    if m:
        ext = m.group(1).lower()
        ext = {"jpeg": "jpg"}.get(ext, ext)
        try:
            raw = base64.b64decode(m.group(2))
        except Exception:
            raw = None
        if raw:
            h = hashlib.sha1(raw).hexdigest()[:16]
            fname = f"{h}.{ext}"
            ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            outpath = ASSETS_DIR / fname
            if not outpath.exists():
                outpath.write_bytes(raw)
            result = f"assets/{fname}"
    else:
        target = (BASE_DIR / src).resolve()
        if target.exists():
            result = os.path.relpath(target, WIKI_DIR)

    _IMG_SRC_CACHE[src] = result
    return result


def convert_images(fragment):
    """<img> 태그를 markdown 이미지 문법으로 치환(다른 태그 스트리핑 전에 실행돼야
    함 — TAG_RE가 나중에 일반 태그를 지울 때 markdown ![]() 문법은 안 건드림)."""
    def repl(m):
        tag = m.group(0)
        src_m = SRC_ATTR_RE.search(tag)
        if not src_m:
            return ""
        relpath = resolve_image_src(src_m.group(1))
        if not relpath:
            return ""
        alt_m = ALT_ATTR_RE.search(tag)
        alt = alt_m.group(1) if alt_m else ""
        alt = alt.replace("[", "(").replace("]", ")")
        return f"\n\n![{alt}]({relpath})\n\n"
    return IMG_TAG_RE.sub(repl, fragment)


def strip_tags(fragment):
    frag = convert_images(fragment)
    frag = BR_RE.sub("\n", frag)
    frag = BLOCK_CLOSE_RE.sub("\n", frag)
    frag = LI_OPEN_RE.sub("\n- ", frag)
    frag = TAG_RE.sub("", frag)
    frag = html_module.unescape(frag)
    lines = [WS_RE.sub(" ", ln).strip() for ln in frag.split("\n")]
    frag = "\n".join(ln for ln in lines if ln)
    return NL_RE.sub("\n\n", frag).strip()


def extract_cards(chapter_text, header_end):
    """chapter_text 안에서 finding-card/callout 시작 위치를 전부 찾고, 각 카드는
    "자기 시작 ~ 다음 카드 시작(또는 챕터 끝)"까지로 정의해 텍스트 추출.
    header_end 이전(챕터 헤더 영역)은 카드 탐색에서 제외."""
    starts = [(m.start(), m.group(1)) for m in CARD_START_RE.finditer(chapter_text)
              if m.start() >= header_end]
    cards = []
    for i, (start, kind) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(chapter_text)
        block = chapter_text[start:end]
        title_m = TITLE_RE.search(block)
        title = strip_tags(title_m.group(1)) if title_m else None
        body = strip_tags(block)
        if title and body.startswith(title):
            body = body[len(title):].strip()
        cards.append(dict(kind=kind, title=title, text=body))
    if not cards:
        # 구식 챕터(finding-card/callout 클래스 없이 색상 인라인 div로만 구성된
        # CH16~59 일부 등) — 챕터 전체(헤더 이후)를 프로즈 한 덩어리로 대체.
        body = strip_tags(chapter_text[header_end:])
        if body:
            cards.append(dict(kind="prose", title=None, text=body))
    return cards


def parse_chapter(chapter_id, chapter_text):
    header_m = HEADER_END_RE.search(chapter_text)
    header_end = header_m.end() if header_m else min(2000, len(chapter_text))
    num_m = HEADER_NUM_RE.search(chapter_text)
    title_m = HEADER_TITLE_RE.search(chapter_text)
    sub_m = HEADER_SUB_RE.search(chapter_text)
    num = strip_tags(num_m.group(1)) if num_m else None
    title = strip_tags(title_m.group(1)) if title_m else None
    subtitle = strip_tags(sub_m.group(1)) if sub_m else None
    cards = extract_cards(chapter_text, header_end)
    return dict(id=chapter_id, num=num, title=title, subtitle=subtitle,
                n_cards=len(cards), cards=cards)


def main():
    text = SRC.read_text()
    starts = [(m.start(), m.group(1)) for m in CHAPTER_START_RE.finditer(text)]
    chapters = []
    for i, (start, chapter_id) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        chapter_text = text[start:end]
        parsed = parse_chapter(chapter_id, chapter_text)
        chapters.append(parsed)
        print(f"  {parsed['id'] or '-':16s} {parsed['num'] or '-':10s} "
              f"{(parsed['title'] or '(제목없음)')[:50]:50s}  [{parsed['n_cards']}장]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(chapters, indent=2, ensure_ascii=False))
    print(f"\n총 {len(chapters)}개 챕터 파싱 완료 → {OUT}")
    total_cards = sum(c["n_cards"] for c in chapters)
    print(f"총 finding-card/callout {total_cards}개")
    zero_card = [c["id"] for c in chapters if c["n_cards"] == 0]
    if zero_card:
        print(f"⚠️ 카드 0개(수동 확인 필요): {zero_card}")


if __name__ == "__main__":
    main()
