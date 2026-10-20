"""CH77/CH78 공통: 본문을 마커(⓪~⑧)로 쪼개 탭 패널로 재배치, 챕터 블록 삽입(TOC·상단 버튼·스크롤 스파이·SUMMARY)."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent.parent / "docs/v5/research_story.html"
MARK = re.compile(r'<div style="margin:16px 0 6px[^>]*>(⓪|①|②|③|④|⑤|⑥-\d+|⑥|⑦|⑧)[ ]')


def split_parts(body_):
    mk = list(MARK.finditer(body_))
    head = body_[:mk[0].start()]
    parts = {}
    for i, m in enumerate(mk):
        end = mk[i + 1].start() if i + 1 < len(mk) else len(body_)
        parts[m.group(1)] = body_[m.start():end]
    return head, parts


def tabs_html(ch, tabs, head, parts, raw=None):
    """tabs: [(탭id, 라벨, [항목...])]. 항목: 'HEAD'(본문 머리) | 마커 문자열(parts 키) | ('html', 문자열)"""
    raw = raw or {}
    bar = "".join(f'<button class="ctab-btn" data-t="{t}" onclick="ctabShow(\'{ch}\',\'{t}\')">{lab}</button>' for t, lab, _ in tabs)
    panels = ""
    for t, lab, items in tabs:
        inner = ""
        for it in items:
            if it == "HEAD":
                inner += head
            elif isinstance(it, tuple):
                inner += it[1]
            else:
                inner += parts.get(it, "")
        panels += f'<div class="ctab-panel" data-t="{t}" style="display:none">{inner}</div>\n'
    return f'<div class="ctab-bar">{bar}</div>\n{panels}<script>ctabInit("{ch}")</script>'


def chapter_block(num, bg, fg, title, subtitle, date, inner):
    return (f'<!--CH{num}--><hr class="chapter-divider">\n<div class="chapter" id="ch{num}">\n  <div class="chapter-header">\n'
            f'    <span class="chapter-num" style="background:{bg};color:#0a0f1a">CH {num}</span>\n    <h2 class="chapter-title">{title}</h2>\n'
            f'    <p class="chapter-subtitle"><span style="color:{fg};font-weight:800">{date}</span> · {subtitle}</p>\n  </div>\n'
            f'{inner}\n</div>\n<!-- ══ /CH{num} ═ -->\n<!--/CH{num}-->\n')


def insert_chapter(s, num, block, prev_marker, toc_label, nav_label, cat, bg, fg, border, nav_bg, summary_row, prev_ids):
    """prev_marker: 이 챕터 앞 챕터의 끝 표식(문자열). prev_ids: (이전 챕터 번호) — TOC·버튼·SUMMARY 행을 그 뒤에 넣는다."""
    s = re.sub(rf"<!--CH{num}-->.*?<!--/CH{num}-->\n?", "", s, flags=re.S)
    k = s.index(prev_marker) + len(prev_marker)
    k = s.index("\n", k) + 1
    s = s[:k] + "\n" + block + s[k:]
    if f'data-target="ch{num}"' not in s:
        p = prev_ids
        toc = s.index(f'<a class="toc-item" href="#ch{p}"'); e = s.index("</a>", toc) + 4
        s = s[:e] + (f'\n    <a class="toc-item" href="#ch{num}" data-target="ch{num}"><span class="toc-badge" style="background:rgba(251,191,36,0.2);color:{fg};font-weight:900">CH {num}</span>'
                     f'<span class="toc-label">{toc_label}</span><span class="toc-cat cat-exp" style="background:rgba(251,191,36,0.15);color:{bg}">{cat}</span></a>') + s[e:]
        nav = s.index(f'<a href="#ch{p}" style="padding:7px 16px'); e = s.index("</a>", nav) + 4
        s = s[:e] + (f'\n    <a href="#ch{num}" style="padding:7px 16px;background:{nav_bg};border:1px solid {border};border-radius:8px;font-size:0.78rem;font-weight:800;color:{fg};text-decoration:none">{nav_label}</a>') + s[e:]
        s = s.replace(f"'ch{p}','next-step','summary'", f"'ch{p}','ch{num}','next-step','summary'", 1) if f"'ch{p}','next-step'" in s else s.replace(f"'ch{p}','ch", f"'ch{p}','ch{num}','ch", 1)
        row = re.search(rf"<tr><td[^>]*><a href='#ch{p}'.*?</tr>", s, flags=re.S)
        s = s.replace(row.group(0), row.group(0) + "\n" + summary_row, 1)
    return s


def balance(s):
    out = []
    for t in ["div", "svg", "table", "tr", "button", "a", "ul", "li"]:
        out.append((t, len(re.findall(f"<{t}[ >]", s)), len(re.findall(f"</{t}>", s))))
    return out


def summary_row(num, title, text, color="#fbbf24"):
    td = 'style="padding:6px 10px;border-bottom:1px solid #1e293b"'
    return (f'<tr><td style="padding:6px 10px;border-bottom:1px solid #1e293b;white-space:nowrap"><a href=\'#ch{num}\' style=\'color:{color}\'>CH{num}</a></td>'
            f'<td {td}>{title}</td><td {td}>{text}</td></tr>')


def replace_summary_row(s, num, title, text, color="#fbbf24"):
    pat = re.compile(rf"<tr><td[^>]*><a href='#ch{num}'.*?</tr>", re.S)
    return pat.sub(lambda m: summary_row(num, title, text, color), s, count=1)


def replace_toc_label(s, num, label):
    pat = re.compile(rf'(<a class="toc-item" href="#ch{num}"[^>]*>.*?<span class="toc-label">)(.*?)(</span>)', re.S)
    return pat.sub(lambda m: m.group(1) + label + m.group(3), s, count=1)
