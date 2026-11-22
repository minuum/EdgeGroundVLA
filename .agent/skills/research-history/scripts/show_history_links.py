#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
메인 리서치 히스토리 웹서버 주소 및 최신 챕터 링크 출력 스크립트.
(커밋 79c36d12 반영: CH1~73은 research_story.html, CH74~는 research_story_egvla2.html, 원본은 research_story_src.html)
"""

import argparse
import re
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = ROOT / "docs"
SRC_FILE = DOCS_DIR / "v5/research_story_src.html"
EG2_FILE = DOCS_DIR / "v5/research_story_egvla2.html"
MAIN_FILE = DOCS_DIR / "v5/research_story.html"

GITHUB_PAGES_BASE = "https://minuum.github.io/EdgeGroundVLA"


def find_active_docs_port():
    """MoNaVLA docs 디렉토리를 서빙 중인 포트 탐색 (8001, 8766, 8002, 8765 순서)"""
    for port in (8001, 8766, 8002, 8765):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.4)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                try:
                    import urllib.request
                    res = urllib.request.urlopen(f"http://127.0.0.1:{port}/v5/research_story.html", timeout=0.8)
                    if res.status == 200:
                        return port
                except Exception:
                    pass
    return 8001


def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def get_latest_chapter():
    target_file = EG2_FILE if EG2_FILE.exists() else (SRC_FILE if SRC_FILE.exists() else MAIN_FILE)
    if not target_file.exists():
        return "ch82", "EGVLA2 최신 분석"

    content = target_file.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r'<div class="chapter" id="(ch\d+)"', content, re.IGNORECASE)
    if not matches:
        matches = re.findall(r'id=["\'](ch\d+)["\']', content, re.IGNORECASE)
    if matches:
        nums = sorted(int(m[2:]) for m in matches if m[2:].isdigit())
        if nums:
            latest = f"ch{nums[-1]}"
            return latest, f"최신 챕터 ({latest.upper()})"
    return "ch82", "EGVLA2 최신 분석"


def page_for_chapter(ch_str: str) -> str:
    m = re.match(r"ch(\d+)", ch_str.lower())
    if m and int(m.group(1)) >= 74:
        return "research_story_egvla2.html"
    return "research_story.html"


def format_links(ch=None):
    port = find_active_docs_port()
    running = is_port_open(port)
    latest_ch, _ = get_latest_chapter()
    target_ch = ch.lower() if ch else latest_ch
    target_page = page_for_chapter(target_ch)

    status_badge = f"🟢 RUNNING (포트 {port})" if running else f"🔴 STOPPED (포트 {port})"

    msg = f"""
┌─────────────────────────────────────────────────────────────┐
│ 📚 MoNaVLA Research History & Interactive Studios ({status_badge}) │
└─────────────────────────────────────────────────────────────┘
• 📖 메인 연구 스토리 (CH1~73 + 요약, 2.1MB):
  👉 http://localhost:{port}/v5/research_story.html

• 🚀 EGVLA2 연구 기록 (CH74~82, 2.3MB) & 최신 챕터 (#{target_ch}):
  👉 http://localhost:{port}/v5/research_story_egvla2.html
  👉 http://localhost:{port}/v5/{target_page}#{target_ch}

• 🧠 모델 세대 레지스트리 & 실제 세션 의사결정(Factor) 스튜디오:
  👉 http://localhost:{port}/v5/model_registry.html#decision-studio

• 🎮 11/13 + 11/16 (110세션 1,458장) 동시 비교 재생기:
  👉 http://localhost:{port}/v5/object_test_compare_1113.html

• 🌐 GitHub Pages 원격 링크:
  👉 {GITHUB_PAGES_BASE}/v5/{target_page}#{target_ch}
  👉 {GITHUB_PAGES_BASE}/v5/model_registry.html
"""
    if not running:
        msg += f"""
※ 로컬 서버 시작 방법:
  cd {DOCS_DIR} && nohup python3 -m http.server {port} > /tmp/monavla_docs_{port}.log 2>&1 &
"""
    return msg.strip()


def main():
    parser = argparse.ArgumentParser(description="리서치 히스토리 및 시연 링크 출력 유틸")
    parser.add_argument("--ch", type=str, default=None, help="이동할 챕터 번호 (예: ch82)")
    args = parser.parse_args()

    print(format_links(args.ch))


if __name__ == "__main__":
    main()
