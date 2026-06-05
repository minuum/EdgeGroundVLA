#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = ROOT / "docs" / "v5" / "research_story.html"

if not FILE_PATH.exists():
    print(f"Error: {FILE_PATH} does not exist!")
    sys.exit(1)

content = FILE_PATH.read_text(encoding="utf-8")

# 1. TOC 부분 교체
# 원래 파일에 ch27이 없는 원래 TOC 상태
old_toc = """    <a class="toc-item" href="#ch24" data-target="ch24"><span class="toc-badge" style="background:rgba(34,197,94,0.15);color:#22c55e">CH 24</span><span class="toc-label">MoNa-Pi 통합 &amp; 243ep 확장</span></a>
    <a class="toc-item" href="#ch25" data-target="ch25"><span class="toc-badge" style="background:rgba(34,197,94,0.15);color:#22c55e">CH 25</span><span class="toc-label">MoNa-Pi 로컬 추론 API</span></a>
    <a class="toc-item" href="#ch26" data-target="ch26"><span class="toc-badge" style="background:rgba(34,197,94,0.15);color:#22c55e">CH 26</span><span class="toc-label">MoNa-Pi 순수 신경망 파이프라인</span></a>
    <a class="toc-item" href="#next-step" data-target="next-step"><span class="toc-badge" style="background:rgba(167,139,250,0.15);color:#a78bfa">NEXT</span><span class="toc-label">Step 3 방향</span></a>"""

new_toc = """    <a class="toc-item" href="#ch24" data-target="ch24"><span class="toc-badge" style="background:rgba(8,186,212,0.15);color:#08bad4">CH 24</span><span class="toc-label">MoNa-Pi 프레임워크 설계 사상</span></a>
    <a class="toc-item" href="#ch25" data-target="ch25"><span class="toc-badge" style="background:rgba(34,197,94,0.15);color:#22c55e">CH 25</span><span class="toc-label">MoNa-Pi 통합 &amp; 243ep 확장</span></a>
    <a class="toc-item" href="#ch26" data-target="ch26"><span class="toc-badge" style="background:rgba(34,197,94,0.15);color:#22c55e">CH 26</span><span class="toc-label">MoNa-Pi 로컬 추론 API</span></a>
    <a class="toc-item" href="#ch27" data-target="ch27"><span class="toc-badge" style="background:rgba(34,197,94,0.15);color:#22c55e">CH 27</span><span class="toc-label">MoNa-Pi 순수 신경망 파이프라인</span></a>
    <a class="toc-item" href="#ch28" data-target="ch28"><span class="toc-badge" style="background:rgba(34,197,94,0.15);color:#22c55e">CH 28</span><span class="toc-label">MoNa-Pi Ablation Study</span></a>
    <a class="toc-item" href="#next-step" data-target="next-step"><span class="toc-badge" style="background:rgba(167,139,250,0.15);color:#a78bfa">NEXT</span><span class="toc-label">6/2 반박대응 &amp; Next</span></a>"""

# 2. 자바스크립트 SECTIONS 부분 수정
old_sections = """  const SECTIONS = ['ch1','ch2','ch3','ch4','ch5','cl-overview','cl-dive',
                    'ch6','ch7','ch8','ch9','ch10','ch11','ch12','ch13','ch14','ch15','ch16','ch17','ch18','ch19','ch20','ch21','ch22','ch23','ch24','ch25','ch26','next-step','summary'];"""

new_sections = """  const SECTIONS = ['ch1','ch2','ch3','ch4','ch5','cl-overview','cl-dive',
                    'ch6','ch7','ch8','ch9','ch10','ch11','ch12','ch13','ch14','ch15','ch16','ch17','ch18','ch19','ch20','ch21','ch22','ch23','ch24','ch25','ch26','ch27','ch28','next-step','summary'];"""


# 치환들 적용
patched = False

# 1. TOC 치환
if old_toc in content:
    content = content.replace(old_toc, new_toc)
    print("Patched TOC successfully!")
    patched = True
else:
    old_toc_rn = old_toc.replace("\n", "\r\n")
    if old_toc_rn in content:
        content = content.replace(old_toc_rn, new_toc.replace("\n", "\r\n"))
        print("Patched TOC (CRLF) successfully!")
        patched = True

# 2. SECTIONS 자바스크립트 치환
if old_sections in content:
    content = content.replace(old_sections, new_sections)
    print("Patched SECTIONS javascript successfully!")
    patched = True
else:
    old_sections_rn = old_sections.replace("\n", "\r\n")
    if old_sections_rn in content:
        content = content.replace(old_sections_rn, new_sections.replace("\n", "\r\n"))
        print("Patched SECTIONS javascript (CRLF) successfully!")
        patched = True

if patched:
    FILE_PATH.write_text(content, encoding="utf-8")
    print("Successfully synchronized TOC and SECTIONS variables in research_story.html!")
else:
    print("Error: Could not apply TOC or SECTIONS patch. Match targets not found.")
    sys.exit(1)
