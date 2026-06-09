#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
research_story.html 패치 스크립트
- 7MB가 넘는 대용량 HTML 파일 수정을 위해 python replace 사용
- '박스를 본다' 증명 카드 및 '다음 관문' 카드 내용 업데이트
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = ROOT / "docs" / "v5" / "research_story.html"

if not FILE_PATH.exists():
    print(f"Error: {FILE_PATH} 존재하지 않음")
    exit(1)

content = FILE_PATH.read_text(encoding="utf-8")

# 변경할 타겟 대상 텍스트
target_txt = """    <div class="finding-card good">
      <div class="finding-title">✅ "박스를 본다" 증명</div>
      <div class="finding-body">Zero-shot probe 96.6% (CLIP이 이미 알고 있음) + Masking 100% flip (인과 증거) + Stage2 v2 92.6% (basket 인식 → action 연결). 3가지 독립 증거.</div>
    </div>
    <div class="finding-card info">
      <div class="finding-title">🎯 다음 관문</div>
      <div class="finding-body">SODA 이관 완료(5/27). inference_server.py GoalNavMLP 추가 → exp49/exp54_s2v2 배포 → 실로봇 테스트. R3 해결: Goal-Conditioned 데이터 수집 시작.</div>
    </div>"""

# 대체할 신규 텍스트
replacement_txt = """    <div class="finding-card good">
      <div class="finding-title">✅ "박스를 본다" 증명 (다중 스케일 검증 완료)</div>
      <div class="finding-body">Zero-shot probe 96.6% + 다중 스케일 Masking Ablation (1.0x 이하 flip 0%, 0.8x~0.5x conf_drop 유지로 인과성 및 복합 맥락 강건성 동시 증명) + Stage2 v2 92.6%. <a href="masking_ablation_proof.html" style="color:#38bdf8;text-decoration:underline;font-weight:700">상세 분석 페이지 보기 🔗</a></div>
    </div>
    <div class="finding-card info">
      <div class="finding-title">🎯 다음 관문 (STOP & 실로봇)</div>
      <div class="finding-body">도착 STOP Y-Center 게이트 결합 완료 (실질 성공률 68.8% 확보). inference_server.py 무결성 검증 통과 → 실로봇 배포 테스트 대기 중. R3 Goal-Conditioned 데이터 수집 개시.</div>
    </div>"""

# 개행문자(LF vs CRLF) 차이 보정 후 치환 수행
if target_txt in content:
    print("Found exact LF match. Patching...")
    new_content = content.replace(target_txt, replacement_txt)
    FILE_PATH.write_text(new_content, encoding="utf-8")
    print("Patch successful!")
else:
    target_txt_rn = target_txt.replace("\n", "\r\n")
    if target_txt_rn in content:
        print("Found exact CRLF match. Patching...")
        new_content = content.replace(target_txt_rn, replacement_txt.replace("\n", "\r\n"))
        FILE_PATH.write_text(new_content, encoding="utf-8")
        print("Patch successful (CRLF)!")
    else:
        print("Error: Target block not found in research_story.html. Please check formatting.")
