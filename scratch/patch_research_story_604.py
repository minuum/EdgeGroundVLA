#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = ROOT / "docs" / "v5" / "research_story.html"

if not FILE_PATH.exists():
    print(f"Error: {FILE_PATH} does not exist!")
    sys.exit(1)

content = FILE_PATH.read_text(encoding="utf-8")

# 1. 자바스크립트 SECTIONS 변수 업데이트
old_sections = """  const SECTIONS = ['ch1','ch2','ch3','ch4','ch5','cl-overview','cl-dive',\r
                    'ch6','ch7','ch8','ch9','ch10','ch11','ch12','ch13','ch14','ch15','ch16','ch17','ch18','ch19','ch20','ch21','next-step','summary'];"""

# 줄바꿈과 공백의 일관성을 위해 \n 및 \r\n 두 버전 모두 대응할 수 있도록 함
old_sections_unix = """  const SECTIONS = ['ch1','ch2','ch3','ch4','ch5','cl-overview','cl-dive',\n                    'ch6','ch7','ch8','ch9','ch10','ch11','ch12','ch13','ch14','ch15','ch16','ch17','ch18','ch19','ch20','ch21','next-step','summary'];"""

new_sections_unix = """  const SECTIONS = ['ch1','ch2','ch3','ch4','ch5','cl-overview','cl-dive',\n                    'ch6','ch7','ch8','ch9','ch10','ch11','ch12','ch13','ch14','ch15','ch16','ch17','ch18','ch19','ch20','ch21','ch22','ch23','ch27','next-step','summary'];"""

sections_patched = False
if old_sections_unix in content:
    content = content.replace(old_sections_unix, new_sections_unix)
    sections_patched = True
else:
    # 혹시 \r\n일 수 있으므로
    old_sections_crlf = old_sections_unix.replace("\n", "\r\n")
    new_sections_crlf = new_sections_unix.replace("\n", "\r\n")
    if old_sections_crlf in content:
        content = content.replace(old_sections_crlf, new_sections_crlf)
        sections_patched = True

if sections_patched:
    print("Successfully patched SECTIONS array!")
else:
    print("Warning: Could not patch SECTIONS array directly, trying fallback matching...")
    # fallback: 조금 더 느슨한 매칭
    fallback_old = "const SECTIONS = ['ch1','ch2','ch3','ch4','ch5','cl-overview','cl-dive',"
    if fallback_old in content:
        # 파일 내용을 보고 직접 바꾼다.
        # 실제 파일은 이전 python3 -c에서 ch27 라인을 중복 삭제하면서 윈도우 스타일 줄바꿈이 깨졌거나 한 부분이 있을 수 있음.
        # 따라서 단순 replace 시도
        idx = content.find("const SECTIONS =")
        if idx != -1:
            end_idx = content.find("];", idx)
            if end_idx != -1:
                old_block = content[idx:end_idx+2]
                new_block = "const SECTIONS = ['ch1','ch2','ch3','ch4','ch5','cl-overview','cl-dive',\n                    'ch6','ch7','ch8','ch9','ch10','ch11','ch12','ch13','ch14','ch15','ch16','ch17','ch18','ch19','ch20','ch21','ch22','ch23','ch27','next-step','summary'];"
                content = content.replace(old_block, new_block)
                sections_patched = True
                print("Successfully patched SECTIONS array via fallback regex-like logic!")

# 2. 로드맵 nextstep-grid 부분 보완 (D안 카드 신설)
old_grid = """  <div class="nextstep-grid">
    <div class="nextstep-card ns-A">"""

new_grid = """  <div class="nextstep-grid" style="display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:14px">
    <div class="nextstep-card ns-D" style="border:2px solid #a78bfa;background:#130b24;border-radius:12px;padding:20px;display:flex;flex-direction:column;justify-content:space-between">
      <div>
        <div class="ns-badge" style="background:#a78bfa;color:#0a0f1a;font-size:0.7rem;font-weight:800;padding:2px 8px;border-radius:10px;display:inline-block;margin-bottom:8px">D 안 — 6/4 결정</div>
        <div class="ns-title" style="color:#d8b4fe;font-weight:800;font-size:0.95rem;margin-bottom:8px">LoRA 아키텍처 재설계 &amp; 제어 튜닝</div>
        <div class="ns-body" style="color:#cbd5e1;font-size:0.8rem;line-height:1.6">
          SigLIP + DINOv2 비전 레이어 전체에 LoRA를 적용하고 LLM 레이어는 제외하여 일반화 성능 극대화. OOD 데이터 증강 및 액션 청크 길이 최적화를 통해 실물 주행 오버슈팅 차단.
        </div>
      </div>
    </div>
    <div class="nextstep-card ns-A">"""

grid_patched = False
if old_grid in content:
    content = content.replace(old_grid, new_grid)
    grid_patched = True
else:
    old_grid_rn = old_grid.replace("\n", "\r\n")
    if old_grid_rn in content:
        content = content.replace(old_grid_rn, new_grid.replace("\n", "\r\n"))
        grid_patched = True

if grid_patched:
    print("Successfully patched nextstep-grid with Option D!")
else:
    print("Error: nextstep-grid match targets not found!")

# 3. 요약 배너 (현재 핵심 결론 5/28 -> 6/4 최신 업데이트)
old_banner = """    <div style="color:#86efac;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px">현재 핵심 결론 (5/28 업데이트)</div>
    <div style="color:#e2e8f0;font-size:1.05rem;font-weight:700;line-height:1.7">
      "basket을 본다는 것을 5-Track으로 증명(R1 ✅)하고,<br>
      CL 96.7%를 4개 모델이 재현하며 안정성 입증(R2-1 ✅).<br>
      <span style="color:#86efac">Stage 1 v2 LoRA: left 91.1%→97.3%(+6.2%p) — 세 방향 97%+ 균등화(R2-2 ✅)</span><br>
      <span style="color:#86efac">Exp57 PaliGemma LoRA: gray basket 100% / red ball 0% → R2-3 해결 ✅</span><br>
      <span style="color:#fbbf24">R2-4는 구조적 한계</span>, <span style="color:#fbbf24">R3는 Goal-Conditioned으로 해결 계획</span>."
    </div>"""

new_banner = """    <div style="color:#86efac;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px">현재 핵심 결론 (6/4 업데이트)</div>
    <div style="color:#e2e8f0;font-size:1.05rem;font-weight:700;line-height:1.7">
      "PaliGemma-3B 기반 그라운딩 성능 및 Closed-Loop 오프라인 96.2% 성공률을 확보하고,<br>
      동기식 수집 데이터 적용 및 π0 기반 로봇 주행 검증으로 <span style="color:#4ade80;font-weight:700">RISE 사업 은상</span>을 수상했습니다.<br>
      <span style="color:#86efac">R3 실물 OOD/조향 병목 극복을 위해 DINOv2 + SigLIP 비전 인코더 전체 LoRA 재학습을 적용하고</span><br>
      <span style="color:#86efac">LLM 레이어 튜닝 제외 및 데이터 증강 확대, 액션 청크 길이 최적화를 진행합니다.</span>"
    </div>"""

banner_patched = False
if old_banner in content:
    content = content.replace(old_banner, new_banner)
    banner_patched = True
else:
    old_banner_rn = old_banner.replace("\n", "\r\n")
    if old_banner_rn in content:
        content = content.replace(old_banner_rn, new_banner.replace("\n", "\r\n"))
        banner_patched = True

if banner_patched:
    print("Successfully patched summary banner to 6/4 updates!")
else:
    print("Error: summary banner match targets not found!")

# 4. finding-card info 부분 업데이트
old_card = """    <div class="finding-card info">
      <div class="finding-title">🎯 다음 관문 (STOP & 실로봇)</div>
      <div class="finding-body">도착 STOP Y-Center 게이트 결합 완료 (실질 성공률 68.8% 확보). inference_server.py 무결성 검증 통과 → 실로봇 배포 테스트 대기 중. R3 Goal-Conditioned 데이터 수집 개시.</div>
    </div>"""

new_card = """    <div class="finding-card info">
      <div class="finding-title">🎯 다음 관문 (LoRA 재설계 & 제어 최적화)</div>
      <div class="finding-body">실로봇 테스트 완료 및 RISE 은상 수상. 실주행 시 발생한 OOD 인식 실패와 조향 오버슈팅 해소를 위해 비전 인코더(SigLIP + DINOv2) 통합 LoRA 학습 및 LLM 제외 튜닝 돌입. 액션 청크 최적화 및 다각도 데이터 증강 적극 도입.</div>
    </div>"""

card_patched = False
if old_card in content:
    content = content.replace(old_card, new_card)
    card_patched = True
else:
    old_card_rn = old_card.replace("\n", "\r\n")
    if old_card_rn in content:
        content = content.replace(old_card_rn, new_card.replace("\n", "\r\n"))
        card_patched = True

if card_patched:
    print("Successfully patched finding-card info to 6/4 updates!")
else:
    print("Error: finding-card info match targets not found!")

# 파일 저장
if sections_patched or grid_patched or banner_patched or card_patched:
    FILE_PATH.write_text(content, encoding="utf-8")
    print("Successfully applied patches to research_story.html!")
else:
    print("Error: No patches were applied.")
    sys.exit(1)
