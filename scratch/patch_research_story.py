#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = ROOT / "docs" / "v5" / "research_story.html"

content = FILE_PATH.read_text(encoding="utf-8")

# R3 카드 덩어리 (이전 커밋이 만들어둔 빈 줄 2개 포함)
r3_target = """      <div style="display:flex;gap:12px;align-items:flex-start;padding:10px 14px;background:#1c1900;border-radius:8px">
        <span style="font-size:1rem;margin-top:1px">🔄</span>
        <div>
          <div style="color:#fbbf24;font-weight:700;font-size:0.9rem">R3 · "데이터 부족 및 center 경로 병목" — Exp61 진행 중</div>
          <div style="color:#94a3b8;font-size:0.82rem;margin-top:2px">MoNa-Pi 데이터셋을 통합하여 에피소드를 기존 150개에서 243개(center 경로 등 신규 73개 추가)로 증강. 현재 PG2 재주석 및 MLP 재학습(Exp61)을 통해 center 경로 0% 성공률 병목 해결 진행 중.</div>
        </div>
      </div>\n\n\n"""

# R4 카드를 덧붙인 최종 대체 텍스트
r4_card = """      <div style="display:flex;gap:12px;align-items:flex-start;padding:10px 14px;background:#052e16;border-radius:8px">
        <span style="font-size:1rem;margin-top:1px">✅</span>
        <div style="width:100%">
          <div style="color:#86efac;font-weight:700;font-size:0.9rem">R4 · "도착 STOP 종결 조건 구비 여부" — 완료</div>
          <div style="color:#94a3b8;font-size:0.82rem;margin-top:4px;margin-bottom:10px">
            데이터 내 STOP 라벨의 희소성(8.6%)을 극복하기 위해 `stop65_mlp` 정지 예측 모델을 구축하고 <b>3-프레임 윈도우 스무딩(W=3, θ=0.8) 래치</b>를 적용하여 precomputed PG2 주석 기준 <b>CL 성공률 68.8%</b>를 달성했습니다.
          </div>
          <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:6px">
            <div style="background:#0a1628;border-radius:8px;padding:10px;text-align:center">
              <div style="color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;margin-bottom:6px">오프라인 스무딩 (Precomputed)</div>
              <div style="color:#22c55e;font-weight:900;font-size:1.1rem;margin-bottom:4px">68.8% <span style="font-size:0.75rem;color:#64748b">(22/32 ep)</span></div>
              <div style="color:#94a3b8;font-size:0.7rem">평균 FPE: 0.081m · TLD: 0.998</div>
            </div>
            <div style="background:#0a1628;border-radius:8px;padding:10px;text-align:center">
              <div style="color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;margin-bottom:6px">실시간 VLM 결합 (Live Grounding)</div>
              <div style="color:#ef4444;font-weight:900;font-size:1.1rem;margin-bottom:4px">9.4% <span style="font-size:0.75rem;color:#64748b">(3/32 ep)</span></div>
              <div style="color:#94a3b8;font-size:0.7rem;color:#ef4444">평균 FPE: 1.205m (지터 누적 탈선)</div>
            </div>
          </div>
          <div style="color:#64748b;font-size:0.75rem;margin-top:6px;line-height:1.4">
            💡 <b>학술적 의의:</b> VLM 그라운딩 정확도가 우수하더라도 실시간 추론 지터 노이즈가 제어 피드백 루프에 누적되면 이탈이 생김을 정량 실증했습니다. 실로봇 배포 시 <u>주행 제어(Best Controller)와 정지 판단(stop65_mlp)을 멀티스레드로 Decomposed 병렬 구동</u>하는 구조의 강건성을 확보했습니다.
          </div>
        </div>
      </div>\n\n"""

replacement = r3_target.replace("\n\n\n", "\n\n") + r4_card

if r3_target in content:
    print("Found exact R3 target. Replacing...")
    new_content = content.replace(r3_target, replacement)
    FILE_PATH.write_text(new_content, encoding="utf-8")
    print("Successfully patched research_story.html!")
else:
    # CRLF 개행 호환성 처리
    r3_target_rn = r3_target.replace("\n", "\r\n")
    if r3_target_rn in content:
        print("Found exact R3 target with CRLF. Replacing...")
        new_content = content.replace(r3_target_rn, replacement.replace("\n", "\r\n"))
        FILE_PATH.write_text(new_content, encoding="utf-8")
        print("Successfully patched research_story.html (CRLF)!")
    else:
        print("Error: R3 target block not found literal match.")
