#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = ROOT / "docs" / "v5" / "research_story.html"

if not FILE_PATH.exists():
    print(f"Error: {FILE_PATH} does not exist!")
    sys.exit(1)

content = FILE_PATH.read_text(encoding="utf-8")

# 삽입할 CH 27 HTML 내용
ch27_html = """
<div class="chapter" id="ch27">
  <div class="chapter-header">
    <span class="chapter-num" style="background:#a78bfa;color:#0a0f1a">CH 27</span>
    <h2 class="chapter-title">MoNa-Pi: π0 (Pi-zero) 기반 Flow Matching VLA 프레임워크 설계 사상</h2>
  </div>
  <p class="chapter-desc">
    본 연구의 모태이자 핵심 설계 사상인 Physical Intelligence의 <strong>π0 (Pi-zero)</strong> 모델을 기반으로, 
    <strong>Flow Matching Action Head</strong>와 <strong>AdaLN-Zero 시간 컨디셔닝</strong>이 적용된 하이브리드 분해형 VLA 프레임워크 <strong>MoNa-Pi</strong>를 정의합니다.
  </p>

  <!-- ① 제안 배경 및 핵심 문제점 -->
  <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:12px;padding:20px;margin-bottom:20px">
    <div style="color:#7dd3fc;font-size:0.82rem;font-weight:700;margin-bottom:12px">① Motivation: π0 모델의 핵심 가치와 연속 액션 제어</div>
    <div style="font-size:0.8rem;color:#94a3b8;line-height:1.8">
      기존 VLA 모델들(예: RT-1, RT-2, Kosmos-2)은 조향과 속도 명령을 유한한 이산 토큰(Discrete Tokens)으로 분류하여 출력하므로 거동이 끊기고 부드럽지 못한 조향 제어를 보였습니다. 
      본 연구는 이를 극복하기 위해 Physical Intelligence의 <strong>π0 (Pi-zero)</strong> 모델을 모티브로 삼아, <strong>Flow Matching(또는 Diffusion)</strong> 기술을 기반으로 연속 액션 공간(Continuous Action Space)에서 최적의 주행 제어 궤적을 직접 생성하는 프레임워크를 수립했습니다.
      <ul style="margin-top:8px;padding-left:20px">
        <li><strong>Flow Matching Action Head</strong>: 9개의 불연속 조향 클래스로 분류하는 대신, 실시간 3차원 연속 액션 벡터 <code style="background:#1e2438;padding:1px 4px;border-radius:3px">[linear_x, linear_y, angular_z]</code>를 예측하여 부드럽고 자연스러운 물리 제어를 도출합니다.</li>
        <li><strong>Action Chunking (Multi-step Prediction)</strong>: 매 프레임 단일 액션만 예측하는 병목에서 탈피, 한 번에 미래의 N-step(예: 16~50 step) 액션 시퀀스를 동시에 생성해 냄으로써 제어 주기를 50Hz 이상으로 대폭 끌어올릴 수 있는 기틀을 마련했습니다.</li>
      </ul>
    </div>
  </div>

  <!-- ② AdaLN-Zero 기반 시간 컨디셔닝 아키텍처 -->
  <div style="background:#07111e;border:1px solid #1e3040;border-radius:12px;padding:20px;margin-bottom:20px">
    <div style="color:#a78bfa;font-size:0.82rem;font-weight:700;margin-bottom:12px">② 핵심 아키텍처: AdaLN-Zero (Adaptive Layer Normalization) 컨디셔닝</div>
    <div style="font-size:0.8rem;color:#94a3b8;line-height:1.8">
      VLA Flow Model의 핵심은 노이즈 제거 과정의 timestep $t$와 VLM이 추출한 시각/언어 조건부 임베딩 $c$를 어떻게 Action Expert 트랜스포머에 투입할 것인가입니다. MoNa-Pi는 π0 논문 및 DiT(Diffusion Transformer)의 핵심인 <strong>AdaLN-Zero 시간 컨디셔닝</strong>을 적용했습니다.
      <ul style="margin-top:8px;padding-left:20px">
        <li><strong>동적 스케일 및 쉬프트 주입:</strong> 단순히 입력 임베딩에 timestep 임베딩을 더하던 기존의 단순 덧셈 방식과 달리, $t$를 MLP에 통과시켜 각 Layer Normalization 블록마다 scale/shift 파라미터 $(\\alpha, \\beta, \\gamma)$로 매핑해 동적으로 곱하고 더해줍니다.</li>
        <div style="background:#040d1a;padding:10px;border-radius:6px;font-family:monospace;font-size:0.75rem;margin:8px 0;border:1px solid #334155;color:#e2e8f0;line-height:1.4">
          // self-attention block<br>
          h = h + &gamma;1 * self_attn( &alpha;1 * norm1(h) + &beta;1 )<br><br>
          // mlp block<br>
          h = h + &gamma;2 * mlp( &alpha;2 * norm3(h) + &beta;2 )
        </div>
        <li><strong>Zero Initialization 효과:</strong> 학습 초기 단계에서 게이팅 변수 $\\gamma$를 0으로 초기화(Zero-init)하여 잔차 연결(Residual Block)이 항등 함수(Identity mapping)로 시작하도록 유도하여 학습 안정성을 극대화합니다.</li>
      </ul>
    </div>
  </div>

  <!-- ③ 타 VLA 논문들과의 아키텍처 정량/정성 비교 -->
  <div style="background:#08111e;border:1px solid #1e3040;border-radius:12px;padding:20px;margin-bottom:20px">
    <div style="color:#fbbf24;font-size:0.82rem;font-weight:700;margin-bottom:12px">③ 주요 VLA 프레임워크와의 구조적 비교 분석 (π0 기반)</div>
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:0.75rem;text-align:center;color:#cbd5e1">
        <thead>
          <tr style="background:#2d1b4e;color:#a78bfa">
            <th style="padding:8px 10px;text-align:left;border:1px solid #4c2885">프레임워크</th>
            <th style="padding:8px 10px;border:1px solid #4c2885">모델 구조</th>
            <th style="padding:8px 10px;border:1px solid #4c2885">시간 컨디셔닝 기법</th>
            <th style="padding:8px 10px;border:1px solid #4c2885">제어 출력 및 빈도</th>
            <th style="padding:8px 10px;border:1px solid #4c2885">필요 데이터 규모</th>
            <th style="padding:8px 10px;border:1px solid #4c2885;text-align:left">학술적 근거 및 출처</th>
          </tr>
        </thead>
        <tbody>
          <tr style="background:#130b24;border-bottom:1px solid #4c2885">
            <td style="padding:8px 10px;text-align:left;font-weight:700">RT-2</td>
            <td style="padding:8px 10px">Single E2E Transformer</td>
            <td style="padding:8px 10px">없음 (이산 토큰 분류)</td>
            <td style="padding:8px 10px;color:#f87171">이산 액션 / ~3Hz</td>
            <td style="padding:8px 10px">수십만 ep + WebLI</td>
            <td style="padding:8px 10px;text-align:left;color:#94a3b8">RT-2 (CoRL 2023), Section 4.2</td>
          </tr>
          <tr style="background:#130b24;border-bottom:1px solid #4c2885">
            <td style="padding:8px 10px;text-align:left;font-weight:700">Octo Policy</td>
            <td style="padding:8px 10px">Transformer + Diffusion Head</td>
            <td style="padding:8px 10px">FiLM (Feature-wise Linear Modulation)</td>
            <td style="padding:8px 10px;color:#fca5a5">연속 액션 / ~10Hz</td>
            <td style="padding:8px 10px">Open X-Embodiment</td>
            <td style="padding:8px 10px;text-align:left;color:#94a3b8">Octo Policy (CoRL 2024), Section 3.1</td>
          </tr>
          <tr style="background:#130b24;border-bottom:1px solid #4c2885">
            <td style="padding:8px 10px;text-align:left;font-weight:700">π0 (Pi-zero)</td>
            <td style="padding:8px 10px">ViT + Gated LLM + Flow Matching (E2E)</td>
            <td style="padding:8px 10px">AdaLN-Zero (Adaptive Layer Norm)</td>
            <td style="padding:8px 10px;color:#4ade80;font-weight:700">연속 액션 / 50Hz</td>
            <td style="padding:8px 10px">수백만 ep + 교차 로봇 데이터</td>
            <td style="padding:8px 10px;text-align:left;color:#94a3b8">π0 (Physical Intelligence 2024), Section 3</td>
          </tr>
          <tr style="background:#091b35;border-bottom:1px solid #1d4ed8">
            <td style="padding:8px 10px;text-align:left;font-weight:700;color:#60a5fa">MoNa-pi (제안)</td>
            <td style="padding:8px 10px;color:#60a5fa;font-weight:700">Decomposed VLM + Flow Matching Head</td>
            <td style="padding:8px 10px;color:#60a5fa;font-weight:700">AdaLN-Zero + BBox 임프린팅 (물리적 분해)</td>
            <td style="padding:8px 10px;color:#4ade80;font-weight:700">연속 액션 / 50Hz (5ms 이하)</td>
            <td style="padding:8px 10px;color:#4ade80;font-weight:700">243 ep (소규모 온디바이스)</td>
            <td style="padding:8px 10px;text-align:left;color:#94a3b8">본 연구, Section 3.1 &amp; 3.2</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</div>
"""

# ch23 블록의 마지막 닫는 태그와 next-step 블록의 시작 사이에 ch27을 삽입함
target_find = '      "엉뚱한 bbox"는 basket 미인식도, 사전학습 부재도 아니다 — <strong>우리가 fine-tune한 LoRA가 grounding을 망가뜨렸다.</strong>\n      베이스 PG2는 basket을 정확·안정적으로 잡으며(12/12, cx_std 0.047), 없는 객체는 거부한다. 근본 해법은 grounding 안정화(베이스 복귀 or LoRA 재설계)다.\n    </div>\n  </div>\n\n</div>\n\n<div class="chapter" id="next-step">'

replacement = '      "엉뚱한 bbox"는 basket 미인식도, 사전학습 부재도 아니다 — <strong>우리가 fine-tune한 LoRA가 grounding을 망가뜨렸다.</strong>\n      베이스 PG2는 basket을 정확·안정적으로 잡으며(12/12, cx_std 0.047), 없는 객체는 거부한다. 근본 해법은 grounding 안정화(베이스 복귀 or LoRA 재설계)다.\n    </div>\n  </div>\n\n</div>\n' + ch27_html + '\n\n<div class="chapter" id="next-step">'

if target_find in content:
    content = content.replace(target_find, replacement)
    FILE_PATH.write_text(content, encoding="utf-8")
    print("CH 27 block successfully inserted!")
else:
    # 윈도우 스타일 줄바꿈 대응
    target_find_rn = target_find.replace("\n", "\r\n")
    if target_find_rn in content:
        content = content.replace(target_find_rn, replacement.replace("\n", "\r\n"))
        FILE_PATH.write_text(content, encoding="utf-8")
        print("CH 27 block (CRLF) successfully inserted!")
    else:
        print("Error: Target pattern to insert CH 27 not found!")
        sys.exit(1)
