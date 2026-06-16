#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = ROOT / "docs" / "v5" / "research_story.html"

if not FILE_PATH.exists():
    print(f"Error: {FILE_PATH} does not exist!")
    sys.exit(1)

content = FILE_PATH.read_text(encoding="utf-8")

# 1. CH27의 테이블 <tbody> 교체
old_tbody = """        <tbody>
          <tr style="border-bottom:1px solid #1e293b">
            <td style="padding:8px 10px;text-align:left;font-weight:700">A1</td>
            <td style="padding:8px 10px">HSV 색상 필터 (GT)</td>
            <td style="padding:8px 10px">150 ep (기본)</td>
            <td style="padding:8px 10px;color:#64748b">없음 (no-aug)</td>
            <td style="padding:8px 10px">92.6%</td>
            <td style="padding:8px 10px;color:#22c55e;font-weight:900;font-size:0.9rem">96.7%</td>
          </tr>
          <tr style="border-bottom:1px solid #1e293b;background:#0a0f1a">
            <td style="padding:8px 10px;text-align:left;font-weight:700;color:#38bdf8">B1</td>
            <td style="padding:8px 10px">PaliGemma2-3B</td>
            <td style="padding:8px 10px">243 ep (MoNa-Pi)</td>
            <td style="padding:8px 10px;color:#64748b">없음 (no-aug)</td>
            <td style="padding:8px 10px">95.7%</td>
            <td style="padding:8px 10px;color:#22c55e;font-weight:900;font-size:0.9rem">70.0%</td>
          </tr>
          <tr style="border-bottom:1px solid #1e293b">
            <td style="padding:8px 10px;text-align:left;font-weight:700">B2</td>
            <td style="padding:8px 10px">PaliGemma2-3B</td>
            <td style="padding:8px 10px">243 ep (MoNa-Pi)</td>
            <td style="padding:8px 10px">Horizontal Flip 단독</td>
            <td style="padding:8px 10px">95.2%</td>
            <td style="padding:8px 10px;color:#fbbf24;font-weight:700">65.0%</td>
          </tr>
          <tr style="border-bottom:1px solid #1e293b;background:#0a0f1a">
            <td style="padding:8px 10px;text-align:left;font-weight:700;color:#38bdf8">B3 (Exp61)</td>
            <td style="padding:8px 10px">PaliGemma2-3B</td>
            <td style="padding:8px 10px">243 ep (MoNa-Pi)</td>
            <td style="padding:8px 10px">Flip + Center 3배 오버샘플링</td>
            <td style="padding:8px 10px">95.5%</td>
            <td style="padding:8px 10px;color:#22c55e;font-weight:900;font-size:0.9rem">70.0%</td>
          </tr>
        </tbody>"""

new_tbody = """        <tbody>
          <tr style="border-bottom:1px solid #1e293b">
            <td style="padding:8px 10px;text-align:left;font-weight:700">A1</td>
            <td style="padding:8px 10px">HSV 색상 필터 (GT)</td>
            <td style="padding:8px 10px">150 ep (기본)</td>
            <td style="padding:8px 10px;color:#64748b">없음 (no-aug)</td>
            <td style="padding:8px 10px">92.6%</td>
            <td style="padding:8px 10px;color:#22c55e;font-weight:900;font-size:0.9rem">96.7%</td>
          </tr>
          <tr style="border-bottom:1px solid #1e293b;background:#0a0f1a">
            <td style="padding:8px 10px;text-align:left;font-weight:700">A2</td>
            <td style="padding:8px 10px">HSV 색상 필터 (GT)</td>
            <td style="padding:8px 10px">150 ep (기본)</td>
            <td style="padding:8px 10px;color:#64748b">없음 (no-aug) (재재학습)</td>
            <td style="padding:8px 10px">92.6%</td>
            <td style="padding:8px 10px;color:#22c55e;font-weight:900;font-size:0.9rem">96.7%</td>
          </tr>
          <tr style="border-bottom:1px solid #1e293b">
            <td style="padding:8px 10px;text-align:left;font-weight:700">A3</td>
            <td style="padding:8px 10px">HSV 색상 필터 (GT)</td>
            <td style="padding:8px 10px">150 ep (기본)</td>
            <td style="padding:8px 10px">Horizontal Flip</td>
            <td style="padding:8px 10px">94.3%</td>
            <td style="padding:8px 10px;color:#fbbf24;font-weight:700">47.6%</td>
          </tr>
          <tr style="border-bottom:1px solid #1e293b;background:#0a0f1a">
            <td style="padding:8px 10px;text-align:left;font-weight:700">A4</td>
            <td style="padding:8px 10px">HSV 색상 필터 (GT)</td>
            <td style="padding:8px 10px">243 ep (MoNa-Pi)</td>
            <td style="padding:8px 10px;color:#64748b">없음 (no-aug)</td>
            <td style="padding:8px 10px">95.2%</td>
            <td style="padding:8px 10px;color:#22c55e;font-weight:900;font-size:0.9rem">100.0%</td>
          </tr>
          <tr style="border-bottom:1px solid #1e293b">
            <td style="padding:8px 10px;text-align:left;font-weight:700;color:#38bdf8">B1</td>
            <td style="padding:8px 10px">PaliGemma2-3B</td>
            <td style="padding:8px 10px">243 ep (MoNa-Pi)</td>
            <td style="padding:8px 10px;color:#64748b">없음 (no-aug)</td>
            <td style="padding:8px 10px">95.7%</td>
            <td style="padding:8px 10px;color:#22c55e;font-weight:900;font-size:0.9rem">70.0%</td>
          </tr>
          <tr style="border-bottom:1px solid #1e293b;background:#0a0f1a">
            <td style="padding:8px 10px;text-align:left;font-weight:700">B2</td>
            <td style="padding:8px 10px">PaliGemma2-3B</td>
            <td style="padding:8px 10px">243 ep (MoNa-Pi)</td>
            <td style="padding:8px 10px">Horizontal Flip 단독</td>
            <td style="padding:8px 10px">95.2%</td>
            <td style="padding:8px 10px;color:#fbbf24;font-weight:700">65.0%</td>
          </tr>
          <tr style="border-bottom:1px solid #1e293b">
            <td style="padding:8px 10px;text-align:left;font-weight:700;color:#38bdf8">B3 (Exp61)</td>
            <td style="padding:8px 10px">PaliGemma2-3B</td>
            <td style="padding:8px 10px">243 ep (MoNa-Pi)</td>
            <td style="padding:8px 10px">Flip + Center 3배 오버샘플링</td>
            <td style="padding:8px 10px">95.5%</td>
            <td style="padding:8px 10px;color:#22c55e;font-weight:900;font-size:0.9rem">70.0%</td>
          </tr>
          <tr style="border-bottom:1px solid #1e293b;background:#0a0f1a">
            <td style="padding:8px 10px;text-align:left;font-weight:700;color:#a78bfa">C1 (Exp63)</td>
            <td style="padding:8px 10px">E2E Kosmos-2</td>
            <td style="padding:8px 10px">243 ep (MoNa-Pi)</td>
            <td style="padding:8px 10px;color:#64748b">없음 (no-aug)</td>
            <td style="padding:8px 10px">78.6%</td>
            <td style="padding:8px 10px;color:#f87171;font-weight:700">18.8%</td>
          </tr>
        </tbody>"""

# 2. Group A 카드의 A2 수치 교체
old_a2_card = """            <!-- A2 -->
            <div style="background:#071221;padding:8px 10px;border-radius:6px">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:#38bdf8;font-weight:700;font-size:0.82rem">A2 (Re-train Baseline)</span>
                <span style="color:#eab308;font-weight:800;font-size:0.82rem">CL 52.4%</span>
              </div>
              <div style="color:#94a3b8;font-size:0.75rem;margin-top:2px">val_acc: 95.5% · FPE: 0.55m · TLD: 1.03</div>
            </div>"""

new_a2_card = """            <!-- A2 -->
            <div style="background:#071221;padding:8px 10px;border-radius:6px">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:#38bdf8;font-weight:700;font-size:0.82rem">A2 (Re-train Baseline)</span>
                <span style="color:#22c55e;font-weight:800;font-size:0.82rem">CL 96.7%</span>
              </div>
              <div style="color:#94a3b8;font-size:0.75rem;margin-top:2px">val_acc: 92.6% · FPE: 0.113m · TLD: 1.009</div>
            </div>"""

# 3. CH27 마지막 부분에 디버깅 노트를 추가
old_ch27_end = """  <!-- ③ 핵심 발견 -->
  <div style="background:#08111e;border:1px solid #1e3040;border-radius:12px;padding:20px">
    <div style="color:#fbbf24;font-size:0.82rem;font-weight:700;margin-bottom:10px">③ 학술적 결론 및 발견</div>
    <div style="font-size:0.8rem;color:#94a3b8;line-height:1.8">
      <ul>
        <li><strong>데이터 스케일의 지배적 영향력 (B1 70%):</strong> 인위적인 이미지 증강이나 기교적 오버샘플링을 거치지 않더라도, 물리적으로 신규 에피소드를 다량 수집하여 편향을 제거한 원본 데이터 확장(243ep) 자체가 Closed-Loop 성공률을 개선하는 가장 강력한 통제 변수임을 실증했습니다.</li>
        <li><strong>단순 증강의 부작용 (B2 65%):</strong> 데이터 다양성을 증가시키기 위한 단순 Horizontal Flip은 조향 신경망에 왜국된 주행 통계를 주입하여 조향 결정을 불안정하게 만들고 성능을 5%p 하락시키는 한계를 확인했습니다.</li>
        <li><strong>인식-제어 Decomposed 구조의 최적성:</strong> 243ep 기준 VLM 그라운딩을 이용한 분리형 제어 아키텍처(B1)가 엔드투엔드 거대 VLA(C1) 대비 저비용 학습 환경에서 일반화 안정성 및 데이터 효율성이 비약적으로 높음을 정량적으로 검증했습니다.</li>
      </ul>
    </div>
  </div>
</div>"""

new_ch27_end = """  <!-- ③ 핵심 발견 -->
  <div style="background:#08111e;border:1px solid #1e3040;border-radius:12px;padding:20px;margin-bottom:20px">
    <div style="color:#fbbf24;font-size:0.82rem;font-weight:700;margin-bottom:10px">③ 학술적 결론 및 발견</div>
    <div style="font-size:0.8rem;color:#94a3b8;line-height:1.8">
      <ul>
        <li><strong>데이터 스케일의 지배적 영향력 (B1 70%):</strong> 인위적인 이미지 증강이나 기교적 오버샘플링을 거치지 않더라도, 물리적으로 신규 에피소드를 다량 수집하여 편향을 제거한 원본 데이터 확장(243ep) 자체가 Closed-Loop 성공률을 개선하는 가장 강력한 통제 변수임을 실증했습니다.</li>
        <li><strong>단순 증강의 부작용 (B2 65%):</strong> 데이터 다양성을 증가시키기 위한 단순 Horizontal Flip은 조향 신경망에 왜곡된 주행 통계를 주입하여 조향 결정을 불안정하게 만들고 성능을 5%p 하락시키는 한계를 확인했습니다.</li>
        <li><strong>인식-제어 Decomposed 구조의 최적성:</strong> 243ep 기준 VLM 그라운딩을 이용한 분리형 제어 아키텍처(B1)가 엔드투엔드 거대 VLA(C1) 대비 저비용 학습 환경에서 일반화 안정성 및 데이터 효율성이 비약적으로 높음을 정량적으로 검증했습니다.</li>
      </ul>
    </div>
  </div>

  <!-- ④ 디버깅 노트 — HSV 주석 부재로 인한 NaN 이슈 -->
  <div style="background:#2a1010;border:1px solid #7f1d1d;border-radius:12px;padding:20px">
    <div style="color:#fca5a5;font-size:0.82rem;font-weight:700;margin-bottom:10px">④ 디버깅 노트 — MoNa-Pi 데이터셋 혼재로 인한 NaN(학습 붕괴) 해결</div>
    <div style="font-size:0.8rem;color:#94a3b8;line-height:1.8">
      <ul>
        <li><strong>현상 및 원인 분석:</strong> 신규로 수집된 73개의 MoNa-Pi 에피소드 데이터셋에 HSV 기반 바운딩 박스(BBox) 주석(annotation)이 누락된 상태에서 학습을 진행(A2)한 결과, HSV와 PG2 그라운더 좌표계가 혼재되며 gradient가 발산(NaN 발생)하고 학습이 붕괴되는 현상이 관찰되었습니다.</li>
        <li><strong>해결 방안 및 Ablation 재설계:</strong> 
          1) HSV BBox 주석이 부재한 73개 에피소드를 포함한 243ep에 대해 단순 HSV 학습을 하려던 대조군 설계(A2, A3)를 150ep 기반으로 제한하여 재재학습을 완료(A2 96.7%, A3 47.6%)했습니다.<br>
          2) 대신, HSV 주석을 호환 처리한 243ep 통합본을 활용하여 성공적으로 A4(no-aug) 모델의 학습을 완료했고, <strong>최종 Closed-Loop 성공률 100.0%</strong>를 달성하여 데이터 확장의 물리적 효과를 최종 입증했습니다.
        </li>
      </ul>
    </div>
  </div>
</div>"""


# 치환 수행
patched = False

# 1. <tbody> 치환
if old_tbody in content:
    content = content.replace(old_tbody, new_tbody)
    print("Patched old_tbody successfully!")
    patched = True
else:
    # 개행 문자 처리 (\r\n)
    old_tbody_rn = old_tbody.replace("\n", "\r\n")
    if old_tbody_rn in content:
        content = content.replace(old_tbody_rn, new_tbody.replace("\n", "\r\n"))
        print("Patched old_tbody (CRLF) successfully!")
        patched = True

# 2. old_a2_card 치환
if old_a2_card in content:
    content = content.replace(old_a2_card, new_a2_card)
    print("Patched old_a2_card successfully!")
    patched = True
else:
    old_a2_card_rn = old_a2_card.replace("\n", "\r\n")
    if old_a2_card_rn in content:
        content = content.replace(old_a2_card_rn, new_a2_card.replace("\n", "\r\n"))
        print("Patched old_a2_card (CRLF) successfully!")
        patched = True

# 3. ch27_end 치환 (디버깅 노트 삽입)
if old_ch27_end in content:
    content = content.replace(old_ch27_end, new_ch27_end)
    print("Patched old_ch27_end successfully!")
    patched = True
else:
    old_ch27_end_rn = old_ch27_end.replace("\n", "\r\n")
    if old_ch27_end_rn in content:
        content = content.replace(old_ch27_end_rn, new_ch27_end.replace("\n", "\r\n"))
        print("Patched old_ch27_end (CRLF) successfully!")
        patched = True

if patched:
    FILE_PATH.write_text(content, encoding="utf-8")
    print("All patches applied to research_story.html successfully!")
else:
    print("Error: Could not apply patches. Literal match targets not found.")
    sys.exit(1)
