#!/usr/bin/env python3
"""포트폴리오·대학원 컨택용 고해상도 시각 카드(SVG + PNG 5종) 자동 생성 스크립트.

노션, PPT, PDF, 이메일 어디에 삽입해도 깔끔하게 어울리는 라이트 학술 테마로 제작한다.
출력 디렉토리: docs/v5/portfolio_kit/
"""
from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs/v5/portfolio_kit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT = "NanumBarunGothic, Noto Sans CJK KR, sans-serif"
MONO = "Noto Sans Mono CJK KR, DejaVu Sans Mono, monospace"


def _b64_img(rel_path: str) -> str:
    """로컬 이미지 파일을 SVG <image> 삽입용 data URI로 인코딩한다."""
    fp = ROOT / rel_path
    ext = fp.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    data = base64.b64encode(fp.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _save_svg_and_png(name: str, svg_str: str, scale: float = 2.0) -> None:
    """SVG를 저장하고 cairosvg로 고해상도 PNG(2x)를 함께 렌더링한다."""
    svg_path = OUT_DIR / f"{name}.svg"
    png_path = OUT_DIR / f"{name}.png"
    svg_path.write_text(svg_str, encoding="utf-8")
    cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), write_to=str(png_path), scale=scale)
    print(f"[OK] {svg_path.name} & {png_path.name} ({png_path.stat().st_size // 1024} KB)")


def _svg_shell(w: int, h: int, body: str, defs: str = "") -> str:
    """공통 라이트 카드 SVG 프레임."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="100%" stop-color="#f8fafc"/>
    </linearGradient>
    <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#1d4ed8"/>
      <stop offset="100%" stop-color="#2563eb"/>
    </linearGradient>
    <linearGradient id="greenGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="100%" stop-color="#10b981"/>
    </linearGradient>
    <linearGradient id="purpleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#6d28d9"/>
      <stop offset="100%" stop-color="#8b5cf6"/>
    </linearGradient>
    <marker id="arrGray" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,1 L7,4 L0,7 Z" fill="#64748b"/>
    </marker>
    <marker id="arrBlue" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,1 L7,4 L0,7 Z" fill="#2563eb"/>
    </marker>
    {defs}
  </defs>
  <rect x="1" y="1" width="{w-2}" height="{h-2}" rx="18" fill="url(#bgGrad)" stroke="#cbd5e1" stroke-width="1.5"/>
  {body}
</svg>'''


def build_01_profile_roadmap() -> None:
    """카드 1: 프로필 및 목표 + Serbot 2 실기 사진 + EGVLA1->2->3 진화 로드맵."""
    W, H = 1100, 410
    serbot_uri = _b64_img("docs/v5/figures/hardware_serbot_combined.png")
    defs = '<clipPath id="clipSerbot"><rect x="26" y="26" width="278" height="330" rx="14"/></clipPath>'

    body = f'''
    <!-- 좌측: Serbot 2 하드웨어 카드 -->
    <rect x="24" y="24" width="282" height="362" rx="16" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <image href="{serbot_uri}" x="26" y="26" width="278" height="320" preserveAspectRatio="xMidYMid meet" clip-path="url(#clipSerbot)"/>
    <rect x="24" y="344" width="282" height="42" rx="10" fill="#f1f5f9"/>
    <text x="165" y="363" text-anchor="middle" font-family="{FONT}" font-size="13" font-weight="700" fill="#0f172a">Serbot 2 + Jetson Orin NX</text>
    <text x="165" y="379" text-anchor="middle" font-family="{FONT}" font-size="10.5" fill="#475569">On-Device Real-Robot Testbed (ROS2 · 10Hz)</text>

    <!-- 우측 상단: 헤더 -->
    <rect x="330" y="26" width="146" height="24" rx="12" fill="#dbeafe"/>
    <text x="403" y="42" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#1d4ed8">RESEARCH PROFILE</text>
    <text x="488" y="43" font-family="{FONT}" font-size="12" font-weight="700" fill="#475569">2027-1 인턴 · 2027-2 석박통합 진학 목표</text>

    <text x="330" y="84" font-family="{FONT}" font-size="27" font-weight="700" fill="#0f172a">이민우 (Minwoo Lee)</text>
    <text x="615" y="83" font-family="{FONT}" font-size="14" font-weight="700" fill="#2563eb">강남대 AI 4학년 · GPA 4.27 / 4.5</text>
    <text x="330" y="110" font-family="{FONT}" font-size="13.5" fill="#334155">온디바이스 로봇 정책(VLA) · 미학습 일반화 진단 · 배포 시점 제어 오케스트레이션 연구</text>

    <!-- 우측 중단: EGVLA1 -> EGVLA2 -> EGVLA3 진화 로드맵 -->
    <text x="330" y="148" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#475569">RESEARCH EVOLUTION — 실기에서 발견한 한계를 다음 세대 구조로 검증</text>

    <!-- Stage 1 -->
    <rect x="330" y="160" width="228" height="156" rx="14" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"/>
    <rect x="344" y="172" width="94" height="20" rx="10" fill="#2563eb"/>
    <text x="391" y="186" text-anchor="middle" font-family="{FONT}" font-size="10.5" font-weight="700" fill="#ffffff">EGVLA 1 · 투고</text>
    <text x="344" y="214" font-family="{FONT}" font-size="15.5" font-weight="700" fill="#0f172a">On-Device 경량 VLA</text>
    <text x="344" y="236" font-family="{FONT}" font-size="11.5" fill="#334155">• OWLv2 + Kosmos-2 (Frozen)</text>
    <text x="344" y="255" font-family="{FONT}" font-size="11.5" fill="#334155">• 1.128M / 459M (0.25%) 학습</text>
    <text x="344" y="274" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#1d4ed8">• 실기 95.0% (300회 92.0%)</text>
    <text x="344" y="299" font-family="{FONT}" font-size="10.5" fill="#64748b">JKSCI 1저자 심사 중</text>

    <line x1="558" y1="238" x2="582" y2="238" stroke="#64748b" stroke-width="2.2" marker-end="url(#arrGray)"/>

    <!-- Stage 2 -->
    <rect x="586" y="160" width="234" height="156" rx="14" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="1.5"/>
    <rect x="600" y="172" width="106" height="20" rx="10" fill="#059669"/>
    <text x="653" y="186" text-anchor="middle" font-family="{FONT}" font-size="10.5" font-weight="700" fill="#ffffff">EGVLA 2 · 실기완료</text>
    <text x="600" y="214" font-family="{FONT}" font-size="15.5" font-weight="700" fill="#0f172a">다물체 일반화·고속화</text>
    <text x="600" y="236" font-family="{FONT}" font-size="11.5" fill="#334155">• Florence-2 + PiZeroFM (Flow)</text>
    <text x="600" y="255" font-family="{FONT}" font-size="11.5" fill="#334155">• 시각 셔플 뒤집힘 27.4%→0.0%</text>
    <text x="600" y="274" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#047857">• 실기 45/45 · 접촉 0건 · 2.09×</text>
    <text x="600" y="299" font-family="{FONT}" font-size="10.5" fill="#64748b">EG1 vs EG2 비교논문 진행 중</text>

    <line x1="820" y1="238" x2="844" y2="238" stroke="#64748b" stroke-width="2.2" marker-end="url(#arrGray)"/>

    <!-- Stage 3 -->
    <rect x="848" y="160" width="228" height="156" rx="14" fill="#f5f3ff" stroke="#c4b5fd" stroke-width="1.5"/>
    <rect x="862" y="172" width="104" height="20" rx="10" fill="#7c3aed"/>
    <text x="914" y="186" text-anchor="middle" font-family="{FONT}" font-size="10.5" font-weight="700" fill="#ffffff">EGVLA 3 · E2E</text>
    <text x="862" y="214" font-family="{FONT}" font-size="15.5" font-weight="700" fill="#0f172a">미분가능 공간 병목</text>
    <text x="862" y="236" font-family="{FONT}" font-size="11.5" fill="#334155">• Soft-Loc (문자열 파싱 0%)</text>
    <text x="862" y="255" font-family="{FONT}" font-size="11.5" fill="#334155">• Decoder LoRA + Action Joint</text>
    <text x="862" y="274" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#6d28d9">• 미학습 정지 18.6% → 100%</text>
    <text x="862" y="299" font-family="{FONT}" font-size="10.5" fill="#64748b">Differentiable Spatial CoT</text>

    <!-- 우측 하단: 핵심 키워드 필 -->
    <rect x="330" y="336" width="174" height="34" rx="17" fill="#f1f5f9" stroke="#cbd5e1"/>
    <text x="417" y="358" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#1e293b">On-Device VLA (Jetson)</text>

    <rect x="516" y="336" width="186" height="34" rx="17" fill="#f1f5f9" stroke="#cbd5e1"/>
    <text x="609" y="358" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#1e293b">Counterfactual Audit</text>

    <rect x="714" y="336" width="174" height="34" rx="17" fill="#f1f5f9" stroke="#cbd5e1"/>
    <text x="801" y="358" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#1e293b">Real-Robot Control</text>

    <rect x="900" y="336" width="176" height="34" rx="17" fill="#f1f5f9" stroke="#cbd5e1"/>
    <text x="988" y="358" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#1e293b">Multi-Agent &amp; RAG</text>
    '''
    _save_svg_and_png("01_profile_roadmap", _svg_shell(W, H, body, defs))


def build_02_edgeground_vla_v1_v2() -> None:
    """카드 2: 대표 연구 EdgeGround-VLA (v1 -> v2 아키텍처 & 실기 정량 성과)."""
    W, H = 1100, 520
    body = f'''
    <!-- 헤더 -->
    <rect x="28" y="24" width="170" height="24" rx="12" fill="#dbeafe"/>
    <text x="113" y="40" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#1d4ed8">FLAGSHIP RESEARCH</text>
    <text x="210" y="41" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#475569">EdgeGround-VLA v1 (On-Device 경량화) → v2 (미학습 다물체 일반화 · 무충돌 고속 제어)</text>
    <text x="28" y="76" font-family="{FONT}" font-size="22" font-weight="700" fill="#0f172a">아키텍처 진화 및 실기(Real-Robot) 정량 성과 비교</text>

    <!-- 좌측: 아키텍처 비교 패널 -->
    <rect x="28" y="96" width="516" height="398" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="46" y="124" font-family="{FONT}" font-size="14" font-weight="700" fill="#0f172a">A. 경량 정책 파이프라인 진화 (Frozen VLM + Lightweight Policy)</text>

    <!-- v1 블록 -->
    <rect x="44" y="140" width="484" height="152" rx="12" fill="#f8fafc" stroke="#cbd5e1"/>
    <rect x="56" y="150" width="124" height="22" rx="11" fill="#2563eb"/>
    <text x="118" y="165" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#ffffff">EGVLA 1 (JKSCI)</text>
    <text x="192" y="165" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#334155">1.128M / 459M 학습 (0.25%) · 10Hz 비동기 + Jitter Hold</text>

    <rect x="56" y="184" width="136" height="58" rx="9" fill="#e2e8f0" stroke="#94a3b8"/>
    <text x="124" y="207" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#0f172a">OWLv2 + Kosmos-2</text>
    <text x="124" y="226" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#475569">Dual Frozen Encoders</text>

    <line x1="192" y1="213" x2="216" y2="213" stroke="#2563eb" stroke-width="2" marker-end="url(#arrBlue)"/>

    <rect x="220" y="184" width="136" height="58" rx="9" fill="#dbeafe" stroke="#60a5fa"/>
    <text x="288" y="207" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#1e3a8a">MLP Action Head</text>
    <text x="288" y="226" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#1d4ed8">8-Class Discrete (1.1M)</text>

    <line x1="356" y1="213" x2="380" y2="213" stroke="#2563eb" stroke-width="2" marker-end="url(#arrBlue)"/>

    <rect x="384" y="184" width="132" height="58" rx="9" fill="#eff6ff" stroke="#93c5fd"/>
    <text x="450" y="207" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#0f172a">Jetson Orin NX</text>
    <text x="450" y="226" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#475569">3-DoF Kiwi-Drive</text>

    <text x="56" y="270" font-family="{FONT}" font-size="11" fill="#475569">✓ 지도·클라우드 없이 온디바이스 주행 입증 (훈련 100회 95.0%, 미학습 포함 300회 92.0%)</text>

    <!-- v2 블록 -->
    <rect x="44" y="308" width="484" height="170" rx="12" fill="#f0fdf4" stroke="#86efac" stroke-width="1.5"/>
    <rect x="56" y="318" width="146" height="22" rx="11" fill="#059669"/>
    <text x="129" y="333" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#ffffff">EGVLA 2 (MOD-DR-Safe)</text>
    <text x="214" y="333" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#065f46">연속 Flow Matching + 기하 정합 + 2.09× 고속 서빙</text>

    <rect x="56" y="352" width="136" height="62" rx="9" fill="#dcfce7" stroke="#4ade80"/>
    <text x="124" y="375" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#064e3b">Florence-2 Grounding</text>
    <text x="124" y="393" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#047857">beam=1 fp16 (285ms)</text>
    <text x="124" y="407" text-anchor="middle" font-family="{FONT}" font-size="9.5" fill="#065f46">Open-Vocab BBox + Vis</text>

    <line x1="192" y1="383" x2="216" y2="383" stroke="#059669" stroke-width="2" marker-end="url(#arrGray)"/>

    <rect x="220" y="352" width="136" height="62" rx="9" fill="#d1fae5" stroke="#10b981"/>
    <text x="288" y="375" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#064e3b">PiZeroFM Head</text>
    <text x="288" y="393" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#047857">50% Swap + Area 정합</text>
    <text x="288" y="407" text-anchor="middle" font-family="{FONT}" font-size="9.5" fill="#065f46">연속 2D 속도 청크 출력</text>

    <line x1="356" y1="383" x2="380" y2="383" stroke="#059669" stroke-width="2" marker-end="url(#arrGray)"/>

    <rect x="384" y="352" width="132" height="62" rx="9" fill="#ecfdf5" stroke="#34d399"/>
    <text x="450" y="375" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#064e3b">S2 Safe Controller</text>
    <text x="450" y="393" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#047857">연속 감속 + 스파이크 필터</text>
    <text x="450" y="407" text-anchor="middle" font-family="{FONT}" font-size="9.5" fill="#065f46">실기 접촉 0건 달성</text>

    <text x="56" y="440" font-family="{FONT}" font-size="11" font-weight="700" fill="#065f46">✓ 바구니만 학습하고도 미학습 의자 15/15 포함 실기 45/45 (접촉 0건) · 방향 오차 0.54°</text>
    <text x="56" y="460" font-family="{FONT}" font-size="10.5" fill="#047857">✓ 언어 게이트(V4R): 비목표 지시어 입력 시 99.5% 제자리 정지</text>

    <!-- 우측: 핵심 정량 성과 카드 + 비교 바 차트 -->
    <rect x="560" y="96" width="512" height="398" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="578" y="124" font-family="{FONT}" font-size="14" font-weight="700" fill="#0f172a">B. 실기 검증 정량 지표 (Real-Robot Benchmarks)</text>

    <!-- 2x2 KPI 카드 -->
    <rect x="578" y="140" width="232" height="82" rx="10" fill="#eff6ff" stroke="#bfdbfe"/>
    <text x="592" y="160" font-family="{FONT}" font-size="11" font-weight="700" fill="#1d4ed8">EGVLA 1 · 훈련/미학습 위치</text>
    <text x="592" y="191" font-family="{FONT}" font-size="24" font-weight="700" fill="#0f172a">95.0% / 92.0%</text>
    <text x="592" y="211" font-family="{FONT}" font-size="10.5" fill="#475569">95/100 (훈련) · 276/300 (미학습 포함)</text>

    <rect x="822" y="140" width="232" height="82" rx="10" fill="#ecfdf5" stroke="#a7f3d0"/>
    <text x="836" y="160" font-family="{FONT}" font-size="11" font-weight="700" fill="#047857">EGVLA 2 · 미학습 물체 실기</text>
    <text x="836" y="191" font-family="{FONT}" font-size="24" font-weight="700" fill="#0f172a">45/45 · 접촉 0건</text>
    <text x="836" y="211" font-family="{FONT}" font-size="10.5" fill="#475569">바구니 30/30 + 미학습 의자 15/15 (11/25)</text>

    <rect x="578" y="232" width="232" height="74" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="592" y="252" font-family="{FONT}" font-size="11" font-weight="700" fill="#334155">추론 서빙 지연 (End-to-End)</text>
    <text x="592" y="280" font-family="{FONT}" font-size="20" font-weight="700" fill="#059669">591ms → 285ms</text>
    <text x="592" y="297" font-family="{FONT}" font-size="10.5" fill="#475569">2.09× 가속 · beam=3 대비 99.7% 일치</text>

    <rect x="822" y="232" width="232" height="74" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="836" y="252" font-family="{FONT}" font-size="11" font-weight="700" fill="#334155">조향 정합 · 파라미터 효율</text>
    <text x="836" y="280" font-family="{FONT}" font-size="20" font-weight="700" fill="#2563eb">0.54° · 0.25%</text>
    <text x="836" y="297" font-family="{FONT}" font-size="10.5" fill="#475569">핸들 각도 차 1° 미만 · 1.128M/459M</text>

    <!-- 하단 바 차트 -->
    <text x="578" y="334" font-family="{FONT}" font-size="12" font-weight="700" fill="#334155">실기 주요 지표 비교 (Baseline vs Proposed)</text>

    <!-- Bar 1: 미학습 의자 정중앙 출발 -->
    <text x="578" y="358" font-family="{FONT}" font-size="11" fill="#475569">의자 중앙 출발</text>
    <rect x="688" y="346" width="290" height="16" rx="4" fill="#f1f5f9"/>
    <rect x="688" y="346" width="290" height="16" rx="4" fill="url(#greenGrad)"/>
    <text x="986" y="358" font-family="{FONT}" font-size="11" font-weight="700" fill="#047857">0% → 100%</text>

    <!-- Bar 2: 근접 무충돌 정지 (의자) -->
    <text x="578" y="388" font-family="{FONT}" font-size="11" fill="#475569">의자 근접 정지</text>
    <rect x="688" y="376" width="290" height="16" rx="4" fill="#f1f5f9"/>
    <rect x="688" y="376" width="81" height="16" rx="4" fill="#fb923c"/>
    <rect x="688" y="376" width="270" height="16" rx="4" fill="url(#greenGrad)" fill-opacity="0.88"/>
    <text x="986" y="388" font-family="{FONT}" font-size="11" font-weight="700" fill="#047857">28% → 93%</text>

    <!-- Bar 3: 비목표 지시어 안전 차단 -->
    <text x="578" y="418" font-family="{FONT}" font-size="11" fill="#475569">부정 명령 정지</text>
    <rect x="688" y="406" width="290" height="16" rx="4" fill="#f1f5f9"/>
    <rect x="688" y="406" width="288" height="16" rx="4" fill="url(#blueGrad)"/>
    <text x="986" y="418" font-family="{FONT}" font-size="11" font-weight="700" fill="#1d4ed8">99.5% (V4R)</text>

    <!-- Bar 4: 추론 지연 단축 -->
    <text x="578" y="448" font-family="{FONT}" font-size="11" fill="#475569">추론 지연(ms)</text>
    <rect x="688" y="436" width="290" height="16" rx="4" fill="#fee2e2"/>
    <rect x="688" y="436" width="140" height="16" rx="4" fill="url(#purpleGrad)"/>
    <text x="986" y="448" font-family="{FONT}" font-size="11" font-weight="700" fill="#6d28d9">285ms (-52%)</text>

    <text x="578" y="478" font-family="{FONT}" font-size="10" fill="#64748b">* 모든 실기 지표는 Jetson Orin NX + Serbot 2 실환경 주행 로그(.h5 / session JSON) 기준</text>
    '''
    _save_svg_and_png("02_edgeground_vla_v1_v2", _svg_shell(W, H, body))


def _draw_scene_overlay(x0: float, y0: float, w: float, h: float, sc: dict, items: list[tuple[str, str, str, float]]) -> str:
    """실기 카메라 사진 위에 BBox와 모델별 행동 벡터/정지 마커를 정밀 오버레이한다."""
    img_uri = _b64_img(f"docs/v5/{sc['img_src']}")
    cid = f"clip_{sc['id']}"
    elems = [
        f'<clipPath id="{cid}"><rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="10"/></clipPath>',
        f'<image href="{img_uri}" x="{x0}" y="{y0}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" clip-path="url(#{cid})"/>',
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="10" fill="none" stroke="#94a3b8" stroke-width="1.2"/>',
    ]
    bb = sc["bbox"]
    side = math.sqrt(max(0.002, float(bb[2])))
    bw = min(0.92, side * 1.15) * w
    bh = min(0.92, side / 1.15) * h
    bx = x0 + max(3.0, min(w - bw - 3.0, float(bb[0]) * w - bw / 2.0))
    by = y0 + max(3.0, min(h - bh - 3.0, float(bb[1]) * h - bh / 2.0))
    elems.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="none" stroke="#38bdf8" stroke-width="2.5"/>')
    elems.append(f'<rect x="{bx:.1f}" y="{max(y0+4, by-18):.1f}" width="150" height="17" rx="4" fill="rgba(15,23,42,0.82)"/>')
    elems.append(f'<text x="{bx+5:.1f}" y="{max(y0+16, by-6):.1f}" font-family="{FONT}" font-size="10" font-weight="700" fill="#7dd3fc">F2 BBox (area={bb[2]:.2f})</text>')

    ox, oy, scale = x0 + w * 0.5, y0 + h - 22.0, 128.0
    ms = sc["models"]
    for key, label, col, yoff in items:
        m = ms[key]
        lx, ly = float(m["cmd"][0]), float(m["cmd"][1])
        if m["stop"]:
            r = 8.0 + abs(yoff) * 0.35
            elems.append(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="{r:.1f}" fill="none" stroke="{col}" stroke-width="3.0"/>')
            lx_pos = ox + r + 8 if yoff >= 0 else ox - r - 118
            elems.append(f'<rect x="{lx_pos:.1f}" y="{oy - 9 + yoff*0.5:.1f}" width="112" height="18" rx="4" fill="rgba(15,23,42,0.84)"/>')
            elems.append(f'<text x="{lx_pos+5:.1f}" y="{oy + 3 + yoff*0.5:.1f}" font-family="{FONT}" font-size="10" font-weight="700" fill="{col}">{html.escape(label)}: STOP</text>')
        else:
            tx = max(x0 + 18, min(x0 + w - 95, ox - ly * scale))
            ty = max(y0 + 24, min(y0 + h - 18, oy - lx * scale))
            elems.append(f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="{col}" stroke-width="3.4" stroke-linecap="round"/>')
            elems.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="4.5" fill="{col}"/>')
            elems.append(f'<rect x="{tx+7:.1f}" y="{ty - 10 + yoff*0.4:.1f}" width="122" height="18" rx="4" fill="rgba(15,23,42,0.84)"/>')
            elems.append(f'<text x="{tx+12:.1f}" y="{ty + 2 + yoff*0.4:.1f}" font-family="{FONT}" font-size="10" font-weight="700" fill="{col}">{html.escape(label)}: GO</text>')
    return "\n".join(elems)


def build_03_research_question_diagnostic() -> None:
    """카드 3: 핵심 연구 질문 — 대체 가능 행동(81.4%) -> 시각 지름길 반사실 감사 & 공간 병목."""
    W, H = 1100, 560
    fam = json.loads((ROOT / "docs/v5/egvla2_fm/family_arch_compare.json").read_text(encoding="utf-8"))
    sc_map = {s["id"]: s for s in fam["scenes"]}
    sc1 = sc_map["sc1_chair_center"]
    sc3 = sc_map["sc3_chair_close"]

    ov1 = _draw_scene_overlay(
        44, 152, 484, 240, sc1,
        [("MOD-A3", "a3 (기존)", "#f87171", -14), ("MOD-A3-AG", "a3-ag23", "#4ade80", 0)],
    )
    ov2 = _draw_scene_overlay(
        572, 152, 484, 240, sc3,
        [("VLM-E2E-Hidden", "EG3-08(Hidden)", "#f87171", -10), ("VLM-E2E-SoftLoc", "Soft-Loc / AG23", "#4ade80", 12)],
    )

    body = f'''
    <!-- 헤더 -->
    <rect x="28" y="24" width="186" height="24" rx="12" fill="#fef3c7"/>
    <text x="121" y="40" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#b45309">CORE RESEARCH QUESTION</text>
    <text x="226" y="41" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#475569">오프라인 630건 오류 중 81.4% 대체 가능 행동 → 시각 지름길(Visual Shortcut) 규명</text>
    <text x="28" y="76" font-family="{FONT}" font-size="22" font-weight="700" fill="#0f172a">왜 저데이터 VLA는 낯선 물체에서 실패하는가? — 반사실 진단과 해결</text>

    <!-- 좌측 장면 카드 -->
    <rect x="28" y="96" width="516" height="340" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="44" y="122" font-family="{FONT}" font-size="13.5" font-weight="700" fill="#0f172a">① 출발 기로 (미학습 의자 정중앙 · 실제 카메라 프레임)</text>
    <text x="44" y="140" font-family="{FONT}" font-size="11" fill="#475569">지시어: "go to the brown wooden chair" (바구니만 300ep 학습)</text>
    {ov1}
    <text x="44" y="414" font-family="{FONT}" font-size="11.5" fill="#334155"><tspan font-weight="700" fill="#dc2626">a3 (기존)</tspan>: 768-d 시각 지름길로 도착 오인 → <tspan font-weight="700">실기 0/5 비출발</tspan></text>
    <text x="44" y="429" font-family="{FONT}" font-size="11.5" fill="#334155"><tspan font-weight="700" fill="#059669">a3-ag23</tspan>: 50% Feature Swap + 기하 정합 → <tspan font-weight="700">100% 정상 출발</tspan></text>

    <!-- 우측 장면 카드 -->
    <rect x="556" y="96" width="516" height="340" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="572" y="122" font-family="{FONT}" font-size="13.5" font-weight="700" fill="#0f172a">② 근접 정지 (미학습 의자 초근접 · area=0.36)</text>
    <text x="572" y="140" font-family="{FONT}" font-size="11" fill="#475569">E2E 정책에서 명시적 공간 병목(Spatial Bottleneck) 유무 Ablation</text>
    {ov2}
    <text x="572" y="414" font-family="{FONT}" font-size="11.5" fill="#334155"><tspan font-weight="700" fill="#dc2626">EG3-08 (Hidden E2E)</tspan>: 기하 병목 부재 → <tspan font-weight="700">의자 정지 18.6% (계속 전진)</tspan></text>
    <text x="572" y="429" font-family="{FONT}" font-size="11.5" fill="#334155"><tspan font-weight="700" fill="#059669">EG3-11 (Soft-Loc) / AG23</tspan>: 미분가능 공간 병목 → <tspan font-weight="700">93~100% 정확 정지</tspan></text>

    <!-- 하단: 반사실 감사 정량 요약 바 -->
    <rect x="28" y="450" width="1044" height="86" rx="12" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="48" y="476" font-family="{FONT}" font-size="13" font-weight="700" fill="#0f172a">반사실 정밀 감사 (Counterfactual Audit) — 동일 BBox에서 이미지 특징만 무작위 셔플</text>

    <rect x="48" y="490" width="320" height="32" rx="8" fill="#fef2f2" stroke="#fecaca"/>
    <text x="208" y="510" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#b91c1c">시각 셔플 정지 뒤집힘: 27.4% → 0.0%</text>

    <rect x="386" y="490" width="324" height="32" rx="8" fill="#ecfdf5" stroke="#a7f3d0"/>
    <text x="548" y="510" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#047857">기하 면적 반응성(S_area): -0.3%p → +100%p</text>

    <rect x="728" y="490" width="324" height="32" rx="8" fill="#f5f3ff" stroke="#ddd6fe"/>
    <text x="890" y="510" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#6d28d9">E2E Soft-Loc 의자 정지: 18.6% → 100.0%</text>
    '''
    _save_svg_and_png("03_research_question_diagnostic", _svg_shell(W, H, body))


def build_04_eg1_vs_eg2_protocol_y2() -> None:
    """카드 4: 후속 비교 논문 프로토콜(pos1~3 x G6~8) & 바닥 접지선(y2) 정지 정합."""
    W, H = 1100, 460
    basket_uri = _b64_img("docs/v5/figures/real_1125/20261125_152041_바구니_trackF_center.jpg")
    chair_uri = _b64_img("docs/v5/figures/real_1125/20261125_155938_의자_trackF_center.jpg")
    defs = (
        '<clipPath id="clipBsk"><rect x="576" y="152" width="228" height="128" rx="8"/></clipPath>'
        '<clipPath id="clipChr"><rect x="820" y="152" width="228" height="128" rx="8"/></clipPath>'
    )

    body = f'''
    <!-- 헤더 -->
    <rect x="28" y="24" width="184" height="24" rx="12" fill="#ede9fe"/>
    <text x="120" y="40" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#6d28d9">ONGOING FOLLOW-UP</text>
    <text x="224" y="41" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#475569">교수님 미팅 확정 2차 비교 논문 — 동일 조건 실기 프로토콜 &amp; 물체 크기 무관 정지 기하</text>
    <text x="28" y="76" font-family="{FONT}" font-size="22" font-weight="700" fill="#0f172a">EGVLA1 vs EGVLA2 공정 비교 실험 설계 &amp; 바닥 접지선(y2) 정지 정합</text>

    <!-- 좌측: 3x3 공통 경로 매트릭스 -->
    <rect x="28" y="96" width="512" height="340" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="46" y="124" font-family="{FONT}" font-size="14" font-weight="700" fill="#0f172a">A. 공통 3×3 실기 경로 매트릭스 (pos1~3 → G6~8)</text>

    <!-- 3x3 SVG 영역 -->
    <rect x="46" y="138" width="476" height="186" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
    <!-- 도착 G6, G7, G8 -->
    <rect x="82" y="152" width="104" height="36" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
    <text x="134" y="175" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#14532d">G6 (도착 A)</text>
    <rect x="232" y="152" width="104" height="36" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
    <text x="284" y="175" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#14532d">G7 (도착 B)</text>
    <rect x="382" y="152" width="104" height="36" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
    <text x="434" y="175" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#14532d">G8 (도착 C)</text>

    <!-- 9개 연결선 -->
    <g stroke="#3b82f6" stroke-opacity="0.45" stroke-width="1.6" stroke-dasharray="4 3">
      <line x1="134" y1="276" x2="134" y2="188"/><line x1="134" y1="276" x2="284" y2="188"/><line x1="134" y1="276" x2="434" y2="188"/>
      <line x1="284" y1="276" x2="134" y2="188"/><line x1="284" y1="276" x2="284" y2="188"/><line x1="284" y1="276" x2="434" y2="188"/>
      <line x1="434" y1="276" x2="134" y2="188"/><line x1="434" y1="276" x2="284" y2="188"/><line x1="434" y1="276" x2="434" y2="188"/>
    </g>
    <rect x="184" y="218" width="200" height="28" rx="14" fill="#ffffff" stroke="#93c5fd"/>
    <text x="284" y="236" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#1d4ed8">동일 9경로 (출발 3 × 도착 3)</text>

    <!-- 출발 pos1, pos2, pos3 -->
    <rect x="82" y="276" width="104" height="34" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="134" y="298" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#1e3a8a">pos1 (출발 1)</text>
    <rect x="232" y="276" width="104" height="34" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="284" y="298" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#1e3a8a">pos2 (출발 2)</text>
    <rect x="382" y="276" width="104" height="34" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="434" y="298" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#1e3a8a">pos3 (출발 3)</text>

    <!-- 하단 비교 요약 -->
    <rect x="46" y="338" width="232" height="82" rx="10" fill="#eff6ff" stroke="#bfdbfe"/>
    <text x="60" y="360" font-family="{FONT}" font-size="12" font-weight="700" fill="#1d4ed8">EGVLA 1 (기준선 · 9조건)</text>
    <text x="60" y="380" font-family="{FONT}" font-size="11" fill="#334155">• 평가 객체: 바구니 1종 (학습)</text>
    <text x="60" y="399" font-family="{FONT}" font-size="11" fill="#334155">• 고정 지시어 · 8-class 이산 제어</text>

    <rect x="290" y="338" width="232" height="82" rx="10" fill="#ecfdf5" stroke="#a7f3d0"/>
    <text x="304" y="360" font-family="{FONT}" font-size="12" font-weight="700" fill="#047857">EGVLA 2 (제안 · 27조건)</text>
    <text x="304" y="380" font-family="{FONT}" font-size="11" fill="#334155">• 바구니 + 미학습 의자·책상 3종</text>
    <text x="304" y="399" font-family="{FONT}" font-size="11" fill="#334155">• 성공률·접촉·잔여거리(cm)·지연 비교</text>

    <!-- 우측: 바닥 접지선(y2) 정지 정합 -->
    <rect x="558" y="96" width="514" height="340" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="576" y="124" font-family="{FONT}" font-size="14" font-weight="700" fill="#0f172a">B. 물체 크기 편차 해결 — 바닥 접지선 (y2 = cy + h/2)</text>
    <text x="576" y="142" font-family="{FONT}" font-size="11" fill="#475569">같은 BBox 면적(0.29)이어도 큰 의자는 멀리서 정지 → 바닥선 y2 기준으로 거리 통일</text>

    <!-- 바구니 11/25 실기 도착 사진 -->
    <image href="{basket_uri}" x="576" y="152" width="228" height="128" preserveAspectRatio="xMidYMid slice" clip-path="url(#clipBsk)"/>
    <rect x="576" y="152" width="228" height="128" rx="8" fill="none" stroke="#94a3b8"/>
    <line x1="576" y1="272" x2="804" y2="272" stroke="#22c55e" stroke-width="3" stroke-dasharray="6 3"/>
    <rect x="584" y="159" width="144" height="20" rx="5" fill="rgba(15,23,42,0.82)"/>
    <text x="590" y="173" font-family="{FONT}" font-size="10.5" font-weight="700" fill="#86efac">바구니 (11/25) · y2=1.00</text>

    <!-- 의자 11/25 실기 도착 사진 -->
    <image href="{chair_uri}" x="820" y="152" width="228" height="128" preserveAspectRatio="xMidYMid slice" clip-path="url(#clipChr)"/>
    <rect x="820" y="152" width="228" height="128" rx="8" fill="none" stroke="#94a3b8"/>
    <line x1="820" y1="254" x2="1048" y2="254" stroke="#f59e0b" stroke-width="3" stroke-dasharray="6 3"/>
    <rect x="828" y="159" width="142" height="20" rx="5" fill="rgba(15,23,42,0.82)"/>
    <text x="834" y="173" font-family="{FONT}" font-size="10.5" font-weight="700" fill="#fde68a">의자 (11/25) · y2=0.82</text>

    <!-- y2 정합 결과 박스 -->
    <rect x="576" y="294" width="472" height="126" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="592" y="318" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#0f172a">Step B (A3C_Y2_s0) 시뮬레이션 검증 완료 (soda 전송 완료)</text>
    <text x="592" y="342" font-family="{FONT}" font-size="11.5" fill="#334155">• <tspan font-weight="700" fill="#059669">바구니 근접 (y2=1.00)</tspan>: 100.0% 정확 정지 유지</text>
    <text x="592" y="363" font-family="{FONT}" font-size="11.5" fill="#334155">• <tspan font-weight="700" fill="#2563eb">의자 원거리 (y2=0.82)</tspan>: 조기 정지 100% → <tspan font-weight="700">0.0%</tspan> (더 가까이 접근 가능)</text>
    <text x="592" y="384" font-family="{FONT}" font-size="11.5" fill="#334155">• <tspan font-weight="700" fill="#6d28d9">반사실 시각 셔플</tspan>: 뒤집힘 <tspan font-weight="700">0.2%</tspan> (외형 과적합 차단 유지)</text>
    <text x="592" y="406" font-family="{FONT}" font-size="10.5" fill="#64748b">※ 현장 기록 CSV · 자동 집계 파이프라인 구축 완료</text>
    '''
    _save_svg_and_png("04_eg1_vs_eg2_protocol_y2", _svg_shell(W, H, body, defs))


def build_05_publications_awards_stack() -> None:
    """카드 5: 논문 · 수상 · 시스템 스택 요약 증명 카드."""
    W, H = 1100, 420
    body = f'''
    <!-- 헤더 -->
    <rect x="28" y="24" width="210" height="24" rx="12" fill="#e0f2fe"/>
    <text x="133" y="40" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#0369a1">TRACK RECORD &amp; STACK</text>
    <text x="250" y="41" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#475569">연구(논문) · 구현(ROS2/Jetson) · 에이전트(LangGraph/RAG) 전 주기 역량</text>
    <text x="28" y="76" font-family="{FONT}" font-size="22" font-weight="700" fill="#0f172a">논문 · 수상 · 프로젝트 및 기술 스택</text>

    <!-- 좌측: 논문 & 수상 -->
    <rect x="28" y="96" width="528" height="298" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="46" y="124" font-family="{FONT}" font-size="14" font-weight="700" fill="#0f172a">Publications &amp; Awards</text>

    <rect x="44" y="138" width="496" height="62" rx="10" fill="#eff6ff" stroke="#bfdbfe"/>
    <rect x="56" y="148" width="84" height="19" rx="9" fill="#2563eb"/>
    <text x="98" y="161" text-anchor="middle" font-family="{FONT}" font-size="10" font-weight="700" fill="#ffffff">1저자 · 심사중</text>
    <text x="150" y="162" font-family="{FONT}" font-size="12" font-weight="700" fill="#0f172a">EdgeGround-VLA (JKSCI, KCI 등재지)</text>
    <text x="56" y="185" font-family="{FONT}" font-size="10.8" fill="#475569">On-Device Goal-Directed Navigation via Dual Frozen Grounding Encoders (2026)</text>

    <rect x="44" y="210" width="496" height="62" rx="10" fill="#ecfdf5" stroke="#a7f3d0"/>
    <rect x="56" y="220" width="84" height="19" rx="9" fill="#059669"/>
    <text x="98" y="233" text-anchor="middle" font-family="{FONT}" font-size="10" font-weight="700" fill="#ffffff">2저자 · 우수상</text>
    <text x="150" y="234" font-family="{FONT}" font-size="12" font-weight="700" fill="#0f172a">RAG-Based Loneliness Analysis (JCCT 10(6), 2024)</text>
    <text x="56" y="257" font-family="{FONT}" font-size="10.8" fill="#475569">IPACT 2024 Outstanding Paper Award · 한국인 외로움 분석 기반 대화형 AI</text>

    <rect x="44" y="282" width="496" height="96" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="58" y="304" font-family="{FONT}" font-size="12" font-weight="700" fill="#0f172a">주요 수상 및 프로젝트</text>
    <text x="58" y="325" font-family="{FONT}" font-size="11.2" fill="#334155">• <tspan font-weight="700">2026-1 캡스톤 디자인 은상</tspan> · <tspan font-weight="700">2025 강남톤 해커톤 대상(공과대학장상)</tspan></text>
    <text x="58" y="345" font-family="{FONT}" font-size="11.2" fill="#334155">• <tspan font-weight="700">Ladi (최우수상)</tspan>: KiwiBM25+FAISS+CrossEncoder · <tspan font-weight="700">Liberty (우수상)</tspan>: LangGraph</text>
    <text x="58" y="365" font-family="{FONT}" font-size="11.2" fill="#334155">• <tspan font-weight="700">Serbot 2 플랫폼 통합</tspan>: Jetson Orin Linux + ROS2 드라이버·주행 제어 스택</text>

    <!-- 우측: 기술 스택 + 링크 -->
    <rect x="572" y="96" width="500" height="298" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="590" y="124" font-family="{FONT}" font-size="14" font-weight="700" fill="#0f172a">Core Technical Stack</text>

    <!-- AI / VLA -->
    <rect x="588" y="138" width="468" height="66" rx="10" fill="#f5f3ff" stroke="#ddd6fe"/>
    <text x="602" y="159" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#6d28d9">AI · Vision-Language-Action (VLA)</text>
    <text x="602" y="182" font-family="{MONO}" font-size="11" fill="#1e293b">PyTorch · HuggingFace · LoRA/PEFT · Flow Matching</text>
    <text x="602" y="196" font-family="{MONO}" font-size="10.5" fill="#475569">Florence-2 · OWLv2 · Kosmos-2 · SigLIP · PaliGemma2</text>

    <!-- Robotics -->
    <rect x="588" y="214" width="468" height="58" rx="10" fill="#ecfdf5" stroke="#a7f3d0"/>
    <text x="602" y="235" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#047857">Robotics · On-Device Deployment</text>
    <text x="602" y="257" font-family="{MONO}" font-size="11" fill="#1e293b">ROS2 · Jetson Orin NX · Async Control (10Hz) · Linux</text>

    <!-- Agent / Systems -->
    <rect x="588" y="282" width="468" height="48" rx="10" fill="#fffbeb" stroke="#fde68a"/>
    <text x="602" y="302" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#b45309">Agent Orchestration &amp; Retrieval</text>
    <text x="602" y="321" font-family="{MONO}" font-size="11" fill="#1e293b">LangGraph · FAISS · KiwiBM25 · CrossEncoder · FastAPI</text>

    <!-- Contact Bar -->
    <rect x="588" y="340" width="468" height="38" rx="10" fill="#0f172a"/>
    <text x="822" y="363" text-anchor="middle" font-family="{MONO}" font-size="11" font-weight="700" fill="#f8fafc">minuum.github.io/EdgeGroundVLA  |  github.com/minuum  |  minwool0357@gmail.com</text>
    '''
    _save_svg_and_png("05_publications_awards_stack", _svg_shell(W, H, body))


def main() -> None:
    build_01_profile_roadmap()
    build_02_edgeground_vla_v1_v2()
    build_03_research_question_diagnostic()
    build_04_eg1_vs_eg2_protocol_y2()
    build_05_publications_awards_stack()
    print(f"\nAll 5 portfolio visual cards saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
