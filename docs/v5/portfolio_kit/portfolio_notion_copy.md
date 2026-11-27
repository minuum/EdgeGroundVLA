# 이민우 (Minwoo Lee) — 연구 포트폴리오

## 1. 프로필 및 목표

강남대 AI 4학년 이민우(GPA 4.27/4.5, 2027.02 졸업 예정). 2027-1 인턴 및 2027-2 석박통합 진학을 목표로 온디바이스 로봇 정책(VLA)·미학습 일반화·배포 시점 오케스트레이션을 연구합니다.

![Serbot 2 + Jetson Orin NX 테스트베드 및 EGVLA 1→2→3 연구 진화 로드맵](https://minuum.github.io/EdgeGroundVLA/v5/portfolio_kit/01_profile_roadmap.png)

---

## 2. 대표 연구 · EdgeGround-VLA (v1 → v2)

v1에서는 OWLv2·Kosmos-2와 경량 헤드(1.128M/459M, 0.25%)를 Jetson Orin NX에 올려 훈련 100회 95.0%, 미학습 포함 300회 92.0%를 달성했습니다.
v2에서는 Florence-2 + Flow Matching(PiZeroFM)과 beam=1 fp16 가속(591ms→285ms, 2.09×)으로 확장해, 바구니만 학습하고도 미학습 의자 15/15 포함 실기 45/45·접촉 0건을 입증했습니다.

![EdgeGround-VLA v1→v2 아키텍처 진화 및 실기 정량 성과 비교](https://minuum.github.io/EdgeGroundVLA/v5/portfolio_kit/02_edgeground_vla_v1_v2.png)

---

## 3. 핵심 연구 질문 · 시각 지름길 규명과 공간 병목

실기 오류 630건 중 81.4%가 대체 가능 행동이었던 발견에서 출발해, 저데이터 VLA가 외형 지름길(Visual Shortcut)에 의존하는 한계를 반사실 시각 셔플 감사로 규명했습니다.
50% Feature Swap과 기하 정합으로 셔플 정지 뒤집힘을 27.4%→0.0%로 제거했고, E2E 구조에서는 미분가능 공간 병목(Soft-Loc)이 미학습 의자 정지를 18.6%→100%로 끌어올림을 보였습니다.

![실기 카메라 프레임 위 모델별 행동 벡터 오버레이 및 반사실 시각 셔플 감사 결과](https://minuum.github.io/EdgeGroundVLA/v5/portfolio_kit/03_research_question_diagnostic.png)

---

## 4. 후속 연구 · EGVLA1 vs EGVLA2 공정 비교 & 바닥 접지선(y2) 정합

물체 크기별 정지 거리 편차를 해결하기 위해 BBox 면적 대신 바닥 접지선(y2 = cy + h/2) 기반 정지 정합(A3C_Y2)을 도입해 시뮬레이션에서 바구니 100% 정지·의자 원거리 조기정지 0%를 확인했습니다.
현재 동일 출발(pos1~3)·도착(G6~8) 9경로·3객체(27조건) 공정 실기 프로토콜로 2차 비교 논문을 진행 중입니다.

![공통 3×3 실기 평가 매트릭스 및 11/25 실기 도착 사진 기반 바닥 접지선(y2) 비교](https://minuum.github.io/EdgeGroundVLA/v5/portfolio_kit/04_eg1_vs_eg2_protocol_y2.png)

---

## 5. 논문 · 수상 · 링크

JKSCI 1저자(심사 중), IPACT 2024 우수논문상(2저자), 2026-1 캡스톤 은상, 2025 강남톤 대상.

- Project Page: [minuum.github.io/EdgeGroundVLA](https://minuum.github.io/EdgeGroundVLA/)
- GitHub: [github.com/minuum](https://github.com/minuum)
- Contact: [minwool0357@gmail.com](mailto:minwool0357@gmail.com) | 010-6757-3689

![논문 · 수상 · 프로젝트 및 핵심 기술 스택 요약](https://minuum.github.io/EdgeGroundVLA/v5/portfolio_kit/05_publications_awards_stack.png)

---

<details>
<summary><b>세부 논문 · 프로젝트 · 수상 · 기술 스택 보기</b></summary>

### Publications & Manuscripts
- EdgeGround-VLA: On-Device Goal-Directed Navigation via Dual Frozen Grounding Encoders — Minwoo Lee (1st author), Inyeop Choe. JKSCI (KCI Registered), Under Review (2026). 1.128M/459M trained, 10Hz async control + Jitter Hold on Jetson Orin NX + Serbot 2.
- Designing Conversational AI Services Using RAG-Based Loneliness Analysis for Koreans — Minwoo Lee (2nd author) et al. JCCT 10(6):625–633, 2024 (IPACT 2024 Outstanding Paper Award).

### Projects
- Serbot 2 Robot Platform Integration (2025.01–03): Jetson Orin Linux + ROS2 driver nodes + locomotion control stack.
- Ladi (2024.10): AI Healthcare Routine Recommender (KiwiBM25 + FAISS + CrossEncoder) — Best Project Award.
- Liberty (2024.05): AI Legal QA Multi-Agent System (LangGraph, PM & Lead Architect) — Excellence Award.

### Awards & Honors
- 2026-1 Kangnam Univ. Capstone Design Competition — Silver Prize
- 2025 Kangnengthon Hackathon — Grand Prize (Dean of College of Engineering Award)
- IPACT 2024 — Outstanding Paper Award
- 2024 Software Academic Festival (1st Prize) / AI Academic Festival (Excellence Award) / K-HTML Hackathon

### Skills
- AI / VLA: PyTorch, HuggingFace, LoRA/PEFT, Flow Matching, Florence-2, OWLv2, Kosmos-2, SigLIP, PaliGemma2, LangGraph
- Robotics: ROS2, Jetson Orin NX, Ubuntu Linux, Asynchronous Control, Git

</details>
