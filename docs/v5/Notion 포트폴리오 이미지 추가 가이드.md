# Notion 포트폴리오 이미지 추가 가이드

Notion 포트폴리오의 시각적 완성도를 높이기 위해, 기존 연구 페이지(`research_story.html`)에 있는 이미지들을 Notion의 어느 위치에 넣어야 하는지 정리한 가이드입니다.

| 순서 | 필요한 이미지 | 원본 이미지 위치 (연구 페이지) | Notion 삽입 위치 | 삽입 목적 및 효과 |
|:---:|:---|:---|:---|:---|
| **1** | **로봇 실물 사진** | 메인 페이지 배경 (`hero_robot.png` 등) | **Notion 페이지 커버 (최상단)** | 교수님이 링크를 열자마자 "실제 하드웨어를 다룰 줄 아는 학생"임을 직관적으로 각인시킵니다. |
| **2** | **Decomposition 아키텍처 다이어그램** | CHAPTER 4의 End-to-End vs Decomposition 비교 그림 | **`MoNaVLA` 섹션의 `Key Achievements` 바로 위** | 복잡한 VLA 파이프라인(PaliGemma + MLP)을 직접 설계했다는 기술적 깊이를 시각적으로 증명합니다. |
| **3** | **Zero-shot Probe 또는 Masking 검증 결과** | `exp54_viz/masking_comparison.png` 또는 `linear_probe_results.png` | **`Key Achievements`의 첫 번째 항목(zero-shot probe) 바로 아래** | "단순히 모델을 돌려본 것이 아니라, 모델이 무엇을 보고 판단하는지 내부를 뜯어보고 분석했다"는 연구자적 자질을 어필합니다. |
| **4** | **BBox Grounding 주행 화면** | `bbox_nav_step0/images/center_straight__260408__f006.jpg` 등 | **`Key Achievements`의 Goal-Conditioned VLA 항목 바로 아래** | 실제 로봇 시점에서 목표물(바스켓/의자)을 어떻게 인식하고 추종하는지 직관적으로 보여줍니다. |
| **5** | **성능 향상 그래프/히트맵** | `bbox_nav_exp51/report_figs/fig1_exp_progression.png` | **`Performance Summary` 표 바로 위 또는 아래** | 59번의 실험을 거치며 성능이 우상향했다는 집요함과 문제 해결 능력을 데이터로 증명합니다. |
| **6** | **3축 옴니휠 로봇 설계도/CAD** | (연구 페이지에 없다면 개인 소장 파일) | **`3-Axis Omni-Wheel Robot` 섹션 바로 아래** | 하드웨어 설계(기구학, 회로) 역량을 텍스트가 아닌 도면으로 확실하게 증명합니다. |

### 💡 이미지 삽입 팁
- 이미지를 넣을 때는 너무 크지 않게 적절히 리사이징하고, 이미지 바로 아래에 캡션(Caption)을 달아 이 이미지가 무엇을 의미하는지 1~2줄의 영어로 설명해 주세요.
- 예: *Figure 1: Decomposition architecture separating visual grounding (PaliGemma) and control policy (MLP).*
