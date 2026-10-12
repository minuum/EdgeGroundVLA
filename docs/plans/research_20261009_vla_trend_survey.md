# 리서치 — 최근 VLA / 모바일 로봇 트렌드 서베이 (2026-10-09)

⚠️ **이 문서는 방향 확정용이 아니라 참고 서베이다.** EGVLA3("MoNa-pi 연속액션 재빌드")는
여러 후보 방향 중 하나일 뿐이며, 아래 내용은 "이쪽으로 가야 한다"는 결론이 아니라
각 기업/연구소가 실제로 무엇에 집중하고 있는지에 대한 출처 기반 정리다. 로드맵
(`docs/v5/project_roadmap_2026q4.html`)의 프레이밍 노드는 이 서베이를 참고해
**사용자가 확정할 때까지 제안 상태로 유지**한다.

모든 항목은 WebSearch(2026-10-09 수행) 결과를 바탕으로 하며, 검증 못 한 2차 출처는
"(2차 출처, 미확인)"로 표시했다.

---

## 1. Flow-Matching — 매니퓰레이션 액션 헤드의 사실상 표준

### Physical Intelligence — π0 / π0.5 / openpi
- 공식 GitHub: **https://github.com/Physical-Intelligence/openpi**
- 회사 사이트: https://www.pi.website/ , π0.5 공지: https://www.pi.website/blog/openpi
- repo README 기준 3개 모델 제공: `π0`(flow-matching 기반 VLA), `π0-FAST`(FAST 토크나이저 기반 autoregressive VLA), `π0.5`(open-world 일반화 강화, knowledge insulation로 학습). **repo는 현재 π0.5 학습/추론에 flow matching 헤드만 지원**한다고 명시.
- 체크포인트는 `gs://openpi-assets`에서 자동 다운로드(`~/.cache/openpi`에 캐시), 10k+ 시간 로봇 데이터로 사전학습된 베이스 체크포인트 다수 제공(DROID 파인튜닝판 `pi05_droid` 포함).
- π0.5 블로그 공지: "모바일 매니퓰레이터로 완전히 새로운 부엌/침실을 치울 수 있다" — **모바일 베이스는 매니퓰레이터를 옮기는 용도로만 등장**, 네비게이션 자체(목표 방향 일반화 등)를 다루는 내용은 없음.

### Intel Embodied Intelligence SDK — π0.5 온디바이스 실행
- 릴리즈 노트: https://docs.openedgeplatform.intel.com/2026.1/edge-ai-suites/robotics-ai-suite/embodied/release-notes.html (2026.1), 2026.0판: https://docs.openedgeplatform.intel.com/2026.0/edge-ai-suites/robotics-ai-suite/embodied/release-notes.html
- 2026.0: "OpenVINO Integration: Enable Pi0.5 pipeline policy with OpenVINO inference optimized on Intel integrated GPU" — **인텔 노트북급 GPU에서의 로컬 추론**이지, 모바일 로봇 자체 온보드 배포는 아님.
- 2026.1: MuJoCo 시뮬레이터 평가용 파이프라인 리팩터, Aloha 파이프라인(시뮬+실로봇) — 둘 다 모바일 베이스 아님. 알려진 이슈로 `CL_OUT_OF_RESOURCES`(i915 드라이버, Intel Ultra 2) 기록돼 있음.

### 관련 연구 동향(arXiv)
- Frequency-Conditioned Flow Matching for VLA — https://arxiv.org/pdf/2609.10405
- VITA: Vision-to-Action Flow Matching Policy — https://arxiv.org/pdf/2507.13231
- ActionCache: Training-Free Acceleration for VLA — https://arxiv.org/pdf/2607.06370 (flow 기반 VLA의 매 제어 스텝마다 여러 denoising step이 필요해 느리다는 점을 효율화 동기로 명시)
- Neuro-Symbolic Safety Guidance via Constrained Flow Matching — https://arxiv.org/pdf/2607.01378

**정리**: flow-matching이 "연속 액션을 표현하는 표준 방법"이라는 전제 위에서 효율화·안전성
쪽 후속 연구가 쌓이고 있음 — 그런데 모두 **매니퓰레이션(팔) 중심**이고, 모바일 베이스의
목표 방향(방위각) 추종 문제를 다루는 사례는 이번 서베이에서 확인되지 않음.

---

## 2. 온디바이스/엣지 경량화 — 두 가지 서로 다른 전략

### Google DeepMind — Gemini Robotics On-Device 2
- 모델 카드(2026-07-30 공개): https://deepmind.google/models/model-cards/gemini-robotics-on-device-2/
- Gemma 기반 온디바이스 VLA, "범용 로봇 매니퓰레이션"용으로 설명, 현재 일부 신뢰 테스터에게만 공개.
- 상위 모델 Gemini Robotics 2(ER 2 포함) 쪽에서는 보행/웅크리기 등 전신 제어 데모(Apptronik Apollo 2)가 나오지만, **이건 메인 VLA 쪽 데모이고 온디바이스판에 귀속된 기능은 아님**. 온디바이스판이 네비게이션/보행을 다룬다는 출처는 확인 못 함.

### AsyncVLA — 원격 대형모델 + 온보드 Edge Adapter
- 논문(Hirose, Glossop, Shah, Levine — UC Berkeley / Toyota / Princeton, 2026-02): arXiv **2602.13476**
  - PDF: https://arxiv.org/pdf/2602.13476 (HTML: https://arxiv.org/html/2602.13476)
  - 체크포인트(HuggingFace, MIT 라이선스): https://huggingface.co/NHirose/AsyncVLA_release , https://huggingface.co/NHirose/AsyncVLA_v0
  - ⚠️ 공식 GitHub 저장소/프로젝트 페이지 URL은 이번 서베이에서 확인 못 함 (2차 출처는 "논문 프로젝트 페이지에 코드 공개"라고 하나 URL 미확인) — HuggingFace 쪽이 현재 확인 가능한 가장 구체적인 공개 지점.
- 핵심 아이디어: 큰 base VLA(OmniVLA, 8.27B)는 **원격 워크스테이션**에서 돌리고, 로봇에는 **경량 Edge Adapter(소형 ViT)**만 얹어 고빈도로 액션을 보정 — distillation이 아니라 end-to-end 공동학습 + "동적 상호작용 우선 가중치" 전략. 통신 지연 최대 6초까지 견디며 베이스라인 대비 성공률 +40%.
  - ⚠️ 이름이 같은 다른 논문 주의: "AsyncVLA: Asynchronous Flow Matching for VLA"(arXiv **2511.14148**, Jiang 등, 다른 저자/다른 접근) — 혼동 금지.

### NVIDIA Isaac GR00T
- 공식 GitHub: **https://github.com/NVIDIA/Isaac-GR00T** (Apache-2.0 코드 라이선스)
- 모델: https://huggingface.co/nvidia/GR00T-N1.7-3B 등 (NVIDIA Open Model License, 상업적 이용 가능)
- VLM 백본은 게이트된 `nvidia/Cosmos-Reason2-2B` — 최초 사용 시 HuggingFace 접근 승인 필요.
- 기술 보고서(GR00T N1): arXiv **2503.14734**.
- **주로 휴머노이드 매니퓰레이션/전신제어** 대상. 모바일 베이스 네비게이션 전용 기능은 이번 서베이에서 확인 안 됨(과거 Isaac Perceptor가 AMR 대상이었으나 GR00T와는 별개 제품군).

**정리**: 온디바이스 경량화에 두 갈래 전략이 보임 — ① Gemini/GR00T류의 "모델 자체를 작게"(완전 온디바이스), ② AsyncVLA류의 "크고 똑똑한 건 원격, 로봇엔 보정용 소형 어댑터만"(비동기 분리). **EGVLA1(그라운딩 frozen + 초경량 헤드, 완전 온디바이스)은 ①에 가까우면서 더 극단적**("원격조차 없음") — 이 포지셔닝 차이는 관련 연구 섹션에 쓸 수 있는 소재.

---

## 3. 학습 없는(Training-free) 네비게이션 — 그라운딩/인식이 핵심이라는 흐름

### OpenFrontier (RSS 2026)
- 논문: Padilla, Sun, Pollefeys, Blum — "OpenFrontier: General Navigation with Visual-Language Grounded Frontiers"
  - arXiv: **2603.05377** (2026-03-05 제출) — abs: https://arxiv.org/abs/2603.05377 , PDF: https://arxiv.org/pdf/2603.05377
  - RSS 2026 (Robotics: Science and Systems XXII) 게재, DOI `10.15607/RSS.2026.XXII.067`
  - 프로시딩: https://www.roboticsproceedings.org/rss22/p067.html , 학회 프로그램: https://roboticsconference.org/program/papers/67/
- 핵심: 탐사 경계(frontier)를 "의미적 앵커"로 삼아 RGB+언어목표만으로 VLM에게 각 frontier의 관련도를 묻고, 3D로 투영해 목표 후보로 사용 — **정책 학습/파인튜닝 전혀 없이** 제로샷 네비게이션. 여러 벤치마크 + 실제 모바일 로봇 배포 보고.
- 저자들이 직접 대비하는 기존 접근: "밀집 3D 복원 + 수작업 목표 지표에 의존하는 기존 방식", "end-to-end VLN/VLA는 보통 상호작용 학습·대규모 데이터·태스크별 파인튜닝이 필요".

### WildOS (2026-02)
- 야외/비정형 환경 대상, ExploRFM(foundation-model 비전 모듈)으로 traversability·frontier·객체 유사도를 동시 예측, 파티클필터로 깊이 범위 밖 목표의 대략적 위치추정까지 수행.
- ⚠️ 정확한 arXiv ID/GitHub를 이번 서베이에서 특정 못함 — 추가 확인 필요.

**정리**: "그라운딩/인식이 네비게이션 성공을 가른다"는 방향은 EGVLA2의 핵심 결론(그라운딩 가용성 ≥80% → 98.8% 성공)과 같은 선 위에 있음. 다만 OpenFrontier류는 액션 자체를 아예 학습 안 하는 극단이고, EGVLA는 "그라운딩은 frozen, 액션 헤드만 초경량 학습"이라는 중간 지점 — 스펙트럼상 위치를 명확히 설명 가능.

---

## 4. 휴머노이드 전신 네비게이션 (참고, 우리 도메인과는 거리 있음)

### TANGO
- 논문: "TANGO: Humanoid Navigation in Cluttered Environments with a Whole-Body VLA" — UC Berkeley / 베이징대 / 칭화대 / HKU / Princeton, arXiv **2609.09158** (2026-09-08)
  - abs/PDF: https://arxiv.org/pdf/2609.09158 , HuggingFace Papers: https://huggingface.co/papers/2609.09158
  - 프로젝트 페이지: **https://tango-vla.github.io** (공식 GitHub 코드 저장소는 이번 서베이에서 미확인)
- 언어 지시 + egocentric RGB → 29-DoF 전신 관절 액션 직접 예측. 전부 시뮬레이션에서 학습(글로벌 경로계획+운동생성+RL 트래킹), Unitree G1 실기 배포.
- ⚠️ RSS 2026 공식 프로그램에서는 확인 못함(2차 출처만 존재) — 게재 여부 불확실.
- 우리 도메인(모바일 베이스 조이스틱 8-class)과는 액션공간이 완전히 다름(29-DoF 전신) — **직접 비교 대상이 아니라 "업계가 네비게이션+VLA를 결합할 때 어느 쪽에 자원을 쓰는지"의 참고 사례**로만 가치 있음.

---

## 5. EGVLA 트랙별 시사점 (제안일 뿐, 확정 아님)

| 트랙 | 참고할 수 있는 지점 | 비고 |
|---|---|---|
| EGVLA1(완료 논문) | AsyncVLA와의 "원격분리 vs 완전온디바이스" 대비를 related work에 한 줄 추가 가능 | 추가 실험 불필요 |
| EGVLA2(그라운더 교체) | OpenFrontier의 "그라운딩이 병목" 결론과 공명 — 우리 실측(gnd%≥80→98.8%)을 더 강하게 뒷받침하는 근거로 인용 가능 | 기존 계획 변경 없음 |
| EGVLA3(이산 vs 연속, 가칭) | "flow-matching이 매니퓰레이션 표준인데 왜 모바일 네비게이션 방향 일반화에서 이산보다 못한가"라는 질문이 서베이상 빈 자리로 보임 | **아직 사용자 확정 안 됨 — 로드맵에 "제안" 상태로만 표시** |

---

## 출처 모음 (URL 전체)

- https://github.com/Physical-Intelligence/openpi
- https://www.pi.website/
- https://www.pi.website/blog/openpi
- https://docs.openedgeplatform.intel.com/2026.1/edge-ai-suites/robotics-ai-suite/embodied/release-notes.html
- https://docs.openedgeplatform.intel.com/2026.0/edge-ai-suites/robotics-ai-suite/embodied/release-notes.html
- https://arxiv.org/pdf/2609.10405
- https://arxiv.org/pdf/2507.13231
- https://arxiv.org/pdf/2607.06370
- https://arxiv.org/pdf/2607.01378
- https://deepmind.google/models/model-cards/gemini-robotics-on-device-2/
- https://arxiv.org/pdf/2602.13476 (AsyncVLA, Hirose et al.)
- https://huggingface.co/NHirose/AsyncVLA_release
- https://huggingface.co/NHirose/AsyncVLA_v0
- https://arxiv.org/pdf/2511.14148 (다른 "AsyncVLA", 혼동 주의 — Jiang et al.)
- https://github.com/NVIDIA/Isaac-GR00T
- https://huggingface.co/nvidia/GR00T-N1.7-3B
- https://arxiv.org/abs/2503.14734 (GR00T N1 기술보고서)
- https://arxiv.org/abs/2603.05377 (OpenFrontier)
- https://www.roboticsproceedings.org/rss22/p067.html
- https://roboticsconference.org/program/papers/67/
- https://arxiv.org/pdf/2609.09158 (TANGO)
- https://huggingface.co/papers/2609.09158
- https://tango-vla.github.io
