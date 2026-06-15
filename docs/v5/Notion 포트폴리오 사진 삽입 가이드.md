# Notion 포트폴리오 사진 삽입 가이드

현재 Notion 포트폴리오 페이지의 시각적 완성도를 높이고 교수님에게 더 강력한 인상을 주기 위해, MoNaVLA 연구 페이지에서 활용 가능한 이미지들을 효과적으로 배치하는 방법을 안내합니다.

## 1. Notion 이미지 삽입 기본 원칙

- **간결함 유지:** Notion 페이지는 CV의 확장판이자 명함 역할을 하므로, 너무 많은 이미지를 넣기보다 핵심적인 내용을 시각적으로 보강하는 데 집중합니다.
- **캡션 활용:** 모든 이미지 아래에는 1~2줄의 간결한 영어 캡션을 달아 이미지가 무엇을 의미하는지 명확히 설명합니다.
- **링크 연결:** 필요한 경우 이미지를 클릭하면 GitHub 레포나 데모 영상으로 연결되도록 설정할 수 있습니다.
- **파일 형식:** PNG, JPG 등 일반적인 이미지 형식을 사용합니다. GIF 애니메이션도 가능합니다.

## 2. 필수 이미지 삽입 가이드 (지금 당장)

| 순서 | 필요한 이미지 | 원본 이미지 위치 (MoNaVLA 연구 페이지) | Notion 삽입 위치 | 삽입 목적 및 효과 |
|:---:|:---|:---|:---|:---|
| **1** | **로봇 실물 사진** | `minuum.github.io/MoNaVLA/` 메인 페이지 배경 (`hero_robot.png` 등) | **Notion 페이지 커버 (최상단)** | 교수님이 링크를 열자마자 "실제 하드웨어를 다룰 줄 아는 학생"임을 직관적으로 각인시킵니다. Notion 페이지 상단의 `Add Cover` 클릭 후 `Upload` 또는 `Link`로 삽입합니다. |
| **2** | **성능 비교 표** | `minuum.github.io/MoNaVLA/v5/research_story.html` (텍스트 표) | **`Research Experience` 섹션의 `MoNaVLA` 설명 바로 아래** | "34% → 81%"와 같은 텍스트 설명보다 시각적인 표가 데이터 기반의 문제 해결 능력을 더 명확하게 보여줍니다. Notion에서 `+` 버튼을 눌러 `Table` 블록을 추가하고 직접 입력합니다. |

## 3. 권장 이미지 삽입 가이드 (추후 보강)

시간적 여유가 있다면 아래 이미지들을 추가하여 포트폴리오의 깊이를 더할 수 있습니다.

| 순서 | 필요한 이미지 | 원본 이미지 위치 (MoNaVLA 연구 페이지) | Notion 삽입 위치 | 삽입 목적 및 효과 |
|:---:|:---|:---|:---|:---|
| **3** | **Decomposition 아키텍처 다이어그램** | `minuum.github.io/MoNaVLA/v5/research_story.html` (CHAPTER 4의 End-to-End vs Decomposition 비교 그림) | **`Research Experience` 섹션의 `MoNaVLA` 설명 중 `Decomposition Architecture` 언급 부분 바로 아래** | 복잡한 VLA 파이프라인(PaliGemma + MLP)을 직접 설계했다는 기술적 깊이를 시각적으로 증명합니다. |
| **4** | **Zero-shot Probe 또는 Masking 검증 결과** | `minuum.github.io/MoNaVLA/v5/research_story.html` (`exp54_viz/masking_comparison.png` 또는 `linear_probe_results.png` 등) | **`Research Experience` 섹션의 `MoNaVLA` 설명 중 `Text attention collapse` 언급 부분 바로 아래** | "단순히 모델을 돌려본 것이 아니라, 모델이 무엇을 보고 판단하는지 내부를 뜯어보고 분석했다"는 연구자적 자질을 어필합니다. |
| **5** | **BBox Grounding 주행 화면 (데모 영상 스크린샷)** | `minuum.github.io/MoNaVLA/` (메인 페이지 데모 영상) | **`Research Experience` 섹션의 `MoNaVLA` 설명 중 `Goal-Conditioned VLA` 항목 바로 아래** | 실제 로봇 시점에서 목표물(바스켓/의자)을 어떻게 인식하고 추종하는지 직관적으로 보여줍니다. 데모 영상에서 핵심 장면을 캡처하여 사용합니다. |
| **6** | **성능 향상 그래프/히트맵** | `minuum.github.io/MoNaVLA/v5/research_story.html` (`bbox_nav_exp51/report_figs/fig1_exp_progression.png` 등) | **`Research Experience` 섹션의 `MoNaVLA` 설명 중 `59+ experiments` 언급 부분 근처** | 59번의 실험을 거치며 성능이 우상향했다는 집요함과 문제 해결 능력을 데이터로 증명합니다. |
| **7** | **3축 옴니휠 로봇 설계도/CAD (개인 소장)** | (연구 페이지에 없다면 개인 소장 파일) | **`Projects` 섹션의 `3-Axis Omni-Wheel Robot` 설명 바로 아래** | 하드웨어 설계(기구학, 회로) 역량을 텍스트가 아닌 도면으로 확실하게 증명합니다. |

## 4. 이미지 삽입 방법 (Notion)

1.  **이미지 다운로드:** MoNaVLA 연구 페이지에서 원하는 이미지를 마우스 오른쪽 버튼 클릭 후 `이미지를 다른 이름으로 저장`하여 다운로드합니다.
2.  **Notion 블록 추가:** 이미지를 삽입할 위치에서 `+` 버튼을 클릭하거나 `/image`를 입력하여 `Image` 블록을 선택합니다.
3.  **이미지 업로드:** `Upload`를 클릭하여 다운로드한 이미지를 선택하거나, `Embed link`를 통해 이미지 URL을 직접 붙여넣습니다.
4.  **캡션 추가:** 이미지 블록 아래에 나타나는 `Add a caption`에 1~2줄의 영어 설명을 추가합니다.
5.  **링크 연결 (선택 사항):** 이미지 블록을 클릭한 후 `Link` 아이콘을 클릭하여 관련 GitHub 레포나 데모 영상 URL을 연결할 수 있습니다.

이 가이드를 활용하여 Notion 포트폴리오를 더욱 풍성하고 전문적으로 만들어 보세요.
