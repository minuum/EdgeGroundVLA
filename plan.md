# Plan: 결론 체크포인트 정리 + 채우기/축약

> 상태: **검토 대기** — 승인 전 코드 작성 없음

---

## 변경 대상 파일

| 파일 | 변경 성격 |
|---|---|
| `docs/v5/research_story.html` | 버그픽스 + 체크포인트 섹션 추가 + SUMMARY 쇄신 |
| `README.md` | CH 카운트 오류 수정 |
| `docs/index.html` | link-card 텍스트 + metrics-bar |

---

## 변경 1 — research_story.html: 버그픽스

**라인 293**: `96.7%` → `96.6%` (HERO KPI 오타)

---

## 변경 2 — research_story.html: CHECKPOINT 섹션 신설 (HERO 직후)

**위치**: HERO `</div>` 직후, `<hr class="chapter-divider">` 앞에 삽입

**내용**: 전체 챕터(CH1→CH36)를 "기반·여정·SOTA 확정" 3단계로 묶어 체크포인트 카드로 나열.  
각 카드: 챕터 번호 + 핵심 1줄 결론 + 링크. 기존 챕터는 그대로 아래에 유지.

```
[기반 구축 — CH1~5]
CH1: V5 데이터셋 9경로 150 ep 정의
CH2: E2E VLA → CL 0% 확인
CH3: HSV grounding 도입
CH4: Decomposition 접근 전환
CH5: 첫 Closed-Loop 검증 개념 확립

[실험 여정 — CH6~32]
CH6~9:  Exp46~52 100% 성공 (but: 작은 데이터셋 과적합)
CH10~12: 5/15·5/22 미팅 — Exp54 설계, 교수님 반박
CH13:   텍스트 경로 구조적 사망 확인 (text attn 0%)
CH14~15: CL이 학술 지표인 이유, 5/27 반박 대응
CH16~18: PaliGemma 전환, 파이프라인 전체 흐름
CH19~21: 타 VLA 비교, Free ep, 연구 방향 정리
CH22~24: 비동기 수집·STOP 진실, grounding 붕괴, Flow Matching 검토
CH25~27: 학술 기여점 5선, 통합 Ablation, 오트래킹 점검
CH28~32: LoRA 검증, RoboVLMs 해부, 의자 전환, exp64 collapse, LoRA 결산

[SOTA 확정 — CH33~36]
CH33: 파이프라인이 범인 (10.3%→96.6%, ×9.4) — cx 소스 무관 확정
CH34: Head Ablation — LSTM=MLP=96.6%, FCHead=93.1%, Linear=69%
CH35: Window Ablation — MLP w≥4 포화, LSTM w=16 FPE 0.080m 최저
CH36: 6/12 미팅 — 실사 테스트 96.6%, 논문 제출 결정
```

TOC 패널에도 `CHECKPOINT` 항목 추가 (CH1 위).

---

## 변경 3 — research_story.html: SUMMARY 섹션 쇄신

**현재 문제**: SUMMARY (line 9412) 내용이 "6/4 업데이트", π0/SigLIP 언급 등 3~4달 전 내용.  
**기존 카드 내용**: legacy.html에 이미 있는 내용이기도 하고, 각 챕터에 상세 기록 있으므로 SUMMARY에서 구버전 배너를 교체하기만 하면 됨.

**교체 내용**:
1. 배너 문구: `"6/4 업데이트"` → `"확정 결론 (CH1→CH36 여정 완료)"`  
   본문: SOTA 한 문장 요약 (96.6% CL, ×9.4 pipeline gap)

2. finding-card 6개 교체:
   - ❌ E2E 실패: text attn 0%, CL 0%
   - 🔑 파이프라인 결정 변수: 10.3% → 96.6% (×9.4)  
   - ✅ Grounding 소스 무관: HSV = PG2 = LoRA 모두 96.6%
   - ✅ Head Ablation: Linear 69% → FCHead 93.1% → LSTM=MLP 96.6%
   - ✅ Window Ablation: MLP w≥4 포화, LSTM w=16 FPE 0.080m
   - ✅ Basket 이중 증명: zero-shot probe 96.6% + masking 9/9 flip

3. **기존 카드(구버전 6개)**: HTML 내에 `<!-- LEGACY SUMMARY (6/4) -->` 주석으로 감싸 보존

---

## 변경 4 — README.md: CH 카운트 수정

라인 89:  
`전체 연구 여정 (CH1→CH33)` → `전체 연구 여정 (CH1→CH36)`

---

## 변경 5 — docs/index.html: 두 곳 업데이트

**5-A. link-card "Full Research Journey"** (line 398~400):
- 제목: `Full Research Journey (CH1→CH33)` → `Full Research Journey (CH1→CH36)`
- 설명: `CH33이 최신 결론.` → `CH36이 최신 결론 (6/12 실사 테스트·논문 제출 결정).`

**5-B. metrics-bar FPE 셀** (line 227~229):
- 현재: `0.094 m` / `FPE · MLP w=4`
- 변경: `0.080 m` / `Best FPE · LSTM w=16`  
  sub-note: `MLP w=4: 0.094 m`  
  (둘 다 96.6% CL — 최저 FPE만 업데이트)

---

## 보존 전략

| 내용 | 보존 위치 |
|---|---|
| CH1~CH36 상세 챕터 전체 | 그대로 research_story.html 하단 유지 |
| SUMMARY 구버전 카드 (6/4) | HTML 주석으로 감싸 파일 내 보존 |
| 구버전 실험 (Exp01~09 등) | legacy.html에 이미 있음 |

---

## 작업 순서

1. [x] research_story.html: 96.7% 버그픽스
2. [x] research_story.html: CHECKPOINT 섹션 + TOC 항목 추가
3. [x] research_story.html: SUMMARY 섹션 쇄신 (구 카드 주석 처리)
4. [x] README.md: CH 카운트 수정
5. [x] docs/index.html: link-card + metrics-bar
6. [ ] git commit + cherry-pick → main push
