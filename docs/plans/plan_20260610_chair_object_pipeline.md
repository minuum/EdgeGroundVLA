# Plan — 의자(Chair) 객체 전환 파이프라인

> 작성: 2026-06-10 · 상태: **검토 대기 (승인 전 구현 금지)**
> 연관 문서: `docs/v5/PRETRAINED_OBJECT_REPLACEMENT_PLAN.md`, `docs/v5/MEETING_20260604_LORA_REBUILD.md`

---

## 0. 배경 / 결정 사항

- 6/4 미팅 피드백 **R3(단일 데이터)·OOD 약점**을 푸는 유일한 경로 = 새 객체 데이터 재수집.
- 객체 후보 분석(`PRETRAINED_OBJECT_REPLACEMENT_PLAN.md`)에서 **Chair/Stool**이 인식 강도 98%/85%로 1순위.
- **사용자 결정(6/10):** "흰 의자가 아니어도 된다. 그냥 의자로." → **색 수식어 제거**, 프롬프트는 `detect chair`.
  - 근거: 6/4 텍스트 변형 함정("grey basket" vs "grey container")을 색에도 적용. 타겟 의자 1개만 두므로 색 수식어 불필요. 색 무관일 때 PaliGemma 인식률이 더 높음(98%).
  - 조건: 배경과 **대비 분명**, 다리보다 **등받이+좌판이 통으로 잡히는 형태**.

## 1. 목표

1. **(이번 단계) 의자 인식 검증** — base PG2가 다양한 의자를 `detect chair`로 안정적으로 잡는지 실측. 색/표현 변형(`chair` vs `white chair` vs `stool` vs `office chair`)이 인식률에 미치는 영향 측정 → 프롬프트 확정.
2. **(다음 단계) 수집 프로토콜 확정** — 350~500 ep, 70% 직진 + 30% 복원, STOP 자동 합성.
3. **(후속) Stage2 MLP 재학습** — 의자 데이터로 decomposition 파이프라인 재학습.

본 plan은 **1번(인식 검증)**의 구현을 다룬다. 2·3번은 1번 결과를 보고 별도 plan으로 확정.

---

## 2. 접근 — 의자 인식 검증 스크립트

신규 파일: **`scripts/probe_chair_recognition.py`**

기존 `scripts/probe_pg2_objects.py`와의 차이:

| | probe_pg2_objects.py (기존) | probe_chair_recognition.py (신규) |
|---|---|---|
| 입력 이미지 | 우리 basket 에피소드 프레임 | **외부 의자 이미지 다종** (Wikimedia 등) |
| 측정 의미 | basket 프레임에서 chair 오탐 | **실제 의자에서 chair 정탐률** |
| phrase | basket 위주 10종 | **chair 변형 4종** (chair/white chair/stool/office chair) |
| 산출 | JSON | JSON + **bbox 오버레이 그리드 PNG** |

### 2.1 이미지 소스 (여러 군데)

재현 가능하고 키 불필요한 **Wikimedia Commons 직링크**를 큐레이션해 다양성 확보:
- 종류: 플라스틱 스툴, 사무용 의자, 목재 의자, 폴딩 체어, 바 스툴, 암체어
- 색: 흰색/검정/나무색/컬러
- 각도: 정면 / 측면 / 비스듬
- 배경: 단색 스튜디오 / 실내 환경

```python
# 큐레이션 목록 (직링크, ~12장)
CHAIR_URLS = [
    ("plastic_stool_white", "https://upload.wikimedia.org/.../white_stool.jpg"),
    ("office_chair_black",   "https://upload.wikimedia.org/.../office_chair.jpg"),
    ("wooden_chair",         "https://upload.wikimedia.org/.../wooden_chair.jpg"),
    # ... 총 ~12장, 종류·색·각도·배경 다양
]
```

다운로드 → `docs/v5/chair_probe/images/<name>.jpg` 캐시 (있으면 skip).

### 2.2 측정

각 (이미지 × phrase)에 `detect <phrase>` 실행:

```python
PHRASES = ["chair", "white chair", "stool", "office chair"]

def detect(img, phrase):
    inp = proc(text=f"<image>detect {phrase}", images=img, return_tensors="pt")...
    gen = model.generate(**inp, max_new_tokens=48, do_sample=False)
    locs = LOC.findall(decoded)        # <loc####> 4개 → bbox
    return cx, cy, area, hit
```

지표(phrase별 집계):
- **hit_rate** — bbox 검출 성공률
- **area_mean / area_std** — bbox 크기 안정성
- **bbox 위치** — cx/cy 분포
- **phrase 비교** — `chair` vs `white chair` vs `stool` 어느 쪽이 hit율↑ / 변형에 강한가

### 2.3 시각 산출

`docs/v5/chair_probe/chair_recognition_grid.png` — 의자별 이미지에 검출 bbox 오버레이한 그리드. research_story에 바로 삽입 가능.

`docs/v5/chair_probe/chair_recognition.json` — 수치 요약.

### 2.4 한계 명시 (환각 방지)

> ⚠️ 스톡 이미지는 **스튜디오 정면 샷** — 로봇의 low-angle(30cm) · 224px · 원거리 POV와 다르다.
> 이 검증은 "PG2가 chair 개념을 아는가 + 프롬프트 선택" 용도의 **필요조건**이지 충분조건이 아니다.
> 최종 검증은 로봇 카메라로 찍은 의자 프레임 몇 장으로 별도 확인 필요.

---

## 3. 수정/생성 파일

| 파일 | 작업 |
|---|---|
| `scripts/probe_chair_recognition.py` | **신규** — 인식 검증 |
| `docs/v5/chair_probe/images/*.jpg` | 다운로드 캐시 |
| `docs/v5/chair_probe/chair_recognition.json` | 수치 산출 |
| `docs/v5/chair_probe/chair_recognition_grid.png` | 시각 산출 |

기존 코드 수정 없음. `third_party/RoboVLMs/` 미접촉.

---

## 4. 트레이드오프 / 리스크

- **스톡 ≠ 로봇 POV**: 위 2.4 한계. 보완책 = 로봇 프레임 별도 확인(다음 단계).
- **Wikimedia 링크 깨짐 가능**: 다운로드 실패 시 해당 이미지 skip, 나머지로 진행 + 로그 경고.
- **`detect white chair`가 색 흔들림에 약할 것**이라는 가설 검증 — 결과로 색 수식어 제거 결정을 뒷받침/반증.

---

## 5. 완료 기준

- [ ] 다양한 의자 ~12장 다운로드 성공
- [ ] phrase 4종 × 이미지 detect 완료, JSON 산출
- [ ] bbox 오버레이 그리드 PNG 산출
- [ ] `chair` vs 변형 phrase hit율 비교표 → 프롬프트 확정 근거 확보
- [ ] (선택) research_story에 의자 인식 검증 섹션 추가

---

## 6. 이후 단계 (별도 plan)

1. 수집 프로토콜 확정 — `PRETRAINED_OBJECT_REPLACEMENT_PLAN.md` §3 기반, "흰 스툴"→"의자(색 무관)"로 갱신, STOP 자동 합성 규칙.
2. 조이스틱 수집 스크립트 점검 (`scripts/...joystick...`).
3. Stage2 MLP 재학습 + closed-loop 평가.
