# KCI 논문 심사 대응 (1차/2차/3차 재심)

<p class="tagline">논문 제출 후 받은 심사위원 리뷰를 서술 보강 vs 실제 실험 필요로 분류해 대응한 기록 — 각 라운드의 지적사항, 그에 대한 ablation/재분석 결과, 재심 진행 상황.</p>

<div class="summary-box" markdown="1">

**압축 요약**

1차 심사 결과는 위원1·3 "수정 후 재심", 위원2 "수정 후 게재"다. 9건의 지적을 사람이 결정할
필요 없이 서술만 보강하면 되는 것과 실제 데이터/실험이 필요한 것으로 나눠 대응했다.
서술 보강 6건(lightweight 용어 재정의, 스코프 한계 명시, Mobility VLA 비교 재검토, frozen
인코더 결합 설명, Fig/Table 인덱스 표기, 하이퍼파라미터 서술 확장)은 영문 복붙 문단까지
`revision_briefing.html`에 준비됐다. 실제 데이터가 필요했던 두 건 중 하나(그라운딩 인코더
기여도 ablation)는 배포와 동일 조건에서 돌려 답을 얻었다: OWLv2 bbox 단독 64.09%, Kosmos-2
vision 단독 73.71%, 융합(배포 구성) 73.87%(best seed 74.13%가 배포 체크포인트 저장값과 정확히
일치) — Kosmos-2 비전이 성능 대부분을 담당하고 OWLv2는 보조적이라는 결론이 PG2 기반 이전
실험(Exp14)과 같은 방향으로 재확인됐다. 다른 하나(실패 케이스 원인 분류)는 시도 결과
**데이터가 존재하지 않는다는 것을 확인**했다 — `episode_log.csv`의 "실패" 기록은 95/100
헤드라인 배치가 아니라 그 이전 개발 단계 디버깅 로그였고, 정작 95/100 배치 자체는 H5에
세션별 성공/실패 라벨이 없다. 대신 원인 분해가 가능한 유일한 자료인 89/100 배치의 미검출
프레임 confidence 분포를 출처를 명시하고 인용하는 쪽으로 정직하게 서술하기로 했다. 안전(충돌
회피) 논의는 현재 파이프라인에 없는 기능이므로 Limitation 절에 명시하는 것으로 처리한다.
유일하게 미결정으로 남은 것은 새 출발위치/카메라/조도 조건 추가 실기 실험 — 로봇 재출동이
필요한 유일한 항목이라 재심 기한을 확인한 뒤 규모를 정하기로 했다.

</div>

## 챕터별 원문 발췌 (시간순)

<div class="chapter-block accent-a" markdown="1">

<div class="chapter-block-head"><span class="chapter-badge">CH 72</span> 1차 심사(수정 후 재심) 대응 — OWLv2/Kosmos-2 기여도 ablation, 실패분류 데이터 한계 확인</div>

<p class="chapter-subtitle-line">2026-09-14 ·
KCI 논문 1차 심사 결과(위원1·3 수정 후 재심, 위원2 수정 후 게재)를 받고, 지적 9건을
"서술로 해결 가능"과 "실제 데이터/실험 필요"로 분류해 대응했다. 그 중 위원1-①·위원3-④가
요구한 그라운딩 인코더 기여도 ablation을 실제로 돌렸고, 위원3-⑤·⑦이 요구한 실패 케이스
분류는 데이터가 없어서 할 수 없다는 것을 확인했다.</p>

<div class="card" markdown="1">

🟢 3줄 요약
① 배포 조건과 동일한 exp73 split(192/33ep, seed=42)에서 OWLv2 bbox 단독(64.09%)·Kosmos-2
vision 단독(73.71%)·융합(73.87%, best seed 74.13%=배포 체크포인트와 정확히 일치)을 3-seed로
비교 — Kosmos-2 비전이 성능 대부분을 담당하고 OWLv2는 보조적이라는, PG2 기반 이전 실험(Exp14)과
같은 방향의 결론을 OWLv2 그라운더에서도 재확인.
② logs/episode_log.csv의 "실패" 74건을 전수 조사한 결과, 이건 95/100 헤드라인
배치(08-07)가 아니라 그 이전 개발 단계(06-26~07-23) 디버깅
세션이었고, 95/100 배치 자체는 H5에 세션별 성공/실패 라벨이 없어(status 전량
manual_stop) 실패 5건의 원인을 재구성할 방법이 없다는 걸 확인 — 환각 없이 "데이터 한계"로
명시하는 쪽을 택함.
③ 서술 보강만으로 해결 가능한 6개 항목(lightweight 용어 정의, 스코프 강조, Mobility VLA
비교 재검토, frozen 인코더 결합 서술, Fig/Table 인덱스, 하이퍼파라미터 서술)은 복붙용 영문
문단까지 revision_briefing.html에 준비해뒀고, 새 조건(출발위치/카메라/조도)
실기 실험은 재심 기한 확인 후 결정하기로 미결정 상태로 남김.

</div>

<div class="card" markdown="1">

**🔬 72-1. OWLv2 단독 / Kosmos-2 단독 / 융합 ablation — 배포 조건 그대로 재현**

scripts/ablate_owlv2_kosmos2_fusion.py로 exp73_v6_vis_cache_stage1v3.pt
(Kosmos-2 vision, 그라운더 무관)의 vis 특징은 그대로 두고 bbox만
bbox_dataset_v6_owl.json(OWLv2)으로 교체 — 배포 체크포인트가 학습된 것과 동일한 절차다.
결과:
조건입력 차원val acc (3-seed)
OWLv2 bbox 단독
24 (4×window6)
64.09% ± 0.34%p
Kosmos-2 vision 단독
1536 (256×window6)
73.71% ± 0.10%p
융합(배포)
1560 (260×window6)
73.87% ± 0.20%p
best seed(74.13%)가 배포 체크포인트 저장값(74.13%)과 정확히 일치해 ablation 파이프라인이
배포 조건과 동일함을 검증했다. Kosmos-2 비전 단독으로 이미 융합 성능의 대부분(73.71/73.87)을
회복하고, OWLv2 bbox 단독도 8-class chance(12.5%)를 크게 상회(64.09%)하지만 둘보다는 못하다 —
두 인코더가 상호보완적이되 시각 표현이 더 결정적이라는
결론. PG2 기반 이전 실험(Exp14: bbox_only 67.4%/image_only 75.6%/융합 76.7%)과 방향이 같아
그라운더를 바꿔도 재현되는 구조적 결론임을 확인.
스크립트
scripts/ablate_owlv2_kosmos2_fusion.py · 결과
docs/v5/bbox_nav_owl/ablation_owlv2_kosmos2_fusion.json

</div>

<div class="card" markdown="1">

**🚧 72-2. 실패 케이스 원인 분류(위원3-⑤·⑦) — 요청받은 데이터가 존재하지 않음을 확인**

logs/episode_log.csv(145행)를 전수 조사했다. "실패" 74건의 메모를 읽어보니
"검증 스크리닝(L2 저장)", "프리뷰 모델 오류", "async모드 테스트" 등 전부 06-26~07-23 개발 단계 디버깅 세션이었다 — 논문에 실린
95/100 헤드라인 배치(08-07)와는 체크포인트/파이프라인 조건이 다른 별개의 로그다.
정작 95/100 배치 자체는 CH64/model_architecture_brief.html에 이미 기록된 대로 H5에
세션별 성공/실패 라벨이 없다(status 전량 manual_stop) — 실패 5건
각각이 어떤 프레임에서 왜 틀렸는지 재구성할 방법이 데이터상 없다.
대안으로 삼은 것: 원인 분해가 실제로 가능한 유일한 데이터는 89/100 배치(07-31)의 미검출
프레임 197장 confidence 분포(0.10~0.20 구간이 54.8%로 지배적 — 이미 model_architecture_brief.html
§5에 있음)뿐이다. 논문에는 이 배치 출처를 명시하고 "다른 배치의 진단 데이터에 근거해
일반화한 것"이라고 정직하게 쓰기로 했다 — 없는 데이터를 만들어내지 않는다는 원칙을
여기서도 지켰다.

</div>

<div class="card" markdown="1">

**📝 72-3. 서술 보강 6건 + 미결정 1건 — 위원별 지적 전체 분류**

서술로 해결(완료, 영문 복붙 문단까지 준비):
lightweight 용어 재정의(위원1-③, 학습파라미터 1.128M vs 추론참여 459.3M 구분),
스코프 한계 명시(위원2, 위원3-②, constrained vocabulary+고정 5목표지점),
Mobility VLA 비교 재검토(위원3-③, 동일조건 아님을 명시), frozen 인코더 결합 서술
보강(위원3-④, ablation 결과로 답변), Fig/Table 인덱스 표기(위원3-⑥, revision_briefing.html에서
이미 재정렬 완료), 하이퍼파라미터 서술 확장(위원3-⑨).
서술로 처리(실험 대상 아님): 충돌회피 안전 논의(위원3-⑧)
— 현재 파이프라인에 없는 기능이므로 Limitation 절에 명시.
미결정 — 사람 판단 대기: 새 출발위치/카메라/조도
조건 추가 실기(위원1-②, 위원3-①) — 유일하게 로봇 재출동이 필요한 항목이라 재심 기한을
확인한 뒤 조건 수·회당 시행 횟수를 역산하기로 함.
전체 트리아지 표
docs/01.paper/revision_briefing.html#round1-review

</div>

<a class="src-link" href="../v5/research_story.html#ch72">→ 원문 전체 보기 (research_story.html#ch72)</a>

</div>
