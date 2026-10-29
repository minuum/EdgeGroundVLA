"""교수님 보고용 그림 위주 슬라이드(16:9). 출력: docs/v5/prof_deck/index.html (+ PDF는 Firefox 캡처를 슬라이스). 수치 출처는 CH79(docs/v5/research_story.html#ch79)."""
from pathlib import Path
D = Path(__file__).resolve().parent.parent / "docs/v5/prof_deck"
CSS = """
*{box-sizing:border-box}body{margin:0;background:#050a14;font-family:'Noto Sans KR','NanumGothic',sans-serif;color:#e2e8f0}
.s{width:1280px;height:720px;padding:34px 54px;background:linear-gradient(160deg,#0a1628,#0a0f1a);position:relative;overflow:hidden;border-bottom:2px solid #050a14}
h1{font-size:40px;margin:0 0 8px;line-height:1.25}h2{font-size:30px;margin:0 0 4px;color:#fff}.sub{font-size:17px;color:#94a3b8;margin-bottom:14px}
.k{display:inline-block;background:#1e3a5f;border-radius:6px;padding:2px 10px;font-size:15px;color:#93c5fd;margin-right:8px}
.row{display:flex;gap:22px;align-items:flex-start}.col{flex:1}img{border-radius:8px;border:1px solid #1e3050;display:block;width:100%}
.pt{font-size:20px;line-height:1.55;margin:7px 0}.pt b{color:#6ee7b7}.warn{color:#fbbf24}.mut{color:#94a3b8;font-size:15px}
.card{background:#0e1d33;border:1px solid #1e3050;border-radius:10px;padding:10px 12px}.cap{font-size:13px;color:#94a3b8;margin-top:5px;line-height:1.4}
.big{font-size:64px;font-weight:900;color:#34d399;line-height:1}.foot{position:absolute;bottom:14px;left:54px;right:54px;font-size:12px;color:#64748b;display:flex;justify-content:space-between}
.flow{display:flex;flex-direction:column;gap:8px;align-items:stretch}.box{background:#12263f;border:1.5px solid #2b4a78;border-radius:8px;padding:8px 12px;font-size:16px;text-align:center}
.box.g{border-color:#34d399;background:#0d2b24}.arr{text-align:center;color:#64748b;font-size:18px;line-height:1}
a{color:#7dd3fc;text-decoration:none}table{border-collapse:collapse;width:100%;font-size:16px}td,th{padding:6px 10px;border-bottom:1px solid #1e293b;text-align:left}th{color:#93c5fd;border-bottom:2px solid #2b4a78}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.g2{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
"""
_N = [0]
def slide(n, body, src=""):
    _N[0] += 1; n = _N[0]
    return f'<section class="s">{body}<div class="foot"><span>{src}</span><span>EdgeGround-VLA · EGVLA2 현재 상황 · {n}</span></div></section>'
def ref(i, title, one, url, img): return f'<div class="card"><img src="img/{img}"><div style="font-size:15px;font-weight:700;margin-top:7px">{title}</div><div class="cap">{one}<br><a href="{url}">{url.replace("https://","")}</a></div></div>'
S = []
S.append(slide(1, '''<div style="margin-top:120px"><span class="k">교수님 보고</span><span class="k">2026-10-29</span></div>
<h1 style="margin-top:20px;font-size:50px">EGVLA2: 단일 백본(Florence-2) 내비게이션 정책<br>— 실기 폐루프 결과와 현재 모델 확정(A3)</h1>
<div class="pt" style="margin-top:34px;font-size:24px">① 무엇을 만들었나 ② 무엇을 시도했나 ③ 실기에서 무엇이 확인됐나 ④ 한계와 다음 단계 ⑤ 참고 문헌</div>
<div class="mut" style="margin-top:60px">상세: <a href="http://localhost:8001/v5/research_story.html#ch79">research_story.html#ch79</a> (📍 현재 상황 탭, 5-2절)</div>'''))
S.append(slide(2, '''<h2>1. 구조 — 두 모델 한눈에</h2><div class="sub">EGVLA1(OWLv2 + Kosmos-2) → EGVLA2(Florence-2 하나로 통합)</div>
<div class="row"><div class="col" style="flex:1.05"><img src="img/edgeground_vla_architecture.jpg" style="max-height:540px;width:auto;margin:auto"><div class="cap" style="text-align:center">EGVLA1: 검출기 + 비전 타워 두 개, MLP 분류 헤드 · 실기 92.0%(276/300)</div></div>
<div class="col"><div class="flow"><div class="box">카메라 이미지</div><div class="arr">▼</div><div class="box"><b>Florence-2</b> 한 번의 인코딩<br><span class="mut">bbox(cx,cy,area) + 이미지 특징 768-d 동시 추출</span></div><div class="arr">▼</div><div class="box">bbox 3프레임 창 12-d + 특징 768-d</div><div class="arr">▼</div><div class="box g"><b>Flow-matching 행동 헤드</b> (π0식 AdaLN, H=8)<br><span class="mut">연속 조이스틱: 방향각 + 속도</span></div><div class="arr">▼</div><div class="box">서빙: 서보 혼합(선택) → 로봇 명령</div></div>
<div class="pt" style="margin-top:14px">· 모델 2개 → <b>1개</b>, 헤드는 분류 → <b>연속 행동</b><br>· 학습: 300 에피소드 · 29,161프레임(pos1)</div></div></div>''', "코드: robovlm_nav/models/egvla2_fm_heads.py · 구조도: docs/v5/figures"))
S.append(slide(3, '''<h2>2. 시도한 것과 결과 요약</h2><div class="sub">오프라인(같은 분할·라벨·지표) 비교 — 사람 방향 오차(°)는 낮을수록 사람 조작과 같음</div>
<div class="row"><div class="col"><img src="img/e2e_bars.png"><div class="cap">E2E(DaViT 미세조정) 변형은 기존 고정 융합 특징(A)을 못 이김. 3분할 평균. 저예산 변형(1 seed, 8 epoch, LoRA 미튜닝)만 시험했으므로 "E2E 무용"이 아니라 "이 변형들은 이득 없음"</div></div>
<div class="col"><table><tr><th>시도</th><th>결과</th></tr>
<tr><td>월드모델식 보조 손실<br><span class="mut">(미래 bbox·특징 변화 예측)</span></td><td>변화 예측은 잘함, <b class="warn">타깃 반응 S는 안 오름</b> → 1단계에서 중단</td></tr>
<tr><td>서보 라벨(타깃으로 향하는 방향 라벨 혼합)</td><td>타깃 반응 S ↑, 오프라인 방향 오차는 ↑(나빠짐)</td></tr>
<tr><td>서버 서보 혼합(A1)</td><td>실기 11/23 — 효과 약함</td></tr></table>
<div class="pt" style="margin-top:16px"><b>핵심</b>: 오프라인 방향 오차는 모델 선택 기준으로 <b>부적합</b>(A3는 +22° 나쁜데 실기는 최고) → 선택은 실기 폐루프로.</div></div></div>''', "수치: e2e_fullstage.json · gap_wm.json · gap_e1b.json (CH79 3~4절)"))
S.append(slide(4, '''<h2>3. 실기 폐루프 결과 (10/28, pos1)</h2><div class="sub">성공 = 수동 기록(episode_log). 구간은 Wilson 95%</div>
<div class="row"><div class="col"><img src="img/first60/success_ci.png"></div><div class="col"><div class="big">60 / 60</div><div class="mut" style="margin-bottom:12px">A3 (servolabel 1.0) 시간순 앞 60회 · 구간 94~100%<br>(같은 날 이어서 한 3회 포함 전체 63/63, 결과 동일)</div>
<div class="pt">· A2(서보 0.5) 14/20, A1(서버 혼합) 11/23</div><div class="pt">· 서보 라벨이 강할수록 성공이 높음</div><div class="pt">· 타깃 5종 위치 모두 성공(위치별 9~16회)</div>
<div class="pt warn">· 단서: ASYNC·EMA 0.0(합의 SYNC/0.5와 다름), 조건별 연속 블록, 기준 A0 1회, pos1만, 수동 판정</div></div></div>''', "집계: ROS_action/dryrun_closedloop_20261028. 60회 = 순수 A3의 시간순 앞 60회(결과를 본 뒤 정한 구분, 모두 성공이라 결론 불변). A1 23·A2 20은 전체"))
S.append(slide(5, '''<h2>3-2. 무엇을 "성공"이라 했나 — 판정 근거</h2><div class="sub">성공의 정의는 현장 관찰자의 수동 기록. 영상 bbox로 그 기록을 사후 교차 확인</div>
<div class="row"><div class="col"><img src="img/frames_a3.png"><div class="cap">A3 성공 세션: 타깃을 정면에 두고 접근, 마지막 프레임에서 바구니가 화면 하단 중앙을 채우며 정지</div></div>
<div class="col"><table><tr><th>구분</th><th>기준</th></tr>
<tr><td><b>정의(수동)</b></td><td>현장에서 사람이 기록한 결과 열(성공/실패)</td></tr>
<tr><td><b>성공 세션의 영상 특징</b></td><td>마지막 타깃 면적 0.20~0.34(정지 기준 0.25 근처), 타깃 소실 0프레임, 평균 13.5스텝</td></tr>
<tr><td><b>실패 세션의 영상 특징</b></td><td>타깃 소실 · 과접근(면적 0.45 이상) · 엉뚱한 물체 추종(A1·A2 실패 16건)</td></tr></table>
<div class="pt warn" style="margin-top:14px">한계: 서면 판정표가 사전에 없었고(핸드오프에는 소실률·정지 거리·성공률 기준만 기재), 조건별 연속 블록이라 판정이 블라인드가 아니었을 가능성이 있다. 그래서 영상 지표로 교차 확인했고, 로봇 측에 실제 사용한 판정 문구와 정지 거리(FPE) 기록을 요청할 예정.</div></div></div>''', "근거: episode_log_snapshot.csv 결과 열 · H5 grounding/bbox · docs/plans/HANDOFF_20261028_ROBOT_TEST_ORDER.md"))
S.append(slide(6, '''<h2>3-3. 목표 위치별 성공 사례 — 시작(pos1)에서 도착까지</h2><div class="sub">A3 성공 세션 중 경로마다 1개(해당 경로 성공 세션의 스텝 수 중앙값에 가까운 것), 진행 1/5~5/5 시점</div>
<div class="row"><div class="col" style="flex:2.2"><img src="img/path_frames.png"></div><div class="col"><div class="pt"><b>공통</b>: 멀리 보이던 바구니(면적 0.02~0.03)가 점점 커져 0.22~0.26에서 정지</div><div class="pt">· 시작 위치는 전부 pos1 (pos2·pos3 미시험)</div><div class="pt warn">· 중앙 행은 마지막 프레임에서 바구니가 화면 오른쪽으로 치우쳐 있음 — 수동 기록은 성공. 정지 위치 정확도는 FPE 기록 필요</div><div class="mut">선택 규칙을 미리 정해 뽑은 사례이며 결과가 좋은 것만 고른 것이 아님</div></div></div>''', "생성: scripts/ch79_path_frames.py"))
S.append(slide(5, '''<h2>4. 판정 편향은 없었나 — 수동 판정과 무관한 H5 확인</h2><div class="sub">영상 bbox만으로 본 마지막 프레임 상태</div>
<div class="row"><div class="col"><img src="img/first60/final_bbox_scatter.png"></div><div class="col"><img src="img/first60/bbox_trajectories.png"></div></div>
<div class="pt" style="margin-top:12px">· A1·A2 <b>실패</b>는 타깃 소실이거나 과접근(면적 0.45↑). A3(H5 보유 56개)는 전부 면적 0.20~0.34, 소실 0.</div><div class="pt">· 판정이 후했다면 A3에도 이런 세션이 섞였어야 함 → <b>편향 증거 없음</b>. <span class="warn">(A3 고유의 다른 실패 유형은 이 방법으로 못 봄)</span></div>''', "그림 생성: scripts/ch79_evidence_figs.py"))
S.append(slide(6, '''<h2>5. 실제 영상 — 성공과 실패의 모습</h2><div class="sub">빨간 상자 = 모델이 잡은 타깃(bbox)</div>
<div class="row"><div class="col"><div class="pt"><b>A3 성공</b> — 바구니에 정면 접근, 면적 0.22~0.32에서 정지</div><img src="img/frames_a3.png"></div>
<div class="col"><div class="pt"><b>A1 실패</b> — ① 검은 판을 바구니로 잘못 잡고 벽으로 ② 바구니를 지나쳐 벽만 보임</div><img src="img/frames_a1.png"></div></div>
<div class="mut" style="margin-top:10px">유형별 2건 예시(시작·중간·마지막 프레임). 전체 통계는 앞 장의 산점도.</div>'''))
S.append(slide(7, '''<h2>6. 평가 코스 — 단계별로 무엇이 가려졌나</h2><div class="sub">오프라인으로 후보를 거르고, 선택은 실기로</div>
<table><tr><th>단계</th><th>측정</th><th>알려준 것</th><th>한계</th></tr>
<tr><td>① 오프라인 방향 오차</td><td>사람 방향과의 차이</td><td>A3 +22° → <b>선택 기준 부적합</b></td><td>사람 조작=정답 가정</td></tr>
<tr><td>② 반사실 민감도 S</td><td>타깃 cx ±0.2 이동 시 행동 변화</td><td>서보 라벨↑ → S↑</td><td>오프라인 지표</td></tr>
<tr><td>③ 구조 비교</td><td>E2E·월드모델 3분할</td><td>대체안이 기존 방식을 못 이김</td><td>저예산 변형만</td></tr>
<tr><td>④ 실기 폐루프</td><td>수동 성공 + H5 교차 확인</td><td><b>A3 63/63</b></td><td>ASYNC·EMA 0.0, pos1</td></tr></table>
<div class="pt" style="margin-top:22px"><b>다음 단계(로봇 측 핸드오프 문서로 전달)</b>: A3 pos2·pos3 일반화 → SYNC·EMA 0.5 합의 설정 시험 → 속도 정렬</div>
<div class="pt warn">보고 문구 원칙: "실기 pos1 63/63", 한계 병기. 300회 실기(EGVLA1 92.0%)와 직접 비교하지 않음(조건 다름)</div>''', "핸드오프: docs/plans/HANDOFF_20261029_REPLY_A3_ADOPTED.md"))
S.append(slide(8, '<h2>7. 참고한 모델·논문 (1) 구조의 뿌리</h2><div class="sub">그림은 각 논문의 Figure 1(발췌)</div><div class="g3">'
  + ref(0, "Florence-2 (Xiao et al., 2023)", "하나의 인코더로 검출·그라운딩 등 통합 — EGVLA2 백본", "https://arxiv.org/abs/2311.06242", "ref_2311.06242.jpg")
  + ref(0, "π0 (Black et al., Physical Intelligence, 2024)", "Flow-matching 행동 헤드(AdaLN) — 우리 헤드의 원형", "https://arxiv.org/abs/2410.24164", "ref_2410.24164.jpg")
  + ref(0, "AsyncVLA (Hirose et al., 2026)", "큰 모델 + 엣지 어댑터 비동기 내비게이션 — 경량화 방향 비교", "https://arxiv.org/abs/2602.13476", "ref_2602.13476.jpg") + '</div><div class="pt" style="margin-top:22px"><b>차용·비교 지점</b>: Florence-2 = 단일 백본 / π0 = 연속 행동을 flow-matching으로 생성 / AsyncVLA = 엣지 실시간성 문제 설정(우리는 단일 소형 백본으로 해결 시도)</div>'))
S.append(slide(9, '<h2>7. 참고한 모델·논문 (2) 인식·특징</h2><div class="sub">EGVLA1과 비교·E2E 시험에 사용</div><div class="g3">'
  + ref(0, "OWLv2 (Minderer et al., 2023)", "개방어휘 검출기 — EGVLA1의 타깃 검출", "https://arxiv.org/abs/2306.09683", "ref_2306.09683.jpg")
  + ref(0, "Kosmos-2 (Peng et al., 2023)", "그라운딩 MLLM — EGVLA1 비전 타워", "https://arxiv.org/abs/2306.14824", "ref_2306.14824.jpg")
  + ref(0, "DaViT (Ding et al., 2022)", "Florence-2의 비전 인코더 — E2E 시험에서 LoRA 적용", "https://arxiv.org/abs/2204.03645", "ref_2204.03645.jpg") + '</div>'))
L = [("RoboVLMs — What Matters in Building VLA Models", "https://arxiv.org/abs/2412.14058", "선행연구 프레임워크(Kosmos-2 백본 학습 레시피)"),
("OpenVLA", "https://arxiv.org/abs/2406.09246", "VLA 대표 베이스라인"), ("NaVILA", "https://arxiv.org/abs/2412.04453", "내비게이션 VLA 비교(실기 지표 방식)"),
("MobilityVLA", "https://arxiv.org/abs/2407.07775", "지시 기반 내비게이션 VLA"), ("TIC-VLA", "https://arxiv.org/abs/2510.03342", "지연 보정 내비게이션 VLA"),
("OpenFrontier", "https://arxiv.org/abs/2603.05377", "그라운딩 기반 일반 내비게이션"), ("TANGO", "https://arxiv.org/abs/2609.09158", "휴머노이드 내비게이션 VLA"),
("π0.7 (월드모델 서브골)", "https://arxiv.org/abs/2604.15483", "월드모델식 시도의 동기"), ("V-JEPA 2", "https://arxiv.org/abs/2506.09985", "잠재 공간 미래 예측(월드모델식 시도의 참고 방향)"),
("DiT (AdaLN-Zero)", "https://arxiv.org/abs/2212.09748", "헤드의 조건 주입 방식"), ("Flow Matching", "https://arxiv.org/abs/2210.02747", "행동 생성 학습 원리"), ("LoRA", "https://arxiv.org/abs/2106.09685", "E2E 시험의 미세조정 방식"),
("Florence-2-base (HF)", "https://huggingface.co/microsoft/Florence-2-base", "사용 체크포인트"), ("OWLv2 ensemble (HF)", "https://huggingface.co/google/owlv2-base-patch16-ensemble", "EGVLA1 검출기"), ("Kosmos-2 (HF)", "https://huggingface.co/microsoft/kosmos-2-patch14-224", "EGVLA1 비전 타워"),
("레포", "https://github.com/minuum/EdgeGroundVLA", "코드·문서(branch inference-integration)")]
rows = "".join(f'<tr><td>{a}</td><td><a href="{u}">{u.replace("https://","")}</a></td><td class="mut">{c}</td></tr>' for a, u, c in L)
S.append(slide(10, f'<h2>7. 참고한 모델·논문 (3) 링크 모음</h2><table style="font-size:14px"><tr><th>자료</th><th>링크</th><th>용도</th></tr>{rows}</table>', "그림 6편 + RoboVLMs·π0.7·V-JEPA 2·DiT·Flow Matching·LoRA는 원문 PDF로 제목 확인. 나머지는 레포 문서의 인용을 그대로 옮김(수치 인용 전 원문 확인 필요)"))
html = f'<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>EGVLA2 교수님 보고 슬라이드</title><style>{CSS}</style></head><body>{"".join(S)}</body></html>'
(D / "index.html").write_text(html, encoding="utf-8"); print(len(S), "slides")
