"""모델 레지스트리용 의사결정 과정 & 실제 세션 판단 근거(Factor) 시각화 모듈.

build_model_registry.py에서 호출하여:
  1) 상단 인터랙티브 '🧠 모델 의사결정 & 판단 근거(Factor) 해부 스튜디오' HTML/JS 블록 생성
  2) 각 모델 카드 내부에 물리적 레이어 구조 대신 '의사결정 메커니즘 & 4대 판단 요인 게이지' 블록 삽입
입력 데이터: docs/v5/egvla2_fm/registry_decision_anatomy.json
"""
from __future__ import annotations
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANATOMY_JSON = ROOT / "docs/v5/egvla2_fm/registry_decision_anatomy.json"

# 모델 카드별 의사결정 프로파일 (조향근거%, 정지근거(면적)%, 낯선물체출발%, 언어게이트, 한줄 의사결정 요약, 스튜디오 연결 키)
DECISION_PROFILES: dict[str, dict] = {
    "EG0-01": dict(
        flow="[카메라 이미지 + 텍스트] ➔ [Kosmos-2 LoRA 직접 분류] ➔ [8방향 이산 토큰]",
        steer=(15, "배경·이미지 통째 암기 (박스 없음)"),
        stop=(10, "이미지 외형 암기"),
        start=(0, "폐루프 주행 0% (경로 복원 실패)"),
        lang="E2E 토큰 혼합 (박스 분해 없음)",
        diag="목표물의 위치(bbox)를 명시적으로 찾지 않고 픽셀에서 바로 행동을 찍어, 시작 위치가 조금만 바뀌어도 방향을 잃습니다.",
        studio_key=None,
    ),
    "EG0-02": dict(
        flow="[GT/검출 BBox + 이미지] ➔ [분해형 MLP] ➔ [8방향 이산 행동]",
        steer=(75, "BBox 위치 조건화 시작"),
        stop=(40, "소규모 데모(45ep) 면적 암기"),
        start=(67, "시뮬 폐루프 66.7%"),
        lang="고정 타깃 (단일 물체)",
        diag="순수 E2E(0%)를 벗어나 '박스 검출 → 행동 결정' 2단계 분해가 필수적임을 처음 입증한 초기 프로토타입입니다.",
        studio_key=None,
    ),
    "EG1-02": dict(
        flow="[OWLv2 BBox(창6, ×3증폭) + Kosmos-2 특징] ➔ [MLP 8클래스 분류] + [컨트롤러 면적 정지]",
        steer=(88, "BBox cx ×3 증폭으로 조향 유도"),
        stop=(70, "컨트롤러 근접 면적 + STOP 클래스"),
        start=(92, "바구니 300세션 92% (pos1~3)"),
        lang="OWLv2 프롬프트 그라운딩",
        diag="박스 좌표를 3배 증폭해 MLP가 박스 위치를 무시하지 못하게 강제한 1세대 실기 기준선(이산 8방향 한계 존재).",
        studio_key=None,
    ),
    "EG2-04": dict(
        flow="[Florence-2 BBox(창3) + 전역 이미지(768)] ➔ [결정적 MLP 회귀] ➔ [연속 속도(lx,ly)]",
        steer=(20, "이미지 배경 지름길 지배"),
        stop=(15, "이미지 외형 지름길"),
        start=(30, "다중 궤적 평균화로 회전/출발 둔화"),
        lang="그라운더 문구만 사용",
        diag="동일 장면에서 좌·우회전이 섞인 데이터를 단순 평균(MSE) 회귀하여 조향 민감도가 크게 떨어집니다.",
        studio_key=None,
    ),
    "EG2-05": dict(
        flow="[Florence-2 BBox(드롭0%) + 이미지(768)] ➔ [PiZeroFM 서보라벨 없음] ➔ [연속 청크]",
        steer=(18, "이미지 암기 지배 (cx 반전 무반응)"),
        stop=(6, "11/16 근접(25~30%) 정지 6%"),
        start=(100, "11/13 의자·책상 중앙 100% 이동"),
        lang="그라운더 문구만 사용",
        diag="서보 라벨이 없어 박스를 좌우로 옮겨도 이미지 기억대로 조향합니다(방향 차이 A3 대비 23.3°).",
        studio_key=None,
    ),
    "EG2-06": dict(
        flow="[BBox(드롭30%) + 이미지(768)] ➔ [PiZeroFM 순수 데모 라벨] ➔ [연속 청크]",
        steer=(12, "시각 지름길 지배 (cx 반전 변화 1.2°)"),
        stop=(23, "시각 지름길 정지 (면적 민감도 0%)"),
        start=(3, "11/13 의자 중앙 3% (비출발)"),
        lang="그라운더 문구만 사용",
        diag="768차원 이미지 특징이 12차원 BBox를 압도하여, 박스가 왼쪽에 있어도 배경 기억대로 직진/우회전합니다(실기 32%).",
        studio_key="a3-no-servo",
    ),
    "EG2-08": dict(
        flow="[BBox(드롭30%) + 이미지(768)] ➔ [PiZeroFM 서보라벨 λ=0.5 혼합] ➔ [연속 청크]",
        steer=(66, "BBox cx 조향 50% + 데모 라벨 50%"),
        stop=(12, "11/16 근접(25~30%) 정지 12%"),
        start=(0, "11/13 의자 중앙 0% (시각 억제 -0.39)"),
        lang="그라운더 문구만 사용",
        diag="방향의 절반만 BBox cx에 정합시킨 중간 모델(실기 70%). 여전히 시각 외형 지름길로 인해 의자 중앙에서 출발하지 못합니다.",
        studio_key="a3-servo0.5",
    ),
    "EG2-09": dict(
        flow="[지시어 BBox(cx,area) + 이미지(768)] ➔ [조향: cx 서보 100% | 정지: 시각 외형 지름길]",
        steer=(98, "BBox cx 기하 조향 98% (반전 +43°)"),
        stop=(41, "바구니 시각 외형 정지 (면적 민감도 +3%p)"),
        start=(0, "의자·책상 중앙 0~4% (시각 억제 -0.28)"),
        lang="그라운더 지시어 박스 검출 (헤드 텍스트 없음)",
        diag="조향은 BBox cx를 완벽히 따르나, 출발/정지 크기는 768-d 이미지 외형에 의존해 '중앙의 낯선 의자/책상 = 이미 도착'으로 오인해 멈춥니다.",
        studio_key="a3",
    ),
    "EG2-11": dict(
        flow="[BBox(드롭30%) + 이미지(30% 확률 0)] ➔ [PiZeroFM 서보라벨 λ=1.0]",
        steer=(96, "BBox cx 기하 조향"),
        stop=(6, "11/16 근접(25~30%) 정지 6%"),
        start=(0, "11/13 의자 중앙 0% (시드0 한계)"),
        lang="그라운더 지시어 박스 검출",
        diag="단순히 이미지 특징을 30% 확률로 0으로 가리는 것만으로는 시드 0의 시각 정지 지름길을 깨지 못했습니다.",
        studio_key=None,
    ),
    "EG2-12": dict(
        flow="[BBox(드롭30%) + 이미지(50% 확률 0)] ➔ [PiZeroFM 서보라벨 λ=1.0]",
        steer=(95, "BBox cx 기하 조향"),
        stop=(6, "11/16 근접(25~30%) 정지 6%"),
        start=(1, "11/13 의자 중앙 1% (기각)"),
        lang="그라운더 지시어 박스 검출",
        diag="이미지 특징 50% 드롭아웃도 의자 중앙 비출발을 해결하지 못하고 바구니 이동률만 낮췄습니다.",
        studio_key=None,
    ),
    "EG2-13": dict(
        flow="[BBox(드롭30%) + 이미지(30% 타 프레임 교체)] ➔ [PiZeroFM 서보라벨 λ=1.0]",
        steer=(97, "BBox cx 기하 조향"),
        stop=(6, "11/16 바구니 근접 정지 30% (정지 상실)"),
        start=(100, "11/13 의자·책상·화분 중앙 100% 출발"),
        lang="그라운더 지시어 박스 검출",
        diag="이미지를 다른 프레임과 섞어 의자 중앙 출발(100%)은 풀었으나, 도착 장면에서도 멈추지 않는 부작용이 발생했습니다.",
        studio_key="a3-featswap30",
    ),
    "EG2-14": dict(
        flow="[지시어 BBox(cx,area) + 이미지(768)] ➔ [조향: cx 서보 98% | 정지 문턱 상승(시드2)]",
        steer=(98, "BBox cx 기하 조향 (A3와 0.7° 차이)"),
        stop=(18, "11/16 근접(25~40%) 정지 12~22% (둔화)"),
        start=(100, "실기 의자·책상 중앙 5/5 (100% 출발)"),
        lang="그라운더 지시어 박스 검출",
        diag="시드 2(a3-s2)는 출발 시각 억제가 사라져 중앙 새 물체 출발(5/5)을 해결했으나, 도착 정지 신호도 늦어져 바구니 옆 6/10 접촉이 발생했습니다.",
        studio_key="a3-s2",
    ),
    "EG2-15": dict(
        flow="[지시어 BBox(드롭10%) + 이미지(768)] ➔ [PiZeroFM 서보라벨 λ=1.0, 시드1]",
        steer=(98, "BBox cx 기하 조향 (반전 +55°)"),
        stop=(63, "11/16 초근접(30~40%) 정지 63% (의자 60%)"),
        start=(100, "11/13 의자·책상·화분 중앙 100% 출발"),
        lang="그라운더 지시어 박스 검출",
        diag="박스 가림을 30%→10%로 낮춰 모델이 이미지 대신 BBox를 90% 이상 신뢰하게 만든 결과, 기존 레시피 중 유일하게 중앙 출발(100%)과 근접 정지(63%)를 함께 달성했습니다.",
        studio_key="a3-drop10-s1",
    ),
    "EG2-16": dict(
        flow="[지시어 BBox(cx,area) + 이미지(50%교체)] ➔ [조향: cx 서보 100% | 정지: area≥0.23 기하 정합]",
        steer=(99, "BBox cx 기하 조향 (A3와 0.54° 차이)"),
        stop=(100, "11/16 근접(25~40%) 정지 100% (면적 민감도 +100%p)"),
        start=(100, "시드0·1 모두 의자·책상 중앙 100% 출발 (오정지 0%)"),
        lang="그라운더 지시어 박스 검출 (드롭인 .pt 호환)",
        diag="조향뿐 아니라 속도 크기(||a||)까지 BBox 면적에 기하학적으로 정합(area_servo_label)하여, 시각 외형 지름길을 원천 차단하고 '중앙 100% 출발 + 근접 90~100% 정지'를 동시 달성했습니다.",
        studio_key="a3-ag23",
    ),
    "EG4-04": dict(
        flow="[고정 바구니 BBox + 이미지(768) + 지시어 텍스트(768)] ➔ [PiZeroFM 1536-d]",
        steer=(95, "긍정 지시어 시 BBox cx 조향"),
        stop=(99, "비바구니/부정 지시어 99.5% 언어 게이트 정지"),
        start=(0, "비바구니 물체 지시 시 설계상 0% 정지"),
        lang="헤드 내부 768-d 텍스트 조건화 (어휘 40+40)",
        diag="지시어 의미를 헤드가 직접 판단해 목표가 아니면 99.5% 멈추지만, 박스가 바구니에 고정되어 다른 물체로 주행하는 역할은 아닙니다.",
        studio_key=None,
    ),
    "EG4-07": dict(
        flow="[고정 바구니 BBox(훼손증강) + 이미지(768) + 텍스트(768)] ➔ [PiZeroFM V4R]",
        steer=(96, "박스 반전 시에도 좌우 94.3% 강건 조향"),
        stop=(99, "비바구니 지시어 99.5% 정지 (언어 게이트)"),
        start=(0, "비바구니 지시 시 설계상 정지 (바구니 전용)"),
        lang="헤드 내부 768-d 텍스트 조건화 + 박스 훼손 강건화",
        diag="V4에 박스 드롭·지터·반전 증강을 더해 시드 편차를 없애고 오검출 저항성을 높인 텍스트 조건화 모델입니다.",
        studio_key=None,
    ),
    "EG3-04": dict(
        flow="[카메라 이미지] ➔ [DaViT 마지막 단 LoRA (디코더 제거)] ➔ [PiZeroFM 직접 출력]",
        steer=(35, "A3 대비 25.1° 오차 (좌우 43%, S=-2.90°)"),
        stop=(26, "11/16 근접 의자 정지 26% · 바구니 45%"),
        start=(45, "비전 백본 과적합으로 실기 시뮬 붕괴"),
        lang="텍스트/디코더 미사용 (바구니 단일 암기)",
        diag="구 명칭 B1(ENC-E2E-1). VLM 디코더를 빼고 300ep로만 학습해 실기 장면에서 방향·정지 모두 붕괴했습니다.",
        studio_key=None,
    ),
    "EG3-06": dict(
        flow="[카메라 이미지] ➔ [DaViT LoRA (디코더 제거)] ➔ [Area-Grounded PiZeroFM]",
        steer=(82, "A3 대비 8.05° (좌우 82.0%, S=+26.75°)"),
        stop=(91, "11/16 근접(≥0.25) 바구니 90% · 의자 93%"),
        start=(100, "11/13 의자·책상 중앙 출발 100%"),
        lang="디코더 미사용 (ENC-E2E-AG)",
        diag="기존 B1에 Area-Grounded 라벨을 적용해 오프라인 시뮬 지표를 크게 회복했으나, VLM 디코더 미사용으로 Open-Vocab 한계가 남습니다.",
        studio_key=None,
    ),
    "EG3-07": dict(
        flow="[카메라 이미지] ➔ [DaViT LoRA + 인코더 Aux BBox 헤드] ➔ [Area-Grounded PiZeroFM]",
        steer=(85, "A3 대비 7.78° (좌우 85.0%, S=+25.25°)"),
        stop=(89, "11/16 근접 바구니 90% · 의자 88% (Aux cx MAE 0.007)"),
        start=(100, "11/13 의자·책상 중앙 출발 100%"),
        lang="디코더 미사용 (ENC-E2E-Aux)",
        diag="인코더 위 보조 BBox 헤드로 generate() 없이 빠르지만 VLM 디코더 사전학습 지식을 쓰지 않는 대조군입니다.",
        studio_key=None,
    ),
    "EG3-08": dict(
        flow="[이미지 + 지시어] ➔ [Florence-2 Enc + BART Decoder(6L) 1-Pass Hidden] ➔ [PiZeroFM (BBox=0)]",
        steer=(75, "A3 대비 7.70° (좌우 74.7%, B1 대비 3.3배 개선)"),
        stop=(59, "바구니≥0.25 100% vs 미학습 의자≥0.25 19%"),
        start=(100, "11/13 의자·책상·화분 중앙 출발 100%"),
        lang="VLM Decoder 6L 직접 연결 (π0형 순수 E2E)",
        diag="외부 BBox를 완전히 0으로 차단한 순수 VLM E2E. 바구니 정지는 100%이나 기하 병목이 없어 미학습 의자 근접 정지(19%)가 한계로 드러납니다.",
        studio_key=None,
    ),
    "EG3-09": dict(
        flow="[이미지 + 지시어] ➔ [Decoder <loc> Softmax 기댓값(미분가능) + Hidden] ➔ [Area-Grounded PiZeroFM]",
        steer=(83, "A3 대비 6.00° (좌우 83.4%, 문자열 파싱 0%)"),
        stop=(96, "11/16 바구니≥0.25 100% · 미학습 의자≥0.25 93%"),
        start=(100, "11/13 의자·책상·화분 중앙 출발 100%"),
        lang="VLM Decoder Open-Vocab 98.8% 보존 (283.8ms, 2.08x)",
        diag="문자열 파싱 없이 Decoder <loc> 분포의 미분가능 Softmax 기댓값을 통과시켜, 미학습 의자 근접 정지를 19%→93%(+74%p)로 해결한 VLM-E2E 1순위입니다.",
        studio_key=None,
    ),
}


def card_decision_html(pn: str) -> str:
    """각 모델 카드 안에 들어갈 '🧠 의사결정 구조 & 판단 요인 프로파일' HTML 반환."""
    p = DECISION_PROFILES.get(pn)
    if not p:
        return ""
    esc = html.escape

    def bar_row(label: str, pct: int, desc: str, col: str) -> str:
        return (
            f'<div class="df-row">'
            f'<span class="df-lbl">{esc(label)}</span>'
            f'<div class="df-bar"><i style="width:{max(3, min(100, pct))}%;background:{col}"></i></div>'
            f'<span class="df-val">{pct}% · {esc(desc)}</span>'
            f'</div>'
        )

    st_pct, st_desc = p["steer"]
    sp_pct, sp_desc = p["stop"]
    go_pct, go_desc = p["start"]
    c_st = "#22c55e" if st_pct >= 80 else ("#f59e0b" if st_pct >= 50 else "#ef4444")
    c_sp = "#22c55e" if sp_pct >= 60 else ("#f59e0b" if sp_pct >= 30 else "#ef4444")
    c_go = "#22c55e" if go_pct >= 80 else ("#f59e0b" if go_pct >= 40 else "#ef4444")

    btn = ""
    if p.get("studio_key"):
        sk = p["studio_key"]
        btn = f'<button type="button" class="df-jump" onclick="window.jumpToStudio(\'{sk}\')">🔍 실제 세션 사진에서 <b>{esc(sk)}</b> 판단 근거 해부 보기 ↑</button>'

    return (
        f'<div class="df-box">'
        f'<div class="df-title">🧠 의사결정 메커니즘 & 판단 근거 (Decision Factors)</div>'
        f'<div class="df-flow">{esc(p["flow"])}</div>'
        f'{bar_row("🎯 조향 근거 (cx)", st_pct, st_desc, c_st)}'
        f'{bar_row("🛑 정지 근거 (면적)", sp_pct, sp_desc, c_sp)}'
        f'{bar_row("🚀 낯선물체 출발", go_pct, go_desc, c_go)}'
        f'<div class="df-lang">🔤 <b>언어 역할:</b> {esc(p["lang"])}</div>'
        f'<div class="df-diag">💡 {esc(p["diag"])}</div>'
        f'{btn}'
        f'</div>'
    )


def studio_css() -> str:
    return """
.studio-wrap{border:2px solid var(--acc);border-radius:14px;background:var(--card);padding:16px 18px;margin:1.2rem 0;box-shadow:0 4px 20px rgba(0,0,0,.08)}
.studio-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:8px;border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:12px}
.studio-head h2{margin:0;font-size:1.28rem;color:var(--acc)}
.cog-flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:10px 0 14px}
.cog-step{border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:var(--bg);position:relative}
.cog-step .st-num{font-size:.72rem;font-weight:800;color:var(--acc);text-transform:uppercase;letter-spacing:.04em}
.cog-step .st-ttl{font-weight:700;font-size:.94rem;margin:2px 0 4px}
.cog-step .st-body{font-size:.8rem;color:var(--mut);line-height:1.5}
.sc-tabs{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
.sc-tab{border:1px solid var(--line);background:var(--bg);color:var(--fg);padding:6px 12px;border-radius:8px;font-size:.82rem;font-weight:600;cursor:pointer;transition:.15s}
.sc-tab[aria-selected="true"]{background:var(--acc);color:var(--bg);border-color:var(--acc)}
.sc-grid{display:grid;grid-template-columns:440px 1fr;gap:16px;align-items:start;margin-top:10px}
@media(max-width:980px){.sc-grid{grid-template-columns:1fr}}
.sc-photo-card{border:1px solid var(--line);border-radius:12px;background:#0b101b;color:#e2e8f0;padding:10px;overflow:hidden}
.sc-canvas{position:relative;width:100%;aspect-ratio:16/9;border-radius:8px;overflow:hidden;background:#000}
.sc-canvas img{width:100%;height:100%;object-fit:cover;display:block}
.sc-canvas svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.sc-meta{font-size:.78rem;color:#94a3b8;margin-top:8px;line-height:1.55}
.sc-meta b{color:#f8fafc}
.m-toggles{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.m-tog{border:1px solid #334155;background:#162032;color:#e2e8f0;padding:3px 8px;border-radius:6px;font-size:.74rem;cursor:pointer;display:flex;align-items:center;gap:5px}
.m-tog.off{opacity:.35}
.m-dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.sc-panel{display:flex;flex-direction:column;gap:10px}
.sc-qbox{border-left:4px solid var(--acc);background:var(--bg);padding:8px 12px;border-radius:0 8px 8px 0;font-size:.84rem}
.sc-qbox .q{font-weight:700;color:var(--fg);margin-bottom:4px}
.sc-qbox .a{color:var(--mut);line-height:1.6}
.ft-table{width:100%;border-collapse:collapse;font-size:.8rem;background:var(--bg);border-radius:8px;overflow:hidden}
.ft-table th,.ft-table td{border:1px solid var(--line);padding:6px 8px;vertical-align:middle}
.ft-table th{background:var(--head);font-size:.75rem;color:var(--mut)}
.ft-table tr.hl-row{outline:2px solid var(--acc);background:var(--bluebg)}
.fbar-wrap{display:flex;align-items:center;gap:6px;min-width:110px}
.fbar{flex:1;height:8px;background:var(--graybg);border-radius:4px;overflow:hidden;position:relative}
.fbar i{display:block;height:100%;border-radius:4px}
.pill-go{display:inline-block;padding:1px 7px;border-radius:99px;font-size:.72rem;font-weight:700;background:var(--okbg);color:var(--ok)}
.pill-stop{display:inline-block;padding:1px 7px;border-radius:99px;font-size:.72rem;font-weight:700;background:var(--badbg);color:var(--bad)}
.df-box{border:1px solid var(--line);border-radius:9px;padding:8px 10px;background:var(--bg);font-size:.78rem;display:flex;flex-direction:column;gap:5px}
.df-title{font-weight:700;font-size:.76rem;color:var(--acc);letter-spacing:.02em}
.df-flow{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.73rem;background:var(--head);padding:4px 7px;border-radius:5px;color:var(--fg)}
.df-row{display:grid;grid-template-columns:6.2rem 70px 1fr;gap:6px;align-items:center;font-size:.74rem}
.df-lbl{color:var(--mut);font-weight:600}
.df-bar{height:7px;background:var(--graybg);border-radius:4px;overflow:hidden}
.df-bar i{display:block;height:100%}
.df-val{color:var(--fg);font-size:.72rem}
.df-lang{font-size:.74rem;color:var(--mut)}
.df-diag{font-size:.75rem;color:var(--fg);line-height:1.5;border-top:1px dashed var(--line);padding-top:4px}
.df-jump{margin-top:2px;border:1px solid var(--acc);background:var(--bluebg);color:var(--acc);border-radius:6px;padding:4px 8px;font-size:.74rem;font-weight:600;cursor:pointer;text-align:center}
.df-jump:hover{filter:brightness(1.08)}
"""


def studio_html() -> str:
    if not ANATOMY_JSON.exists():
        return ""
    data = json.loads(ANATOMY_JSON.read_text(encoding="utf-8"))
    data_js = json.dumps(data, ensure_ascii=False)
    return f"""
<section class="studio-wrap" id="decision-studio">
  <div class="studio-head">
    <div>
      <h2>🧠 모델 의사결정 과정 &amp; 실제 세션 판단 근거(Factor) 해부 스튜디오</h2>
      <p class="d" style="margin:2px 0 0">물리적 레이어(MLP·AdaLN)가 아니라, <b>실제 로봇 카메라 사진</b>에서 모델이 <b>무엇(BBox 위치·박스 면적·시각 외형 지름길·언어 게이트)을 근거로 출발·조향·정지를 결정하는지</b> 반사실(Counterfactual) 실험으로 분해한 뷰어입니다.</p>
    </div>
    <div class="m">임계값 기준: 명령 크기 <code>||cmd|| &lt; 0.20</code> = 정지(idle) · 컨트롤러 근접 정지 <code>area ≥ 0.25</code> (2프레임)</div>
  </div>

  <div class="cog-flow">
    <div class="cog-step">
      <div class="st-num">Stage 1 · 지각 (Perception)</div>
      <div class="st-ttl">🎯 지시어 그라운딩 &amp; 3프레임 창</div>
      <div class="st-body">Florence-2가 지시어 물체의 <code>(cx, cy, area, has_bbox)</code> 3프레임 시계열(12-d)과 전역 이미지 특징(768-d)을 추출합니다.</div>
    </div>
    <div class="cog-step">
      <div class="st-num">Stage 2 · 조향 근거 (Steering Factor)</div>
      <div class="st-ttl">🧭 기하 <code>BBox cx</code> vs 배경 기억</div>
      <div class="st-body"><b>서보 라벨(λ=1.0)</b> 적용 모델(<code>a3</code>, <code>a3-s2</code>, <code>a3-ag23</code>)은 박스 <code>cx</code> 반전 시 조향각이 43~55° 즉시 반전되나, <code>A1(no-servo)</code>은 배경만 외워 1.2°만 변합니다.</div>
    </div>
    <div class="cog-step">
      <div class="st-num">Stage 3 · 출발·정지 근거 (Speed Factor)</div>
      <div class="st-ttl">🛑 기하 <code>BBox area</code> vs 시각 외형 지름길</div>
      <div class="st-body">기존 <code>a3</code>·<code>a3-s2</code>는 면적 변화(0.05→0.35)에 무반응(+3%p)이고 <b>768-d 시각 외형</b>으로 멈춥니다(의자 중앙 비출발·바구니 충돌 원인). 반면 <b><code>a3-ag23</code>은 면적에 100% 반응</b>해 멈춥니다.</div>
    </div>
    <div class="cog-step">
      <div class="st-num">Stage 4 · 제어 게이트 (Controller Gate)</div>
      <div class="st-ttl">⚙️ <code>idle</code> 감속 &amp; <code>near_area</code> 래치</div>
      <div class="st-body">모델이 <code>&lt;0.20</code>을 내면 즉시 <code>idle(속도 0)</code>로 관성을 끊고, <code>area ≥ 0.25</code>가 2프레임 연속이면 <code>STOPPED(near_goal)</code>로 세션을 종료합니다.</div>
    </div>
  </div>

  <div class="sc-tabs" id="sc-tabs" role="tablist"></div>

  <div class="sc-grid">
    <div class="sc-photo-card">
      <div class="sc-canvas">
        <img id="sc-img" src="" alt="실기 세션 프레임" loading="lazy" decoding="async">
        <svg id="sc-svg" viewBox="0 0 416 234"></svg>
      </div>
      <div class="m-toggles" id="sc-toggles"></div>
      <div class="sc-meta" id="sc-meta"></div>
    </div>

    <div class="sc-panel">
      <div class="sc-qbox" id="sc-qbox"></div>
      <div style="overflow-x:auto">
        <table class="ft-table" id="sc-table">
          <thead>
            <tr>
              <th>모델 (레지스트리 PN)</th>
              <th>이 프레임 최종 판단<br><span class="m">크기 ||cmd|| (방향각)</span></th>
              <th>요인 ① 기하 BBox 기본 출력<br><span class="m">중립 배경(vis_ref) 시 크기</span></th>
              <th>요인 ② 시각 외형 개입량 (Δvis)<br><span class="m">현재 이미지가 속도를 깎았는가?</span></th>
              <th>요인 ③ 면적 감속 민감도<br><span class="m">area 0.05→0.35 시 감속량</span></th>
              <th>요인 ④ cx 반전 조향 반응<br><span class="m">박스 좌우 반전 시 Δθ</span></th>
              <th>상황별 판단 근거 진단</th>
            </tr>
          </thead>
          <tbody id="sc-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</section>
<script>
(function(){{
  var SC = {data_js};
  var curSc = 0;
  var hlModel = null;
  var visModels = {{"a3":true, "a3-s2":true, "a3-ag23":true, "a3-drop10-s1":true, "a3-no-servo":false, "a3-servo0.5":false, "a3-featswap30":false}};
  var ORDER = ["a3", "a3-s2", "a3-ag23", "a3-drop10-s1", "a3-servo0.5", "a3-no-servo", "a3-featswap30"];

  function diagnose(scId, mKey, m) {{
    if (scId === "sc1_chair_center" || scId === "sc4_desk_center") {{
      if (m.stop) return "🔴 <b>시각 외형 억제(" + (m.vis_effect>=0?"+":"") + m.vis_effect.toFixed(2) + ")</b>: 기하 BBox는 전진(" + m.bbox_geom_mag.toFixed(2) + ")을 원하나 중앙의 낯선 물체 외형을 '도착'으로 오인해 멈춤";
      if (m.area_drop_05_to_35 >= 0.4) return "🟢 <b>기하 BBox 주도 출발</b>: 시각 외형 간섭(" + m.vis_effect.toFixed(2) + ") 없이 원거리(area<0.18) 기하 조건으로 힘차게 출발";
      return "🟢 <b>출발 성공</b>: 시각 억제가 약해(" + m.vis_effect.toFixed(2) + ") 정지 문턱(0.20)을 넘어 출발함";
    }}
    if (scId === "sc2_basket_close") {{
      if (!m.stop) return "🔴 <b>근접 정지 실패(" + m.mag.toFixed(2) + ")</b>: 면적(0.36)이 커져도 기하 감속이 약해 계속 전진 ➔ <b>실기 바구니 5cm 밀음 원인</b>";
      if (m.area_drop_05_to_35 >= 0.4) return "🟢 <b>기하 면적(area=0.36≥0.23) 정지</b>: 이미지 외형과 무관하게 박스 면적만으로 100% 즉시 정지(0.01)";
      return "🟡 <b>시각 외형 의존 정지</b>: 기하 면적 반응은 약하나(+" + m.area_drop_05_to_35.toFixed(2) + ") 근접 바구니 시각 특징(" + m.vis_effect.toFixed(2) + ") 덕에 멈춤";
    }}
    if (scId === "sc3_chair_close") {{
      if (!m.stop) return "🔴 <b>낯선 물체 정지 실패(" + m.mag.toFixed(2) + ")</b>: 바구니 외형이 아니라서 시각 정지가 안 걸리고 의자 코앞(area=0.36)에서 전진 지속";
      if (m.area_drop_05_to_35 >= 0.4) return "🟢 <b>물체 불문 기하 면적 정지</b>: 처음 보는 의자 앞에서도 area=0.36 기하 요인으로 정확히 정지(0.02)";
      return "🟢 <b>근접 정지 성공</b>: 낮은 박스 드롭아웃(10%) 덕분에 의자 근접 시에도 안정적으로 감속·정지";
    }}
    if (scId === "sc5_side_steer") {{
      if (m.mirror_ang_diff < 10) return "🔴 <b>박스 위치 무시(Δθ=" + m.mirror_ang_diff.toFixed(1) + "°)</b>: 박스를 반대로 옮겨도 배경 이미지 기억에 끌려감";
      return "🟢 <b>BBox cx 기하 조향(Δθ=" + m.mirror_ang_diff.toFixed(1) + "°)</b>: 타깃 cx=0.28(좌측)을 따라 정확히 좌측 조향";
    }}
    if (m.vis_effect < -0.10) return "🟡 <b>원거리 시각 주춤(" + m.vis_effect.toFixed(2) + ")</b>: 낯선 시각 특징 때문에 정지 경계(0.20) 근처까지 속도 급감";
    return "🟢 <b>일정 속도 전진</b>: 소형 박스(area=0.008)에서도 시각 흔들림 없이 안정적 전진";
  }}

  function render() {{
    var sc = SC[curSc];
    // Tabs
    var tabsHtml = SC.map(function(s, idx) {{
      return '<button type="button" class="sc-tab" role="tab" aria-selected="' + (idx===curSc) + '" data-i="' + idx + '">' + s.tab + '</button>';
    }}).join('');
    document.getElementById('sc-tabs').innerHTML = tabsHtml;
    document.querySelectorAll('#sc-tabs .sc-tab').forEach(function(b) {{
      b.onclick = function() {{ curSc = parseInt(b.dataset.i, 10); render(); }};
    }});

    // Photo & SVG overlay (416x234)
    document.getElementById('sc-img').src = sc.img_src || sc.img_b64;
    var W = 416, H = 234;
    var b = sc.bbox; // [cx, cy, area, has]
    var svg = '';
    if (b[3] >= 0.5) {{
      var side = Math.sqrt(b[2] * W * H);
      var x0 = b[0] * W - side / 2, y0 = b[1] * H - side / 2;
      svg += '<rect x="' + x0.toFixed(1) + '" y="' + y0.toFixed(1) + '" width="' + side.toFixed(1) + '" height="' + side.toFixed(1) + '" fill="rgba(250,204,21,0.10)" stroke="#facc15" stroke-width="2.2" stroke-dasharray="5 3"/>';
      svg += '<rect x="' + Math.max(2, x0).toFixed(1) + '" y="' + Math.max(14, y0 - 16).toFixed(1) + '" width="138" height="15" rx="3" fill="rgba(0,0,0,0.78)"/>';
      svg += '<text x="' + (Math.max(2, x0) + 4).toFixed(1) + '" y="' + (Math.max(14, y0 - 16) + 11).toFixed(1) + '" fill="#fde047" font-size="10" font-weight="700">BBox cx=' + b[0].toFixed(2) + ' · area=' + b[2].toFixed(3) + '</text>';
    }}
    var ox = W / 2, oy = H - 14;
    svg += '<circle cx="' + ox + '" cy="' + oy + '" r="4.5" fill="#fff" stroke="#000" stroke-width="1.5"/>';

    var stopLabels = [];
    ORDER.forEach(function(k) {{
      if (!visModels[k] || !sc.models[k]) return;
      var m = sc.models[k];
      var col = m.col;
      if (m.stop) {{
        stopLabels.push('<tspan fill="' + col + '" font-weight="700">■ ' + k + ' 정지(' + m.mag.toFixed(2) + ')</tspan>');
      }} else {{
        var len = Math.min(m.mag, 1.0) * 95;
        var ux = m.cmd[1] / Math.max(m.mag, 1e-6);
        var uy = -m.cmd[0] / Math.max(m.mag, 1e-6);
        var x2 = ox + ux * len, y2 = oy + uy * len;
        var px = -uy, py = ux;
        var w = (hlModel === k) ? 5.5 : 3.6;
        svg += '<line x1="' + ox + '" y1="' + oy + '" x2="' + x2.toFixed(1) + '" y2="' + y2.toFixed(1) + '" stroke="#000" stroke-width="' + (w+2) + '" stroke-linecap="round" opacity="0.55"/>';
        svg += '<line x1="' + ox + '" y1="' + oy + '" x2="' + x2.toFixed(1) + '" y2="' + y2.toFixed(1) + '" stroke="' + col + '" stroke-width="' + w + '" stroke-linecap="round"/>';
        svg += '<polygon points="' + (x2+ux*8).toFixed(1) + ',' + (y2+uy*8).toFixed(1) + ' ' + (x2+px*5.5).toFixed(1) + ',' + (y2+py*5.5).toFixed(1) + ' ' + (x2-px*5.5).toFixed(1) + ',' + (y2-py*5.5).toFixed(1) + '" fill="' + col + '" stroke="#000" stroke-width="0.8"/>';
        svg += '<text x="' + (x2 + ux*14).toFixed(1) + '" y="' + (y2 + uy*10).toFixed(1) + '" text-anchor="middle" fill="' + col + '" stroke="#000" stroke-width="2.5" paint-order="stroke" font-size="10.5" font-weight="800">' + k + ' (' + m.mag.toFixed(2) + ')</text>';
      }}
    }});
    if (stopLabels.length) {{
      svg += '<rect x="6" y="' + (H - 26) + '" width="' + (W - 12) + '" height="20" rx="4" fill="rgba(0,0,0,0.75)"/>';
      svg += '<text x="12" y="' + (H - 12) + '" font-size="10.5">' + stopLabels.join('  ·  ') + '</text>';
    }}
    document.getElementById('sc-svg').innerHTML = svg;

    // Model toggle chips
    var togHtml = ORDER.map(function(k) {{
      var m = sc.models[k];
      return '<button type="button" class="m-tog ' + (visModels[k]?'':'off') + '" data-k="' + k + '"><span class="m-dot" style="background:' + m.col + '"></span>' + k + '</button>';
    }}).join('');
    document.getElementById('sc-toggles').innerHTML = togHtml;
    document.querySelectorAll('#sc-toggles .m-tog').forEach(function(btn) {{
      btn.onclick = function() {{
        var k = btn.dataset.k;
        visModels[k] = !visModels[k];
        render();
      }};
    }});

    document.getElementById('sc-meta').innerHTML =
      '<b>세션:</b> <code>' + sc.sid + '</code> (프레임 ' + (sc.fi+1) + '/' + sc.frame_total + ') · <b>물체·위치:</b> ' + sc.obj + ' · ' + sc.pos +
      '<br><b>지시어:</b> <code>"' + sc.ins + '"</code> · <b>실기 결과:</b> ' + sc.real_outcome;

    document.getElementById('sc-qbox').innerHTML =
      '<div class="q">❓ ' + sc.title + '</div>' +
      '<div class="a"><b>핵심 질문:</b> ' + sc.question + '<br><b>요인 분해 결론:</b> ' + sc.key_finding + '</div>';

    // Factor Breakdown Table
    var rowsHtml = ORDER.map(function(k) {{
      var m = sc.models[k];
      var isHl = (hlModel === k);
      var badge = m.stop ? '<span class="pill-stop">정지 ' + m.mag.toFixed(2) + '</span>' : '<span class="pill-go">이동 ' + m.mag.toFixed(2) + '</span>';
      var geomPct = Math.min(100, Math.round(m.bbox_geom_mag * 100));
      var visSign = m.vis_effect >= 0 ? '+' : '';
      var visCol = m.vis_effect < -0.15 ? '#ef4444' : (m.vis_effect > 0.15 ? '#f59e0b' : '#64748b');
      var visPct = Math.min(100, Math.round(Math.abs(m.vis_effect) * 120));
      var areaPct = Math.min(100, Math.max(0, Math.round(m.area_drop_05_to_35 * 125)));
      var areaCol = m.area_drop_05_to_35 >= 0.4 ? '#22c55e' : (m.area_drop_05_to_35 >= 0.15 ? '#f59e0b' : '#ef4444');
      var cxPct = Math.min(100, Math.round(m.mirror_ang_diff / 55 * 100));
      var cxCol = m.mirror_ang_diff >= 20 ? '#22c55e' : '#ef4444';

      return '<tr class="' + (isHl ? 'hl-row' : '') + '">' +
        '<td><span class="m-dot" style="background:' + m.col + ';margin-right:5px"></span><b>' + k + '</b><br><a href="#' + m.pn + '" class="m">' + m.pn + '</a></td>' +
        '<td>' + badge + '<br><span class="m">lx=' + m.cmd[0].toFixed(2) + ', ly=' + m.cmd[1].toFixed(2) + ' (' + m.ang_deg.toFixed(0) + '°)</span></td>' +
        '<td><div class="fbar-wrap"><div class="fbar"><i style="width:' + geomPct + '%;background:#38bdf8"></i></div><b>' + m.bbox_geom_mag.toFixed(2) + '</b></div></td>' +
        '<td><div class="fbar-wrap"><div class="fbar"><i style="width:' + visPct + '%;background:' + visCol + '"></i></div><b style="color:' + visCol + '">' + visSign + m.vis_effect.toFixed(2) + '</b></div></td>' +
        '<td><div class="fbar-wrap"><div class="fbar"><i style="width:' + areaPct + '%;background:' + areaCol + '"></i></div><b>' + (m.area_drop_05_to_35>=0?'+':'') + m.area_drop_05_to_35.toFixed(2) + '</b></div></td>' +
        '<td><div class="fbar-wrap"><div class="fbar"><i style="width:' + cxPct + '%;background:' + cxCol + '"></i></div><b>' + m.mirror_ang_diff.toFixed(1) + '°</b></div></td>' +
        '<td>' + diagnose(sc.id, k, m) + '</td>' +
      '</tr>';
    }}).join('');
    document.getElementById('sc-tbody').innerHTML = rowsHtml;
  }}

  window.jumpToStudio = function(modelKey) {{
    hlModel = modelKey;
    visModels[modelKey] = true;
    render();
    var el = document.getElementById('decision-studio');
    if (el) el.scrollIntoView({{behavior: 'smooth', block: 'start'}});
  }};

  render();
}})();
</script>
"""

