"""CH77(학습 분야) 개요: 3줄 요약, 학습 파이프라인 흐름, 핵심 수치 카드, 학습 관련 결정 기록."""
import json

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT


def card(big, label, col):
    return (f'<div style="flex:1;min-width:150px;background:#0d1420;border:1px solid {col};border-radius:10px;padding:12px;text-align:center">'
            f'<div style="font-size:1.5rem;font-weight:900;color:{col}">{big}</div><div style="font-size:0.74rem;color:#94a3b8;margin-top:4px;line-height:1.4">{label}</div></div>')


def block(tbl, H, st):
    s = json.load(open(FM / "summary_s1_full300.json"))["aggregate"]; l = json.load(open(FM / "summary_s1_full300_lopo.json"))["aggregate"]
    hc = json.load(open(FM / "human_ceiling.json"))["results"]["같은 셀(목표 위치+경로)"]; sw = json.load(open(FM / "smoothing_sweep.json"))
    c5, cl = s["C_pi0_adaln"]["angle_deg"], l["C_pi0_adaln"]["angle_deg"]
    call = ('<div class="callout critical">🟡 <strong>CH77 = 학습 분야 정리 (300건 수집 → 오프라인 평가 → 배포 체크포인트)</strong><br>'
            f'① <strong>연속 조이스틱 300건(15칸×20)으로 방향이 사람 변동성 한계 수준까지 학습된다</strong>: 같은 목표 위치 방향 오차 {c5:.1f}° ≈ 사람끼리 {hc["pairwise_mean"]:.1f}°·사람 합의 {hc["consensus_mean"]:.1f}°.<br>'
            f'② <strong>줄일 수 있는 오차는 일반화에 있다</strong>: 안 본 목표 위치 {cl:.1f}°(+{cl - c5:.1f}°). 헤드·서브골·epoch·데이터량의 효과(각 1° 안팎)보다 위치 일반화 차이(6° 이상)가 훨씬 크다.<br>'
            '③ <strong>FM 헤드는 MLP 회귀와 정확도가 같고(차 ≤0.003) 서브골은 정확도를 거의 안 바꾼다(≤0.4°)</strong>. 대신 드롭아웃 서브골(FM-π0)이 결측에 견고하고 K=8+EMA 평활이 흔들림을 사람 수준으로 낮춘다. <b>시작 구간은 경로 모호성으로 다봉(K 평균이 모드를 뭉갬)</b>이고 지연 보상·좌우 반전 증강은 기각했다. 서빙·실기 준비는 CH78, 분석·시사점은 CH79.</div>')
    flow = V.flow([("수집 300건", G, "15칸×20, 검증 통과"), ("학습 세트", B, "299→300, 라벨 없음 28건 reserve"), ("S1 5-fold×5seed", B, "헤드 4종 비교"), ("LOPO", A, "안 본 목표 위치"),
                   ("서브골 ablation", P, "30분할·곡선"), ("평활·천장", P, "K×EMA, 사람 일치도"), ("배포 ckpt", G, "FM-π0+드롭아웃30%")])
    cards = ('<div style="display:flex;gap:10px;flex-wrap:wrap;margin:10px 0">'
             + card(f"{c5:.1f}°", "같은 목표 위치 방향 오차(FM-π0, 5-fold)", G) + card(f"{cl:.1f}°", "안 본 목표 위치(FM-π0, LOPO)", R)
             + card(f'{hc["consensus_mean"]:.1f}~{hc["pairwise_mean"]:.1f}°', "사람 조작 일치도 천장", MUT) + card("≤0.003", "FM vs MLP 회귀 MSE 차", B)
             + card("≤0.4°", "서브골 유무 효과(3 seed)", A) + card(f'{sw["fold0"]["K8|a0.5"]["jitter"]:.3f}', "K=8+EMA 0.5 저크(사람 0.045~0.063)", G) + '</div>')
    dec = tbl(["결정", "내용", "근거"], [
        ["학습 세트", "15칸(5 목표 위치×3경로)×20=300건, 라벨 없는 10/13 파일럿 28건은 reserve(학습 제외)", "plan §11, CH77 ⓪-A"],
        ["az(회전)", "학습하지 않고 규칙으로 둠(첫 적용 az=0)", "az R² 약 0.1, 마스킹해도 같음"],
        ["epoch·조기 종료", "250 epoch + inner-val 조기 종료로 비교, 배포는 120 epoch(FM-π0)", "곡선: FM-π0 최적 약 80~160, LOPO에서는 일찍 과적합(⑥-6)"],
        ["헤드", "FM-π0(π0형) 권장, 회귀 헤드를 비교군으로 함께 실기 평가", "안 본 위치 최우수(16.9° vs FM-MLP 20.3°), 서브골 결측에 강함"],
        ["서브골", "bbox 서브골은 유지하되 학습 때 30% 드롭아웃", "정확도 효과 ≤0.4°, 드롭아웃은 손실 없이 결측 견고(FM-π0 10.1°·16.8°)"],
        ["평활", "K=8, EMA 새 값 가중 0.5로 시작(드라이런으로 확정)", "저크 0.056≈사람 수준, 지연 ≤2프레임(⑥-9)"],
        ["평가 구성", "오프라인은 5-fold·LOPO, 실기는 3 시작 위치×5 목표 위치×20 (CH78)", "EGVLA1과 같은 조건 비교"],
        ["지연 보상", "하지 않음(청크 첫 스텝 사용)", "뒤 스텝이 더 나쁨(⑥-13)"],
        ["좌우 반전 증강", "채택하지 않음", "안 본 위치 +0.6°, 10쌍 중 7쌍 악화(⑥-15)"],
        ["시작 구간 서빙", "미정(P1): K 평균 / 모드 선택 / 경로 조건", "시작 구간 다봉, 장애물 배치 확인 필요"]])
    return (call + H("학습 분야 흐름") + V.wrap("수집에서 배포 체크포인트까지", flow, "각 단계의 결과는 탭별로 정리, 서빙·실기는 CH78", kind="flow")
            + H("핵심 수치 한눈에") + cards + H("③ 학습 관련 결정 기록") + dec)
