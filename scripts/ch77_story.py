"""CH77 '오프라인 지표 vs 실기 성능' 분석 섹션(구조도·플로우·읽는 법). gen_ch77.py가 호출."""
import json
import math
from pathlib import Path

import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT


def _load(tag):
    f = FM / f"summary_s1_{tag}.json"
    return json.load(open(f)) if f.exists() else None


def _best():
    for tag in ("full300", "full299"):
        d = _load(tag)
        if d:
            lo = _load(tag + "_lopo")
            return d, lo, d["episodes"]
    return None, None, 0


def block(tbl, H, st):
    d, lo, n = _best()
    ang = d["aggregate"]["C_pi0_adaln"]["angle_deg"] if d else float("nan")
    ang_lo = lo["aggregate"]["C_pi0_adaln"]["angle_deg"] if lo else float("nan")
    lat = lambda a, L=100: L * math.sin(math.radians(a))
    arch = V.flow([
        ("카메라 프레임", MUT, "BGR→RGB"), ("Florence-2 한 번", B, "bbox + 768-d 특징 (817ms Jetson fp32)"),
        ("FM 액션 헤드", P, "방향·크기 (lx, ly) 청크 H=8"), ("속력 규칙·섀도", A, "m→throttle, 첫 적용 s≤0.7, az=0"), ("모터 명령", G, "실제 이동 → 다음 프레임")])
    split = (
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin:8px 0">'
        f'<div style="flex:1 1 300px;border:2px solid {G};border-radius:10px;padding:10px 12px;background:{G}12"><b style="color:{G}">✅ 오프라인(이미 측정)</b>'
        '<div style="font-size:0.78rem;margin-top:4px">기록된 300개 에피소드에서 <b>"사람이 이 장면에서 어떻게 조작했나"를 모델이 맞히는지</b> 본다.<br>'
        '• 방향 오차·R²(축별)·GO/STOP F1<br>• 5-fold: 같은 장소의 안 본 에피소드<br>• LOPO: 안 본 <b>위치</b>까지 일반화</div></div>'
        f'<div style="flex:1 1 300px;border:2px solid {R};border-radius:10px;padding:10px 12px;background:{R}12"><b style="color:{R}">🤖 실기(로봇에서만 앎)</b>'
        '<div style="font-size:0.78rem;margin-top:4px">모델 출력이 <b>실제로 바구니에 도달하게 하는지</b>는 오프라인으로 알 수 없다.<br>'
        '• 한 번 틀린 방향이 다음 프레임에서 바뀐 장면으로 누적/보정되는지<br>• 지연(젯슨 817ms)·속력·미끄럼<br>• 학습 때 없던 상태(분포 밖)로 들어갔을 때 복구</div></div></div>')
    why = tbl(["왜 오프라인 ≠ 실기인가", "내용", "이번 프로젝트의 근거"], [
        ["정답이 사람 조작", "R²·각도는 '사람과 얼마나 같은가'다. 사람과 달라도 바구니에 도달할 수 있고, 같아도 실패할 수 있다", "FM≈MLP 회귀(MSE 차 ≤0.003)인데 실기는 다를 수 있음"],
        ["오차 누적 vs 보정", "폐루프는 매 프레임 다시 본다. 방향이 조금 틀려도 보정되지만, 크게 틀린 자리에서 복구를 학습한 적이 없을 수도 있다", "V5 Exp11: val 좋아도 폐루프 0% (FPE 1.45m 누적)"],
        ["지연", "추론 817ms 동안 로봇은 옛 명령으로 계속 움직임(26cm/s 기준 약 21cm)", "서버 180ms vs 젯슨 817ms"],
        ["학습 분포 밖", "수집은 사람이 주행한 경로뿐이라, 모델이 만든 경로(벗어난 상태)는 학습 데이터에 거의 없다", "LOPO에서 방향 오차 증가가 그 신호"]])
    num = tbl(["오프라인 수치", "값", "실기 감각으로 번역(1m 주행 시 옆으로 벗어나는 양 = 1m×sinθ)"], [
        ["5-fold 방향 오차 (C)", f"{ang:.1f}° ({n}ep)", f"약 {lat(ang):.0f}cm — 같은 장소에서 처음 보는 에피소드일 때"],
        ["LOPO 방향 오차 (C)", f"{ang_lo:.1f}°" if lo else "대기", f"약 {lat(ang_lo):.0f}cm — 처음 보는 위치일 때(더 현실적인 하한)" if lo else "—"],
        ["지연 중 이동", "817ms", f"약 {0.817 * 26.4:.0f}cm (최대 속력 26cm/s) — 판단 전에 이미 움직인 거리"]])
    chk = tbl(["로봇에서 볼 것", "방법", "합격 판정(제안)", "순서"], [
        ["서버 경로 무결성", "같은 이미지를 서버 `action_cont`와 캐시 특징으로 각각 돌려 비교(BGR/RGB 포함)", "bbox·특징 차 ≈ 0", "1"],
        ["지연", "젯슨·서버 fp32, FM 헤드 포함 측정", "추론 후 이동 거리 허용 범위", "2"],
        ["드라이런", "로봇 정지, 출력만 로그 → 사람 조작과 겹쳐 보기", "방향이 수집 때와 비슷한지", "3"],
        ["폐루프 300회", "15칸(5위치×3경로)×20회, 첫 적용 섀도·az=0·s≤0.7", "EGVLA1 실기 92%(300건 기준)와 같은 성공 정의로 비교", "4"]])
    return (H("⑥ 오프라인 지표와 실기 성능 — 무엇이 증명됐고 무엇은 로봇에서만 아나")
            + '<div style="font-size:0.82rem;line-height:1.6;margin:6px 0">지금까지의 R²·방향 오차는 <b>기록된 데이터에서 "사람 조작을 얼마나 따라 하나"</b>를 잰 값이다. '
              '실제로 <b>로봇이 바구니까지 가는지</b>는 폐루프에서만 알 수 있다. 아래는 이 둘이 파이프라인의 어디에 해당하는지 보여 준다.</div>'
            + V.wrap("추론 파이프라인 (한 프레임이 모터 명령이 되기까지)", arch, "오프라인 지표는 가운데 '액션 헤드'까지만 본다. 속력 규칙·모터·다음 프레임으로 이어지는 되먹임은 실기에서만 확인된다", kind="flow")
            + split + why + H("오프라인 수치를 실기 감각으로 번역")+ num
            + f'{st("주의", R)} 위 번역은 <b>열린 고리(보정 없음)</b> 상한이다. 실제 폐루프는 매 프레임 다시 보고 고치므로 이보다 작을 수 있지만, 복구 능력은 오프라인 지표로 알 수 없다. 성공 기준(허용 오차)도 기존 프로토콜 정의를 따른다.'
            + H("실기에서 확인할 순서") + chk
            + '<div style="font-size:0.8rem;margin:8px 0">현재 위치: ✅ 오프라인 완료 → <b>다음은 1번(서버 경로 무결성)</b>. 젯슨 접근이 필요하며, 이 결과가 나오기 전에는 폐루프 성능을 주장하지 않는다.</div>')
