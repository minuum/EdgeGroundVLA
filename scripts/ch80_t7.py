"""CH80 ▶ T7 탭: 실기 세션 재생 — A3 체크포인트 vs 저장 학습한 V0·V4 (3시드). 원자료 docs/v5/egvla2_fm/t7_replay{,_s1,_s2}.json (생성: scripts/t7_compare.py)"""
import json
from pathlib import Path
FM = Path(__file__).resolve().parent.parent / "docs/v5/egvla2_fm"
P = lambda t: f'<div style="font-size:0.82rem;line-height:1.8;margin:6px 0">{t}</div>'


def block(tbl, H, st):
    R = [json.loads((FM / f"t7_replay{x}.json").read_text()) for x in ["", "_s1", "_s2"]]
    f = lambda g, nd=1, sc=1: " / ".join(f"{g(r) * sc:.{nd}f}" for r in R)
    r0 = R[0]
    o = [H("T7 — 기록된 실기 세션 재생: 100% 성공한 A3와 텍스트 조건화 V4를 같은 프레임에서 비교"),
         P("<b>질문</b>: 실기에서 455/455 성공한 A3(EG2-09, <code>servolabel1.0</code>)와, 텍스트를 받는 V4(EG4-04)·이미지 전용 V0(EG4-00)가 <b>같은 실기 프레임</b>에서 얼마나 다르게 내는가.<br>"
           f"<b>방법</b>: A3 세션 {r0['sessions']}개(10/28~11/05, <code>servo_blend</code> 0.0)의 {r0['frames']:,}프레임. 입력은 세션 H5의 원본 이미지와 서버가 낸 박스. V0·V4는 300에피소드 전체로 120ep 저장 학습(시드 0·1·2, 체크포인트 6개 <code>runs/egvla2_fm/textcond/</code>). "
           "A3는 서빙 정책 코드 그대로 재생. 모든 값은 시드 0 / 1 / 2 순서."),
         H("점검 — 재생 환경이 맞는가"),
         tbl(["항목", "값"], [["A3 재생 vs 기록된 실행 액션 각도", f"평균 {r0['a3_replay_vs_stored']['ang_mean']:.2f}°, 90% {r0['a3_replay_vs_stored']['ang_p90']:.2f}° 이내"],
                              ["좌우 부호 일치", f"{r0['a3_replay_vs_stored']['lat_agree'] * 100:.1f}%"],
                              ["입력 특징 분포(세션 vs 학습)", "평균 이동 학습 표준편차의 0.14배(이미지)·0.11배(텍스트), 표준편차 비율 0.91·0.95"]]),
         H("결과 (A3의 명령을 기준으로)"),
         tbl(["지표", "V0 (이미지 전용)", "V4 (텍스트 조건화)"], [
             ["정상 박스 — A3와 각도 차이(°)", f(lambda r: r["v0_vs_a3"]["ang_mean"]), f(lambda r: r["v4_vs_a3"]["ang_mean"])],
             ["정상 박스 — 좌우 일치(%)", f(lambda r: r["v0_vs_a3"]["lat_agree"], 1, 100), f(lambda r: r["v4_vs_a3"]["lat_agree"], 1, 100)],
             ["박스 좌우 반전 — A3 명령과 각도 차이(°)", f(lambda r: r["mirror"]["v0_vs_a3"]), f(lambda r: r["mirror"]["v4_vs_a3"])],
             ["박스 좌우 반전 — 좌우 일치(%)", f(lambda r: r["mirror"]["v0_lat_agree"], 0, 100), f(lambda r: r["mirror"]["v4_lat_agree"], 0, 100)],
             ["A3 자신의 반전 시 좌우 일치(%)", f(lambda r: r["mirror"]["a3_lat_agree"], 0, 100), "(참고)"]]),
         tbl(["V4에 미학습 문구를 넣었을 때(실기 프레임)", "정지율(%)"], [
             ["장면에 없는 물체 14개", f(lambda r: r["v4_text"]["neg_phrases_stop"], 1, 100)],
             ["같은 색 다른 물체 4개", f(lambda r: r["v4_text"]["hard_neg_stop"], 1, 100)],
             ["동의 표현(긍정) 16개", f(lambda r: r["v4_text"]["held_pos_stop"], 1, 100)]]),
         H("읽는 법"),
         P("• <b>정상 장면</b>: V4가 V0보다 3시드 모두 A3와 더 가깝다(차이는 1~2°). 둘 다 A3와 3~5° 안이라 크게 어긋나지 않는다.<br>"
           "• <b>박스가 반대편일 때</b>: V0는 A3처럼 좌우가 뒤집힌다(일치 9~12%). V4는 3시드 모두 V0보다 낫지만 <b>시드 편차가 크다</b>(좌우 일치 42~94%). 반대편 오검출을 막는다고 단정할 수 없다.<br>"
           "• <b>문구</b>: 없는 물체·같은 색 다른 물체 지시는 3시드 모두 거의 100% 정지한다. 동의 표현은 약 6.5~7.6%가 정지(정상 정지 1~3%)로 일부 부작용이 남는다."),
         H("한계"),
         P("• 기준이 <b>A3가 실제로 낸 명령</b>이라 '비슷하다'만 잰다. V4가 낫거나 실기에서 성공한다는 증거가 아니다. V4·V0 체크포인트는 실기 미검증.<br>"
           "• 박스 반전은 정상 이미지에 박스만 뒤집은 <b>합성 조건</b>이다. 실제 오검출 분포와 다르다.<br>"
           "• 시드 3개는 같은 300에피소드에서 시드만 바꾼 것이다. 로컬 H5 이미지가 있는 프레임은 5,845개(로그 기준 6,178), 13개 세션은 박스·액션이 이미지보다 1행 많아 이미지 길이에 맞췄다.<br>"
           "• V5(토큰 어텐션)는 포함하지 않았다(T5에서 V4 대비 이득 없음, 시드 불안정).<br>"
           "파일: <code>scripts/t7_extract_session_feats.py</code> → <code>t7_train_save.py</code> → <code>t7_compare.py</code> · 계획서 <code>docs/plans/plan_20261109_t7_session_replay.md</code>")]
    return "".join(o)
