#!/usr/bin/env python3
"""research_story.html에 CH77(EGVLA2 데이터 확대 대비 브리핑)을 (재)생성한다. 멱등: <!--CH77-->…<!--/CH77--> 교체.
데이터: docs/v5/egvla2_fm/validation_batch.json, summary_s1*.json. 실행: .venv/bin/python3 -I scripts/gen_ch77.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gen_ch76_viz as V  # noqa: E402
import ch_common as CC  # noqa: E402

HTML, FM = V.HTML, V.FM
A, AL, G, R, B, P, MUT = V.A, V.AL, V.G, V.R, V.B, V.P, V.MUT


def tbl(head, rows):
    h = "".join(f'<th style="text-align:left;padding:7px 10px;border-bottom:2px solid {A};color:{AL}">{x}</th>' for x in head)
    o = f'<table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:0.8rem"><tr style="background:#0a1220">{h}</tr>'
    for r in rows:
        o += "<tr>" + "".join(f'<td style="padding:6px 10px;border-bottom:1px solid #1e293b">{x}</td>' for x in r) + "</tr>"
    return o + "</table>"


def H(t):
    return f'<div style="margin:16px 0 6px;padding-bottom:4px;border-bottom:2px solid {A}55;color:{AL};font-size:0.92rem;font-weight:800">{t}</div>'


def st(label, col):
    return f'<span style="display:inline-block;background:{col}22;color:{col};border:1px solid {col};border-radius:6px;padding:0 8px;font-size:0.74rem;font-weight:800">{label}</span>'


TABS = [("t1", "📌 개요·진행", ["①", "③", "④"]), ("t2", "📖 용어·대조표", ["⓪"]), ("t3", "🔀 이산 vs 연속·궤적", ["⑤", "⑥-4"]),
        ("t4", "🧠 헤드·FM 평가", ["②", "⑥-8", "⑥-3", "⑥-9"]), ("t5", "🎯 서브골·학습 곡선", ["⑥-5", "⑥-6"]), ("t10", "📐 심화 실험", ["⑥-18"]), ("t6", "🤖 EGVLA1 vs 2·실기", ["⑥-2", "⑥-7", "⑥-10", "⑥"]), ("t7", "🧰 산출물", ["⑦"])]


def tabify(body_):
    """본문을 최상위 소제목(⓪~⑦) 단위로 잘라 CH74식 탭 버튼(ctab)으로 재배치한다."""
    mk = list(re.finditer(r'<div style="margin:16px 0 6px[^>]*>(⓪|①|②|③|④|⑤|⑥-\d|⑥|⑦)[ ]', body_))
    head = body_[:mk[0].start()]
    parts = {}
    for i, m in enumerate(mk):
        end = mk[i + 1].start() if i + 1 < len(mk) else len(body_)
        parts[m.group(1)] = body_[m.start():end]
    bar = "".join(f'<button class="ctab-btn" data-t="{t}" onclick="ctabShow(\'ch77\',\'{t}\')">{lab}</button>' for t, lab, _ in TABS)
    panels = ""
    for t, lab, keys in TABS:
        inner = (head if t == "t1" else "") + "".join(parts.get(k, "") for k in keys)
        panels += f'<div class="ctab-panel" data-t="{t}" style="display:none">{inner}</div>\n'
    missing = [k for k in parts if not any(k in ks for _, _, ks in TABS)]
    assert not missing, missing
    return f'<div class="ctab-bar">{bar}</div>\n{panels}<script>ctabInit("ch77")</script>'


def body():
    Vd = json.load(open(FM / "validation_batch.json"))["aggregate"]
    s1_old = json.load(open(FM / "summary_s1.json"))["aggregate"]
    s1 = json.load(open(FM / "summary_s1_full300.json"))["aggregate"]
    lo = json.load(open(FM / "summary_s1_full300_lopo.json"))["aggregate"]
    iso, nz = Vd["az_spike"]
    ep = FM / "summary_s1_ep250.json"
    ep_txt = ""
    if ep.exists():
        a3 = json.load(open(ep))["aggregate"]
        ep_txt = (f"<li><b>epoch 확대(250, 3 seed)</b>: 방향 오차 FM-MLP {s1_old['A_mlp_fm']['angle_deg']:.1f}→{a3['A_mlp_fm']['angle_deg']:.1f}°, "
                  f"FM-FiLM {s1_old['B2_film_res_fm']['angle_deg']:.1f}→{a3['B2_film_res_fm']['angle_deg']:.1f}°, FM-π0 {s1_old['C_pi0_adaln']['angle_deg']:.1f}→{a3['C_pi0_adaln']['angle_deg']:.1f}° (자세한 표는 76-10)</li>")
    else:
        ep_txt = f"<li><b>epoch 확대(250, 3 seed)</b>: 진행 중 — 끝나면 이 항목과 76-10에 반영</li>"
    prog = V.flow([("수집 300/300 완료", G, "15칸×20, 로봇 쪽 대조 완료"), ("원샷 파이프라인", B, "캐시→검증→세트→S1→LOPO→체크포인트"),
                   ("서빙 정책(기본 꺼짐)", G, "F2 fused → FM → 속력 규칙"), ("지연 측정", R, "젯슨·서버 fp32, 로봇 접근 필요"), ("드라이런 → 폐루프 300회", P, "3 시작 위치×5 목표 위치×20")])
    stat = tbl(["영역", "상태", "내용"], [
        ["신규 v3 데이터 검증", st("완료", G), f"300건 읽기 문제 0, 제외 0, az 스파이크 {iso}/{nz}={iso / nz:.0%}(합격). 속력 단계 {Vd['level_dist']['0.4']:.0%}/{Vd['level_dist']['0.7']:.0%}/{Vd['level_dist']['1.0']:.0%}, 속력~거리 ρ=-0.09(가이드 이전)"],
        ["S1 학습(300ep, 5-fold×5seed·250epoch)", st("완료", G), f"lx R² {s1['C_pi0_adaln']['r2_mean_axis'][0]:.2f}, ly R² {s1['C_pi0_adaln']['r2_mean_axis'][1]:.2f}, az≈0.1. FM(FM-MLP/FM-FiLM/FM-π0) ≈ MLP 회귀. 안 본 목표 위치(LOPO)는 FM-π0 방향 오차 {lo['C_pi0_adaln']['angle_deg']:.1f}°(A {lo['A_mlp_fm']['angle_deg']:.1f}°). 같은 목표 위치 {s1['C_pi0_adaln']['angle_deg']:.1f}°"],
        ["epoch 확대 재시험", st("진행 중" if not ep.exists() else "완료", A if not ep.exists() else G), "최적 epoch이 상한 100 근처여서 250으로 재시험(방향 오차 소폭 개선)"],
        ["속력 보간 규칙", st("구현", G), "<code>continuous_speed.py</code>: STOP 0.2, 클램프 0.4~0.7, 섀도, az=0. 테스트 6개 통과"],
        ["FM 헤드 공유 모듈", st("구현", G), "<code>egvla2_fm_heads.py</code>. 분리 전후 가중치·샘플 출력 비트 동일"],
        ["서빙 정책·서버 훅", st("구현(꺼짐)", G), "<code>VLA_ACTION_MODE=continuous</code>일 때 <code>action_cont</code> 추가, <b>단계 A 하이브리드 구동</b>(<code>VLA_DRIVE_SOURCE</code>)·EMA·서브골 결측 처리. 원본 입력에서 학습 캐시와 일치, <b>JPEG q80은 4.1° 어긋나 q95 이상 필요</b>"],
        ["300ep 원샷 파이프라인", st("완료", G), "<code>run_egvla2_pipeline.py</code>: 캐시→검증→세트→S1→LOPO→체크포인트 6단계 실행 완료(2026-10-16)"],
        ["수집(로봇 세션)", st("완료", G), "300/300. 로봇 쪽에서 불량 파일 정리 후 20/20×15칸"],
        ["배포 체크포인트", st("저장", G), "FM-π0 + 서브골 드롭아웃30%, 120 epoch(권장)과 대조용(드롭아웃 없음). 로봇 쪽 전달은 rsync로 당겨 가기(핸드오프 §15)"],
        ["젯슨 점검(서버 경로·지연)", st("완료", G), "점검 A pass(방향 차 3.01°). 왕복 지연 <b>892ms</b>(Florence-2 821ms + FM·기타 54ms)"],
        ["드라이런", st("진행 예정", A), "수집 에피소드를 서버에 재생(<code>replay_episodes_through_server.py</code>). pos2·pos3 시작 신규 수집분으로 시작 위치 일반화도 오프라인 측정"],
        ["폐루프 평가", st("미착수", MUT), "300회 = 3 시작 위치(pos1~3)×5 목표 위치×20 (EGVLA1과 동일 구성, 확정), 첫 적용 섀도·az=0·s≤0.7. 양쪽 모두 pos1 하나로 학습, pos2·pos3은 제로샷 추론(시작 위치 일반화)"]])
    dec = tbl(["결정(사용자, 10/15)", "내용"], [
        ["폐루프 규모", "300회(성공 정의는 기존 프로토콜 유지). 구성은 EGVLA1과 같은 <b>3 시작 위치×5 목표 위치×20</b>으로 확정(경로는 수집 보조라 평가 축 아님, 사용자 확인 2026-10-18)"],
        ["속력 출처", "첫 적용은 섀도(규칙 속력으로 구동, 모델 속력은 기록만)"],
        ["az", "첫 적용은 az=0(이동만 모델, 회전은 기존 규칙)"],
        ["판정 기준", "|ρ(속력, 면적)|≥0.3, 60~100ep — 제안값 수용"],
        ["az 스파이크", "재수집 없이 진행(15%라 합격), 학습은 az 손실 가중↓/마스크 타깃 비교"],
        ["일정", "300ep가 모이기 전에는 빡세게 하지 않고, 모이면 바로 실행할 수 있게 사전 구현"]])
    body_ = (
        f'<div class="callout critical">🟡 <strong>3줄 요약</strong><br>'
        f'① 300건(15칸×20) 수집·검증 완료: <strong>읽기 문제 0, az 스파이크 9%로 합격</strong>. 다만 속력은 가이드 이전 수집이라 거리와 무관(ρ=-0.09).<br>'
        f'② S1(300ep): FM 헤드는 방향 축을 학습(방향 오차 약 10.4~11.3°)하지만 <strong>단순 MLP 회귀와 동급</strong>, az는 학습되지 않는다. 안 본 목표 위치(LOPO)에서는 FM-π0이 가장 낫다(16.9° vs FM-MLP 20.3°).<br>'
        f'③ 학습·배포 체크포인트(FM-π0+드롭아웃30)·서빙 경로·추론 서버 단계 A까지 끝났다. 젯슨 서버 경로 점검(pass)·지연 측정(892ms)까지 끝났다. <strong>남은 것</strong>은 드라이런(수집 에피소드 서버 재생)과 폐루프 3 시작 위치×5 목표 위치×20이며, 서버로 보내는 JPEG는 q95 이상이어야 한다.</div>'
        + __import__("ch77_summary").block(tbl, H, st)
        + H("① 지금 어디까지 왔나") + V.wrap("진행 흐름", prog, "300건 기준(2026-10-16 최종 재실행)", kind="flow") + stat
        + H("② 수치 한눈에") + V.wrap("S1 축별 R² (300ep, 5-fold × 5 seed, 250 epoch)", V.legend([("FM-MLP", B), ("FM-FiLM", G), ("FM-π0", A)]) + V.vgroups(["lx", "ly", "az"], [(V.HNAME[n], c, s1[n]["r2_mean_axis"]) for n, c in zip(["A_mlp_fm", "B2_film_res_fm", "C_pi0_adaln"], (B, G, A))], 1.0, fmt="{:.2f}"), "MLP 회귀와 MSE 차 ≤0.003", kind="chart")
        + f"<ul style='margin:6px 0 6px 18px'><li>신규 속력 단계 분포: 느림 {Vd['level_dist']['0.4']:.0%} / 보통 {Vd['level_dist']['0.7']:.0%} / 빠름 {Vd['level_dist']['1.0']:.0%}, 중간 크기 비율 {Vd['mid_ratio']:.0%}</li>"
        + f"<li>방향 오차: FM-MLP {s1['A_mlp_fm']['angle_deg']:.1f}° / FM-FiLM {s1['B2_film_res_fm']['angle_deg']:.1f}° / FM-π0 {s1['C_pi0_adaln']['angle_deg']:.1f}°, GO/STOP F1 ≈0.95</li>{ep_txt}</ul>"
        + H("③ 결정 기록") + dec
        + H("④ 막힌 것·의존") + "<ul style='margin:6px 0 6px 18px'><li><b>로봇 세션</b>: v3 구동 FM-π0·게이지 코드가 repo에 없음(푸시 요청), 남은 셀 수집, <code>scenario_progress.json</code> 갱신, rsync <code>--delete</code> 제거 요청(핸드오프 §14)</li><li><b>젯슨 접근</b>: FM 헤드 포함 전체 지연 측정(fp32), 드라이런</li><li><b>데이터</b>: 속력~거리 상관은 수집 가이드(§12.3/12.4) 적용 후 재측정해야 함(사용자 승인 대기)</li></ul>"
        + __import__("ch77_traj").block(tbl, H, st)
        + __import__("ch77_story").block(tbl, H, st)
        + __import__("ch77_compare").block(tbl, H, st)
        + __import__("ch77_serving").block(tbl, H, st)
        + __import__("ch77_startpos").block(tbl, H, st)
        + __import__("ch77_ceiling").block(tbl, H, st)
        + __import__("ch77_anatomy").block(tbl, H, st)
        + __import__("ch77_units").block(tbl, H, st)
        + __import__("ch77_deep").block(tbl, H, st)
        + __import__("ch77_outcome").block(tbl, H, st)
        + __import__("ch77_latmodes").block(tbl, H, st)
        + __import__("ch77_flip").block(tbl, H, st)
        + __import__("ch77_fm_extras").block(tbl, H, st)
        + __import__("ch77_smoothing").block(tbl, H, st)
        + __import__("ch77_replay").block(tbl, H, st)
        + __import__("ch77_subgoal").block(tbl, H, st)
        + __import__("ch77_curves").block(tbl, H, st)
        + H("⑦ 산출물") + tbl(["종류", "경로"], [
            ["plan", "<code>plan_20261014_inference_continuous_speed.md</code>, <code>plan_20261015_egvla2_300ep_pipeline.md</code>, <code>plan_20261016_egvla2_serving_path.md</code>"],
            ["코드", "<code>run_egvla2_pipeline.py</code>, <code>build_egvla2_main_set.py</code>, <code>train_egvla2_s1.py</code>, <code>replay_egvla2_policy.py</code>, <code>robovlm_nav/serve/egvla2_fm_policy.py</code>, <code>robovlm_nav/models/egvla2_fm_heads.py</code>, <code>continuous_speed.py</code>"],
            ["결과", "<code>validation_batch.json</code>, <code>summary_s1*.json</code> (<code>docs/v5/egvla2_fm/</code>)"]])
        + f"{st('한계', R)} S1은 한 장소 수집 데이터의 에피소드 단위 hold-out이라 장소·환경 일반화는 아직 모른다. 서빙 경로는 점검용 체크포인트로 동작만 확인했고 성능 수치는 최종 체크포인트(300ep 후)가 필요하다.")
    return CC.split_parts(body_)


def update_vis(s):
    """VIS G-4 패널(S1 FM 헤드 축별 R²)을 300건 결과로 다시 그린다."""
    pat = re.compile(r'<div style="background:#0a1628;border:1px solid #1e3050;border-radius:10px;padding:14px 16px"><div style="color:#fcd34d;font-size:0.8rem;font-weight:800;margin-bottom:8px">G-4\..*?CH76-10</a></div></div>', re.S)
    if not pat.search(s):
        return s
    a = json.load(open(FM / "summary_s1_full300.json"))["aggregate"]
    rows = []
    for n, lab, c in (("A_mlp_fm", "FM-MLP", B), ("B2_film_res_fm", "FM-FiLM", G), ("C_pi0_adaln", "FM-π0", A)):
        for i, ax in ((0, "lx"), (1, "ly")):
            v = a[n]["r2_mean_axis"][i]
            rows.append((f"{lab} {ax} R²", v, c, f"{v:.2f}"))
    panel = ('<div style="background:#0a1628;border:1px solid #1e3050;border-radius:10px;padding:14px 16px"><div style="color:#fcd34d;font-size:0.8rem;font-weight:800;margin-bottom:8px">'
             'G-4. S1 FM 헤드 축별 R² (300건, 5-fold×5seed)</div>' + V.hbars(rows, 1.0, w=150)
             + '<div style="font-size:0.72rem;color:#64748b;margin-top:6px">MLP 회귀와 MSE 차 ≤0.003, az R²≈0.1. 방향 오차 FM-π0 10.4°(같은 목표 위치)·16.9°(안 본 목표 위치). 사람 조작 천장 약 10~13°. '
             "<a href='#ch77' style='color:#94a3b8'>CH77</a> · <a href='#ch78' style='color:#94a3b8'>CH78</a></div></div>")
    return pat.sub(lambda m: panel, s, count=1)


TABS77 = [("t1", "📌 개요·결과", [("html", "OVERVIEW")]), ("t2", "📖 용어·대조표", ["⓪"]), ("t3", "🔀 이산 vs 연속·궤적", ["⑤", "⑥-4"]),
          ("t4", "🧠 헤드·FM 평가", ["②", "⑥-8", "⑥-3", "⑥-9"]), ("t5", "🎯 서브골·학습 곡선", ["⑥-5", "⑥-6"]), ("t10", "📐 심화 실험", ["⑥-18"]), ("t6", "🧬 실험군 구조", [("html", "ARCH")]), ("t7", "🔬 학습 대조 해설", [("html", "CONTRAST")]),
          ("t8", "🔍 방향 오차 해부", ["⑥-11", "⑥-16", "⑥-17", "⑥-13", "⑥-15"]), ("t9", "🧰 산출물(학습)", ["⑦"])]


def main():
    s = HTML.read_text(encoding="utf-8")
    head, parts = body()
    import ch77_overview, ch77_contrast, ch77_arch
    raw = {"OVERVIEW": ch77_overview.block(tbl, H, st), "CONTRAST": ch77_contrast.block(tbl, H, st), "ARCH": ch77_arch.block(tbl, H, st)}
    tabs = [(t, lab, [("html", raw[it[1]]) if isinstance(it, tuple) else it for it in items]) for t, lab, items in TABS77]
    inner = CC.tabs_html("ch77", tabs, "", parts)
    blk = CC.chapter_block(77, "#fbbf24", "#fcd34d", "EGVLA2 학습 분야 정리 — 300건 수집에서 오프라인 평가·배포 체크포인트까지",
                           "학습 분야의 결과와 대조 해설. 서빙·점검·로드맵·시사점은 CH78. 세부는 CH76-10.", "2026-10-20", inner)
    s = CC.insert_chapter(s, 77, blk, "/CH76 ═ -->", "", "", "", "", "", "", "", "", 76)
    s = CC.replace_toc_label(s, 77, "EGVLA2 학습 분야 정리 — 300건·헤드·서브골·대조 해설")
    s = CC.replace_summary_row(s, 77, "EGVLA2 학습 분야 정리", "300건 수집·검증·학습, 헤드(MLP 회귀/FM-MLP/FM-FiLM/FM-π0)·서브골 ablation(3 seed)·학습 곡선·평활·사람 일치도 천장. 같은 위치 방향 오차 10.4°≈사람 변동성, 안 본 위치 16.9°가 남은 과제, FM≈MLP·서브골 효과 ≤0.4°, 시작 구간 다봉(K 평균이 모드를 뭉갬), 지연 보상·반전 증강 기각, 방향 오차 해부·실험군 구조 해부")
    s = re.sub(r"현재 상태 \(CH1→CH\d+ · [0-9-]+\)", "현재 상태 (CH1→CH78 · 2026-10-20)", s, count=1)
    s = update_vis(s)
    s = re.sub(r'(<a href="#ch77" style="[^"]*">)[^<]*(</a>)', r'\g<1>🧭 CH77 EGVLA2 학습 분야 정리\g<2>', s, count=1)
    HTML.write_text(s, encoding="utf-8")
    for t, a, b in CC.balance(s):
        print(t, a, b)


if __name__ == "__main__":
    main()
