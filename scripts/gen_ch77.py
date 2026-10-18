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
        ("t4", "🧠 헤드·FM 평가", ["②", "⑥-3"]), ("t5", "🎯 서브골·학습 곡선", ["⑥-5", "⑥-6"]), ("t6", "🤖 EGVLA1 vs 2·실기", ["⑥-2", "⑥"]), ("t7", "🧰 산출물", ["⑦"])]


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
        ep_txt = (f"<li><b>epoch 확대(250, 3 seed)</b>: 방향 오차 A {s1_old['A_mlp_fm']['angle_deg']:.1f}→{a3['A_mlp_fm']['angle_deg']:.1f}°, "
                  f"B2 {s1_old['B2_film_res_fm']['angle_deg']:.1f}→{a3['B2_film_res_fm']['angle_deg']:.1f}°, C {s1_old['C_pi0_adaln']['angle_deg']:.1f}→{a3['C_pi0_adaln']['angle_deg']:.1f}° (자세한 표는 76-10)</li>")
    else:
        ep_txt = f"<li><b>epoch 확대(250, 3 seed)</b>: 진행 중 — 끝나면 이 항목과 76-10에 반영</li>"
    prog = V.flow([("수집 300/300 완료", G, "15칸×20, 로봇 쪽 대조 완료"), ("원샷 파이프라인", B, "캐시→검증→세트→S1→LOPO→체크포인트"),
                   ("서빙 정책(기본 꺼짐)", G, "F2 fused → FM → 속력 규칙"), ("지연 측정", R, "젯슨·서버 fp32, 로봇 접근 필요"), ("드라이런 → 폐루프 300회", P, "3 시작 위치×5 목표 위치×20")])
    stat = tbl(["영역", "상태", "내용"], [
        ["신규 v3 데이터 검증", st("완료", G), f"300건 읽기 문제 0, 제외 0, az 스파이크 {iso}/{nz}={iso / nz:.0%}(합격). 속력 단계 {Vd['level_dist']['0.4']:.0%}/{Vd['level_dist']['0.7']:.0%}/{Vd['level_dist']['1.0']:.0%}, 속력~거리 ρ=-0.09(가이드 이전)"],
        ["S1 학습(300ep, 5-fold×5seed·250epoch)", st("완료", G), f"lx R² {s1['C_pi0_adaln']['r2_mean_axis'][0]:.2f}, ly R² {s1['C_pi0_adaln']['r2_mean_axis'][1]:.2f}, az≈0.1. FM(A/B2/C) ≈ MLP 회귀. 안 본 목표 위치(LOPO)는 C 방향 오차 {lo['C_pi0_adaln']['angle_deg']:.1f}°(A {lo['A_mlp_fm']['angle_deg']:.1f}°). 같은 목표 위치 {s1['C_pi0_adaln']['angle_deg']:.1f}°"],
        ["epoch 확대 재시험", st("진행 중" if not ep.exists() else "완료", A if not ep.exists() else G), "최적 epoch이 상한 100 근처여서 250으로 재시험(방향 오차 소폭 개선)"],
        ["속력 보간 규칙", st("구현", G), "<code>continuous_speed.py</code>: STOP 0.2, 클램프 0.4~0.7, 섀도, az=0. 테스트 6개 통과"],
        ["FM 헤드 공유 모듈", st("구현", G), "<code>egvla2_fm_heads.py</code>. 분리 전후 가중치·샘플 출력 비트 동일"],
        ["서빙 정책·서버 훅", st("구현(꺼짐)", G), "<code>VLA_ACTION_MODE=continuous</code>일 때만 <code>action_cont</code> 추가. 서빙 추출기 vs 캐시: bbox 0.0, 특징 0.002 차이"],
        ["300ep 원샷 파이프라인", st("완료", G), "<code>run_egvla2_pipeline.py</code>: 캐시→검증→세트→S1→LOPO→체크포인트 6단계 실행 완료(2026-10-16)"],
        ["수집(로봇 세션)", st("완료", G), "300/300. 로봇 쪽에서 불량 파일 정리 후 20/20×15칸"],
        ["지연 측정·헤드 선택", st("대기", R), "최종 체크포인트와 젯슨 접근 필요. 후보 C(방향) vs A(지연)"],
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
        f'② S1(300ep): FM 헤드는 방향 축을 학습(방향 오차 약 10.4~11.3°)하지만 <strong>단순 MLP 회귀와 동급</strong>, az는 학습되지 않는다. 안 본 목표 위치(LOPO)에서는 C가 가장 낫다(16.9° vs A 20.3°).<br>'
        f'③ 학습·체크포인트·서빙 경로 재생 점검까지 끝났다. <strong>남은 것은 실기</strong>(서버 경로 점검, 지연 측정, 드라이런, 폐루프 300회)이며 이 값이 있어야 EGVLA1 92%와 비교할 수 있다.</div>'
        + __import__("ch77_summary").block(tbl, H, st)
        + H("① 지금 어디까지 왔나") + V.wrap("진행 흐름", prog, "300건 기준(2026-10-16 최종 재실행)", kind="flow") + stat
        + H("② 수치 한눈에") + V.wrap("S1 축별 R² (300ep, 5-fold × 5 seed, 250 epoch)", V.legend([("A", B), ("B2", G), ("C", A)]) + V.vgroups(["lx", "ly", "az"], [(n.split("_")[0], c, s1[n]["r2_mean_axis"]) for n, c in zip(["A_mlp_fm", "B2_film_res_fm", "C_pi0_adaln"], (B, G, A))], 1.0, fmt="{:.2f}"), "MLP 회귀와 MSE 차 ≤0.003", kind="chart")
        + f"<ul style='margin:6px 0 6px 18px'><li>신규 속력 단계 분포: 느림 {Vd['level_dist']['0.4']:.0%} / 보통 {Vd['level_dist']['0.7']:.0%} / 빠름 {Vd['level_dist']['1.0']:.0%}, 중간 크기 비율 {Vd['mid_ratio']:.0%}</li>"
        + f"<li>방향 오차: A {s1['A_mlp_fm']['angle_deg']:.1f}° / B2 {s1['B2_film_res_fm']['angle_deg']:.1f}° / C {s1['C_pi0_adaln']['angle_deg']:.1f}°, GO/STOP F1 ≈0.95</li>{ep_txt}</ul>"
        + H("③ 결정 기록") + dec
        + H("④ 막힌 것·의존") + "<ul style='margin:6px 0 6px 18px'><li><b>로봇 세션</b>: v3 구동 C·게이지 코드가 repo에 없음(푸시 요청), 남은 셀 수집, <code>scenario_progress.json</code> 갱신, rsync <code>--delete</code> 제거 요청(핸드오프 §14)</li><li><b>젯슨 접근</b>: FM 헤드 포함 전체 지연 측정(fp32), 드라이런</li><li><b>데이터</b>: 속력~거리 상관은 수집 가이드(§12.3/12.4) 적용 후 재측정해야 함(사용자 승인 대기)</li></ul>"
        + __import__("ch77_traj").block(tbl, H, st)
        + __import__("ch77_story").block(tbl, H, st)
        + __import__("ch77_compare").block(tbl, H, st)
        + __import__("ch77_fm_extras").block(tbl, H, st)
        + __import__("ch77_replay").block(tbl, H, st)
        + __import__("ch77_subgoal").block(tbl, H, st)
        + __import__("ch77_curves").block(tbl, H, st)
        + H("⑦ 산출물") + tbl(["종류", "경로"], [
            ["plan", "<code>plan_20261014_inference_continuous_speed.md</code>, <code>plan_20261015_egvla2_300ep_pipeline.md</code>, <code>plan_20261016_egvla2_serving_path.md</code>"],
            ["코드", "<code>run_egvla2_pipeline.py</code>, <code>build_egvla2_main_set.py</code>, <code>train_egvla2_s1.py</code>, <code>replay_egvla2_policy.py</code>, <code>robovlm_nav/serve/egvla2_fm_policy.py</code>, <code>robovlm_nav/models/egvla2_fm_heads.py</code>, <code>continuous_speed.py</code>"],
            ["결과", "<code>validation_batch.json</code>, <code>summary_s1*.json</code> (<code>docs/v5/egvla2_fm/</code>)"]])
        + f"{st('한계', R)} S1은 한 장소 수집 데이터의 에피소드 단위 hold-out이라 장소·환경 일반화는 아직 모른다. 서빙 경로는 점검용 체크포인트로 동작만 확인했고 성능 수치는 최종 체크포인트(300ep 후)가 필요하다.")
    return tabify(body_)


def main():
    s = HTML.read_text(encoding="utf-8")
    s = re.sub(r"<!--CH77-->.*?<!--/CH77-->\n?", "", s, flags=re.S)
    ch = ('<!--CH77--><hr class="chapter-divider">\n<div class="chapter" id="ch77">\n  <div class="chapter-header">\n'
          '    <span class="chapter-num" style="background:#fbbf24;color:#0a0f1a">CH 77</span>\n'
          '    <h2 class="chapter-title">EGVLA2 데이터 확대 대비 브리핑 — 신규 배치 검증, S1 학습, 사전 구현, 남은 일</h2>\n'
          '    <p class="chapter-subtitle"><span style="color:#fcd34d;font-weight:800">2026-10-15</span> · 300건 수집·검증·학습·오프라인 평가까지의 결과와 남은 실기 일정. 세부는 CH76-10.</p>\n  </div>\n'
          + body() + '\n</div>\n<!-- ══ /CH77 ═ -->\n<!--/CH77-->\n')
    k = s.index("/CH76 ═ -->") + len("/CH76 ═ -->")
    k = s.index("\n", k) + 1
    s = s[:k] + "\n" + ch + s[k:]
    if 'data-target="ch77"' not in s:
        toc = s.index('<a class="toc-item" href="#ch76"'); e = s.index("</a>", toc) + 4
        s = s[:e] + '\n    <a class="toc-item" href="#ch77" data-target="ch77"><span class="toc-badge" style="background:rgba(251,191,36,0.2);color:#fcd34d;font-weight:900">CH 77</span><span class="toc-label">EGVLA2 데이터 확대 대비 브리핑 — 검증·S1·사전 구현·남은 일</span><span class="toc-cat cat-exp" style="background:rgba(251,191,36,0.15);color:#fbbf24">신규분기</span></a>' + s[e:]
        nav = s.index('<a href="#ch76" style="padding:7px 16px'); e = s.index("</a>", nav) + 4
        s = s[:e] + '\n    <a href="#ch77" style="padding:7px 16px;background:#2a1f06;border:1px solid #fbbf24;border-radius:8px;font-size:0.78rem;font-weight:800;color:#fcd34d;text-decoration:none">🧭 CH77 EGVLA2 확대 대비 브리핑</a>' + s[e:]
        s = s.replace("'ch75','ch76','next-step','summary'", "'ch75','ch76','ch77','next-step','summary'", 1)
        # SUMMARY 타임라인
        row = re.search(r"(<tr><td[^>]*><a href='#ch76'.*?</tr>)", s, flags=re.S)
        new = row.group(1).replace("#ch76", "#ch77").replace(">CH76<", ">CH77<").replace("연속 조이스틱 수집·FM 시험", "EGVLA2 확대 대비 브리핑")
        new = re.sub(r"(<td style=\"[^\"]*\">)(?!<a)(?!연속|EGVLA2)([^<]*)(</td></tr>)$", lambda m: m.group(1) + "신규 v3 191건 검증 합격, S1 FM≈MLP, 서빙 경로 사전 구현(꺼짐), 300ep 원샷 파이프라인 준비" + m.group(3), new, flags=re.S)
        s = s.replace(row.group(1), row.group(1) + "\n" + new, 1)
        s = s.replace("현재 상태 (CH1→CH76 · 2026-10-15)", "현재 상태 (CH1→CH77 · 2026-10-15)", 1)
    HTML.write_text(s, encoding="utf-8")
    for t in ["div", "svg", "table", "tr", "button", "a", "ul", "li"]:
        print(t, len(re.findall(f"<{t}[ >]", s)), len(re.findall(f"</{t}>", s)))


if __name__ == "__main__":
    main()
