"""CH77 🧬 실험군 구조 해부: FiLM·AdaLN의 출처, MLP 회귀를 쓰는 이유, 4개 실험군(MLP 회귀 / FM-MLP / FM-FiLM / FM-π0)의 구조·입력 경로·비용·결과 대조. 코드(egvla2_fm_heads.py) 기준."""
import json

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT


def J(n):
    return json.load(open(FM / n))


def block(tbl, H, st):
    stt, s, l = J("head_arch_stats.json"), J("summary_s1_full300.json")["aggregate"], J("summary_s1_full300_lopo.json")["aggregate"]
    ab, cv, ex = J("subgoal_ablation.json"), J("ablation_curves.json"), J("fm_head_extras.json")
    ang = lambda a, k: f'{a[k]["angle_deg"]:.1f}°'
    miss = lambda k: sum(x[f"{k}|full"]["missing"]["angle_deg"] for kk, x in ab.items() if kk.startswith("lopo")) / sum(1 for kk in ab if kk.startswith("lopo"))
    reg = [min(cv[sp]["MLP_regression|full"], key=lambda r: abs(r["ep"] - 60))["angle"] for sp in ("fold0", "lopo_center", "lopo_strong_right")]
    jit = lambda k: ex["fold0"]["metrics"][k]["frame_jitter"]
    box = lambda t, b: f'<div style="margin:14px 0 4px;font-size:0.86rem;font-weight:800;color:#fcd34d">{t}</div>{b}'
    origin = tbl(["방식", "원래 어디서 쓰던 것", "우리 쓰임", "확인 수준"], [
        ["<b>FiLM</b> (feature-wise linear modulation)", "Perez 외 2018, 시각 질의응답에서 질문 벡터로 이미지 특징을 채널별 γ·β로 변조하는 조건부 층. 이후 RT-1이 언어 지시로 이미지 인코더를 변조하는 데 사용", "<b>FM-FiLM</b>: bbox(서브골)와 시간 t로 γ·β를 만들어 이미지 특징을 변조. 레포의 CH73 '그라운딩 서브골 조향(FiLM)'(V6 steered 78.41 vs concat 76.76)에서 이어짐", "일반 지식(원문 재확인 안 함)"],
        ["<b>AdaLN-Zero</b>", "DiT(Peebles & Xie 2023)의 조건부 정규화: 시간·조건으로 정규화의 scale·shift와 잔차 gate를 만들고 gate를 0으로 초기화", "<b>FM-π0</b>의 블록마다 시간 t로 scale·shift·gate 생성(gate 0 초기화)", "일반 지식(원문 재확인 안 함)"],
        ["<b>π0의 시간 조건</b>", "π0: flow 시간을 노이즈 섞인 액션과 <b>병합</b>해 트랜스포머에 넣음. π0.5: 시간을 별도 MLP로 만들어 액션 expert 층마다 <b>adaptive RMSNorm</b>으로 주입", "FM-π0의 시간 주입은 <b>π0.5의 adaRMS·DiT AdaLN-Zero 계열</b>이며 π0 원논문 방식은 아님. 이름의 'π0'는 '액션 청크 토큰 + VLM 조건 prefix + 액션 expert' 구조를 가리킴", "π0.5 논문(arXiv 2504.16054) 서술과 2차 문서로 확인"],
        ["<b>flow-matching 헤드</b>", "π0 등 VLA의 액션 생성, 노이즈→액션 속도장을 학습해 몇 스텝으로 샘플링", "FM-MLP·FM-FiLM·FM-π0 모두 같은 flow-matching 손실(속도장 MSE), Euler 10스텝, K=8 평균", "코드 확인"]])
    why = tbl(["MLP 회귀를 쓰는 이유", "설명"], [
        ["<b>대조군(필수)</b>", "FM이 이득을 주는지 알려면 <b>같은 입력·같은 청크 출력</b>을 쓰는 비생성 모델과 비교해야 한다. 회귀가 없으면 'FM이 좋다/나쁘다'를 말할 근거가 없다"],
        ["EGVLA1의 연장선", "EGVLA1 헤드는 MLP였다. 연속 액션으로 바꿨을 때 'MLP 헤드 그대로(회귀)'와 'FM 헤드'를 같은 조건에서 가르는 것이 10/12 합의(MLP→FM)의 검증 질문"],
        ["서빙 후보", "회귀는 <b>가장 부드럽고(저크 0.085)·가장 빠르다</b>(헤드 0.05ms). 정확도가 동급이면 실기에서 더 단순한 선택지가 된다 → 폐루프에서 비교군으로 함께 평가 권장"],
        ["결과", "S1에서 정확도 차 ≤0.003, 다봉 이점 없음 — 그래서 'FM을 쓸 정확도상의 이유는 없다'는 결론이 나왔다(CH77 헤드·FM 평가 탭)"]])
    arch = tbl(["", "MLP 회귀(대조군)", "FM-MLP (구 A)", "FM-FiLM (구 B2)", "FM-π0 (구 C)"], [
        ["출력·학습", "청크 24값(8스텝×lx·ly·az) 직접 회귀, MSE", "노이즈 x_t → 속도장, flow-matching 손실", "같음", "같음(토큰별 속도)"],
        ["이미지 특징 768-d", "LayerNorm → 은닉에 concat", "LN+Linear(128)+GELU 후 concat", "LN+Linear(128) → <b>γ·β로 변조</b>", "LN+Linear(128) → <b>시각 토큰(prefix)</b>"],
        ["서브골(bbox 창 12-d) 넣는 법", "특징과 concat", "특징·시간·x_t와 한 벡터로 concat", "<b>bbox+시간 → MLP → γ·β</b>로 특징 변조(FiLM)", "<b>서브골 토큰(prefix)</b>, 청크 토큰이 어텐션으로 참조"],
        ["시간 t 넣는 법", "없음", "임베딩 64-d를 concat", "FiLM(γ·β) + 입력 concat", "<b>AdaLN-Zero</b>(층마다 scale·shift·gate)"],
        ["노이즈 액션 x_t 넣는 법", "—", "24-d를 flatten해 concat", "flatten해 trunk 앞에서 concat", "스텝별 Linear + 위치 임베딩 = 청크 토큰 8개"],
        ["본체", "MLP 1층(256)", "MLP 256-256-24", "잔차 MLP 블록 2개(256)", "AdaLN-Zero 블록 2층(d=128, 어텐션 4헤드, MLP 4배)"],
        ["파라미터 수", f'{stt["MLP_regression"]["params"]:,}', f'{stt["A_mlp_fm"]["params"]:,}', f'{stt["B2_film_res_fm"]["params"]:,}', f'<b>{stt["C_pi0_adaln"]["params"]:,}</b>'],
        ["헤드 연산 시간(이 머신 GPU, K=8·10스텝)", f'{stt["MLP_regression"]["head_ms"]} ms(1회)', f'{stt["A_mlp_fm"]["head_ms"]} ms', f'{stt["B2_film_res_fm"]["head_ms"]} ms', f'<b>{stt["C_pi0_adaln"]["head_ms"]} ms</b> (젯슨 FM·기타 약 54ms 실측)'],
        ["방향 오차: 같은 위치(5-fold) / 안 본 위치(LOPO)", f"{reg[0]:.1f}° / {min(reg[1:]):.1f}~{max(reg[1:]):.1f}° (곡선 실험 60ep, seed 1)", f'{ang(s, "A_mlp_fm")} / {ang(l, "A_mlp_fm")}', f'{ang(s, "B2_film_res_fm")} / {ang(l, "B2_film_res_fm")}', f'<b>{ang(s, "C_pi0_adaln")} / {ang(l, "C_pi0_adaln")}</b>'],
        ["서브골 결측 시 방향 오차(안 본 위치)", "—", "미측정", f'{miss("B2_film_res_fm"):.1f}° (크게 악화)', f'{miss("C_pi0_adaln"):.1f}°'],
        ["프레임 저크(평활 전)", f'{jit("MLP_regression"):.3f}', f'{jit("A_mlp_fm"):.3f}', f'{jit("B2_film_res_fm"):.3f}', f'{jit("C_pi0_adaln"):.3f}'],
        ["최적 epoch(5-fold)", "약 50", f'{s["A_mlp_fm"]["best_epoch"]:.0f}', f'{s["B2_film_res_fm"]["best_epoch"]:.0f}', f'{s["C_pi0_adaln"]["best_epoch"]:.0f}']])
    def fl(steps):
        return V.flow(steps, "row")
    diag = (V.wrap("MLP 회귀", fl([("특징 768 + bbox 12", MUT), ("concat", MUT), ("MLP", B, "256"), ("청크 24", G)]), kind="flow")
            + V.wrap("FM-MLP (구 A)", fl([("특징→128 · bbox 12 · 시간 64 · x_t 24", MUT), ("한 벡터로 concat", MUT), ("MLP 256-256", B), ("속도장 24", G)]), "서브골은 평범한 입력", kind="flow")
            + V.wrap("FM-FiLM (구 B2)", fl([("bbox + 시간", A, "MLP→γ·β"), ("특징 128을 변조", P, "(1+γ)h+β"), ("+ x_t, 시간", MUT), ("잔차 MLP ×2", B), ("속도장 24", G)]), "서브골이 이미지 특징을 직접 조향", kind="flow")
            + V.wrap("FM-π0 (구 C)", fl([("prefix: 서브골 토큰 + 시각 토큰", A), ("청크 토큰 8개(x_t)", MUT), ("AdaLN-Zero 블록 ×2", P, "시간 조건 · 셀프+prefix 어텐션"), ("토큰별 속도 3", G)]), "서브골은 어텐션이 필요할 때 참조", kind="flow"))
    read = (f'{st("읽는 법", A)}<ul style="margin:6px 0 6px 18px;font-size:0.8rem">'
            f'<li><b>MLP 회귀 → FM-MLP</b>: 출력만 바뀐다(회귀 → 노이즈에서 샘플링). 정확도는 같고(MSE 차 ≤0.003) 저크가 {jit("A_mlp_fm") / jit("MLP_regression"):.1f}배 → <b>FM 자체는 이 데이터에서 이득 없음</b>.</li>'
            f'<li><b>FM-MLP → FM-FiLM</b>: 서브골이 입력 벡터의 일부에서 <b>특징 변조기</b>가 된다. 방향 오차 {ang(s, "A_mlp_fm")}→{ang(s, "B2_film_res_fm")}(같은 위치), {ang(l, "A_mlp_fm")}→{ang(l, "B2_film_res_fm")}(안 본 위치)로 개선되지만 <b>서브골이 사라지면 크게 악화</b>({miss("B2_film_res_fm"):.1f}°) — 변조 경로가 서브골에 강하게 의존.</li>'
            f'<li><b>FM-FiLM → FM-π0</b>: 서브골이 prefix 토큰이 되어 어텐션으로 선택적으로 참조되고 시간은 층마다 정규화로 들어간다. 안 본 위치 {ang(l, "B2_film_res_fm")}→{ang(l, "C_pi0_adaln")}, 결측 {miss("B2_film_res_fm"):.1f}°→{miss("C_pi0_adaln"):.1f}°. 대가는 파라미터 {stt["C_pi0_adaln"]["params"] / stt["B2_film_res_fm"]["params"]:.1f}배, 헤드 연산 {stt["C_pi0_adaln"]["head_ms"] / stt["B2_film_res_fm"]["head_ms"]:.1f}배.</li>'
            '<li><b>왜 FM-π0가 결측에 덜 취약한가</b>는 <b>가설</b>이다(어텐션이 서브골 토큰을 필요할 때만 쓰도록 학습되어 의존이 낮다). 검증하지 않았다.</li></ul>'
            f'{st("한계", R)} 구조를 바꾸면 <b>파라미터 수·층 수도 함께 바뀌어</b> "구조 효과"만 분리하지 못했다(파라미터 수를 맞춘 통제 실험은 하지 않음). FM-FiLM은 초기형 B가 방향 축 az를 허위 생성(R² −5.2)해서 잔차·시간 입력을 더한 B2로 수정한 것이며 S1·배포 모두 B2다.')
    return (H("⑥-12 실험군 구조 해부 — FiLM의 출처, MLP 회귀의 이유, 네 실험군의 정확한 차이")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">같은 입력(이미지 특징 768-d + bbox 창 12-d)과 같은 출력(액션 청크 8×3)을 쓰는 네 실험군이 <b>무엇이 다른지</b>를 코드(<code>robovlm_nav/models/egvla2_fm_heads.py</code>) 기준으로 정리했다.</div>'
            + box("① 쓰인 방식들은 어디서 왔나", origin) + box("② 왜 FM 헤드 말고 MLP 회귀도 학습했나", why) + box("③ 구조 비교표", arch) + box("④ 구조도", diag) + read)
