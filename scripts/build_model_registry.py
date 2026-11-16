"""모델 세대 레지스트리: 스펙(아래 SPEC) + 파일 해시·크기 + 로봇 episode_log 집계 -> docs/v5/MODEL_REGISTRY.json, model_registry.html.
실기 수치는 origin/monavla-driving:logs/episode_log.csv의 체크포인트 열 집계(운영자 수동 판정). 스펙에 없는 값은 None(미확인)으로 둔다.
실행: .venv/bin/python3 scripts/build_model_registry.py  (git fetch 후)"""
import csv, collections, hashlib, io, json, pathlib, subprocess, datetime, html
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "docs/v5"
GENS = {
 "G0": ("V5 시기(Kosmos-2)", "순수 E2E → 박스 분해. 폐기, 기록만."),
 "G1": ("EGVLA1", "OWLv2 + Kosmos-2 + MLP(exp73). 실기 기준선."),
 "G2": ("EGVLA2", "Florence-2 단일 + FM 헤드, 300에피소드. 현재 배포(A3=EG2-09)."),
 "G3": ("E2E LoRA", "Florence-2 LoRA + FM. 오프라인 부진, 실기 없음, 보류."),
 "G4": ("텍스트 조건화", "V0~V5. 평가 중 학습·폐기, 체크포인트 없음. 오프라인만."),
}
def E(pn, gen, name, ck=None, path=None, arch="", train="", offline="", status="", note="", key=None):
    return dict(pn=pn, gen=gen, name=name, ckpt=ck, path=path, arch=arch, train=train, offline=offline, status=status, note=note, log_key=key if key is not None else ck)
SPEC = [
 E("EG0-01","G0","V5 Exp11","epoch=14 ckpt","runs/v5_nav/kosmos/mobile_vla_v5_exp11/2026-04-16/v5-exp11-google-robot-8cls","Kosmos-2(고정)+LoRA, 구글로봇 사전학습, 8클래스, 순수 E2E","","val_loss 1.010, PM 58.6%","폐기","폐루프(시뮬) 0%, FPE 1.45m",key="__none__"),
 E("EG0-02","G0","Exp14 Step2","","docs/v5/bbox_nav_step1","박스+이미지 MLP(분해)","45ep 794프레임","PM 75.9%(5시드 76.6±1.6%)","폐기","폐루프(시뮬) 66.7%, FPE 0.55m",key="__none__"),
 E("EG1-00","G1","구버전(체크포인트 미기록)","", None,"","","","추적 불가","로그 6/26~7/23, 체크포인트 열 공란",key=""),
 E("EG1-01","G1","OWL 트랙 hold-aware","exp73_owl_trackF_v6_mlp_holdaware_seed0.pt",None,"OWLv2 박스 + MLP","","","기록"),
 E("EG1-02","G1","EGVLA1 기준선","exp73_owl_stage1v3_v6_mlp.pt","runs/v5_nav/mlp/exp73_stage1v3/exp73_owl_stage1v3_v6_mlp.pt","MLP 1560→512→128→8(8클래스), window 6, bbox_scale 3, V6 arm","데이터: mobile_vla_dataset_v5(h5 515개, CH76 ④·사용자 확인). 에폭: CH70이 exp73 원본 값 300을 언급(이 체크포인트 자체에는 기록 없음). CH78: EGVLA1·2 모두 pos1만 학습","val_acc 74.1%(V6 split, 체크포인트 기록)","기준선","3×5×20 폐루프 기준선. 위치별 pos1 95·pos2 91·pos3 90 (episode_log·메모리)"),
 E("EG1-03","G1","Florence-2 grounder + MLP","exp73_florence2_grounder_v6_mlp.pt","runs/v5_nav/mlp/exp73_stage1v3/exp73_florence2_grounder_v6_mlp.pt","Florence-2 그라운더 + MLP(V6)","미확인","미확인","실기 로그 없음","8/20 파일. EGVLA2 전환 이전의 중간 산물로 보이나 미확인",key="__none__"),
 E("EG1-04","G1","Florence-2 phrase grounder + MLP","exp73_florence2_phrase_grounder_v6_mlp.pt","runs/v5_nav/mlp/exp73_stage1v3/exp73_florence2_phrase_grounder_v6_mlp.pt","Florence-2 문구 그라운더 + MLP(V6)","미확인","미확인","실기 로그 없음","8/21 파일",key="__none__"),
 E("EG2-01","G2","A_mlp_fm","A_mlp_fm.pt","runs/egvla2_fm/A_mlp_fm.pt","MLP+FM","250ep, 300ep","각도 11.26°(층화) / 20.28°(LOPO)","오프라인",key="__none__"),
 E("EG2-02","G2","B2_film_res_fm","B2_film_res_fm.pt","runs/egvla2_fm/B2_film_res_fm.pt","FiLM 잔차+FM","250ep, 300ep","10.81° / 17.91°","오프라인",key="__none__"),
 E("EG2-03","G2","C_pi0_adaln","C_pi0_adaln.pt","runs/egvla2_fm/C_pi0_adaln.pt","PiZeroFM AdaLN-Zero","250ep, 300ep","10.39° / LOPO 미출력","오프라인",key="__none__"),
 E("EG2-04","G2","MLP 회귀(결정적)","MLP_regression_drop30_e60.pt","runs/egvla2_fm/deploy/MLP_regression_drop30_e60.pt","MLP 회귀, 드롭아웃 30%","60ep, 300ep","","실기 비교군 후보",key="__none__"),
 E("EG2-05","G2","C adaln 드롭0","C_pi0_adaln_e120.pt","runs/egvla2_fm/deploy/C_pi0_adaln_e120.pt","adaln, 드롭아웃 0","120ep, 300ep","","대조",key="__none__"),
 E("EG2-06","G2","C adaln 드롭30 (CH79의 A1)","C_pi0_adaln_drop30_e120.pt","runs/egvla2_fm/deploy/C_pi0_adaln_drop30_e120.pt","adaln, 드롭아웃 30%","120ep, 300ep","","실기 부진","오프라인 좋았으나 실기 낮음"),
 E("EG2-07","G2","C e1b tuned","C_pi0_e1b_tuned_drop30_e120.pt","runs/egvla2_fm/deploy/C_pi0_e1b_tuned_drop30_e120.pt","서보 τ=0.22 gain=2.6","120ep, 300ep","","실기 기록 없음",key="__none__"),
 E("EG2-08","G2","서보 라벨 λ=0.5 (CH79의 A2)","C_pi0_servolabel0.5_drop30_e120.pt","runs/egvla2_fm/deploy/C_pi0_servolabel0.5_drop30_e120.pt","서보 라벨 λ=0.5","120ep, 300ep","","시험"),
 E("EG2-09","G2","A3 서보 라벨 λ=1.0","C_pi0_servolabel1.0_drop30_e120.pt","runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt","서보 라벨 λ=1.0","120ep, 300ep","","현재 배포","실기 로그 중 11/04 253건은 pos4~6×G6~G8 수집(EG1과 프로토콜 다름)"),
 E("EG2-10","G2","299건 버전 A/B2/C","", "runs/egvla2_fm_299/","A_mlp_fm·B2·C_pi0_adaln","299건(29,062프레임)","","보존","300번째 반영 이전의 학습 결과를 보존한 폴더(계획서 §119). 서빙에 쓰이는지는 미확인",key="__none__"),
 E("EG3-01","G3","E2E LoRA 스모크","e2e_vla_florence2_lora_latest.pt","runs/egvla2_e2e_lora/e2e_vla_florence2_lora_latest.pt","Florence-2 LoRA r=8(0.72%)+PiZeroFM","25ep, 15에피소드","in-sample 2.82°, 미학습 4개 15.78°","보류","412.95ms. 학습 스크립트 기본값 --num_episodes 15로 돌린 스모크(300으로 확장하지 않은 이유는 기록 없음)",key="__none__"),
 E("EG3-02","G3","A_frozen_fused","","docs/v5/egvla2_fm/e2e_general.json","frozen 융합 특징","300에피소드 6프레임 간격 약 4,860장, 3분할(fold0·lopo_center·lopo_strong_right)","held-out 방향 19.6°(3분할 평균)","결과만",key="__none__"),
 E("EG3-03","G3","B0 DaViT frozen","","docs/v5/egvla2_fm/e2e_general.json","인코더 고정 120ep/20ep","300에피소드 6프레임 간격 약 4,860장, 3분할","29.6° / 32.7°","결과만",key="__none__"),
 E("EG3-04","G3","B1 LoRA 마지막 단계","","docs/v5/egvla2_fm/e2e_general.json","LoRA 마지막 단계 20ep","300에피소드 6프레임 간격 약 4,860장, 3분할","27.7°","결과만",key="__none__"),
 E("EG3-05","G3","B2 LoRA 전 단계 384px","","docs/v5/egvla2_fm/e2e_fullstage.json","LoRA 전 단계 8ep","300에피소드 6프레임 간격 약 4,860장, 3분할","30.9°, 라이브 36.1°","결과만",key="__none__"),
]
_A3C = "A3 레시피(λ=1.0, 드롭아웃 30%, 120ep, 300에피소드) + 이미지 특징 처리, 시드 0. 목표=가운데 위치(plan_20261115_a3_improvement S3)"
SPEC += [
 E("EG2-11","G2","A3C-F30 (특징 30% 0으로)","A3C_F30_s0.pt","runs/egvla2_fm/a3c/A3C_F30_s0.pt","adaln + 이미지 특징을 30% 확률로 0",_A3C,
   "의자·가운데 이동 0%(A3 0%), 바구니·가운데 64%(A3 80%), 화분·가운데 77%(A3 92%), 11/13 세션 끝 정지 20%(A3 32%)","폐기(오프라인 기각)","의자 문제를 못 고치고 이동만 줄어듦(시드 0 하나뿐 — 시드 0은 이 장면에서 원래 안 움직이는 시드). 재생 지표이며 성공률 아님",key="__none__"),
 E("EG2-12","G2","A3C-F50 (특징 50% 0으로)","A3C_F50_s0.pt","runs/egvla2_fm/a3c/A3C_F50_s0.pt","adaln + 이미지 특징을 50% 확률로 0",_A3C,
   "의자·가운데 이동 2%, 바구니·가운데 61%, 화분·가운데 74%, 의자·오른쪽 68%(A3 80%)","폐기(오프라인 기각)","F30과 같은 결론",key="__none__"),
 E("EG2-13","G2","A3C-S30 (특징 30% 다른 프레임으로 교체)","A3C_S30_s0.pt","runs/egvla2_fm/a3c/A3C_S30_s0.pt","adaln + 이미지 특징을 30% 확률로 같은 배치 다른 프레임 것으로 교체",_A3C,
   "의자·가운데 이동 100%(A3 0%), 바구니·가운데 100%, 화분·가운데 100%. 단 11/13 세션 끝 3프레임 정지 5%(A3 32%), T7 A3 기록 80세션 끝 정지 0.4%(A3 5%)","보류(정지 상실, 서빙 안 함)","가운데는 풀리나 도착 정지를 잃음(시드 0 하나뿐이라 시드 편차와 구분 안 됨). 서빙 후처리가 보완하는지는 미검증",key="__none__"),
]
SPEC.append(E("EG2-14","G2","A3 레시피 시드 1·2 + 앙상블 (A3C-D30)","A3C_D30_s1.pt · A3C_D30_s2.pt","runs/egvla2_fm/a3c/A3C_D30_s1.pt","A3(EG2-09)와 같은 레시피, 시드만 1·2(드롭 30%, λ=1.0, 120ep, 300에피소드). 앙상블 = 시드 0·1·2 첫 명령 평균",_A3C,
  "11/13 의자·가운데 이동 100%(A3 0%), 바구니·가운데 91~98%, T7 A3 기록 452세션: 기록된 A3와 이동/정지 일치 98.9~99.3%(A3 시드0 99.3%), 세션 끝 정지 4~6%(A3 5.5%), 방향 차이 중앙값 0.4~1.0°. 11/13 세션 끝 정지 10~19%(A3 32%, 의자·가운데 제외 시 A3 27% 대 시드1 11%·시드2 21%)",
  "후보: a3-s2 실기 45/45(11/16, 운영자 판정, 비맹검 여부 미확인) · a3-s1·앙상블 실기 없음","박스 가림 0·10·30%의 시드 1·2 여섯 개가 모두 의자·가운데를 움직이고 시드 0만 안 움직임 → 배포 A3(시드 0)의 개별 편차로 보임. 실기(a3-s2): 의자·가운데 5/5, 책상·가운데 5/5이나 바구니·의자에서 A3보다 가까이 가서 끝남(바구니 종료 면적 0.30~0.45 대 A3 0.23, 접촉 보고 6건: 바구니 옆 10회 중, 나머지는 안 부딪힘 1·언급 없음 3). 11/16 프레임 재생: 박스 면적 0.25~0.40에서 정지 판단 a3 41~63% 대 a3-s2 12~22%(a3가 더 일찍 멈춤). 교체 전 추가 검토 필요",key="A3C_D30_s2.pt"))
for pn,nm,ar,of,nt in [("EG4-00","V0 이미지만","A3 레시피","부정 정지 8%, 박스 반대편 시 서보각 20.3°·좌우 9%",""),("EG4-01","V1 고정 문구","","결과 JSON 참고(text_cond_t2)",""),("EG4-02","V2 패러프레이즈","","결과 JSON 참고(text_cond_t2)",""),
    ("EG4-03","V3 +부정 20%","","부정 정지 81.5%",""),("EG4-04","V4 어휘 40+40","","부정 정지 99.5%, 긍정 +3.1pp(기준 +3 미달), C 정상 2.1°·반대편 4.8°",""),
    ("EG4-05","V0R 박스 훼손 증강","","반대편 −12.8°, 정상 +4.6° 악화",""),("EG4-06","V5 토큰 어텐션","MultiheadAttention 768/4head","부정 정지 94.6%(시드 3/9가 81~83%), 같은 색 물체 77.9±37%(9회 중 2회 7~11% 붕괴), 동의 표현 정지 11.2%(V0 7.9, V4 10.4) — V4 대비 이득 없음","")]:
    SPEC.append(E(pn,"G4",nm,"", "docs/v5/egvla2_fm/text_cond_t4.json" if pn!="EG4-06" else "docs/v5/egvla2_fm/text_cond_t5.json",ar,"3분할×시드3, 120ep, 프레임 1/9",of,"오프라인만",nt,key="__none__"))

_F = {"EG4-00": "EG4-00_V0_full", "EG4-01": "EG4-01_V1_full", "EG4-02": "EG4-02_V2_full", "EG4-03": "EG4-03_V3_full", "EG4-04": "EG4-04_V4_full", "EG4-05": "EG4-05_V0R_full", "EG4-06": "EG4-06_V5_full"}
for _r in SPEC:   # T7(11/09~10): 변형 7종을 300에피소드 전체로 저장 학습(시드 0; V0·V4는 시드 1·2도) → 체크포인트·학습 정보 기록
    if _r["pn"] in _F:
        _f = _F[_r["pn"]]; _extra = "" if _r["pn"] not in ("EG4-00", "EG4-04") else "(+시드 1·2 `_s1`·`_s2`)"
        _r.update(ckpt=_f + ".pt", path="runs/egvla2_fm/textcond/" + _f + ".pt", log_key="__none__",
                  train="300에피소드 전체(프레임 1/9 = 3,377), 120ep, 시드 0" + _extra + (" · T2와 학습 조건 다름(T4 캐시 재학습)" if _r["pn"] in ("EG4-01", "EG4-02") else ""),
                  status="오프라인 + 시연 갤러리" + (" + A3 세션 재생(T7)" if _r["pn"] in ("EG4-00", "EG4-04") else "") + ", 실기 미검증")
        if _r["pn"] == "EG4-00": _r["note"] = "T7: A3 세션 452개 재생에서 A3와 3.7~4.6° 차이, 박스 반전 시 좌우 일치 9~12%"
        if _r["pn"] == "EG4-04": _r["note"] = "T7: A3와 2.3~2.8° 차이, 박스 반전 좌우 일치 42~94%(시드 편차 큼), 미학습 부정 문구 정지 99.7~100%"
SPEC.append(E("EG4-07","G4","V4R 박스 훼손 증강(V4+)","EG4-07_V4R_full.pt","runs/egvla2_fm/textcond/EG4-07_V4R_full.pt","V4 + 박스 훼손 증강(드롭 30·지터 20·반전 10·정상 40)","300에피소드 전체(프레임 1/9 = 3,377), 120ep, 시드 0(+시드 1~9 `_s1`~`_s9`)",
  "T7 10시드: A3와 각도 3.17±0.66°·좌우 94.5%, 박스 반전 좌우 94.3%(최저 90.2, 표준편차 1.5%p), 부정 정지 99.5%, 긍정 정지 6.7%","오프라인 + A3 세션 재생(T7) + 서빙 경로 구현·동등성 검증, 실기 미검증",
  "수치는 A3와의 유사도이지 성공률이 아님. 박스를 항상 gray basket으로 찾으므로 다른 물체로는 구동하지 못함(보완 관계, 대체 아님)",key="__none__"))
for _r in SPEC:
    if _r["pn"] == "EG4-04": _r["offline"] = "T7 10시드: A3와 각도 2.80±0.39°·좌우 95.7%, 박스 반전 좌우 86.0%(최저 42, 표준편차 15%p), 부정 정지 99.5%, 긍정 정지 6.4%"
    if _r["pn"] == "EG4-04": _r["note"] = "수치는 A3와의 유사도이지 성공률이 아님. 시드 편차가 큼(박스 반전 좌우 일치 최저 42%) → V4R이 이 편차를 줄임. 박스는 gray basket 고정이라 다른 물체로는 구동하지 못함"
    if _r["pn"] in ("EG4-04", "EG4-07"): _r["status"] = _r["status"].replace("실기 미검증", "서빙 경로 구현·동등성 검증, 실기 미검증") if "서빙" not in _r["status"] else _r["status"]
SHORT = {  # 고정 명명 규칙: <main>[-<modifier>…][-s<seed>], 영어 소문자·하이픈, 메인 모델명이 항상 맨 앞. 시드는 0이 아닐 때만 -s<N>
 "EG0-01": "kosmos-v5-exp11", "EG0-02": "exp14-step2", "EG1-00": "eg1-unrecorded", "EG1-01": "eg1-owl-holdaware", "EG1-02": "eg1-baseline", "EG1-03": "eg1-f2-grounder", "EG1-04": "eg1-f2-phrase-grounder",
 "EG2-01": "eg2-mlp-fm", "EG2-02": "eg2-film-res-fm", "EG2-03": "eg2-pi0-adaln", "EG2-04": "eg2-mlp-regression", "EG2-05": "a3-no-servo-drop0", "EG2-06": "a3-no-servo", "EG2-07": "a3-e1b-tuned", "EG2-08": "a3-servo0.5", "EG2-09": "a3", "EG2-10": "eg2-299ep",
 "EG2-11": "a3-featdrop30", "EG2-12": "a3-featdrop50", "EG2-13": "a3-featswap30", "EG2-14": "a3-s1 · a3-s2 · a3-ens3",
 "EG3-01": "e2e-lora-smoke", "EG3-02": "e2e-frozen-fused", "EG3-03": "e2e-davit-frozen", "EG3-04": "e2e-lora-last-stage", "EG3-05": "e2e-lora-all-stages-384",
 "EG4-00": "v0", "EG4-01": "v1-fixed-phrase", "EG4-02": "v2-paraphrase", "EG4-03": "v3-neg20", "EG4-04": "v4", "EG4-05": "v0r", "EG4-06": "v5-token-attn", "EG4-07": "v4r",
}
RS = "research_story.html"; VO = "vlaops_dashboard.html"
LINKS = {  # PN -> [(라벨, v5 기준 상대 URL)]
 "EG1-02": [("CH70 헤드 구조·손실 실험(exp73, epoch 300 언급)", RS+"#ch70"), ("CH78 ⑥-2 EGVLA1 vs EGVLA2(276/300)", RS+"#ch78"), ("CH77 EGVLA1 비교", RS+"#ch77"), ("CH76 ④ V6 학습 데이터", RS+"#ch76")],
 "EG2-01": [("CH77 학습·오프라인(FM-MLP)", RS+"#ch77")], "EG2-02": [("CH77 (FM-FiLM)", RS+"#ch77")], "EG2-03": [("CH77 (FM-π0)", RS+"#ch77")],
 "EG2-04": [("CH77 MLP 회귀 대조", RS+"#ch77"), ("CH78 실기 비교군", RS+"#ch78")], "EG2-05": [("CH77", RS+"#ch77"), ("CH79", RS+"#ch79")],
 "EG2-06": [("CH79 폐루프(10/22~28, A1)", RS+"#ch79")], "EG2-07": [("CH79", RS+"#ch79")], "EG2-08": [("CH79 10/28 비교(A2)", RS+"#ch79")],
 "EG2-09": [("CH79 A3 채택·10/29·11/04 일반화", RS+"#ch79"), ("VLAops", VO)],
 "EG3-01": [("CH79 E2E 일반 비교", RS+"#ch79")], "EG3-02": [("CH79 E2E 일반 비교", RS+"#ch79")], "EG3-03": [("CH79 E2E 일반 비교", RS+"#ch79")], "EG3-04": [("CH79 E2E 일반 비교(마지막 단계 LoRA)", RS+"#ch79")], "EG3-05": [("CH79 E2E 일반 비교(전 단계 LoRA@384)", RS+"#ch79")],
}
for _pn in ("EG4-00","EG4-01","EG4-02","EG4-03","EG4-04","EG4-05","EG4-06","EG4-07"):
    LINKS[_pn] = [("CH80 텍스트 조건화 E2E", RS+"#ch80"), ("VLAops 🔤 문구·표현별 테스트", VO)]
for _pn in ("EG2-11","EG2-12","EG2-13","EG2-14"): LINKS[_pn] = [("A3 개선 계획서·결과(7절)", "../plans/plan_20261115_a3_improvement.md")]
for _pn in ("EG4-04","EG4-07"): LINKS[_pn].append(("CH81 종합 브리핑(서빙·11/13 실험)", RS+"#ch81"))
LINKS["EG2-09"].append(("CH81 11/13 물체 실험 65건", RS+"#ch81")); LINKS["EG2-09"].append(("11/13 재생 페이지", "object_test_1113.html"))
DATASETS = [  # id, 이름, 경로, 설명, 사용 모델(PN), 결과 링크
 ("DS-01","EGVLA2 학습셋 300건","ROS_action/mobile_vla_dataset_egvla2","10/13~10/16 수집(디렉터리 H5 371개 중 학습 세트 300건 = meta/egvla2_v3_main.json, 29,161프레임). 속성: 시작 위치·경로 라벨","EG2-01~09, EG3-*, EG4-*", [("CH76 수집 점검", RS+"#ch76"),("CH77", RS+"#ch77")]),
 ("DS-02","V5 데이터셋(EGVLA1 학습)","ROS_action/mobile_vla_dataset_v5","h5 515개 로컬 확인. CH76 ④: V6 학습 데이터, minum 쪽이 유일 원본(로봇 쪽 복사본은 정리됨, 별도 백업 없음). CLAUDE.md의 V5 150건은 이 중 일부로 보이나 미확인","EG1-02(사용자 확인), EG0-*(추정)", [("CH76 ④ 데이터 보관 상태", RS+"#ch76")]),
 ("DS-03","폐루프 드라이런 10/22·10/28","ROS_action/dryrun_closedloop_20261022, ..._20261028","EGVLA2 폐루프 드라이런 원본(각각 657MB, 2.1GB)","EG2-06, EG2-08", [("CH79", RS+"#ch79")]),
 ("DS-04","시작 위치 드라이런","ROS_action/dryrun_startpos","시작 위치 재생용 드라이런","EG2-*", [("CH78 시작 위치 재생", RS+"#ch78")]),
 ("DS-05","실기 세션 H5(10/22~10/29)","docs/inference_sessions/session_20261022~29_*.h5","A3·adaln·서보라벨 실기 세션(10/22 19, 10/28 102, 10/29 96)","EG2-06, EG2-08, EG2-09", [("CH79 📅 10/29", RS+"#ch79")]),
 ("DS-06","11/04·11/05 수집 297건","docs/inference_sessions/session_20261104_*.h5, session_20261105_*.h5","pos4~6 × G6~G8 × 3궤적 × 11. 11/04 252 + 11/05 45(로그 행 없음)","EG2-09", [("CH79 📆 11/04", RS+"#ch79")]),
 ("DS-07","텍스트 어휘·특징 캐시","runs/egvla2_fm/text_vocab_cache.npz, text_vocab_tokens.npz","T4/T5용 문구 116개의 융합 인코더 특징(stride 9, 프레임 약 3.2k)","EG4-*", [("VLAops T4·T5", VO)]),
 ("DS-08","EGVLA1 실기 세션 92%(8/7~9/19)","/home/minum/MoNaVLA/inference_sessions_recv/20260807/h5 (100) + 20261013_missing_backfill (200)","이 서버(minum)에 있음: episode_log의 300개 session_id 전부 H5 존재(300/300), 모델명 stage2_v2, 종료 manual_stop. 저장소 밖 경로(14GB 수신 폴더 일부). 백업 위치는 별도 확인 필요","EG1-02", []),
]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

def robot():
    try:
        txt = subprocess.check_output(["git", "show", "origin/monavla-driving:logs/episode_log.csv"], cwd=ROOT).decode("utf-8")
    except Exception as e:
        return {}, f"episode_log 읽기 실패: {e}"
    g = collections.defaultdict(list)
    for x in csv.DictReader(io.StringIO(txt)): g[x["체크포인트"]].append(x)
    out = {}
    for k, v in g.items():
        ds = sorted(x["날짜"] for x in v if x["날짜"]); ok = sum(x["결과"] == "성공" for x in v)
        out[k] = dict(n=len(v), ok=ok, rate=round(ok / len(v) * 100, 1), first=ds[0][:10] if ds else None, last=ds[-1][:10] if ds else None)
    # 11/16 a3-s2 실기 45세션: origin episode_log에 아직 없어 로봇 서버에서 받은 슬라이스(docs/v5/egvla2_fm/a3c/real_1116_*)에서 병합
    try:
        sl = list(csv.DictReader(open(D / "egvla2_fm/a3c/real_1116_episode_log_slice.csv", encoding="utf-8")))
        if sl:
            ds = sorted(x["날짜"] for x in sl); ok = sum(x["결과"] == "성공" for x in sl); out["A3C_D30_s2.pt"] = dict(n=len(sl), ok=ok, rate=round(ok / len(sl) * 100, 1), first=ds[0][:10], last=ds[-1][:10])
    except Exception:
        pass
    return out, None

SESS_ROOTS = [ROOT / "docs/inference_sessions", pathlib.Path("/home/minum/MoNaVLA/inference_sessions_recv")]
def h5_index():
    idx = set()
    for r in SESS_ROOTS:
        if r.exists(): idx |= {p.name[8:-3] for p in r.rglob("session_*.h5")}
    return idx

def sessions():
    """checkpoint -> 세션 목록(날짜·결과·로컬 H5 유무·필름스트립 유무). docs/v5/egvla2_fm/model_sessions.json 저장."""
    try:
        txt = subprocess.check_output(["git", "show", "origin/monavla-driving:logs/episode_log.csv"], cwd=ROOT).decode("utf-8")
    except Exception:
        return {}
    fs = set()
    for f in ("filmstrips1029.json", "filmstrips1104.json"):
        try: fs |= {x["sid"] for x in json.load(open(D / "egvla2_fm" / f))}
        except Exception: pass
    out = collections.defaultdict(list); hx = h5_index()
    for x in csv.DictReader(io.StringIO(txt)):
        sid = x["session_id"]
        out[x["체크포인트"] or "(공란)"].append(dict(sid=sid or None, date=x["날짜"][:16], ok=x["결과"] == "성공", route=x["경로"], steps=x["steps"],
            h5=bool(sid) and sid in hx, film=sid in fs))
    # 11/13 물체 실험 65세션: origin episode_log에 아직 없어 objtest_data.json(H5 기준)에서 병합
    try:
        OB = json.load(open(D / "egvla2_fm" / "objtest_data.json"))
    except Exception:
        OB = []
    a3 = "C_pi0_servolabel1.0_drop30_e120.pt"; have = {x["sid"] for x in out.get(a3, [])}
    for x in OB:
        sid = x["sid"]
        if sid in have: continue
        out[a3].append(dict(sid=sid, date=f"{sid[:4]}-{sid[4:6]}-{sid[6:8]} {sid[9:11]}:{sid[11:13]}", ok=x["res"] == "성공", route=x["route"], steps=str(x["n"]), h5=sid in hx, film=False,
                            obj=x["obj"], pos=x["pos"], ins=x["ins"], exp="1113", res=x["res"]))
    try:
        sl = list(csv.DictReader(open(D / "egvla2_fm/a3c/real_1116_episode_log_slice.csv", encoding="utf-8"))); js = json.load(open(D / "egvla2_fm/a3c/real_1116_sessions_bbox_area.json"))
        key = "A3C_D30_s2.pt"; have2 = {x["sid"] for x in out.get(key, [])}
        for x in sl:
            sid = x["session_id"]
            if sid in have2: continue
            ins = js.get(sid, {}).get("instruction", "")
            out[key].append(dict(sid=sid, date=x["날짜"][:16], ok=x["결과"] == "성공", route=x["경로"], steps=x["steps"], h5=sid in hx, film=False, obj={"go to the gray basket": "바구니", "go to the brown wooden chair": "의자", "go to the small desk": "책상"}.get(ins, ""),
                                pos={"trackF_center": "가운데", "trackA_weak_right": "오른쪽", "trackA_weak_left": "왼쪽"}.get(x["경로"], ""), ins=ins, exp="1116", res=x["결과"]))
    except Exception:
        pass
    return out

def main():
    rb, err = robot(); ss = sessions(); rows = []
    _o = [x for x in ss.get('C_pi0_servolabel1.0_drop30_e120.pt', []) if x.get('exp') == '1113']; OBJ113 = (sum(x['ok'] for x in _o), len(_o)) if _o else None
    for e in SPEC:
        r = dict(e); p = ROOT / e["path"] if e["path"] else None
        r["exists"] = bool(p and p.exists()); r["bytes"] = p.stat().st_size if r["exists"] and p.is_file() and p.suffix in (".pt", ".ckpt") else None
        r["sha256"] = sha(p) if r["bytes"] and r["bytes"] < 5e8 and p.suffix in (".pt", ".ckpt") else None
        r["robot"] = rb.get(e["log_key"]) if e["log_key"] not in (None, "__none__") else None
        r["links"] = LINKS.get(e["pn"], []); r["short"] = SHORT[e["pn"]]
        sl = ss.get(e["log_key"] or "(공란)", []) if e["log_key"] not in (None, "__none__") else []
        r["sessions"] = dict(n=len(sl), h5_local=sum(x["h5"] for x in sl), film=sum(x["film"] for x in sl)) if sl else None
        rows.append(r)
    (D / "egvla2_fm" / "model_sessions.json").write_text(json.dumps(ss, ensure_ascii=False), encoding="utf-8")
    (D / "MODEL_REGISTRY.json").write_text(json.dumps(dict(generated=datetime.datetime.now().isoformat(timespec="minutes"), gens={k: list(v) for k, v in GENS.items()}, datasets=[dict(id=d[0], name=d[1], path=d[2], desc=d[3], used_by=d[4], links=d[5]) for d in DATASETS], models=rows, robot_log_error=err), ensure_ascii=False, indent=1), encoding="utf-8")
    esc = html.escape
    def links_cell(r):
        a = "<br>".join(f'<a href="{u}">{esc(l)}</a>' for l, u in r["links"])
        se = r["sessions"]
        if se: a += f'<br><span class=m>세션 {se["n"]}건 · 로컬 H5 {se["h5_local"]} · 필름스트립 {se["film"]}</span> <a href="#ses-{r["pn"]}">목록</a>'
        return a
    def cell(r):
        rr = r["robot"]
        if rr: return f'<b>{rr["ok"]}/{rr["n"]} = {rr["rate"]}%</b><br><span class=m>{rr["first"]}~{rr["last"]}</span>'
        return '<span class=m>없음</span>'
    esc = html.escape
    def wilson(k, n, z=1.96):
        p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; w = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / d
        return round((c - w) * 100), round((c + w) * 100)
    def lvl(r):
        if r["pn"] == "EG2-09": return "deploy", "현재 배포"
        if r["robot"]: return "real", "실기 기록 있음"
        if any(w in r["status"] for w in ("폐기", "보류", "추적 불가", "결과만")): return "retired", "폐기·보류·기록만"
        return "off", "오프라인만"
    for r in rows: r["level"], r["level_name"] = lvl(r)
    cnt = {gk: sum(r["gen"] == gk for r in rows) for gk in GENS}
    try: OB = json.load(open(D / "egvla2_fm" / "objtest_summary.json"))
    except Exception: OB = None
    css = """:root{--bg:#fff;--fg:#1d2330;--mut:#5b6577;--line:#d9dee7;--card:#f7f9fc;--head:#eef1f6;--acc:#1f5fbf;--ok:#1b7f4b;--okbg:#e3f4ea;--warn:#a55a00;--warnbg:#fdf0dc;--bad:#b3322e;--badbg:#fbe6e5;--gray:#5b6577;--graybg:#edf0f4;--blue:#1f5fbf;--bluebg:#e4edfb}
@media(prefers-color-scheme:dark){:root{--bg:#14171d;--fg:#e4e8ef;--mut:#9aa4b5;--line:#2c3340;--card:#1a1e26;--head:#1d222b;--acc:#7aa7ee;--ok:#5cc489;--okbg:#17301f;--warn:#e0a050;--warnbg:#33260f;--bad:#ef8a85;--badbg:#3a1c1b;--gray:#9aa4b5;--graybg:#232832;--blue:#7aa7ee;--bluebg:#1c2a44}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,'Noto Sans KR',sans-serif}main{max-width:1320px;margin:0 auto;padding:18px 16px 70px}
h1{font-size:1.55rem;margin:.2rem 0}h2{margin:2.2rem 0 .3rem;font-size:1.2rem;scroll-margin-top:120px}h2 .n{color:var(--mut);font-weight:400;font-size:.9rem}p.d{margin:.2rem 0 .8rem;color:var(--mut);font-size:.88rem}
a{color:var(--acc)}code{font-size:.8em;background:var(--head);padding:1px 5px;border-radius:3px;word-break:break-all}.m{color:var(--mut);font-size:.8rem}
.note{border-left:4px solid var(--warn);background:var(--warnbg);padding:.5rem .9rem;margin:.8rem 0;font-size:.88rem;border-radius:0 6px 6px 0}
.bar{position:sticky;top:0;z-index:20;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;margin:0 -16px;padding-left:16px;padding-right:16px;display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center}
.bar a.g,.chip{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:3px 11px;border-radius:99px;font-size:.82rem;text-decoration:none;cursor:pointer;user-select:none}
.chip[aria-pressed=true]{background:var(--acc);color:var(--bg);border-color:var(--acc)}.bar input{padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);width:220px;max-width:100%;margin-left:auto}
.sep{width:1px;height:20px;background:var(--line)}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;margin:.8rem 0}.fl{flex:1 1 170px;border:1px solid var(--line);border-radius:10px;padding:9px 12px;background:var(--card);text-decoration:none;color:var(--fg)}
.fl b{display:block;font-size:.95rem}.fl.cur{border:2px solid var(--ok);background:var(--okbg)}.fl .t{font-size:.76rem;color:var(--mut)}.ar{align-self:center;color:var(--mut)}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:.8rem 0}.k{border:1px solid var(--line);border-radius:10px;padding:10px 14px;background:var(--card)}.k .v{font-size:1.5rem;font-weight:700;line-height:1.2}.k .l{font-size:.78rem;color:var(--mut)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(370px,1fr));gap:12px}
.card{border:1px solid var(--line);border-radius:12px;background:var(--card);padding:12px 14px;display:flex;flex-direction:column;gap:8px;scroll-margin-top:120px}
.card.deploy{border:2px solid var(--ok)}.card.hide{display:none}
.top{display:flex;gap:8px;align-items:flex-start;justify-content:space-between}.sn{font:700 1.08rem/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--fg);margin-top:1px}.sn:empty{display:none}.pn{font-weight:700;color:var(--acc);font-size:.8rem;white-space:nowrap}.nm{font-weight:700;font-size:1rem;line-height:1.35}
.bd{display:inline-block;padding:2px 9px;border-radius:99px;font-size:.72rem;font-weight:600;white-space:nowrap}
.bd.deploy,.bd.real{background:var(--okbg);color:var(--ok)}.bd.off{background:var(--bluebg);color:var(--blue)}.bd.retired{background:var(--graybg);color:var(--gray)}.bd.srv{background:var(--warnbg);color:var(--warn)}
.mx{display:grid;grid-template-columns:1fr 1fr;gap:8px}.mb{border:1px solid var(--line);border-radius:8px;padding:6px 9px;background:var(--bg);font-size:.8rem;min-height:3.6rem}.mb .h{font-size:.68rem;font-weight:700;color:var(--mut);letter-spacing:.03em}
.mb.rl b{font-size:1.15rem;color:var(--ok)}.x13{margin-top:6px;border:2px solid var(--warn);background:var(--warnbg);border-radius:8px;padding:6px 9px}.h13{font-size:.72rem;font-weight:700;color:var(--warn);margin-bottom:3px}.b13{display:grid;grid-template-columns:4.6rem 1fr auto;gap:6px;align-items:center;font-size:.8rem;margin:2px 0}.pb2{height:10px;background:var(--bg);border:1px solid var(--line);border-radius:5px;overflow:hidden}.pb2 i{display:block;height:100%;background:var(--warn)}.pb{height:6px;background:var(--graybg);border-radius:4px;margin:3px 0;overflow:hidden}.pb i{display:block;height:100%;background:var(--ok)}
.mb.na{color:var(--mut)}dl{margin:0;display:grid;grid-template-columns:4.2rem 1fr;gap:3px 8px;font-size:.82rem}dt{color:var(--mut)}dd{margin:0}
.lk{display:flex;flex-wrap:wrap;gap:5px}.lk a{font-size:.74rem;border:1px solid var(--line);border-radius:6px;padding:1px 7px;text-decoration:none;background:var(--bg)}
details{font-size:.8rem}summary{cursor:pointer;color:var(--mut)}details.ses{border:1px solid var(--line);border-radius:8px;padding:4px 10px;margin:.5rem 0;background:var(--card)}
table{border-collapse:collapse;width:100%;display:block;overflow-x:auto;font-size:.82rem}th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{background:var(--head)}.no{color:var(--bad)}
.leg{display:flex;flex-wrap:wrap;gap:8px 16px;font-size:.8rem;margin:.4rem 0}.pipe{list-style:none;margin:.5rem 0;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px}
.pipe li{border:1px solid var(--line);border-radius:10px;padding:8px 12px;background:var(--card);font-size:.84rem}.pipe .s{font-weight:700;display:block}.pipe .done{border-left:5px solid var(--ok)}.pipe .wait{border-left:5px solid var(--warn)}.pipe .open{border-left:5px solid var(--bad)}
@media(max-width:560px){.grid{grid-template-columns:1fr}.mx{grid-template-columns:1fr}}"""
    # ── 구성 요소 ──
    def extra113(r):
        if not (r['pn'] == 'EG2-09' and OBJ113): return ''
        def bar(lbl, k, n):
            pc = round(k / n * 100)
            return '<div class=b13><span>%s</span><div class=pb2><i style="width:%d%%"></i></div><b>%d/%d · %d%%</b></div>' % (lbl, pc, k, n, pc)
        sd, ct = (OB["side"], OB["center"]) if OB else (None, None)
        sub = (bar("옆(좌·우)", sd[1], sd[0]) + bar("가운데", ct[1], ct[0])) if OB else ""
        return ('<div class=x13><div class=h13>11/13 물체 실험 · 별도 집계(위 455와 합치지 않음)</div>' + bar("전체", OBJ113[0], OBJ113[1]) + sub +
                '<div class=m>지시어·물체·위치를 바꾼 실험, 운영자 수동 판정. 옆 40 + 가운데 24 = 64세션, 나머지 1세션(화분)은 CSV 행이 없어 성공 아님으로 집계</div></div>')
    def metric_real(r):
        rr = r["robot"]
        if rr: return f'<div class="mb rl"><div class=h>실기(운영자 판정 로그)</div><b>{rr["ok"]}/{rr["n"]}</b> <span class=m>{rr["rate"]}%</span><div class=pb><i style="width:{rr["rate"]}%"></i></div><span class=m>{rr["first"]} ~ {rr["last"]}</span>{extra113(r)}</div>'
        return '<div class="mb na"><div class=h>실기</div>기록 없음</div>'
    def card(r):
        badges = f'<span class="bd {r["level"]}">{r["level_name"]}</span>'
        if r["pn"] in ("EG4-04", "EG4-07"): badges += ' <span class="bd srv">서빙 준비(soda 전달 대기)</span>'
        off = r["offline"] or "—"
        rows_dl = [("구조", r["arch"]), ("학습", r["train"]), ("상태", r["status"]), ("비고", r["note"])]
        dl = "".join(f"<dt>{k}</dt><dd>{esc(v)}</dd>" for k, v in rows_dl if v)
        ck = (f'<code>{esc(r["ckpt"])}</code> ' if r["ckpt"] else "") + (f'<span class=m>{round(r["bytes"]/1e6,1)}MB</span> ' if r["bytes"] else "") + (f'<code>sha {r["sha256"][:10]}</code>' if r["sha256"] else "")
        miss = '<span class=no>파일 없음</span> ' if r["path"] and not r["exists"] else ""
        det = f'<details><summary>파일·경로</summary>{miss}{ck}<br><span class=m>{esc(r["path"] or "경로 없음")}</span></details>'
        se = r["sessions"]; ses = f'<span class=m>세션 {se["n"]}건 · 로컬 H5 {se["h5_local"]} · 필름스트립 {se["film"]}</span> <a href="#ses-{r["pn"]}">목록</a>' if se else ""
        lk = "".join(f'<a href="{u}">{esc(l)}</a>' for l, u in r["links"])
        return (f'<article class="card {r["level"]}" id="{r["pn"]}" data-lv="{r["level"]}" data-q="{esc((r["pn"]+" "+r["short"]+" "+r["name"]+" "+(r["ckpt"] or "")+" "+r["arch"]).lower())}">'
                f'<div class=top><div><span class=pn>{r["pn"]}</span><div class=sn>{"★ " if r["robot"] else ""}{esc(r["short"])}</div><div class=nm>{esc(r["name"])}</div></div><div>{badges}</div></div>'
                f'<div class=mx><div class=mb><div class=h>오프라인</div>{esc(off)}</div>{metric_real(r)}</div><dl>{dl}</dl>{det}'
                f'<div>{ses}</div><div class=lk>{lk}</div></article>')
    body = [f'<h1>모델 세대 레지스트리</h1><p class=d>생성 {datetime.datetime.now():%Y-%m-%d %H:%M} · 출처: MANIFEST, episode_log.csv(origin/monavla-driving), docs/v5/egvla2_fm/*.json · PN = EG{{세대}}-{{순번}}(제안안) · <a href="research_story.html#ch81">CH81 종합 브리핑</a></p>']
    nreal = sum(r["level"] in ("real", "deploy") for r in rows)
    kp = [(f'{len(rows)}', "등록 모델"), (f'{nreal}', "실기 기록이 있는 모델"), ("A3 = EG2-09", "현재 배포(실기 455/455 기록, 운영자 판정)"), ("V4 · V4R", "텍스트 조건화 — 서빙 경로 구현, 실기 미검증")]
    body.append('<div class=kpi>' + "".join(f'<div class=k><div class=v>{a}</div><div class=l>{b}</div></div>' for a, b in kp) + '</div>')
    fl = []
    for gk, (gn, gd) in GENS.items():
        cur = " cur" if gk == "G2" else ""
        fl.append(f'<a class="fl{cur}" href="#gen-{gk}"><b>{gk} · {gn}</b><span class=t>{cnt[gk]}개{" · 현재 배포 세대" if gk=="G2" else ""}</span></a>')
    body.append('<div class=flow>' + '<span class=ar>→</span>'.join(fl[:3]) + '<span class=ar>⇢</span>' + fl[3] + fl[4] + '</div>')
    body.append('<div class=note><b>읽는 법</b> · <b>실기</b>는 운영자 수동 판정(블라인드 아님, 성공 기준 문서화 안 됨, 좌우 정렬 미반영)이고, <b>오프라인</b> 수치는 세대마다 평가 방식이 달라 서로 비교하지 않습니다. 세대를 잇는 비교는 같은 프로토콜의 실기뿐입니다. V4·V4R의 재생 수치는 A3와의 <i>유사도</i>이며 성공률이 아닙니다.'
                '<div class=leg><span><span class="bd deploy">현재 배포</span> 서버에 올라가 있음</span><span><span class="bd real">실기 기록 있음</span></span><span><span class="bd off">오프라인만</span> 실기 기록 없음</span><span><span class="bd retired">폐기·보류·기록만</span></span><span><span class="bd srv">서빙 준비</span> 코드 구현·검증 완료, 로봇 전달 전</span><span><b>★</b> 이름 앞: 실기 기록이 있는 모델</span></div></div>')
    nm_rows = "".join("<tr><td>%s</td><td><code>%s</code></td><td>%s</td><td>%s</td></tr>" % (r["pn"], esc(r["short"]), ("★ %d/%d" % (r["robot"]["ok"], r["robot"]["n"])) if r["robot"] else "—", esc(r["name"])) for r in rows)
    body.append('<h2 id=naming>모델 이름 규칙 (고정)</h2><p class=d><code>&lt;main&gt;[-&lt;modifier&gt;…][-s&lt;seed&gt;]</code> · 영어 소문자와 하이픈 · <b>메인 모델명이 항상 맨 앞</b> · 수식어는 main과 다른 점만(예: <code>a3-no-servo</code> = a3에서 서보 라벨을 뺀 것) · 시드는 0이 아닐 때만 <code>-s1</code> · ★ = 실기 기록이 있는 모델</p>'
                '<details><summary>전체 이름표 (PN ↔ 고정 이름 ↔ 실기)</summary><table><thead><tr><th>PN</th><th>고정 이름</th><th>실기(로그)</th><th>기존 표기</th></tr></thead><tbody>' + nm_rows + '</tbody></table></details>')
    # 상태 패널: V4/V4R 서빙
    body.append('<h2 id=status>현재 진행 상태 · V4/V4R 서빙</h2><p class=d>출처: <code>docs/plans/plan_20261115_v4_soda_minimal_patch.md</code>. 로봇 구동 단계는 단계마다 별도 승인.</p><ul class=pipe>'
        '<li class=done><span class=s>✔ 서빙 경로 구현</span>VLA_EGVLA2_TEXT=1(기본 꺼짐), 커밋 a7b3822d</li>'
        '<li class=done><span class=s>✔ 이 서버 동등성(저장소 서버)</span>문구 코사인 최저 0.99989, 각도 중앙값 0.08°(V4)/0.07°(V4R), 정지 일치 100%</li>'
        '<li class=done><span class=s>✔ 최소 패치 승인(A·B)</span>새 파일 1 + 기존 파일 분기 삽입. soda 파일 덮어쓰기 없음</li>'
        '<li class=done><span class=s>✔ soda 파일 복사본 검증</span>패치 적용본(soda 방식 호출)에서 같은 기준으로 재검증</li>'
        '<li class=done><span class=s>✔ soda 전달(11/16 00:38 보고)</span>새 파일 1 + 패치 +6줄 + 체크포인트 4개(v4·v4r·a3-s1·a3-s2) 적용, 해시·py_compile·git diff 확인. 실행 중인 서버는 재시작 전이라 새 코드를 아직 읽지 않음</li>'
        '<li class=wait><span class=s>⏳ 서버 기동 방식</span>동환이 결정(운영서버 위 모델 추가 로드는 Jetson 리부팅 전례로 금지)</li>'
        '<li class=wait><span class=s>⏳ 구동 시험</span>바구니 → 4물체 × G2/G3/G4, V4·V4R 시드 1. 동환 입회, 단계별 승인</li>'
        '<li class=open><span class=s>? 미해명</span>동등성 점검의 박스 중심오차 최대값 9.0(기준 0.02): 일부 프레임에서 박스 유무가 캐시와 다름. 중앙값은 0.001</li></ul>')
    if OB:
        a, s_, c = OB["all"], OB["side"], OB["center"]
        def wl(t): lo, hi = wilson(t[1], t[0]); return f'{t[1]}/{t[0]} ({round(t[1]/t[0]*100)}%, 95% 구간 {lo}~{hi}%)'
        body.append('<h2 id=obj>11/13 물체 실험 (A3, 지시어 변경)</h2><p class=d>출처: 동환 실기 65세션(H5)과 episode_log 슬라이스. 지시어 형태가 물체·위치와 겹쳐 있어 분리 불가. 운영자 수동 판정.</p>'
            f'<div class=kpi><div class=k><div class=v>{wl(a)}</div><div class=l>전체</div></div><div class=k><div class=v>{wl(s_)}</div><div class=l>옆(좌·우)</div></div><div class=k><div class=v>{wl(c)}</div><div class=l>가운데(CSV 행 없는 화분 1세션 제외)</div></div></div>'
            '<p><a href="object_test_compare_1113.html"><b>원본 실기·4개 모델 동시 비교 재생</b></a> · <a href="object_test_compare_1113_standalone.html">단독 실행판(프레임 내장)</a> · <a href="object_test_1113.html">재생 페이지</a> · <a href="research_story.html#ch81">CH81 분석(실패 사례·A3 대비 탭)</a></p>')
    # 필터 바
    chips = "".join(f'<span class=chip role=button tabindex=0 aria-pressed="{"true" if k=="all" else "false"}" data-f="{k}">{t}</span>' for k, t in (("all", "전체"), ("deploy", "현재 배포"), ("real", "실기 기록"), ("off", "오프라인만"), ("retired", "폐기·보류")))
    navg = "".join(f'<a class=g href="#gen-{gk}">{gk} {gn}</a>' for gk, (gn, gd) in GENS.items())
    body.append(f'<div class=bar>{navg}<span class=sep></span>{chips}<input id=q placeholder="PN·이름·체크포인트 검색"></div>')
    for gk, (gn, gd) in GENS.items():
        rs = [r for r in rows if r["gen"] == gk]
        body.append(f'<h2 id="gen-{gk}">{gk} · {gn} <span class=n>{len(rs)}개</span></h2><p class=d>{gd}</p><div class=grid>' + "".join(card(r) for r in rs) + '</div>')
    EXT = [  # 외부 모델(참고). 값은 논문·요약에서 확인한 것만, 못 찾은 칸은 미확인.
      ("우리 A3 (EG2-09)", "모바일 로봇, 바구니 주행", "Florence-2", "박스를 만들고 FM 헤드가 연속 명령", "있음(미분 끊김)", "300에피소드", "—", "—"),
      ("OpenVLA", "로봇 팔 조작", "Llama 2 + DINOv2·SigLIP, 7B", "행동 차원별 256구간 토큰", "없음", "약 97만 에피소드(Open X-Embodiment)", "LoRA 미세조정(파라미터 약 1.4%)", "https://arxiv.org/abs/2406.09246"),
      ("π0", "범용 로봇 조작", "PaliGemma 3B + 액션 전문가(약 300M)", "flow matching 연속 청크(50스텝), 최대 50Hz", "없음", "미확인", "우리 헤드(AdaLN-Zero FM)가 이 계열. 다만 π0는 VLM과 함께 학습, 우리는 Florence-2를 고정", "https://arxiv.org/abs/2410.24164"),
      ("NaVILA", "다리 로봇 언어 지시 내비게이션", "VLM(세부 미확인)", "언어형 중간 행동(\"75cm 전진\") → 로코모션 RL 정책", "언어형 중간 행동", "미확인", "단일 프레임 입력, LiDAR 높이지도는 2차 요약 기준(미확인)", "https://arxiv.org/abs/2412.04453"),
    ]
    def _src(u): return '<a href="' + u + '">논문</a>' if u.startswith("http") else ""
    er = "".join("<tr><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (esc(e[0]), esc(e[1]), esc(e[2]), esc(e[3]), esc(e[4]), esc(e[5]), esc(e[6]), _src(e[7])) for e in EXT)
    body.append('<h2 id=ext>외부 VLA 모델과 비교 (참고)</h2><p class=d>논문·요약 검색으로 확인한 값만 적었고 확인 못한 칸은 미확인. 데이터 규모가 수백 배 달라 성능을 직접 비교하는 표가 아님. 구조 차이를 보는 용도.</p>'
                f'<table><thead><tr><th>모델</th><th>용도</th><th>백본</th><th>행동 출력</th><th>중간 단계</th><th>학습 데이터</th><th>우리와의 관계</th><th>출처</th></tr></thead><tbody>{er}</tbody></table>')
    def alinks(ls): return " ".join('<a href="' + u + '">' + esc(l) + '</a>' for l, u in ls)
    dsr = "".join(f'<tr><td class=pn>{d[0]}</td><td><b>{esc(d[1])}</b></td><td><code>{esc(d[2])}</code></td><td>{esc(d[3])}</td><td>{esc(d[4])}</td><td>{alinks(d[5])}</td></tr>' for d in DATASETS)
    body.append(f'<h2 id=ds>데이터셋</h2><p class=d>경로는 저장소 기준. 사용 모델은 학습·평가에 쓰인 데이터. 확인 못한 항목은 (추정)·미확인.</p><table><thead><tr><th>ID</th><th>이름</th><th>경로</th><th>설명</th><th>사용 모델</th><th>결과 페이지</th></tr></thead><tbody>{dsr}</tbody></table>')
    sec = []
    for r in rows:
        sl = ss.get(r["log_key"] or "(공란)", []) if r["log_key"] not in (None, "__none__") else []
        if not sl: continue
        items = "".join(f'<tr><td><code>{x["sid"] or "-"}</code></td><td>{x["date"]}</td><td>{esc(x["route"])}</td><td>{"성공" if x["ok"] else ("<span class=m>CSV 행 없음(성공 아님으로 집계)</span>" if x.get("res") == "기록 없음" else "<span class=no>실패</span>")}</td><td>{x["steps"]}</td><td>{"✓" if x["h5"] else "—"}</td><td>{"✓" if x["film"] else ""}</td><td>{esc(x["obj"] + " · " + x["pos"] + " · " + x["ins"]) if x.get("obj") else ""}</td></tr>' for x in sl)
        sec.append(f'<details class=ses id="ses-{r["pn"]}"><summary><b class=pn>{r["pn"]}</b> {esc(r["name"])} — 세션 {len(sl)}건</summary><table><thead><tr><th>session_id</th><th>시각</th><th>경로 라벨</th><th>결과</th><th>steps</th><th>로컬 H5</th><th>필름스트립</th><th>11/13: 물체·위치·지시어</th></tr></thead><tbody>{items}</tbody></table></details>')
    body.append('<h2 id=sess>모델별 실기 세션</h2><p class=d>episode_log의 session_id 기준. 로컬 H5 = 이 서버에 파일이 있는 세션. 필름스트립 = CH79에 사진이 있는 세션(10/29·11/04분). 11/13 세션은 origin 로그에 아직 없음(위 11/13 항목 참고).</p>' + "".join(sec))
    js = """<script>(function(){var f='all',q='';function ap(){var n=0;document.querySelectorAll('.card').forEach(function(c){var ok=(f==='all'||c.dataset.lv===f||(f==='real'&&c.dataset.lv==='deploy'))&&(!q||c.dataset.q.indexOf(q)>=0);c.classList.toggle('hide',!ok);if(ok)n++});
document.querySelectorAll('.grid').forEach(function(g){g.previousElementSibling.style.display=g.querySelector('.card:not(.hide)')?'':'none'})}
document.querySelectorAll('.chip').forEach(function(c){function go(){f=c.dataset.f;document.querySelectorAll('.chip').forEach(function(x){x.setAttribute('aria-pressed',x===c)});ap()}c.onclick=go;c.onkeydown=function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();go()}}});
document.getElementById('q').oninput=function(e){q=e.target.value.toLowerCase();ap()}})()</script>"""
    (D / "model_registry.html").write_text(f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>모델 세대 레지스트리</title><style>{css}</style></head><body><main>{"".join(body)}</main>{js}</body></html>', encoding="utf-8")
    print(len(rows), "models; robot log error:", err)
main()
