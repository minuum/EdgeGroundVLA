"""CH77 ⑥-7: 학습-서빙 일치 대조(EGVLA1 vs EGVLA2), JPEG 품질 영향, 배포 체크포인트·추론 서버 현황. 데이터: serving_equiv*.json, deploy/MANIFEST.json"""
import json

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT


def block(tbl, H, st):
    q = json.load(open(FM / "serving_equiv_jpeg.json"))["results"]; e = json.load(open(FM / "serving_equiv.json"))["results"]
    man = json.load(open(ROOT_MANIFEST)) if ROOT_MANIFEST.exists() else {}
    jq = [(k.split("|")[1].replace("jpeg", "JPEG q"), v["action_angle_diff_mean"], R if v["action_angle_diff_mean"] > 1 else G) for k, v in q.items() if k.startswith("fp16|jpeg")]
    bars = Z.bars(jq, 8, "°")
    dt = tbl(["입력 경로", "최종 방향 차이(평균/최대)", "특징 코사인(평균/최소)", "판정"], [
        ["원본 프레임, fp16 (학습 캐시와 같은 조건)", f"{e['fp16|raw']['action_angle_diff_mean']:.3f}° / {e['fp16|raw']['action_angle_diff_max']:.3f}°", f"{e['fp16|raw']['feat_cos_mean']:.6f} / {e['fp16|raw']['feat_cos_min']:.6f}", st("일치", G)],
        ["원본 프레임, fp32 (메인 배포 dtype)", f"{e['fp32|raw']['action_angle_diff_mean']:.3f}° / {e['fp32|raw']['action_angle_diff_max']:.3f}°", f"{e['fp32|raw']['feat_cos_mean']:.6f} / {e['fp32|raw']['feat_cos_min']:.6f}", st("무시 가능", G)],
        ["대시보드 JPEG q80, fp16", f"{e['fp16|jpeg80']['action_angle_diff_mean']:.1f}° / {e['fp16|jpeg80']['action_angle_diff_max']:.1f}°", f"{e['fp16|jpeg80']['feat_cos_mean']:.4f} / {e['fp16|jpeg80']['feat_cos_min']:.4f}", st("어긋남", R)],
        ["JPEG q95, fp16", f"{q['fp16|jpeg95']['action_angle_diff_mean']:.2f}° / {q['fp16|jpeg95']['action_angle_diff_max']:.2f}°", f"{q['fp16|jpeg95']['feat_cos_mean']:.5f} / {q['fp16|jpeg95']['feat_cos_min']:.5f}", st("일치", G)]])
    cmp_ = tbl(["", "EGVLA1 (OWLv2+Kosmos-2+MLP)", "EGVLA2 (Florence-2+FM)"], [
        ["학습 특징", "Stage 1(Kosmos-2 비전+image_proj, 5-class OWL 라벨) 256-d를 오프라인 캐시 + 오프라인 bbox 어노테이션(V6 json)", "Florence-2 한 번 호출: bbox(4-d)+풀링 768-d를 오프라인 캐시(fp16)"],
        ["서버 특징", "같은 Stage1Encoder + <b>온라인 OWL-v2 그라운딩</b>(임계값 0.20, 회복 가드, 그라운딩 캐시 재사용)", "같은 Florence-2 추출기(<code>FusedF2Extractor</code>), 매 프레임 그라운딩"],
        ["전처리 일치 장치", "<code>resize_for_vlm</code>(224)을 학습·추론에서 동일 호출", "학습 <code>[:,:,::-1]</code> 후 원본 해상도로 Florence-2 처리 — 이번 점검으로 일치 확인"],
        ["JPEG q80 전송", "224로 축소해 영향이 작았을 가능성(미측정)", f"<b>영향 큼</b>: 방향 평균 {e['fp16|jpeg80']['action_angle_diff_mean']:.1f}°, q95 이상 필요"],
        ["서버 전용 규칙", "프리뷰 ROT 재시도, STOP(근접/학습), 근접 오버라이드, 래치, 그라운딩 캐시 — <b>92%에 포함</b>", "단계 A는 기존 규칙 위에 연속 방향만 대체, 단계 B(단독)는 규칙 이식 결정 필요"],
        ["학습=서버?", st("같지 않음 (bbox 생성·규칙 계층 차이)", A), st("원본·q95에서 일치, 규칙 계층은 미정", A)]])
    ck = tbl(["항목", "내용", "상태"], [
        ["배포 체크포인트(권장)", "<code>C_pi0_adaln_drop30_e120.pt</code> — 헤드 C, 300건 전체, 120 epoch, 서브골 드롭아웃 30%" + (f", sha256 <code>{man.get('C_pi0_adaln_drop30_e120.pt', {}).get('sha256', '')[:8]}</code>" if man else ""), st("저장", G)],
        ["대조 체크포인트", "<code>C_pi0_adaln_e120.pt</code> — 드롭아웃 없음", st("저장", G)],
        ["서빙 정책", "서브골 결측 시 bbox 창 전체를 0으로(드롭아웃 체크포인트 기본), EMA 평활 옵션(<code>VLA_EGVLA2_EMA</code>)", st("구현", G)],
        ["서버 단계 A", "<code>VLA_DRIVE_SOURCE=egvla2_hybrid</code>: 이동 프레임에서만 연속 방향·속력으로 대체, STOP·ROT는 기존 판단", st("구현(기본 꺼짐)", G)],
        ["대시보드", "<code>drive</code> 적용, <code>VLA_JPEG_QUALITY</code>(EGVLA2는 95 이상)", st("구현(기본 꺼짐)", G)],
        ["서버 단계 B", "EGVLA2 단독 서버 — 정지 규칙·타깃 소실 탐색·평활 값 결정 필요", st("미착수", MUT)],
        ["로봇 쪽 전달·점검", "체크포인트 rsync, 서버 경로 점검, 지연 측정, 드라이런", st("대기(로봇 접근)", R)]])
    return (H("⑥-7 학습-서빙 일치와 추론 서버 — EGVLA1 vs EGVLA2, JPEG 품질, 배포 현황")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">오프라인 성능이 실기에서 유지되려면 <b>서버가 학습 때와 같은 입력·같은 특징</b>을 내야 한다. 아래는 그 일치 여부를 EGVLA1과 EGVLA2로 대조하고, 이번에 찾은 JPEG 품질 문제와 배포 준비 현황을 정리한 것이다.</div>'
            + V.wrap("학습-서빙 대조", cmp_, "EGVLA1 항목은 레포 코드·문서 확인 범위이며 EGVLA1의 학습-서빙 차이를 정량 측정한 기록은 확인하지 못했다", kind="table")
            + '<div style="margin:12px 0 4px;font-size:0.82rem;font-weight:800;color:#fcd34d">JPEG 품질별 최종 방향 어긋남 (평균, 빨강 = 1° 초과)</div>' + bars
            + '<div style="font-size:0.74rem;color:#64748b;margin:2px 0 8px">같은 프레임 20장을 JPEG 왕복 후 Florence-2에 넣어 학습 캐시 입력과 정책 출력 방향을 비교. 대시보드 기본 q80 → 4.1°.</div>'
            + dt + V.wrap("배포 체크포인트와 추론 서버 현황", ck, "기본값은 모두 기존 동작(기본 꺼짐). 켜는 법은 plan_20261019 §6·핸드오프 §15", kind="table")
            + f'{st("실기 전 필수", R)} 서버로 보내는 JPEG 품질을 95 이상으로 올리고 색 보정(<code>apply_cc</code>)은 끈다. 이 두 조건이 안 맞으면 오프라인 결과(방향 오차 약 10~17°)에 평균 4° 안팎의 입력 잡음이 더해진다.')


from pathlib import Path  # noqa: E402
ROOT_MANIFEST = Path(__file__).resolve().parent.parent / "runs/egvla2_fm/deploy/MANIFEST.json"
