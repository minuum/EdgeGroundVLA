# MoNaVLA 위키 (핵심 요약)

VLM 기반 모바일 로봇 내비게이션 프로젝트. Kosmos-2(frozen) + LoRA 백본으로
바스켓을 찾아가는 VLA(Vision-Language-Action) 모델을 학습/평가/실로봇 배포.

## 서버 역할
- **soda (이 서버, Jetson Orin)**: 실로봇 추론/데모, 대시보드(7800), 추론서버(8001)
- **minum**: 학습/평가/교수 대응

## 현재 최선 모델
- **decomposition (실로봇 배포 중)**: Exp71 계열 — BBox+Image MLP/Transformer head
  - `runs/v5_nav/mlp/exp71_window6/action_transformer.pt` (window=6, val_acc 99.2%)
  - `runs/v5_nav/mlp/exp71_window3_bboxscale3/action_transformer.pt` (window=3, bbox_scale=3.0, val_acc 84.4%, FWD+R recall 개선)
- **end-to-end (참고용)**: V5 Exp11 — PM 58.6%, closed-loop 0% (FPE 1.45m 누적오류로 미사용)

## 그라운더 (A/B)
- **PG2 (PaliGemma2)**: 검출 안정적이나 로딩 ~120s, 가끔 full-frame 환각(`area>0.9` 필터로 차단)
- **OWL-v2**: 로딩 빠름, `VLA_OWLV2_AREA_SCALE`(기본 3.0)로 PG2 대비 area 축소분 보정.
  단점: detection flicker (has_bbox 40~60% 확률로 꺼짐) — 조사 중

## 액션 공간 (8-class, V5)
| idx | 이름 | 비고 |
|---|---|---|
| 0 | STOP | 데이터 없음, 에피소드 끝에 합성 |
| 1 | FORWARD | 71~74% 차지 (class imbalance 주범) |
| 2 | LEFT | strafe |
| 3 | RIGHT | strafe |
| 4 | FWD+LEFT | 대각선 |
| 5 | FWD+RIGHT | 대각선 |
| 6 | ROT_L | 제자리 회전 좌, ~0.8% |
| 7 | ROT_R | 제자리 회전 우, ~0.8% |

## 핵심 파일
| 목적 | 경로 |
|---|---|
| 학습 | `robovlm_nav/train.py configs/xxx.json` |
| 데이터셋 | `robovlm_nav/datasets/nav_h5_dataset_impl.py` |
| 추론 서버 | `robovlm_nav/serve/stage2_v2_inference_server.py` (포트 8001) |
| 대시보드 | `robovlm_nav/serve/mona_dashboard.py` (포트 7800) |
| PM 평가 | `scripts/test_v5_pm_dm.py` |
| Closed-loop 평가 | `scripts/sim/evaluate_closed_loop_v5.py` |
| Attention 분석 | `scripts/measure_attention.py` |
| 세션 전송(→minum) | `scripts/sync/push_inference_session_to_minum.sh` |

## 알려진 금지사항 / 함정

> [!critical]
> **Google-robot backbone으로 `generate()` 절대 호출 금지** — "Tin Tin Tin Roof..." 식
> 무한반복 텍스트 생성이 발생함. `third_party/RoboVLMs/`도 수정 금지.

> [!warn]
> Jetson Orin에서 서버(8001) 실행 중 PG2 그라운더를 직접 로드하면 OOM 발생 —
> 반드시 서버 내리고 테스트할 것.

> [!warn]
> **극단cx(cx>0.75 강한우 / cx<0.25 강한좌) 구간에서 grounding이 정상 검출돼도
> FORWARD로 고착되는 현상** — 학습데이터의 1.4~3.2%뿐이라 발생. 자세한 내용은
> 📡 최신현황 탭 또는 `docs/v5/closed_loop_eval/CH61_OWL_LIVE_FAILURE_AND_FIX.md` 참고.

- `go.sh --drive`는 존재하지 않는 MODE — 대시보드만 재시작하려면 `--mona-dash`
- 조이스틱: 다른 로봇에 옮겼다 복귀 시 HID 스트림 죽음 → 물리 재연결만 해결
- text attention = 0% (Google-robot post-training이 text 경로 붕괴시킴, 우리 LoRA와 무관)

## 실측 캘리브레이션
- W(전진) 1회 ≈ 12~13cm
- ROT az = ±0.25

---
*이 문서는 CLAUDE.md 프로젝트 컨텍스트의 요약본입니다. 전체 내용/최신 실험 이력은 `/home/soda/MoNaVLA/CLAUDE.md` 참고.*
