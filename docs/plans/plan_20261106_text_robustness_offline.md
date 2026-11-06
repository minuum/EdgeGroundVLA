# plan: 지시문·물체 교체 가능성 오프라인 점검 (2026-11-06)
배경: 교수님(11/05) 요청 — ① 지시문 변경(go to the gray bin) ② 물체 교체+단어 변경에도 가는지. 사용자 지시 "서버에서 가능한 건 다 해보고, 진짜 E2E VLA가 되는지도 확인".
구조 사실(코드): `precompute_v6_florence2_fused.run_batch` — 768-d 풀링 특징은 이미지 인코더 출력(`img_feats.mean`)으로 **텍스트와 무관**. 텍스트는 디코더가 내는 bbox에만 작용 → 현재 정책은 "텍스트→bbox→헤드" 파이프라인이며 헤드는 텍스트를 직접 받지 않음.
점검(읽기 전용, 학습 없음): 10/29 H5에서 24세션×3프레임을 뽑아 phrase 변형별로 Florence-2 bbox를 구하고 ①기준('gray basket') 대비 검출률·cx·면적 ②A3 헤드(동일 노이즈)의 첫 스텝 명령 차이 ③오답 phrase의 오검출률을 잰다.
스크립트: `scripts/eval_text_robustness.py` → `docs/v5/egvla2_fm/text_robustness.json`. 실기 물체 교체는 사진이 없어 오프라인 불가.
