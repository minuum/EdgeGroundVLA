# 로봇서버 Grounding 실시간 테스트 가이드

작성: 2026-05-27

---

## 목적

Kosmos-2가 로봇 카메라 이미지에서 텍스트 phrase로 객체를 실제로 인식하는지 확인.

- "gray basket" → basket bbox 출력?
- "red ball" → 다른 bbox 또는 no detection? (R2-3 직접 증명)
- adapter 유무에 따른 인식 품질 비교

---

## 1. 파일 전송

```bash
# minum 머신에서 실행
scp scripts/run_grounding_realtime.py robot-server:/home/user/MoNaVLA/scripts/

# Exp56 adapter도 (학습 완료 후)
rsync -av runs/v5_nav/grounding/exp56/ robot-server:/home/user/MoNaVLA/runs/v5_nav/grounding/exp56/
```

---

## 2. 기본 실행 (로봇서버에서)

```bash
cd /home/user/MoNaVLA

# 순수 Kosmos-2 (adapter 없음) — baseline
python3 scripts/run_grounding_realtime.py \
    --source /path/to/episode_260408_123008_target_center_straight_path__core__fixed_center.h5

# 여러 phrase 비교 (R2-3 데모)
python3 scripts/run_grounding_realtime.py \
    --source /path/to/episode_xxx.h5 \
    --phrases "gray basket" "red ball" "person" "white wall"

# Exp56 adapter 사용 (학습 완료 후)
python3 scripts/run_grounding_realtime.py \
    --source /path/to/episode_xxx.h5 \
    --adapter /home/user/MoNaVLA/runs/v5_nav/grounding/exp56 \
    --phrases "gray basket" "red ball"

# HTTP 서버로 실시간 확인 (브라우저에서 열람)
python3 scripts/run_grounding_realtime.py \
    --source /path/to/episode_xxx.h5 \
    --phrases "gray basket" "red ball" \
    --serve
```

---

## 3. 결과 확인

**브라우저 (--serve 옵션):**
```
http://[robot-server-IP]:7860/realtime_test/live.html
```
→ 2초마다 자동 갱신, 각 phrase별 bbox 오버레이 이미지 표시

**파일:**
```
docs/v5/grounding_demo/realtime_test/
├── frame_0000.jpg     ← 전체 phrase 나란히
├── frame_0001.jpg
├── latest_gray_basket.jpg  ← 최신 프레임만
├── latest_red_ball.jpg
├── live.html
└── results.json
```

---

## 4. 예상 결과 (교수님 데모용)

| Phrase | 예상 결과 | 의미 |
|--------|----------|------|
| "gray basket" | bbox 정확히 출력 | 텍스트 → 올바른 객체 |
| "red ball" | no bbox 또는 wrong bbox | 다른 텍스트 → 다른 결과 |
| "person" | no bbox (복도에 사람 없음) | 다른 텍스트 → 다른 결과 |
| "white wall" | 벽 영역 bbox | 다른 물체도 인식 |

→ "gray basket" hit rate vs "red ball" hit rate 비교가 핵심 증거

---

## 5. Adapter 비교 실험

```bash
# 1) 순수 Kosmos-2
python3 scripts/run_grounding_realtime.py \
    --source /path/to/episode_xxx.h5 \
    --out-dir docs/v5/grounding_demo/pure_kosmos

# 2) 72-frame LoRA (기존)
python3 scripts/run_grounding_realtime.py \
    --source /path/to/episode_xxx.h5 \
    --adapter docs/v5/bbox_nav_step1/grounding_lora \
    --out-dir docs/v5/grounding_demo/lora_72frames

# 3) Exp56 (2624 frame LoRA)
python3 scripts/run_grounding_realtime.py \
    --source /path/to/episode_xxx.h5 \
    --adapter runs/v5_nav/grounding/exp56 \
    --out-dir docs/v5/grounding_demo/exp56

# 결과 비교: results.json의 hit rate 확인
cat docs/v5/grounding_demo/pure_kosmos/results.json
cat docs/v5/grounding_demo/lora_72frames/results.json
cat docs/v5/grounding_demo/exp56/results.json
```

---

## 6. 빠른 검증 (max-frames 10)

네트워크/서버 상황이 불확실하면 먼저 10프레임만:

```bash
python3 scripts/run_grounding_realtime.py \
    --source /path/to/episode_xxx.h5 \
    --max-frames 10 \
    --phrases "gray basket" "red ball"
```

---

## 주의

- Kosmos-2 추론: GPU 기준 ~0.5-1초/프레임 → `--fps 1.0` 적정
- H5 경로: `/home/user/MoNaVLA/ROS_action/mobile_vla_dataset_v5/episode_xxx.h5`
- 결과 이미지는 `docs/v5/grounding_demo/` 아래에 저장 (기존 HTTP 서버로 열람 가능)
