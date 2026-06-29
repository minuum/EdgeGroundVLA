---
name: grounding-session-pipeline
description: 그라운딩 검증 세션(logs/grounding_sessions/gnd_*.jsonl + .mp4)을 정리·분석하고 minum 서버로 전달하는 전체 파이프라인. "세션 분석해줘", "S7 분석", "그라운딩 세션 minum으로 보내줘" 같은 요청에 사용.
---

# Grounding Session Pipeline

PG2 그라운딩 검증 세션 하나를 받아서 분리 → 프레임 추출 → HSV 교차검증 →
시각 스폿체크 → git push(양 브랜치) → minum rsync → minum에서 재현 확인까지
한 번에 처리하는 절차.

## 0. 트리거 확인

사용자가 "세션 분석", "방금 수집한 거 확인", "S{N} 분석/전송" 등을 요청하면 이 스킬 적용.

## 1. idle-gap 세션 분리 (필요시)

`scripts/gradio_inference_dashboard.py`의 `_gnd_ensure_log()`는 idle 10분 이상이면
자동으로 새 jsonl을 만들지만(2026-06-20 수정 이후), **그 이전에 쌓인 파일이거나
직접 의심되면** 먼저 확인:

```bash
python3 scripts/eval/split_grounding_session.py \
    logs/grounding_sessions/<파일>.jsonl --dry-run
```

여러 세션으로 쪼개져야 하면 `--dry-run` 빼고 다시 실행 → 실제 분리 파일 생성.

## 2. mp4 → 프레임 강제 추출

각 세션 jsonl 레코드 수와 대응하는 mp4가 있는지 확인 (`ls logs/grounding_sessions/*.mp4`,
타임스탬프로 매칭). 프레임 이미지가 없으면:

```bash
mkdir -p docs/v5/grounding_frames/s{N}
ffmpeg -y -i logs/grounding_sessions/gnd_<세션>.mp4 \
    -vf fps=1 docs/v5/grounding_frames/s{N}/frame_%04d.jpg -loglevel error
ls docs/v5/grounding_frames/s{N}/ | wc -l   # jsonl 레코드 수와 일치하는지 확인
```

세션 번호 `s{N}`은 기존 `docs/v5/grounding_frames/`에 있는 가장 큰 번호 다음으로 부여.

## 3. HSV 교차검증

```bash
python3 scripts/eval/test_hsv_recovery.py \
    --jsonl logs/grounding_sessions/gnd_<세션>.jsonl \
    --frames docs/v5/grounding_frames/s{N}
```

PG2 fallback 프레임을 HSV classical detector(`scratch/patch_hsv_annotations.py` 동일 로직)로
재검출해 "진짜 복구" / "의심(배경 오탐 추정)" / "HSV도 실패"로 분류.

## 4. 시각 스폿체크 (절대 생략하지 말 것)

**"진짜 복구"로 분류된 항목도 그대로 믿지 않는다.** 최소 3~5건을 Read 툴로
직접 열어서 HSV의 cx와 실제 바스켓 위치가 맞는지 확인:

```
Read docs/v5/grounding_frames/s{N}/frame_00{idx}.jpg
```

- 벽 얼룩/가구/콘센트를 바스켓으로 오탐하는 경우가 실제로 다발 확인됨(S6 n=25/30/44/52, S7 n=6).
- 오탐이 섞여 있으면 보고 시 "HSV 진짜 복구 X건" 같은 숫자를 그대로 쓰지 말고,
  스폿체크로 확인된 신뢰도를 명시할 것 (예: "스폿체크 5건 중 2건 오탐 확인 — 전체 신뢰 불가").
- **cx 위치가 아니라 area 크기가 핵심 판별 기준이다.** 처음엔 `cx>0.85`(화면 가장자리)를
  의심 신호로 썼으나, S6 시각검증 결과 cx가 중앙·좌측이어도 area<0.04인 검출은 거의 전부
  의자 다리·벽 얼룩 등 잡음이었다(n=25 cx=0.02, n=44 cx=0.04, n=52 cx=0.81 — cx는 제각각인데
  전부 area<0.001로 작았고 전부 오탐). 반대로 area≥0.04(근접 촬영으로 바스켓이 화면을
  크게 채움)는 비교적 신뢰 가능. `test_hsv_recovery.py`는 area<0.04를 suspect 기준으로 사용.
- **area 기준도 완벽하지 않다.** S7 n=6은 area=0.20(기준 통과)인데도 벽 변색 얼룩을 오탐 —
  큰 면적의 배경 잡음도 존재 가능. 가장 신뢰도 높은 신호는 "area 충분히 크다 + 인접
  프레임들과 cx가 일관되게 클러스터링"(예: 연속 2~3프레임이 비슷한 cx로 수렴)이지만,
  PG2 자체가 여러 프레임 연속으로 동일한 오탐을 내는 경우(근접-미스 bbox, S7 f39~46)엔
  이마저도 무력화되므로 최종적으로는 항상 수동 확인이 필요하다.

## 5. git 커밋 + 양 브랜치 push

이미지(`docs/v5/grounding_frames/s{N}/`)와 분석 스크립트만 git 대상 —
`logs/`는 `.gitignore` 처리되어 있어 jsonl/mp4는 커밋되지 않는다.

```bash
git add docs/v5/grounding_frames/s{N}/ scripts/eval/
git commit -m "feat(grounding): S{N} 세션 분석 — <한줄요약>"
git push origin monavla-driving
git push origin monavla-driving:inference-integration
```

## 6. minum으로 rsync (jsonl만 — 이미지는 git으로 이미 전달됨)

```bash
rsync -avz logs/grounding_sessions/gnd_<세션>.jsonl \
    minum@100.101.73.21:/home/minum/MoNaVLA/logs/grounding_sessions_from_soda/

ssh minum "cd ~/26CS/MoNaVLA && git pull origin inference-integration && \
    cp /home/minum/MoNaVLA/logs/grounding_sessions_from_soda/gnd_<세션>.jsonl logs/grounding_sessions/"
```

## 7. minum에서 재현 확인

```bash
ssh minum "cd ~/26CS/MoNaVLA && python3 scripts/eval/test_hsv_recovery.py \
    --jsonl logs/grounding_sessions/gnd_<세션>.jsonl \
    --frames docs/v5/grounding_frames/s{N}"
```

soda와 동일한 숫자가 나오는지 확인 — 다르면 git pull이 안 됐거나 이미지가 빠진 것.

## 참고

- [[feedback_pg2_direct_load]] — PG2 직접 로드 테스트 시 서버 동시 실행 금지 규칙은 이 파이프라인과 무관(여긴 HSV만 사용, GPU/PG2 안 씀)
- minum SSH alias: `~/.ssh/config`의 `minum` (HostName 100.101.73.21)
- jsonl/mp4는 `.gitignore`(`logs/`) 대상 — git push만으론 minum에 전달 안 됨, rsync 필수
