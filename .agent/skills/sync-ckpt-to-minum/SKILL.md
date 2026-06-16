---
name: sync-ckpt-to-minum
description: MoNaVLA 실험 체크포인트를 minum 서버로 rsync 전송. 실험명(예: v5_exp38, exp38)을 받아 best ckpt + config를 자동으로 찾아 전송한다. --all 플래그로 전체 run 디렉토리 전송 가능.
---

# Sync Checkpoint to Minum

실험 체크포인트를 `minum` 서버로 전송하는 스킬.

## 실제 스크립트

```
scripts/sync/push_exp_to_minum.sh
```

## 사용법

```bash
# 단일 실험 (best ckpt만)
bash scripts/sync/push_exp_to_minum.sh exp38

# v5_ 접두사도 허용
bash scripts/sync/push_exp_to_minum.sh v5_exp38

# 복수 실험 한 번에
bash scripts/sync/push_exp_to_minum.sh exp35 exp36 exp37 exp38

# 전체 run 디렉토리 (모든 ckpt 포함, 대용량 주의)
bash scripts/sync/push_exp_to_minum.sh exp38 --all
```

## 동작 방식

1. `runs/v5_nav/kosmos/mobile_vla_v5_{exp}/` 탐색
2. 각 서브디렉토리에서 `val_loss` 최솟값 ckpt 선택 (best ckpt)
3. 대응하는 `configs/mobile_vla_v5_{exp}*.json` 탐색
4. `rsync -avz --relative` 로 `minum:/home/billy/25-1kp/MoNaVLA/` 전송
   - `--relative` 플래그 → 로컬 경로 구조 그대로 유지

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MINUM_HOST` | `minum` | SSH 호스트명 (`.ssh/config`에 정의) |
| `MINUM_PATH` | `/home/billy/25-1kp/MoNaVLA` | 원격 프로젝트 루트 |

## 주의사항

- ckpt 1개당 약 6.7GB. `exp35~38` best만 해도 ~27GB.
- `--all` 사용 시 exp당 4개 ckpt × 6.7GB = ~27GB.
- SSH 연결: `~/.ssh/config`의 `minum` alias 사용.

## 에이전트 사용 예시

사용자가 "exp38 minum으로 보내줘" 등의 요청 시:

```bash
bash scripts/sync/push_exp_to_minum.sh exp38
```

복수 실험이면:

```bash
bash scripts/sync/push_exp_to_minum.sh exp35 exp36 exp37 exp38
```
