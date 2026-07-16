# V5/V6 → LeRobotDataset 변환 (상호운용용 export)

> 기존 RoboVLMs 학습 파이프라인은 그대로 두고, 다른 VLA 프레임워크와의
> 상호운용을 위해 로컬 전용으로 export하는 부가 산출물. Hub push 없음.

## 실행 환경

**minum에서만 실행 가능.** `lerobot`은 Python >= 3.12를 요구하는데
soda는 3.10.12라 설치 불가. minum은 3.12.3이라 OK.

```bash
# minum에서 최초 1회 (이미 구성됨: /home/minum/26CS/MoNaVLA/.venv_lerobot)
python3.12 -m venv .venv_lerobot
source .venv_lerobot/bin/activate
pip install h5py numpy lerobot   # 'lerobot[core_scripts]'는 evdev 빌드
                                  # 실패함(Python.h 없음, 우리는 안 씀)
```

## 스크립트

```
scripts/convert/h5_v6_to_lerobot.py
```

```bash
source /home/minum/26CS/MoNaVLA/.venv_lerobot/bin/activate
python3 scripts/convert/h5_v6_to_lerobot.py \
  --dataset-dir /home/minum/26CS/MoNaVLA/ROS_action/mobile_vla_dataset_v5 \
  --out /home/minum/26CS/MoNaVLA/lerobot_export/v6_mixed \
  --repo-id monavla/v6_mixed \
  --schema both        # v5 | v6 | both(기본, 둘 다 하나로 섞음)
  --fps 6               # 전역 고정값(실측 근사, 아래 참조)
```

## 2026-07-16 변환 결과 (minum `/home/minum/26CS/MoNaVLA/lerobot_export/v6_mixed`)

- 420 에피소드 / 18,409 프레임 / task 15종
- 664MB (원본 H5 19GB 대비 AV1 인코딩으로 ~28배 압축)
- 스킵 2건: 손상 파일 1건(`target_right_left_path` 레거시, `bad object
  header version number`), 해상도 불일치 1건(`episode_260607_111159...`,
  480x640 — 나머지 전부 720x1280)

## 스키마 매핑

V5/V6 H5를 자동판별해 하나의 LeRobotDataset으로 병합:

| feature | 내용 |
|---|---|
| `observation.images.cam_front` | H5 `images`(V6) 또는 `observations/images`(V5), video(AV1) 인코딩 |
| `action` | `[linear_x, linear_y, angular_z]` float32 |
| `action.event_type` | V6: H5 `action_event_types` 그대로. V5: 원본에 없어 `"unknown"` 채움 |
| `task` | V5: H5 `language_instruction` 그대로 사용. V6: `cx_position`+`cx_path` 조합해 자연어 문장 생성(예: `"Navigate a left curve approach from the strong left extreme starting position..."`) |

## 알려진 제약

1. **fps는 실측이 아니라 고정 근사치(기본 6)**. V6 89프레임/14.47초
   ≈ 6.15fps 실측을 반올림한 값이고, V5는 애초에 timestamp 정보가 H5에
   없어 실측 자체가 불가능 — 두 소스 다 이 고정값으로 timestamp가
   계산됨. 프레임 간 실제 시간 간격이 중요한 학습(예: 속도 추정)에는
   부정확할 수 있음.
2. **`observation.state` 없음** — 원본 H5가 로봇 상태(odometry 등)를
   저장하지 않아 이 feature 자체를 정의하지 않음.
3. **비디오 로드에 시스템 ffmpeg 필요** — minum엔 현재 `ffmpeg`/
   `libavutil` 시스템 라이브러리가 없어 `LeRobotDataset(...)`로 로드해
   `dataset[i]`로 프레임을 실제로 디코딩하려면 사전에 설치 필요:
   ```bash
   sudo apt install ffmpeg   # 또는 libavutil56/57 등 배포판에 맞는 패키지
   ```
   (변환/쓰기 자체는 파이썬 번들 인코더로 되므로 위 제약과 무관하게 성공함 —
   메타데이터/parquet 직접 읽기로 검증 완료. 실제 학습 로더로 프레임
   디코딩할 때만 필요)
4. **비표준 커스텀 컬럼**: `action.event_type`은 공식 LeRobot 표준
   스키마에 없는 필드라, 다른 LeRobot 기반 도구가 이 컬럼을 인식 못 하고
   무시할 수 있음(문제 없이 동작은 함, 정보만 활용 안 될 뿐).

## 참고 문서 (2026-07-16 확인)

- [LeRobotDataset v3.0 공식 문서](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [LeRobotDataset v3.0 발표 블로그](https://huggingface.co/blog/lerobot-datasets-v3)
- [Porting Large Datasets to v3.0 (DROID 예제, feature 네이밍 참고)](https://huggingface.co/docs/lerobot/en/porting_datasets_v3)
