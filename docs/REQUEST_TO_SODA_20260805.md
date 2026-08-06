# soda 요청서 — 2026-08-05

> 보낸 쪽: minum (`inference-integration`)
> 배경: 교수님 논문 초안(`EdgeGround-VLA_..._0805.docx`) 보완 작업 중, **8개 절 중 7개는
> 우리 자료로 닫혔고 1개 절이 로봇 쪽 정보 없이는 채워지지 않습니다.**
> 참고: [논문 드래프트 보완](https://minuum.github.io/MoNaVLA/v5/paper_redfill_worklist.html)
> · [절별 초고](https://minuum.github.io/MoNaVLA/01.paper/redfill_draft_0805.html)

---

## 우선순위 요약

| # | 항목 | 주행 필요 | 급함 | 왜 |
|---|---|:---:|:---:|---|
| **A** | 미러증강 헤드 실기 100건 검정 | ✅ | **높음** | 논문 주요 수치(89/100)를 바꿀 수 있는 **유일한** 항목 |
| **B** | 젯슨 배포 절차 문서화 | ❌ | **높음** | 논문 6절이 이것 없이는 못 닫힘 |
| **C** | 디코더 제거 젯슨 실측 | ❌ | 중간 | 6절의 −70% 수치가 로컬 측정뿐 |
| **D** | negative 프레임 300~500장 | ⚠️ 촬영만 | 중간 | has_bbox 학습 공백을 서빙 우회가 아니라 학습으로 풀려면 필요 |
| **E** | 기구 편향 검정 4항목 | ❌ | 낮음 | CH66에서 미검증으로 남긴 항목 |
| **F** | 100건 테스트 세션 H5 원본 재확보 | ❌ | 중간 | 운영 threshold(0.20) 등 핵심 설정값을 직접 증거로 확인 못 하는 중 |

**B가 논문 일정에는 가장 급합니다** (주행 불필요). **A는 결과가 논문 수치를 바꿉니다.**

---

## A. 미러증강 헤드 실기 100건 검정 ⭐

### 왜 이걸 해야 하는지 (근거가 새로 생겼습니다)

오프라인에서 **배포 헤드가 cx를 좌측 조향의 근거로 쓰지 못한다**는 것을 확인했습니다.
검출기를 빼고 **cx만 통제 변수로 스윕**한 결과입니다(val 결정 시점 501개, 3 seed).

| cx | 0.05 | 0.25 | 0.45 | 0.65 | 0.85 | 0.95 | Spearman |
|---|---|---|---|---|---|---|---|
| **배포 (holdaware)** | −0.088 | −0.062 | −0.032 | −0.004 | −0.067 | −0.145 | **+0.185** |
| **미러증강** | **+0.141** | **+0.037** | −0.049 | −0.006 | −0.019 | −0.029 | **−0.575** |

> 값은 좌질량 = softmax(좌계열 2,4,6) − softmax(우계열 3,5,7). **양수 = 좌측 선호.**

- **배포 헤드는 cx를 화면 맨 왼쪽(0.05)까지 밀어도 좌질량이 −0.088** — 19단계 어디에서도
  양수가 되지 않습니다. 즉 **어떤 cx에서도 좌측을 선호하지 않습니다.**
- 곡선이 단조가 아니라 **역U자**이고, 분포 안(0.3~0.7)에서도 기울기가 **+0.143로 역전**입니다.
- **미러증강 헤드는 cx 응답이 복구**돼 있습니다(Spearman −0.575, 극좌 기울기 −0.516).

이게 **100건 테스트의 최저 구간이 좌측**이었던 것과 방향이 맞습니다:

| 위치 | 성공률 |
|---|---|
| 중앙 | 100% |
| 약우 | 95% |
| 강우 | 90% |
| **강좌** | **80%** |
| **약좌** | **80%** |

### 요청 내용

`exp73_owl_trackF_v6_mlp_mirroraug_seed*.pt` 3개 중 하나를 배포해서
**기존과 동일한 프로토콜로 100건**(5개 위치 × 20회)을 돌려 주세요.

체크포인트 위치 (minum 서버):
```
runs/v5_nav/mlp/exp73/exp73_owl_trackF_v6_mlp_mirroraug_seed0.pt
runs/v5_nav/mlp/exp73/exp73_owl_trackF_v6_mlp_mirroraug_seed1.pt
runs/v5_nav/mlp/exp73/exp73_owl_trackF_v6_mlp_mirroraug_seed2.pt
```
메타데이터: `head=exp73_mlp`, `window=6`, `bbox_scale=3.0`, `stride=5`, `grounder=owlv2`,
`arm=v6_mirroraug`, `val_acc=0.7505`, `fixed_bias=0.0088`
→ **기존 배포 헤드와 입력 규격이 완전히 동일**합니다. 서버 코드 수정 불필요.

### ⚠️ 꼭 지켜 주실 것 — 지난번에 우리가 틀렸던 지점입니다

1. **`runtime_config`에 체크포인트가 남는지 먼저 확인**해 주세요.
   지난 감사에서 `episode_log.csv`의 체크포인트 열이 전부 NaN이라 어떤 모델이 돌았는지
   알 수 없었고, **H5 `attrs['runtime_config']`만이 유일한 근거**였습니다.
   그것 때문에 CH64 64-11 결론을 철회했습니다.
2. **위치별 균등 20회**를 지켜 주세요. 위치를 섞으면 좌측 개선 여부를 볼 수 없습니다
   (좌측이 정확히 우리가 보려는 구간입니다).
3. **다른 설정은 바꾸지 말아 주세요** — threshold 0.20, skip_n 3 등 그대로.
   헤드만 바꾼 단일 변수 비교여야 합니다.
4. **결과가 나빠도 그대로 보고**해 주세요. 오프라인 개선이 실기로 이어진다는 보장이 없다는 걸
   우리가 이미 한 번 겪었습니다(64-11).

### 판정 기준 (미리 고정)

- **주 지표**: 강좌·약좌 성공률 (현재 각 80%)
- 부지표: 전체 100건 성공률 (현재 89/100), 우측 구간이 떨어지지 않는지
- **좌측이 오르고 우측이 유지되면 교체**, 좌측만 오르고 우측이 내려가면 **총합으로 판단**

---

## B. 젯슨 배포 절차 문서화 (주행 불필요) ⭐ 논문 일정상 가장 급함

논문 6절 "Deploy the trained model on the robot"이 **절 전체가 비어 있고**, 우리 쪽에서
아는 것은 여기까지입니다:

> 체크포인트 `.pt`(헤드) + `stage1_v2_projs.pt`(image_proj)를 젯슨으로 전달 →
> `stage2_v2_inference_server.py`가 로드 → HTTP API로 프레임 수신, 액션 반환.

**필요한 것 (각 항목 2~3줄이면 충분합니다)**

1. **환경 구성** — JetPack 버전, PyTorch/torchvision 버전, transformers 버전,
   설치 방식(pip/conda/docker 중 무엇인지)
2. **모델 파일 배치** — 젯슨 상의 실제 경로, 파일 전달 방법(scp/rsync/git 중 무엇인지)
3. **서버 실행** — 실행 명령 전문, 포트, systemd/supervisor 등록 여부와 자동시작 방식
4. **ROS 연결** — 어느 노드가 서버를 호출하는지, 이미지 토픽/속도 명령 토픽 이름,
   호출 주기
5. **첫 실행 시 주의점** — 모델 로딩 시간, 워밍업 필요 여부

형식은 자유입니다. 이 파일 아래에 이어 쓰거나 별도 md로 주셔도 됩니다.

---

## C. 디코더 제거 젯슨 실측 (주행 불필요)

Kosmos-2 언어 디코더(1.361B)는 **호출이 0회**여서 로드 자체를 제거했습니다.
`VLA_KOSMOS_VISION_ONLY=1`(기본값)로 동작합니다.

로컬 측정 결과:

| | 제거 전 | 제거 후 |
|---|---|---|
| peak 호스트 RAM | 10.59GB | **3.20GB (−70%)** |
| GPU 메모리 | 0.607GB | 0.607GB (**변화 없음**) |

**요청**: 젯슨에서 같은 값을 재봐 주세요.
- `VLA_KOSMOS_VISION_ONLY=0` / `=1` 두 조건에서 서버를 띄우고
- **모델 로딩 완료 직후**의 호스트 RAM(`tegrastats` 또는 `free -m`)과 GPU 메모리
- 가능하면 **모델 로딩 소요 시간**도 같이

**왜 필요한가** — 논문에 "−70%"를 쓰려는데 현재 값이 로컬 측정뿐입니다. 젯슨은 메모리
구조가 달라(통합 메모리) 수치가 다르게 나올 수 있고, 그러면 서술을 고쳐야 합니다.

---

## D. negative 프레임 300~500장

### 왜 필요한가

학습 데이터 16,599 프레임 중 **검출 실패(`has_bbox=False`) 프레임이 0개(0.00%)**입니다.
그래서 모델은 "지금 목표를 못 보고 있다"는 신호에 대응하는 법을 **배운 적이 없고**,
현재는 그 공백을 서빙 제어 규칙(회전 후 강제 재검출 등)으로 우회하고 있습니다.

같은 공백이 검출기 쪽에도 있습니다 — 장면에 **없는 물체**를 요청했을 때도
200프레임 중 **5.0%에서 박스를 냈습니다**(신뢰도 중앙 0.267).

### 요청 내용

**바구니가 프레임에 없는 상태**의 이미지 300~500장. 주행은 필요 없고 **촬영만** 하면 됩니다.

포함해 주시면 좋은 상황:
- 바구니가 시야 밖 (로봇을 돌려놓은 상태)
- 바구니가 다른 물체에 가려진 상태
- 빈 바닥·벽만 보이는 상태
- **조명 조건을 섞어** 주세요 (기존 데이터에 lighting_diff 라벨이 있던 것처럼)

기존 H5 포맷 그대로면 가장 좋고, 단순 이미지 묶음도 괜찮습니다.
**바구니가 없다는 것만 확실**하면 됩니다(라벨은 우리가 붙입니다).

---

## E. 기구 편향 검정 4항목 (주행 불필요)

좌우 비대칭 원인을 추적하면서, **모델 쪽 원인은 찾았습니다**(위 A의 cx 응답 붕괴).
다만 기구 쪽 가능성은 **아직 검증하지 않은 상태로 남겨** 뒀습니다.
교수님이 지적하신 항목들입니다:

1. **카메라 마운트 편향** — 카메라 광축이 로봇 전방 중심과 일치하는지.
   확인법: 로봇을 벽에서 일정 거리에 정면으로 세우고, 벽 중앙에 표식을 붙인 뒤
   촬영 이미지에서 표식의 cx가 0.5인지 확인
2. **무게 중심 치우침** — 정지 상태에서 좌/우 바퀴 하중 차이
3. **휠 동작 차이** — 같은 명령에 대한 좌/우 회전 바퀴의 실제 회전량.
   이전에 "휠 제어에 문제가 있다"고 들은 적이 있어 확인이 필요합니다
4. **직진 명령 시 드리프트** — 순수 FORWARD만 일정 시간 주었을 때 좌/우로 밀리는지와 방향

**우선순위는 1번과 4번**입니다. 두 항목만으로도 "기구 편향이 유의미한가"를 가릴 수 있고,
둘 다 몇 분이면 됩니다. **결과가 "편향 없음"이어도 그것 자체가 논문에 쓸 근거**입니다
(모델 원인이라는 주장을 강화합니다).

---

## F. 100건 테스트 세션 H5 원본 재확보

### 왜 필요한지

논문 초안을 보완하면서 서빙 제어 설정값들을 코드와 세션 기록으로 하나하나 대조했습니다.
그 과정에서 **7/23 세션 27건**의 `runtime_config`는 로컬에 남아 있어 확인했지만, 결과가
갈렸습니다.

| 항목 | 27건 기록값 | 논문에 쓰려는 값 | 상태 |
|---|---|---|---|
| `owlv2_thresh` | **0.25** (27/27) | 0.20 | ⚠️ 불일치 — 확인 필요 |
| `stop_mode` | 키 없음 | `learned` | ⚠️ 미확인 |
| `force_reground_on_miss` | 키 없음(기본 off) | — | ⚠️ 미확인 |

이 27건은 **PG2-448 그라운더 시기**의 세션이라, OWL-v2 threshold 0.20을 실제로 쓴
**100건 테스트(중앙/약우/강우/약좌/강좌 각 20회) 당시의 원본 H5**와는 다른 세션입니다.
100건 테스트의 원본 파일이 지금 로컬(`inference_sessions_recv/`)에 없어서, 논문에 쓰려는
"운영 threshold 0.20"이라는 서술을 **직접 증거로 확인하지 못하고 있습니다** — 지금은
soda 보고와 CH64 64-17 기준으로만 적어둔 상태입니다.

### 요청 내용

**100건 테스트 세션의 원본 H5 파일 전체**(또는 최소한 각 위치별 1~2개 샘플)를
`monavla-driving` 브랜치나 별도 전송으로 다시 보내주세요.

확인하고 싶은 것은 각 세션의 `attrs['runtime_config']` 필드입니다 — 특히:
- `owlv2_thresh` 실제 값 (0.20이 맞는지)
- `stop_mode` 값
- `checkpoint_path`, `grounder_model` (어떤 모델이 돌았는지)

### 왜 이게 중요한지 (지난 실수를 반복하지 않으려는 것)

CH64 64-11에서 **실행 설정을 대조하지 않은 채 결론을 내려 철회**한 적이 있습니다
(체크포인트 전달 전 세션의 성공을 신모델 덕으로 잘못 돌렸던 사례). 이번엔 논문에 쓰기
전에 먼저 확인하려는 것이고, **급하지 않습니다** — B(젯슨 배포 절차)가 훨씬 급합니다.

## 회신 방법

이 파일 아래에 이어 쓰거나, `docs/` 아래 별도 md로 커밋해 주시면 됩니다.
`monavla-driving` 브랜치에 올려주시면 확인합니다.

**A(실기 100건)만 시간이 걸리고 B·C·E는 각각 짧습니다.**
논문 일정 때문에 **B를 먼저** 주시면 6절을 바로 닫을 수 있습니다.

---

## soda 회신 — B. 젯슨 배포 절차 문서화 (2026-08-07)

현재 실제로 돌고 있는 젯슨(soda, Orin) 기준으로 확인한 값입니다. 코드/systemd 상태를
직접 조회한 값이라 추정 아님.

### 1. 환경 구성

| 항목 | 값 |
|---|---|
| L4T / JetPack | R36.3.0 (JetPack 6.x) |
| Python | 3.10.12 (`/usr/bin/python3`, 시스템 파이썬) |
| PyTorch | 2.3.0 (`torch.cuda.is_available()=True`) |
| transformers | 4.45.2 |
| 설치 방식 | **pip --user** (`~/.local/lib/python3.10/site-packages`) — conda/docker 아님 |

### 2. 모델 파일 배치

- 실제 경로: `~/MoNaVLA/runs/v5_nav/mlp/<exp>/<checkpoint>.pt` (헤드), image_proj는
  `.vlms/` 하위 또는 헤드 체크포인트에 내장(현재 배포 헤드는 `stage1_v2_projs.pt` 별도
  전달 없이 헤드 `.pt` 안에 image_proj 가중치 포함 — exp73 계열 확인).
- 전달 방법: **minum → soda 직접 rsync** (Tailscale, `100.85.118.58`, `.agent/skills/deploy-stage2-v2/SKILL.md`에 절차 고정돼 있음).
  ```bash
  rsync -avz --relative <ckpt> soda@100.85.118.58:~/MoNaVLA/runs/v5_nav/mlp/...
  # 또는 minum 쪽 scripts/deploy/rsync_stage2_v2.sh
  ```
- git으로는 체크포인트 자체를 커밋하지 않음(바이너리 크기 때문) — 코드/문서만 git, 모델은 rsync.

### 3. 서버 실행

포트 2개, 둘 다 `systemd --user` **transient unit**(`--collect`)으로 기동 중:

```bash
systemd-run --user --unit=vla-stage2 --collect --working-directory=/home/soda/MoNaVLA \
  /usr/bin/python3 -m robovlm_nav.serve.stage2_v2_inference_server --port 8001

systemd-run --user --unit=vla-mona-dash --collect --working-directory=/home/soda/MoNaVLA \
  /bin/bash -c 'exec /usr/bin/python3 robovlm_nav/serve/mona_dashboard.py --port 7800 >> logs/mona_dashboard.log 2>&1'
```

> **⚠️ 2026-08-07 정정**: `--working-directory` 필수입니다. 빠뜨리면 유닛 기본 작업
> 디렉터리가 `$HOME`(`/home/soda`)이 되어 코드 안의 상대경로(`robovlm_nav/serve/...`,
> `logs/...`)를 못 찾고 곧바로(수십 ms 내) 죽습니다 — 실제로 이 문서를 처음 쓸 때는
> 없어도 됐는데(기존 유닛을 `restart`로만 썼어서 원래 속성이 유지됐던 것), 유닛을
> 처음부터 새로 만들 때는 반드시 명시해야 한다는 걸 재현 중에 발견했습니다.

- `vla-stage2`(8001): 추론 서버 본체. ROS 없음, 순수 FastAPI.
- `vla-mona-dash`(7800): 대시보드 + **ROS 연결 지점**(아래 4번). `vla-stage2`를 HTTP로 호출.
- 설정 복원: checkpoint_path/threshold/stop_mode 등은 `logs/stage2_runtime_state.json`에
  자동 저장되고 재기동 시 자동 복원됨(`_persist_runtime_state()`/`_restore_runtime_state_env()`).

**⚠️ 자동시작 관련 정정 사항 — 논문에 "자동시작" 서술 넣으면 안 됩니다.**
`loginctl`로 linger는 켜져 있지만(로그인 없이도 유저 서비스 실행 가능), 이 두 유닛은
**transient**(`systemd-run --collect`)라서 **재부팅하면 사라지고 등록된 unit file도 없습니다**
(`systemctl --user list-unit-files 'vla-*'` → `STATE: transient`, enable된 파일 없음).
즉 **현재는 재부팅 시 수동으로 위 두 명령을 다시 실행해야 합니다.** 자동시작이 필요하면
`~/.config/systemd/user/*.service` 파일로 승격하는 작업이 별도로 필요합니다(아직 안 함).

### 4. ROS 연결

- **`vla-stage2`(추론 서버, 8001) 자체는 ROS를 전혀 쓰지 않습니다.** ROS는 오직
  `vla-mona-dash`(7800) 안의 `MoNaROSNode`(`robovlm_nav/serve/mona_dashboard.py`)에만 있습니다.
- 이미지 입력: 토픽 구독이 아니라 **서비스 호출** — `camera_interfaces.srv.GetImage`,
  서비스명 `get_image_service`. 실제 카메라 프로세스는 별도 `ros2 run camera_pub
  usb_camera_service_server`(ROS_DOMAIN_ID=42, rmw_fastrtps_cpp)이고, 대시보드가
  10Hz로 폴링(`time.sleep(0.05)` 루프).
- 속도 명령 출력: `geometry_msgs/Twist`를 **`/cmd_vel`** 토픽에 퍼블리시
  (`self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10, ...)`).
  주행 모드는 ASYNC 10Hz 연속 발행(`ASYNC_INTERVAL=0.10`, 300ms jitter hold) 방식.
- 호출 주기 요약: `usb_camera_service_server`(카메라) → 10Hz GetImage 서비스 응답 →
  `vla-mona-dash`가 `http://localhost:8001/predict` HTTP 호출(`INFER_URL`) →
  받은 action을 `/cmd_vel` Twist로 10Hz 퍼블리시. 즉 **ROS 토픽은 출력(/cmd_vel)에만
  있고, 입력(카메라)은 ROS 서비스, 추론 자체는 순수 HTTP**입니다.

### 5. 첫 실행 시 주의점

- 모델 로딩 시간(젯슨 실측, C항목과 동일 측정): Kosmos-2 vision-only(`VLA_KOSMOS_VISION_ONLY=1`,
  기본값) 기준 **7.47초**. 기존 전체 로드는 37.59초 — vision-only가 로컬(오히려 느려짐)과
  반대로 젯슨에서는 5배 더 빠름.
- GPU 메모리는 로딩 직후 **0.608GB로 고정**, vision-only 전환 전후 변화 없음(디코더는
  GPU에 안 올라가 있었다는 뜻).
- 워밍업 필요: 그라운더(현재 OWL-v2)는 **첫 호출이 콜드스타트라 지연이 큼** — 서버
  기동 직후 곧바로 주행 시작하지 말고, 헬스체크(`curl :8001/health`) 통과 후 더미
  프레임 1회 `/predict` 호출로 워밍업 권장(과거 PG2 시절 "첫 그라운딩 호출 시 미웜업 →
  빈 결과 반환" 이슈가 코드 주석에 남아 있음, `stage2_v2_inference_server.py:2076` 부근).
