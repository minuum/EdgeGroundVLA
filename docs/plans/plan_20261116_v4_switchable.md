# v4·v4r도 `/model/switch_family`로 교체되게 하는 계획 (승인 대기 — 코드 변경 없음)

작성 2026-11-16. 요청: "v4, v4r도 스위치로 적용될 수 있도록". 선행: `plan_20261116_soda_test_protocol.md`(교체 방식), `plan_20261115_v4_soda_minimal_patch.md` §8.

## 1. 지금 안 되는 이유 (소스: 받은 soda 코드)
- `switch_family`(stage2_v2_inference_server.py 2294행~)는 `VLA_EGVLA2_CKPT`·EMA·DTYPE·SERVO_*만 `os.environ`에 쓰고 `load_service_from_env()`를 부른다. **`VLA_EGVLA2_TEXT`는 설정하지 못한다**(요청 모델 `SwitchFamilyRequest`에 필드 없음).
- soda에 적용된 변경 B는 `VLA_EGVLA2_TEXT=1`일 때만 텍스트 경로를 탄다 → 서버 프로세스 환경에 그 변수가 없으면 v4·v4r 체크포인트(`fdim=1536`)를 A3용 정책으로 읽다가 차원 불일치로 실패한다.

## 2. 선택지
| 안 | 내용 | 수정 파일 | 장단점 |
|:--|:--|:--|:--|
| **A (권장)** 체크포인트 구조로 자동 판별 | `load_service_from_env()`가 체크포인트의 `fdim==1536 and bdim==12`이면 텍스트 경로를 자동 선택. `VLA_EGVLA2_TEXT`는 "1=강제 켬, 0=강제 끔, 미설정=자동" | **soda의 `egvla2_inference_server.py` 1곳**(이미 적용한 변경 B의 조건 한 줄 + 판별 함수) | stage2 서버·대시보드 무수정, 요청 형식 그대로, 환경변수 신경 쓸 필요 없음. A3(`fdim=768`)는 영향 없음 |
| B | `SwitchFamilyRequest`에 `egvla2_text: bool` 추가, 핸들러가 환경변수 설정 | stage2 서버(2.4천 줄, soda 미커밋 수정 포함) + 요청하는 쪽(대시보드·curl) | 명시적이나 큰 파일을 건드리고 대시보드 UI도 바뀜 |
| C | 재시작할 때 환경변수로 켬 | 없음 | v4 시험 때마다 서버 재시작·`a3`↔`v4` 교체 불가 |

## 3. 권장안 A의 변경 (코드는 승인 후)
soda `robovlm_nav/serve/egvla2_inference_server.py` — 이미 들어간 6줄 블록의 조건을 바꾸고 판별 함수를 추가한다.
```python
def _is_text_ckpt(path) -> bool:
    """텍스트 조건화(V4·V4R) 체크포인트인지 구조로 판별: 문구 특징 포함(fdim=1536), 박스 창 12-d. A3는 fdim=768."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return int(ck.get("fdim", 0)) == 1536 and int(ck.get("bdim", 0)) == 12

# load_service_from_env() 안, 기존 `if os.getenv("VLA_EGVLA2_TEXT", "0") == "1":` 를 아래로 교체
_txt = os.getenv("VLA_EGVLA2_TEXT")          # 미설정=자동 판별, "1"=강제 켬, "0"=강제 끔
if _txt == "1" or (_txt is None and _is_text_ckpt(ckpt)):
    ...(기존 텍스트 경로 블록 그대로)...
```
- 이 저장소의 같은 파일에도 동일 조건을 반영한다(저장소 서버는 별도 호출 방식이라 동작 확인만).
- `TextCondPolicy`는 이미 `fdim!=1536 or bdim!=12`면 ValueError를 내므로 잘못된 강제(`=1`+A3 체크포인트)는 즉시 실패한다.
- `egvla2_text_policy.py`, stage2 서버, 대시보드는 **수정하지 않는다.**

## 4. 반드시 알아둘 점: 서버 재시작 1회가 필요하다
Python은 이미 불러온 모듈을 다시 읽지 않는다. 운영서버가 이미 `egvla2_inference_server`를 한 번이라도 import했다면(11/13에 egvla2를 썼으므로 그럴 가능성이 높음) **디스크의 새 코드(변경 B와 이번 변경)는 서버 재시작 전까지 적용되지 않는다.**
- 따라서 **`a3-s1`·`a3-s2` 시험은 재시작 없이** 교체만으로 먼저 한다(새 코드 불필요).
- v4·v4r 시험 전에 **서버를 한 번 재시작**하고, 그 뒤에는 `a3`↔`a3-s2`↔`v4`↔`v4r` 모두 API 교체로 오간다. 재시작 시점은 동환이 정하며 재시작하면 `phrase_from_request`가 꺼지므로 `/health`로 다시 확인한다.
- (확인 필요) 재시작 후 어느 패밀리·체크포인트로 올라오는지(`_persist_runtime_state`)는 로봇 쪽 Claude에게 읽기 전용으로 확인 요청.

## 5. 검증 절차 (승인 후, 이 서버에서 먼저)
1. soda 파일 복사본(받은 파일 + 적용된 변경 B) 위에 변경을 적용하고 `py_compile`.
2. `VLA_EGVLA2_TEXT` **미설정**으로 `VLA_EGVLA2_CKPT`만 바꿔 `load_service_from_env()`를 호출(= `switch_family`가 하는 일):
   - v4·v4r 체크포인트 → 텍스트 경로로 로드되는지(`meta.variant`, 서비스 `extractor` 종류).
   - a3·a3-s1·a3-s2 체크포인트 → 기존 A3 경로로 로드되는지.
   - 같은 프로세스에서 A3 → v4 → A3 순으로 교체해도 GPU 메모리·결과가 정상인지(동시 로드 없음).
3. 기존 동등성 검증(`check_soda_overlay_equiv.py`)을 **환경변수 없이** 다시 실행 — 기준 불변(문구 코사인 ≥0.999, 각도 중앙값 ≤1°, 정지 일치 ≥95%), 이전 결과와 동일해야 함.
4. 강제 값 시험: `VLA_EGVLA2_TEXT=1`+A3 체크포인트는 오류, `=0`+v4 체크포인트는 오류(차원 불일치)로 끝나는지.

## 6. 전달 (검증 후, 단계별 승인)
- 패키지: soda의 **현재 파일(원본 + 변경 B) 대비 통합 diff** 1개(수 줄)와 SHA256SUMS. 로봇 쪽 Claude가 `patch --dry-run` 성공 시에만 적용, `py_compile`·`git diff --stat`(+몇 줄) 확인. 체크포인트는 이미 전달됨.
- 서버 재시작과 구동 시험은 동환 입회, 별도 승인.

## 7. 하지 않는 것
stage2 서버·대시보드 수정, 서버 재시작(이쪽에서), 모델 동시 로드, 로봇 구동, push.

## 8. 승인 요청
1. 안 A로 간다 — 예/아니오(B 또는 C 희망 시 지정)
2. 검증(§5)을 이 서버에서 진행한 뒤 패키지를 만든다 — 예/아니오
3. 재시작 1회를 v4 시험 전에 동환과 정한다 — 확인

## 9. 진행 결과 (2026-11-16, 승인: 안 A 검증→패키지, 서버 재시작 허가)
- 검증(이 서버, soda 파일 복사본, `VLA_EGVLA2_TEXT` 미설정): A3→v4→A3→v4r→a3-s2→v4→a3-s1→A3 순 `load_service_from_env()` 호출이 매번 올바른 경로로 로드(`docs/v5/egvla2_fm/soda_switch_verify.json`), GPU 메모리 0.94GB 일정. 강제 값: `=1`+A3 체크포인트 → ValueError, `=0`+v4 → KeyError('window')(오류로 끝남, 메시지는 불친절).
- 동등성 재실행(환경변수 없이): 이전 결과와 모든 수치 동일.
- 이 저장소 서버에도 같은 조건 반영(`egvla2_inference_server.py`, +9/-1, py_compile 통과).
- 패치 v2: soda 현재 파일(원본+B) 대비 +7/-1, `patch --dry-run` 후 적용 결과가 검증본과 바이트 동일. 패키지 `_outbox_soda_20261116/soda_package_v2_20261116.tar.gz`(저장소에는 커밋하지 않음).
- 서버 재시작: 사용자가 허가함. 단 이쪽에서는 soda에 접속하지 않으며, 로봇 쪽 Claude/동환이 README 조건(로봇 정지·세션 없음·동환 확인·재시작 전후 `/health` 기록)을 만족할 때만 수행.
