# Plan — 카메라 전송 압축(#1) + HDF5 JPEG 저장(#3)

> 작성 2026-06-10. 승인 전 구현 금지.
> 결정: #1 = 전체 CompressedImage 전환 / #3 = HDF5 JPEG q=90 + 하위호환 로더.
> 근거: 다른 VLA 시스템 비교(Open-X JPEG, LeRobot MP4, Mobile ALOHA JPEG-in-HDF5, ROS2 CompressedImage q80). 모델 입력 224라 720p raw는 학습 이득 0.

---

## #1 — GetImage 서비스: raw Image → CompressedImage (JPEG)

### 변경 파일
1. **`ROS_action/src/camera_interfaces/srv/GetImage.srv`**
   - `sensor_msgs/Image image` → `sensor_msgs/CompressedImage image`
2. **카메라 서버 (응답 인코딩)** — 2개
   - `ROS_action/src/camera_pub/camera_pub/camera_publisher_usb_service.py:202-214`
   - `ROS_action/src/camera_pub/camera_pub/camera_publisher_continuous.py:137-145`
   - `self.bridge.cv2_to_imgmsg(frame,'bgr8')` → `self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')`
     (cv_bridge 표준 메서드; quality는 기본 ~80. 필요시 cv2.imencode 수동.)
3. **소비자 (응답 디코딩)** — 11곳, 패턴 동일 `imgmsg_to_cv2(res.image,'bgr8')` → `compressed_imgmsg_to_cv2(res.image,'bgr8')`:
   - `scripts/gradio_data_collector.py` (_camera_loop:593, _capture_post_sync:536)
   - `scripts/gradio_inference_dashboard.py` (~789, ~797)
   - `scripts/gradio_grounding_demo.py` (~181)
   - `ROS_action/src/mobile_vla_package/mobile_vla_package/api_client_node.py` (~161)
   - `ROS_action/src/mobile_vla_package/mobile_vla_package/mobile_vla_data_collector.py` (~1181)
   - `ROS_action/src/mobile_vla_package/mobile_vla_package/vla_inference_node.py` (~114)
   - `ROS_action/src/vla_inference/vla_inference/vla_inference_node.py` (~121)
4. **import**: 각 파일에서 `from sensor_msgs.msg import Image` 쓰던 곳에 `CompressedImage` 추가 (srv가 자동 처리하므로 대부분 bridge 호출만 바꾸면 됨).

### 빌드/재기동
- `colcon build --packages-select camera_interfaces camera_pub mobile_vla_package vla_inference` (srv 변경 → 의존 패키지 재빌드)
- `source ROS_action/install/setup.bash`
- 카메라 서비스 + 모든 소비자(콜렉터/대시보드/추론) **전부 재기동** (구 인터페이스 노드는 신 서버와 통신 불가).

### 리스크
- 소비자 1곳이라도 누락 시 런타임 디코드 에러 → 11곳 전부 수정 필수.
- 실행 중 노드는 재빌드 후 반드시 재시작. 혼재 금지.

---

## #3 — HDF5 저장: gzip raw → JPEG bytes (q=90) + 하위호환 로더

선례: Mobile ALOHA `act-plus-plus/compress_data.py` (`cv2.imencode('.jpg', q=50)` + vlen + 길이배열).

### 저장 (`scripts/gradio_data_collector.py` save_h5 ~726-751)
```python
# 현재: imgs = [BGR2RGB ...]; create_dataset('observations/images', np.array(imgs), compression='gzip')
# 변경: RGB 프레임을 JPEG(q=90) bytes로 vlen 저장
dt = h5py.vlen_dtype(np.uint8)
ds = f.create_dataset('observations/images', (len(imgs),), dtype=dt)
for i, rgb in enumerate(imgs):
    buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
    ds[i] = np.frombuffer(buf.tobytes(), np.uint8)
ds.attrs['format'] = 'jpeg'   # 로더 감지용 (선택)
```
- 주의: RGB 배열을 인코딩→디코딩하면 동일 RGB 배열 복원(JPEG는 색순서 무관). 기존 로더가 RGB 가정과 일치.

### 로더 (`robovlm_nav/datasets/nav_h5_dataset_impl.py`)
- `len(hf['observations']['images'])` (라인 134/257): vlen이어도 프레임 수 = len → 그대로 동작.
- 이미지 읽기 (라인 534/542): **per-frame ndim으로 raw/JPEG 자동 분기 (하위호환)**:
```python
raw = images_src[t]
if getattr(raw, 'ndim', 2) == 1:        # JPEG bytes (vlen 1D)
    img_array = cv2.imdecode(raw, cv2.IMREAD_COLOR)
else:                                    # 기존 raw 3D 배열 (V5)
    img_array = raw
img = Image.fromarray(img_array.astype(np.uint8))
```
- → 기존 V5(raw 4D) + 신규 V5-2(JPEG vlen) **혼용 안전**. 기존 데이터 마이그레이션 불필요.

### proxy 호환 (`robovlm_nav/serve/proxy_inference_server.py:610-620`)
- `handle["observations"]["images"][:]` 후 `.astype` 하는 경로 → 동일 ndim 분기 추가 (구 데이터 추론용; V5-2는 아직 미사용이나 안전 위해).

### 기타 필드
- actions / timestamps / language_instruction / attrs → **변경 없음**.

---

## 진행 순서 (승인 후)
1. **#3 먼저** (저위험, 2~3파일): save_h5 + 로더 분기 → 신규 H5 1개 저장/로드 왕복 테스트.
2. **#1** (고위험): srv → 카메라서버 → 11소비자 → colcon 빌드 → 전체 재기동.

## Verification
- #3: 콜렉터로 더미 에피소드 저장 → `h5py`로 images dtype=vlen 확인 → 로더 `__getitem__`으로 디코딩 정상(이미지 shape 복원) + 기존 V5 raw도 여전히 로드되는지 둘 다 테스트.
- #1: 재빌드+재기동 후 콜렉터/대시보드 카메라 피드 정상, GetImage 응답이 CompressedImage이고 모든 소비자 디코딩 OK. 대역폭 체감(피드 부드러움) 확인.
- 회귀: 8082 추론 서버가 카메라 쓰면 그쪽도 디코딩 정상인지.

## 기대 효과
- #1: 전송 대역폭 ~10×↓ (피드 끊김 해소 + 수집/추론 round-trip 단축).
- #3: H5 파일 크기 대폭↓ (q=90 JPEG), 학습 품질 손실 사실상 0 (모델 224 입력).
