#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

from camera_interfaces.srv import GetImage
from std_srvs.srv import Empty
import threading

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class USBCameraServiceServer(Node):
    def __init__(self):
        super().__init__('usb_camera_service_server')
        
        self.bridge = CvBridge()

        self.failed_reads = 0
        self.buffer_lock = threading.Lock()  # 스레드 안전성을 위한 락
        
        self.latest_frame = None
        self.is_running = True
        
        # 카메라 초기화
        if not self.init_camera():
            self.get_logger().info('🎨 가상 카메라 모드로 전환 (USB 카메라 시뮬레이션)')
            self.latest_frame = self.generate_virtual_frame()

        # 백그라운드 프레임 캡처 스레드 구동
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        try:
            self.srv = self.create_service(GetImage, 'get_image_service', self.get_image_callback)
            self.reset_srv = self.create_service(Empty, 'reset_camera_service', self.reset_camera_callback)
            self.get_logger().info('✅ get_image_service 서비스 서버 준비 완료!')
            self.get_logger().info('✅ reset_camera_service 서비스 서버 준비 완료!')
            self.get_logger().info('⏳ USB 카메라 이미지 요청 대기 중...')
        except Exception as e:
            self.get_logger().error(f"❌ USB 카메라 서비스 서버 시작 실패: {e}. 'colcon build' 후 'source install/setup.bash'를 다시 실행했는지, 그리고 패키지 구조가 올바른지 확인하세요.")
            rclpy.shutdown()

    def init_camera(self):
        """카메라를 초기화합니다. Jetson CSI 카메라를 우선 시도하고, 실패하면 USB 카메라를 시도합니다."""
        import threading as _threading
        import time as _time
        try:
            # 1. Jetson CSI 카메라 시도 (open 8초 timeout — Argus 초기화 여유, hang 방지)
            self.get_logger().info('📷 Jetson CSI 카메라 시도 중...')
            # framerate 30/1 — 추론 루프(3~10Hz)에는 60fps가 불필요하고, videoconvert(CPU 소프트웨어
            # 컬러 변환)가 프레임레이트에 비례해 CPU를 소모하므로 30으로 낮춰 부하를 절반으로 줄인다.
            gst_str = (
                "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
                "format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                "videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
            )
            _csi_cap = [None]
            def _try_csi():
                _csi_cap[0] = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            _t = _threading.Thread(target=_try_csi, daemon=True)
            _t.start()
            _t.join(timeout=8.0)
            if _t.is_alive() or _csi_cap[0] is None:
                self.get_logger().warn('⚠️ Jetson CSI 카메라 open timeout (8s) — USB로 전환')
            else:
                self.cap = _csi_cap[0]
                if self.cap.isOpened():
                    # 웜업: Argus가 실제 프레임을 낼 때까지 대기. 1장 이상 성공해야 CSI 인정.
                    # (이전엔 warmup 실패해도 무조건 성공 선언 → 캡처루프 영구 실패하던 버그 수정)
                    self.get_logger().info('🔥 Jetson CSI 카메라 웜업 중...')
                    ok = 0
                    for i in range(20):
                        ret, _ = self.cap.read()
                        if ret:
                            ok += 1
                            if ok >= 2:  # 2프레임 연속 확보 → 안정
                                break
                        else:
                            _time.sleep(0.2)  # 프레임 생성 대기
                    if ok >= 1:
                        self.get_logger().info(f'✅ Jetson CSI 카메라 연결 성공! (warmup 확보 {ok})')
                        self.camera_type = "Jetson CSI"
                        self.failed_reads = 0
                        return True
                    else:
                        self.cap.release()
                        self.get_logger().warn('⚠️ CSI 웜업 프레임 0장 — 실제 캡처 불가, USB로 전환')
                else:
                    self.cap.release()
                    self.get_logger().warn('⚠️ Jetson CSI 카메라 연결 실패')
        except Exception as e:
            self.get_logger().warn(f'⚠️ Jetson CSI 카메라 초기화 실패: {e}')
            if self.cap:
                self.cap.release()
        
        # 2. USB 카메라 시도 (CAP_V4L2 명시 — Jetson에서 GStreamer hang 방지)
        for camera_id in range(4):  # 0, 1, 2, 3번 카메라 시도
            self.get_logger().info(f'📷 USB 카메라 {camera_id} 시도 중...')

            # CAP_V4L2로 직접 열기 (Jetson OpenCV는 기본이 GStreamer라 hang 발생)
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
            
            if self.cap.isOpened():
                # 카메라 설정
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
                
                # 카메라 웜업: 처음 몇 프레임은 불안정할 수 있으므로 미리 읽어서 버림
                self.get_logger().info(f'🔥 USB 카메라 {camera_id} 웜업 중...')
                for i in range(5):
                    ret, _ = self.cap.read()
                    if not ret:
                        self.get_logger().warn(f'웜업 프레임 {i+1}/5 읽기 실패')
                    else:
                        self.get_logger().info(f'웜업 프레임 {i+1}/5 완료')
                
                self.get_logger().info(f'✅ USB 카메라 {camera_id} 연결 성공!')
                self.camera_type = f"USB {camera_id}"
                self.failed_reads = 0
                return True
            else:
                self.cap.release()
                self.get_logger().warn(f'⚠️ USB 카메라 {camera_id} 연결 실패')
        
        # 모든 카메라 연결 실패
        self.get_logger().error('❌ 모든 카메라 연결 실패')
        self.cap = None
        self.camera_type = "가상 카메라"
        return False

    def _capture_loop(self):
        """백그라운드에서 실시간으로 프레임을 캡처하여 최신 프레임을 유지합니다."""
        import time
        TARGET_FRAME_DUR = 1.0 / 30  # 33.3ms — CAP_PROP_FPS=30과 동일
        # 2026-07-30: mona_dashboard 쪽에서 그라운딩/캐시 스텝 구분 없이 프레임
        # 중복(픽셀단위 동일) 비율이 15~32%로 잡히는 게 확인됨 — 캐시 스텝(80ms
        # 간격)과 실제 그라운딩 스텝(2000ms 간격) 둘 다 중복률이 비슷해서, "요청이
        # 너무 잦아서"가 아니라 이 캡처 루프 자체가 가끔 초 단위로 멈추는 것으로
        # 의심됨 — cap.read()가 실제로 얼마나 걸리는지, 목표(33ms) 대비 얼마나
        # 밀리는지 직접 계측해서 확인.
        SLOW_READ_WARN_S = 0.2  # 이 이상 걸리면 즉시 경고(단발성 블로킹 감지용)
        STATS_WINDOW = 300      # 약 10초치(30fps 기준) 롤링 통계
        read_durations: list = []
        self.get_logger().info('🌀 백그라운드 카메라 캡처 루프 시작')
        while rclpy.ok() and self.is_running:
            loop_start = time.monotonic()
            if self.cap is not None and self.cap.isOpened():
                # V4L2 드라이버는 버퍼링 때문에 read()가 거의 즉시 반환되는 경우가 많아
                # 드라이버 블로킹에만 의존하면 busy-loop가 되어 CPU 코어를 거의 다 점유함.
                # 루프 종료 시점에 목표 주기(33ms)까지 남은 시간만큼 명시적으로 sleep한다.
                read_start = time.monotonic()
                ret, frame = self.cap.read()
                read_dur = time.monotonic() - read_start
                if read_dur > SLOW_READ_WARN_S:
                    self.get_logger().warn(
                        f'🐌 [Capture Loop] cap.read() 지연 {read_dur*1000:.0f}ms '
                        f'(목표 {TARGET_FRAME_DUR*1000:.0f}ms) — 이 구간 동안 소비자는 '
                        f'같은 프레임을 재사용하게 됨')
                read_durations.append(read_dur)
                if len(read_durations) > STATS_WINDOW:
                    read_durations.pop(0)
                if len(read_durations) == STATS_WINDOW:
                    avg = sum(read_durations) / len(read_durations)
                    mx = max(read_durations)
                    slow_n = sum(1 for d in read_durations if d > SLOW_READ_WARN_S)
                    self.get_logger().info(
                        f'📊 [Capture Loop 통계 {STATS_WINDOW}프레임] avg={avg*1000:.1f}ms '
                        f'max={mx*1000:.0f}ms slow(>{SLOW_READ_WARN_S*1000:.0f}ms)={slow_n}건 '
                        f'실측fps={1.0/avg if avg > 0 else 0:.1f}')
                    read_durations.clear()
                if ret:
                    with self.buffer_lock:
                        self.latest_frame = frame
                        self.failed_reads = 0
                else:
                    with self.buffer_lock:
                        self.failed_reads += 1
                    if self.failed_reads % 30 == 0:
                        self.get_logger().warn(f'⚠️ [Capture Loop] 프레임 읽기 실패 ({self.failed_reads}회 누적)')
            else:
                with self.buffer_lock:
                    self.latest_frame = self.generate_virtual_frame()

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, TARGET_FRAME_DUR - elapsed))

    def reset_camera_callback(self, request, response):
        """USB 카메라를 완전히 재시작하여 버퍼를 초기화합니다."""
        with self.buffer_lock:
            self.get_logger().info('🔄 USB 카메라 스트림 완전 재시작 중...')
            
            # 기존 카메라 해제
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.get_logger().info('📴 기존 USB 카메라 스트림 해제 완료')
            
            # 잠시 대기 (하드웨어 안정화)
            import time
            time.sleep(0.5)
            
            # 카메라 재초기화
            if self.init_camera():
                self.get_logger().info('✅ USB 카메라 스트림 재시작 완료 - 버퍼 초기화됨!')
                self.latest_frame = None
                self.failed_reads = 0
            else:
                self.get_logger().info('🎨 가상 USB 카메라 모드로 전환')
                self.latest_frame = self.generate_virtual_frame()
                
        return response

    def flush_camera_buffer(self):
        """더미 함수 (백그라운드 스레드가 항상 최신 프레임을 캡처하므로 플러시 불필요)"""
        pass

    def get_fresh_frame(self):
        """버퍼를 플러시할 필요 없이 백그라운드에서 캡처된 최신 프레임을 즉시 가져옵니다."""
        with self.buffer_lock:
            if self.latest_frame is None:
                if self.cap is None:
                    return self.generate_virtual_frame(), "가상 카메라"
                return None, "준비 중"
            
            frame = self.latest_frame.copy()
            
            if self.failed_reads >= 15:
                self.get_logger().error('❌ 카메라 하드웨어 문제 감지 - 가상 카메라로 전환')
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                self.cap = None
                self.failed_reads = 0
                return self.generate_virtual_frame(), "가상 카메라 (자동 전환)"
            
            # Jetson CSI 카메라는 180도 회전 필요
            if hasattr(self, 'camera_type') and self.camera_type == "Jetson CSI":
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                
            return frame, f"실제 {self.camera_type}"

    def get_image_callback(self, request, response):
        frame, camera_type = self.get_fresh_frame()
        
        if frame is not None:
            # CompressedImage(JPEG) 전송 — raw bgr8 대비 대역폭 ~10x↓
            response.image = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
            response.image.header.stamp = self.get_clock().now().to_msg()
            response.image.header.frame_id = 'usb_camera_frame'
            self.get_logger().info(f'📸 {camera_type} 최신 이미지 서비스 요청 처리 완료!')
        else:
            self.get_logger().error('❌ USB 카메라 이미지 캡처/생성 실패 - 서비스 요청에 빈 이미지 반환')
            response.image = CompressedImage()

        return response
    
    def generate_virtual_frame(self):
        height, width = 720, 1280
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [50, 100, 50]
        cv2.putText(frame, f'Mobile VLA Virtual USB Camera', 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.circle(frame, (width // 2, height // 2), 50, (0, 0, 255), -1)
        cv2.putText(frame, 'USB', (width // 2 - 30, height // 2 + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def destroy_node(self):
        self.is_running = False
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = USBCameraServiceServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

