#!/usr/bin/env python3
"""
Mobile VLA API Client - RoboVLMs Style & Multi-threaded Safe
"""
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import sys, tty, termios
import os
import time
import threading
import requests
import base64
import cv2
import numpy as np
from collections import deque

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from camera_interfaces.srv import GetImage

try:
    from pop.driving import Driving
    ROBOT_AVAILABLE = True
except ImportError:
    # print("Warning: pop.driving not available. Using ROS only.")
    ROBOT_AVAILABLE = False


class MobileVLAAPIClient(Node):
    def __init__(self):
        super().__init__('mobile_vla_api_client')
        
        # ✅ KEY FIX: ReentrantCallbackGroup 사용 (데드락 방지)
        self.callback_group = ReentrantCallbackGroup()
        
        # 액션 매핑
        self.WASD_TO_CONTINUOUS = {
            'w': {"linear_x": 1.15, "linear_y": 0.0, "angular_z": 0.0},
            'a': {"linear_x": 0.0, "linear_y": 1.15, "angular_z": 0.0},
            's': {"linear_x": -1.15, "linear_y": 0.0, "angular_z": 0.0},
            'd': {"linear_x": 0.0, "linear_y": -1.15, "angular_z": 0.0},
            'q': {"linear_x": 1.15, "linear_y": 1.15, "angular_z": 0.0},
            'e': {"linear_x": 1.15, "linear_y": -1.15, "angular_z": 0.0},
            'z': {"linear_x": -1.15, "linear_y": 1.15, "angular_z": 0.0},
            'c': {"linear_x": -1.15, "linear_y": -1.15, "angular_z": 0.0},
            ' ': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        }
        self.STOP_ACTION = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        
        # Exp47 path_type 키를 직접 쓰거나 env로 override. 자연어 미매칭 시 bbox cx 자동 추론.
        # 유효 키: center_straight, left_left, right_right, left_right, right_left, ...
        self.default_instruction = os.getenv("VLA_INSTRUCTION", "center_straight")
        
        self.inference_mode = False
        self.inference_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.latency_history = deque(maxlen=100)
        self.last_action = [0.0, 0.0]
        
        self.api_server_url = os.getenv("VLA_API_SERVER", "http://localhost:8000")
        self.api_key = os.getenv("VLA_API_KEY", "")
        self.vlm_model = os.getenv("VLA_VLM_MODEL", "kosmos")
        
        if ROBOT_AVAILABLE:
            try:
                self.driver = Driving()
                self.throttle = 50
            except Exception as e:
                self.get_logger().error(f"Failed to init driving: {e}")
                self.driver = None
        else:
            self.driver = None
        
        # ✅ KEY FIX: 모든 클라이언트/퍼블리셔에 callback_group 적용
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10, callback_group=self.callback_group)
        self.get_image_client = self.create_client(GetImage, 'get_image_service', callback_group=self.callback_group)
        
        self.cv_bridge = CvBridge()
        
        # 서비스 대기
        self.wait_for_service()
        
        self.print_welcome()
        
        # 키보드 스레드
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        # ✅ KEY FIX: Timer 대신 별도 스레드로 추론 루프 실행 (데드락 완전 회피)
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        self.get_logger().info(f"✅ 시스템 준비 완료! API: {self.api_server_url}")

    def wait_for_service(self):
        try:
            while not self.get_image_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().info('get_image_service 서비스 대기 중...')
                if not rclpy.ok():
                    sys.exit()
            self.get_logger().info('✅ get_image_service 서비스 연결 완료!')
        except Exception as e:
            self.get_logger().error(f"❌ 서비스 연결 실패: {e}")

    def print_welcome(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info("🤖 Mobile VLA API Client - Multi-Threaded")
        self.get_logger().info("=" * 60)
        self.get_logger().info("📋 조작: W/A/S/D(이동), Space(정지), I(추론), R(테스트), T(진단)")
        self.get_logger().info("=" * 60)

    def keyboard_loop(self):
        while rclpy.ok():
            key = self.get_key()
            self.handle_key_input(key)
            time.sleep(0.01)
    
    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def handle_key_input(self, key: str):
        if key == '\x03':  # Ctrl+C
            rclpy.shutdown()
            sys.exit()
        elif key == 'i':
            self.inference_mode = not self.inference_mode
            status = "🟢 ON" if self.inference_mode else "🔴 OFF"
            self.get_logger().info(f"🎮 추론 모드: {status}")
            if not self.inference_mode:
                self.publish_cmd_vel(self.STOP_ACTION, "inference_off")
        elif key == 'r':
            # 별도 스레드에서 실행 (키보드 루프 블로킹 방지)
            threading.Thread(target=self.run_single_inference_test).start()
        elif key == 't':
            threading.Thread(target=self.run_system_test).start()
        elif key in self.WASD_TO_CONTINUOUS:
            if not self.inference_mode:
                action = self.WASD_TO_CONTINUOUS[key]
                self.publish_cmd_vel(action, f"manual_{key}")
                if key != ' ':
                    pass # 로그 너무 많아서 생략

    def get_latest_image_via_service(self, max_retries: int = 3):
        """서비스로 카메라 이미지 가져오기 (Deadlock-free & Safe Timeout)"""
        for attempt in range(max_retries):
            try:
                self.get_logger().info(f"📸 [시도 {attempt+1}/{max_retries}] 이미지 서비스 요청...")
                request = GetImage.Request()
                future = self.get_image_client.call_async(request)
                
                # ✅ KEY FIX: threading.Event를 사용하여 안전하게 타임아웃 대기
                # Future.result()에는 timeout 인자가 없으므로 이 방식이 필수입니다.
                event = threading.Event()
                def done_callback(f):
                    event.set()
                future.add_done_callback(done_callback)
                
                # 타임아웃 대기 (2.0초)
                if event.wait(timeout=2.0):
                    response = future.result()
                    if response and response.image.data:
                        cv_image = self.cv_bridge.imgmsg_to_cv2(response.image, "bgr8")
                        self.get_logger().info(f"✅ 이미지 수신 성공 ({cv_image.shape})")
                        return cv_image
                    else:
                        self.get_logger().warn(f"⚠️ [시도 {attempt+1}] 빈 이미지 수신")
                else:
                    self.get_logger().warn(f"⚠️ [시도 {attempt+1}] 서비스 호출 타임아웃 (2.0s)")
                    # Future 취소 시도
                    try:
                        future.cancel()
                    except:
                        pass
                    
            except Exception as e:
                self.get_logger().error(f"❌ [시도 {attempt+1}] 서비스 호출 에러: {e}")
                
            time.sleep(0.5)
            
        return None

    def inference_loop(self):
        """추론 루프 (별도 스레드)"""
        while rclpy.ok():
            if self.inference_mode:
                try:
                    # 1. 카메라 이미지 획득 (여기서 result() 대기하므로 안전)
                    image = self.get_latest_image_via_service(max_retries=1)
                    
                    if image is not None:
                        # 2. 리사이즈
                        resized = cv2.resize(image, (224, 224))
                        # 3. 추론 호출
                        self.run_inference(resized)
                    
                except Exception as e:
                    self.get_logger().error(f"추론 루프 에러: {e}")
                    
            time.sleep(0.1) # 10Hz

    def run_inference(self, image: np.ndarray):
        try:
            _, buffer = cv2.imencode('.jpg', image)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            
            response = requests.post(
                f"{self.api_server_url}/predict",
                json={
                    "image": img_b64, 
                    "instruction": self.default_instruction,
                    "vlm_model": self.vlm_model
                },
                headers=headers,
                timeout=2.5
            )
            
            if response.status_code == 200:
                data = response.json()
                action_list = data["action"]
                latency = data["latency_ms"]
                
                self.inference_count += 1
                self.success_count += 1
                self.latency_history.append(latency)
                
                action = {
                    "linear_x": float(action_list[0]),
                    "linear_y": float(action_list[1]),
                    "angular_z": 0.0
                }
                
                self.publish_cmd_vel(action, "inference")
                
                if self.inference_count % 10 == 0:
                    avg = sum(self.latency_history) / len(self.latency_history)
                    self.get_logger().info(
                        f"✅ [INFERENCE#{self.inference_count}] Latency: {latency:.0f}ms (avg: {avg:.0f}ms) | ACTION: {action_list}"
                    )
            else:
                self.fail_count += 1
                
        except Exception as e:
            self.fail_count += 1
            if self.fail_count % 10 == 0:
                self.get_logger().error(f"추론 실패: {e}")

    def run_single_inference_test(self):
        self.get_logger().info("🧪 추론 테스트 (단일)")
        image = self.get_latest_image_via_service()
        if image is None: 
            self.get_logger().error("❌ 이미지 획득 실패")
            return
            
        resized = cv2.resize(image, (224, 224))
        self.get_logger().info(f"📡 API 요청: {self.api_server_url}")
        
        try:
            _, buffer = cv2.imencode('.jpg', resized)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            headers = {}
            if self.api_key: headers["X-API-Key"] = self.api_key
            
            start = time.time()
            resp = requests.post(
                f"{self.api_server_url}/predict",
                json={
                    "image": img_b64, 
                    "instruction": self.default_instruction,
                    "vlm_model": self.vlm_model
                },
                headers=headers, timeout=15.0
            )
            elap = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                self.get_logger().info(f"✅ 성공! Action: {data['action']}, Latency: {data['latency_ms']:.0f}ms")
                action = {"linear_x": float(data['action'][0]), "linear_y": float(data['action'][1]), "angular_z": 0.0}
                self.publish_cmd_vel(action, "test")
            else:
                self.get_logger().error(f"❌ 실패: {resp.status_code} {resp.text}")
        except Exception as e:
            self.get_logger().error(f"❌ 예외: {e}")

    def run_system_test(self):
        self.get_logger().info("🧪 시스템 테스트")
        # 1. Camera
        img = self.get_latest_image_via_service()
        self.get_logger().info(f"📸 카메라: {'✅ OK' if img is not None else '❌ FAIL'}")
        # 2. API
        try:
            r = requests.get(f"{self.api_server_url}/health", timeout=2.0)
            if r.status_code == 200:
                d = r.json()
                loaded = d.get("model_loaded", False)
                self.get_logger().info(f"📡 API 서버: 연결됨, 모델={'✅ 로드됨' if loaded else '⚠️ 미로드'}")
            else:
                self.get_logger().info(f"📡 API 서버: 응답 코드 {r.status_code}")
        except:
             self.get_logger().info("📡 API 서버: 연결 실패")

    def publish_cmd_vel(self, action: dict, source: str):
        twist = Twist()
        twist.linear.x = float(action["linear_x"])
        twist.linear.y = float(action["linear_y"])
        twist.angular.z = float(action.get("angular_z", 0.0))
        self.cmd_pub.publish(twist)
        
        if ROBOT_AVAILABLE and self.driver:
            try:
                if any(abs(v) > 0.1 for v in action.values()):
                    if abs(action.get("angular_z", 0.0)) > 0.1:
                        self.driver.spin(int(action["angular_z"] * self.throttle))
                    elif abs(action["linear_x"]) > 0.1 or abs(action["linear_y"]) > 0.1:
                        angle = np.degrees(np.arctan2(action["linear_y"], action["linear_x"]))
                        self.driver.move(int(angle), self.throttle)
                else:
                    self.driver.stop()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = MobileVLAAPIClient()
    
    # ✅ KEY FIX: MultiThreadedExecutor 사용
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if ROBOT_AVAILABLE and node.driver:
            node.driver.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
