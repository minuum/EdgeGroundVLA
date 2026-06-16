import time
import threading
import numpy as np
from datetime import datetime
from geometry_msgs.msg import Twist

try:
    from pop.driving import Driving
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False

class VLAControlManager:
    """
    Unified Precision Control Library for Mobile VLA.
    Implements 0.4s Bang-Bang control with robust multi-stop logic.
    """
    def __init__(self, node, default_throttle=50, move_duration=0.4):
        self.node = node  # ROS2 Node instance for logging and publishing
        self.throttle = default_throttle
        self.move_duration = move_duration
        self.command_counter = 0
        self.movement_timer = None
        self.movement_lock = threading.Lock()
        self.current_action = {"lx": 0.0, "ly": 0.0, "az": 0.0}
        self.STOP_ACTION = {"lx": 0.0, "ly": 0.0, "az": 0.0}

        # Initialize Hardware
        if ROBOT_AVAILABLE:
            try:
                self.driver = Driving()
                self.node.get_logger().info("✅ [VLA-Control] Hardware Driving interface initialized.")
            except Exception as e:
                self.node.get_logger().error(f"❌ [VLA-Control] HW Init Fail: {e}")
                self.driver = None
        else:
            self.driver = None

    def publish_and_move(self, lx, ly, az, source="unknown"):
        """Publishes to ROS and commands Hardware directly."""
        self.command_counter += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        is_stop = (abs(lx) < 0.01 and abs(ly) < 0.01 and abs(az) < 0.01)
        action_type = "STOP" if is_stop else "MOVE"
        
        # 1. Logging (Data Collector Style)
        log_msg = (f"📤 [CMD#{self.command_counter}] {timestamp} | "
                   f"Src: {source} | Type: {action_type} | "
                   f"Action: lx={lx:.2f}, ly={ly:.2f}, az={az:.2f}")
        self.node.get_logger().info(log_msg)

        # 2. ROS Topic Publishing
        twist = Twist()
        twist.linear.x = float(lx)
        twist.linear.y = float(ly)
        twist.angular.z = float(az)
        
        # Check if node has cmd_pub attribute, otherwise use internal publisher if provided in subclass
        if hasattr(self.node, 'cmd_pub'):
            self.node.cmd_pub.publish(twist)

        # 3. Hardware Control
        if ROBOT_AVAILABLE and self.driver:
            try:
                if action_type == "MOVE":
                    if abs(az) > 0.1:
                        # Rotation (Spin)
                        self.driver.spin(int(np.sign(az) * self.throttle))
                    else:
                        # translation move
                        angle = np.degrees(np.arctan2(ly, lx))
                        if angle < 0: angle += 360
                        self.driver.move(int(angle), self.throttle)
                else:
                    self.driver.stop()
            except Exception as e:
                self.node.get_logger().error(f"❌ [VLA-Control] HW Drive Error: {e}")
        
        return log_msg

    def robust_stop(self, source="robust_stop", count=5):
        """Sends multiple stop packets to clear ROS/HW buffers."""
        for i in range(count):
            self.publish_and_move(0.0, 0.0, 0.0, source=f"{source}_{i+1}")
            time.sleep(0.05)
        self.current_action = self.STOP_ACTION.copy()

    def move_and_stop_timed(self, lx, ly, az, source="precision_mode"):
        """Standard VLA Move-then-Stop (0.4s) sequence."""
        with self.movement_lock:
            # Cancel existing timer if any
            if self.movement_timer:
                self.movement_timer.cancel()

            # Start Movement
            self.current_action = {"lx": lx, "ly": ly, "az": az}
            log_msg = self.publish_and_move(lx, ly, az, source=source)

            # Start Stop Timer
            def _timed_stop():
                self.robust_stop(source=f"{source}_autostop")

            self.movement_timer = threading.Timer(self.move_duration, _timed_stop)
            self.movement_timer.daemon = True
            self.movement_timer.start()

            return log_msg

    def move_and_stop_ramped(self, lx, ly, az, source="ramp_mode"):
        """Soft ramp-up (50ms, 3-step: 33%→67%→100%) then full speed for remaining duration."""
        with self.movement_lock:
            if self.movement_timer:
                self.movement_timer.cancel()

            self.current_action = {"lx": lx, "ly": ly, "az": az}

            ramp_dur = 0.05       # 50ms 총 ramp-up 시간
            ramp_steps = 3
            step_dur = ramp_dur / ramp_steps  # ~16.7ms per step

            # Soft ramp-up: 33% → 67% → 100%
            for i in range(1, ramp_steps + 1):
                scale = i / ramp_steps
                self.publish_and_move(
                    lx * scale, ly * scale, az * scale,
                    source=f"{source}_ramp{i}"
                )
                time.sleep(step_dur)

            # Full speed for remaining duration
            log_msg = self.publish_and_move(lx, ly, az, source=source)

            remaining = self.move_duration - ramp_dur  # 0.35s

            def _timed_stop():
                self.robust_stop(source=f"{source}_autostop")

            self.movement_timer = threading.Timer(remaining, _timed_stop)
            self.movement_timer.daemon = True
            self.movement_timer.start()

            return log_msg
