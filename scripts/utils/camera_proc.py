"""
카메라 프로세스 제어 유틸 — ros2 run camera_pub usb_camera_service_server

각 Gradio 페이지에서 import 해서 사용:
    from scripts.utils.camera_proc import camera_control_widget
    camera_control_widget()   # gr.Blocks 컨텍스트 안에서 호출
"""
from __future__ import annotations

import subprocess
import time

import gradio as gr

import os
from pathlib import Path
_project_root = Path(__file__).resolve().parents[2]

ROS_SETUP     = "/opt/ros/humble/setup.bash"
ROS_WS_SETUP  = os.getenv("VLA_ROS_WS", str(_project_root / "ROS_action")) + "/install/setup.bash"
KILL_PATTERN  = "usb_camera_service_server"
_ROS_DIST = "/opt/ros/humble"
_ROS_WS   = "/home/soda/MoNaVLA/ROS_action/install"
START_CMD = (
    f"export PATH={_ROS_DIST}/bin:$PATH; "
    f"export PYTHONPATH={_ROS_DIST}/local/lib/python3.10/dist-packages"
    f":{_ROS_DIST}/lib/python3.10/site-packages"
    f":{_ROS_WS}/camera_interfaces/local/lib/python3.10/dist-packages"
    f":{_ROS_WS}/camera_pub/local/lib/python3.10/dist-packages:$PYTHONPATH; "
    f"export LD_LIBRARY_PATH={_ROS_DIST}/lib"
    f":{_ROS_WS}/camera_interfaces/lib:{_ROS_WS}/camera_pub/lib:$LD_LIBRARY_PATH; "
    f"source {_ROS_DIST}/setup.bash 2>/dev/null; "
    f"cd {_ROS_WS} && source setup.bash 2>/dev/null; "
    "export ROS_DOMAIN_ID=42; "
    "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp; "
    "nohup ros2 run camera_pub usb_camera_service_server "
    "> /tmp/camera_proc.log 2>&1 &"
)


def is_camera_running() -> bool:
    r = subprocess.run(["pgrep", "-f", KILL_PATTERN], capture_output=True)
    return r.returncode == 0


def get_camera_pid() -> str:
    r = subprocess.run(["pgrep", "-f", KILL_PATTERN], capture_output=True, text=True)
    pids = r.stdout.strip().split()
    return ", ".join(pids) if pids else ""


def camera_status_text() -> str:
    if is_camera_running():
        return f"🟢 카메라 실행 중  pid={get_camera_pid()}"
    return "🔴 카메라 정지됨"


def start_camera():
    """generator — Start 버튼 handler"""
    if is_camera_running():
        yield f"🟢 이미 실행 중  pid={get_camera_pid()}"
        return
    subprocess.Popen(["bash", "-c", START_CMD])
    spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    for i in range(10):
        yield f"{spinners[i % 10]} 카메라 시작 중... ({i+1}s)"
        time.sleep(1)
        if is_camera_running():
            yield f"🟢 시작됨  pid={get_camera_pid()}"
            return
    yield f"⚠️ 타임아웃 — 로그: /tmp/camera_proc.log"


def stop_camera() -> str:
    subprocess.run(["pkill", "-f", KILL_PATTERN])
    time.sleep(0.4)
    return camera_status_text()


def camera_control_widget():
    """
    gr.Blocks 컨텍스트 안에서 호출 → 카메라 Start/Stop 위젯 한 Row 추가.

    반환: (cam_status, cam_start_btn, cam_stop_btn)
    - cam_start_btn / cam_stop_btn 은 내부에서 click 미연결 상태로 반환.
    - 호출측에서 반드시 직접 연결할 것:
        cam_start_btn.click(fn=start_camera, outputs=cam_status)
        cam_stop_btn.click(fn=stop_camera,   outputs=cam_status)
    - timer 연동이 필요하면 .then() 체인 추가:
        cam_start_btn.click(...).then(lambda: gr.update(active=True), outputs=timer)
    """
    with gr.Row():
        cam_status = gr.Textbox(
            value=camera_status_text(),
            label="카메라 프로세스",
            interactive=False,
            scale=4,
        )
        cam_start = gr.Button("▶ 카메라 시작", variant="primary", scale=1, size="sm")
        cam_stop  = gr.Button("■ 정지",        variant="stop",    scale=1, size="sm")
        cam_ref   = gr.Button("↻",                                scale=0, size="sm")

    cam_ref.click(fn=camera_status_text, outputs=cam_status)

    return cam_status, cam_start, cam_stop
