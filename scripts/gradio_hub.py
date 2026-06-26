#!/usr/bin/env python3
"""
MoNaVLA 서비스 허브 — 통합 포털
모든 Gradio / FastAPI 서비스의 상태 확인 + 원클릭 접속

Port: 7860

Usage:
  python3 scripts/gradio_hub.py
  python3 scripts/gradio_hub.py --port 7860
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr


# ─── 서비스 정의 ──────────────────────────────────────────────────────────────

SERVICES = [
    {
        "name":   "Object Recognition Demo",
        "port":   7863,
        "script": "scripts/gradio_grounding_demo.py",
        "cmd":    "python3 scripts/gradio_grounding_demo.py",
        "desc":   "Kosmos-2 그라운딩 — 웹캠/이미지 bbox 테스트 + Alias 비교",
        "group":  "Demo",
    },
    {
        "name":   "Inference Dashboard",
        "port":   7865,
        "script": "scripts/gradio_inference_dashboard.py",
        "cmd":    "python3 scripts/gradio_inference_dashboard.py",
        "desc":   "메인 추론 대시보드 — GoalNav 로봇 제어",
        "group":  "Robot",
        "timeout": 180,
    },
    {
        "name":   "Trial Logger",
        "port":   7862,
        "script": "scripts/real_robot_trial_logger.py",
        "cmd":    "python3 scripts/real_robot_trial_logger.py",
        "desc":   "실로봇 trial 기록 — 3-position 평가",
        "group":  "Robot",
    },
    {
        "name":   "Data Collector",
        "port":   8081,
        "script": "scripts/gradio_data_collector.py",
        "cmd":    "python3 scripts/gradio_data_collector.py",
        "desc":   "조이스틱 데이터 수집 — H5 에피소드 기록",
        "group":  "Data",
    },
    {
        "name":   "Session Eval",
        "port":   7861,
        "script": "scripts/gradio_session_eval.py",
        "cmd":    "python3 scripts/gradio_session_eval.py",
        "desc":   "H5 에피소드 품질 평가 — basket 가시성 분석",
        "group":  "Eval",
    },
    {
        "name":   "H5 Analyzer",
        "port":   7866,
        "script": "scripts/gradio_offline_h5_analyzer.py",
        "cmd":    "python3 scripts/gradio_offline_h5_analyzer.py",
        "desc":   "오프라인 H5 분석기 — 프레임 탐색 / 액션 분포",
        "group":  "Eval",
    },
    {
        "name":   "CL Dashboard",
        "port":   7867,
        "script": "scripts/gradio_cl_dashboard.py",
        "cmd":    "python3 scripts/gradio_cl_dashboard.py",
        "desc":   "Closed-Loop 평가 결과 + 실로봇 주행 로그",
        "group":  "Eval",
    },
    {
        "name":   "Monitor",
        "port":   8080,
        "script": "scripts/monitor_dashboard.py",
        "cmd":    "python3 scripts/monitor_dashboard.py",
        "desc":   "시스템 모니터링 대시보드",
        "group":  "System",
    },
    {
        "name":   "Stage2 v2 Server ⭐",
        "port":   8001,
        "script": "robovlm_nav/serve/stage2_v2_inference_server.py",
        "cmd":    "env VLA_S2V2_STAGE2=runs/v5_nav/mlp/stop_lastN/stop_N1.pt VLA_STOP_MODE=learned python3 robovlm_nav/serve/stage2_v2_inference_server.py --port 8001",
        "desc":   "SOTA 추론 서버 — stop_N1 (CL 96.6%, FPE 0.102m) + PG2 grounding + learned STOP",
        "group":  "System",
        "path":   "/health",
        "timeout": 180,
    },
    {
        "name":   "E2E Server [legacy]",
        "port":   8000,
        "script": "robovlm_nav/serve/inference_server.py",
        "cmd":    "python3 robovlm_nav/serve/inference_server.py",
        "desc":   "[legacy] end-to-end Kosmos VLA 서버 — 비교 테스트용",
        "group":  "System",
        "path":   "/health",
        "timeout": 180,
    },
]

GROUP_COLOR = {
    "Demo":   "#1a6b3c",
    "Robot":  "#1a3d6b",
    "Data":   "#5a3d1a",
    "Eval":   "#4a1a6b",
    "System": "#3d3d3d",
}


# ─── 유틸 ─────────────────────────────────────────────────────────────────────

def get_server_ip() -> str:
    # 환경변수로 명시 가능 (Tailscale 등 특정 IP 고정)
    override = os.getenv("VLA_HUB_IP")
    if override:
        return override
    # Tailscale 인터페이스(tailscale0) IP 우선
    try:
        import subprocess
        out = subprocess.check_output(
            ["ip", "addr", "show", "tailscale0"], text=True, timeout=2
        )
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("inet ") and "/" in line:
                return line.split()[1].split("/")[0]
    except Exception:
        pass
    # fallback: 기본 라우팅 IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "localhost"


def is_port_up(port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=timeout):
            return True
    except Exception:
        return False


def get_pid_on_port(port: int) -> str:
    try:
        result = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=2,
        )
        for line in result.stdout.splitlines():
            if str(port) in line and "pid=" in line:
                pid = line.split("pid=")[1].split(",")[0]
                return pid
    except Exception:
        pass
    return ""


# ─── HTML 렌더 ────────────────────────────────────────────────────────────────

def render_hub_html(server_ip: str) -> str:
    groups: dict[str, list] = {}
    for svc in SERVICES:
        g = svc["group"]
        groups.setdefault(g, []).append(svc)

    cards_html = ""
    for group, svcs in groups.items():
        color = GROUP_COLOR.get(group, "#333")
        cards_html += f"""
        <div style="margin-bottom:24px">
          <div style="font-size:12px;font-weight:600;color:#888;text-transform:uppercase;
                      letter-spacing:1px;margin-bottom:10px;padding-left:4px">{group}</div>
          <div style="display:flex;flex-wrap:wrap;gap:12px">
        """
        for svc in svcs:
            up = is_port_up(svc["port"])
            pid = get_pid_on_port(svc["port"]) if up else ""
            dot_color = "#2ecc71" if up else "#e74c3c"
            dot_label = f"UP  pid={pid}" if up else "DOWN"
            url = f"http://{server_ip}:{svc['port']}{svc.get('path', '')}"
            port = svc["port"]

            if up:
                # 실행 중: Open 버튼 + Stop 버튼
                action_html = f"""
                <div style="display:flex;gap:8px">
                  <a href="{url}" target="_blank"
                     style="flex:3;display:block;text-align:center;padding:7px 0;border-radius:6px;
                            text-decoration:none;font-size:13px;font-weight:500;
                            background:#2a5a3a;color:#2ecc71;border:1px solid #2ecc71">Open →</a>
                  <button onclick="triggerStop({port})"
                     style="flex:1;padding:7px 0;border-radius:6px;font-size:12px;font-weight:500;
                            background:#3d1a1a;color:#e74c3c;border:1px solid #c0392b;cursor:pointer">■ Stop</button>
                </div>"""
            else:
                # 꺼져 있음: Start 버튼
                action_html = f"""
                <button onclick="triggerStart({port})"
                   style="width:100%;padding:8px 0;border-radius:6px;font-size:13px;font-weight:600;
                          background:#1a3a2a;color:#27ae60;border:1px solid #27ae60;cursor:pointer;
                          transition:all 0.2s" onmouseover="this.style.background='#2a5a3a'"
                          onmouseout="this.style.background='#1a3a2a'">▶ Start</button>"""

            cards_html += f"""
            <div style="width:280px;background:#1e1e1e;border:1px solid #333;border-radius:10px;
                        padding:16px;border-top:3px solid {color}">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                <div style="width:10px;height:10px;border-radius:50%;background:{dot_color};
                            box-shadow:0 0 6px {dot_color}"></div>
                <span style="font-size:11px;color:{dot_color};font-family:monospace">{dot_label}</span>
                <span style="margin-left:auto;font-size:11px;color:#666;font-family:monospace">:{port}</span>
              </div>
              <div style="font-weight:600;color:#e0e0e0;margin-bottom:6px">{svc['name']}</div>
              <div style="font-size:12px;color:#888;margin-bottom:8px;line-height:1.5">{svc['desc']}</div>
              <div id="hub-status-{port}"
                   style="font-size:11px;font-family:monospace;min-height:16px;
                          margin-bottom:8px;color:#f39c12"></div>
              {action_html}
            </div>
            """
        cards_html += "</div></div>"

    up_count = sum(1 for svc in SERVICES if is_port_up(svc["port"]))
    total = len(SERVICES)
    ts = time.strftime("%H:%M:%S")

    # JS: Start/Stop 버튼 → hidden Gradio textbox + 트리거 버튼 클릭
    js_script = """
    <script>
    var _spinners = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'];
    function triggerStart(port) {
      window.__hub_action = 'start:' + port;
      var sd = document.getElementById('hub-status-' + port);
      if (sd) {
        sd.style.color = '#f39c12';
        var i = 0;
        clearInterval(window['__sp_' + port]);
        window['__sp_' + port] = setInterval(function() {
          var el = document.getElementById('hub-status-' + port);
          if (el) { el.textContent = _spinners[i % 10] + ' 시작 중...'; i++; }
          else { clearInterval(window['__sp_' + port]); }
        }, 120);
      }
      var btn = document.querySelector('#hub-action-btn button');
      if (btn) btn.click();
    }
    function triggerStop(port) {
      window.__hub_action = 'stop:' + port;
      var sd = document.getElementById('hub-status-' + port);
      if (sd) { sd.style.color = '#e74c3c'; sd.textContent = '■ 종료 중...'; }
      var btn = document.querySelector('#hub-action-btn button');
      if (btn) btn.click();
    }
    </script>
    """

    return f"""
    {js_script}
    <div style="background:#111;color:#e0e0e0;font-family:'Segoe UI',sans-serif;
                padding:24px;border-radius:12px;min-height:400px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px">
        <div>
          <div style="font-size:22px;font-weight:700;color:#fff">MoNaVLA Hub</div>
          <div style="font-size:13px;color:#666;margin-top:4px">{server_ip} &nbsp;·&nbsp;
            <span style="color:#2ecc71">{up_count}</span>/<span style="color:#888">{total}</span> running
            &nbsp;·&nbsp; updated {ts}
          </div>
        </div>
      </div>
      {cards_html}
    </div>
    """


# ─── 서비스 시작/종료 ─────────────────────────────────────────────────────────

def start_service(svc: dict) -> str:
    if is_port_up(svc["port"]):
        return f"already running on :{svc['port']}"
    log_name = svc["script"].replace("/", "_").replace(".py", "")
    log_path = ROOT / "logs" / f"{log_name}.log"
    cmd = f"nohup {svc['cmd']} > {log_path} 2>&1 &"
    subprocess.Popen(
        cmd, shell=True, cwd=str(ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        time.sleep(1)
        if is_port_up(svc["port"]):
            return f"started on :{svc['port']}"
    return f"timeout — check {log_path}"


def stop_service(port: int) -> str:
    pid = get_pid_on_port(port)
    if not pid:
        return "not running"
    try:
        subprocess.run(["kill", pid], check=True)
        return f"killed pid={pid}"
    except Exception as e:
        return f"error: {e}"


# ─── Gradio UI ────────────────────────────────────────────────────────────────

def build_hub(server_ip: str) -> gr.Blocks:
    svc_names = [f"{s['name']} (:{s['port']})" for s in SERVICES]
    port_map = {svc["port"]: svc for svc in SERVICES}

    with gr.Blocks(title="MoNaVLA Hub") as demo:
        # 상태 출력 — 카드보다 위에, 눈에 잘 띄게
        ctrl_out = gr.Textbox(label="서비스 상태", lines=2, interactive=False,
                              placeholder="Start / Stop 결과가 여기에 표시됩니다...")

        hub_html = gr.HTML(value=render_hub_html(server_ip))

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")

        # 카드 버튼 → window.__hub_action → js=로 port_input에 주입 → Python에 전달
        port_input  = gr.Textbox(value="", visible=False, elem_id="hub-port-input")
        action_btn  = gr.Button("__action__", visible=False, elem_id="hub-action-btn")

        def do_refresh():
            return render_hub_html(server_ip), ""

        def do_action(cmd: str):
            """cmd 형식: 'start:7863' 또는 'stop:7865' — generator로 실시간 상태 스트리밍"""
            if not cmd or ":" not in cmd:
                yield "", gr.update()
                return
            action, port_str = cmd.split(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                yield "❌ 잘못된 포트", gr.update()
                return

            svc = port_map.get(port)
            if not svc:
                yield f"❌ 포트 {port} 서비스 없음", gr.update()
                return

            if action == "stop":
                result = stop_service(port)
                yield f"■ {svc['name']} 종료: {result}", render_hub_html(server_ip)
                return

            if action != "start":
                yield f"❓ 알 수 없는 액션: {action}", gr.update()
                return

            # ── Start: 매초 상태 업데이트 ──────────────────────────────────
            if is_port_up(port):
                yield f"이미 실행 중: {svc['name']} :{port}", render_hub_html(server_ip)
                return

            log_name = svc["script"].replace("/", "_").replace(".py", "")
            log_path = ROOT / "logs" / f"{log_name}.log"
            launch_cmd = f"setsid nohup {svc['cmd']} > {log_path} 2>&1 &"
            subprocess.Popen(launch_cmd, shell=True, cwd=str(ROOT),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            max_wait = svc.get("timeout", 60)
            spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            for i in range(max_wait):
                sp = spinners[i % len(spinners)]
                # 로그 마지막 줄도 함께 표시
                try:
                    last_log = log_path.read_text(errors="replace").strip().splitlines()
                    log_tail = last_log[-1][:60] if last_log else ""
                except Exception:
                    log_tail = ""
                hint = f"  │ {log_tail}" if log_tail else ""
                yield f"{sp} {svc['name']} 시작 중... ({i+1}/{max_wait}s){hint}", gr.update()
                time.sleep(1)
                if is_port_up(port):
                    pid = get_pid_on_port(port)
                    yield f"✅ {svc['name']} 시작됨 ({i+1}s)  pid={pid}", render_hub_html(server_ip)
                    return

            yield f"⚠️ {svc['name']} 타임아웃 ({max_wait}s) — 로그 확인: logs/{log_path.name}", render_hub_html(server_ip)

        refresh_btn.click(do_refresh, outputs=[hub_html, ctrl_out])
        action_btn.click(
            fn=do_action,
            inputs=[port_input],
            outputs=[ctrl_out, hub_html],
            js="() => [window.__hub_action || '']",
        )

        # 기존 드롭다운 방식도 유지 (Service Control 아코디언)
        with gr.Accordion("Advanced Service Control", open=False):
            with gr.Row():
                svc_dd  = gr.Dropdown(choices=svc_names, label="서비스 선택", scale=3)
                start_b = gr.Button("Start", variant="primary", scale=1)
                stop_b  = gr.Button("Stop",  variant="stop",    scale=1)
            adv_out = gr.Textbox(label="결과", lines=2)

            def do_start(name):
                idx = svc_names.index(name)
                return start_service(SERVICES[idx])

            def do_stop(name):
                idx = svc_names.index(name)
                return stop_service(SERVICES[idx]["port"])

            start_b.click(do_start, inputs=svc_dd, outputs=adv_out).then(
                lambda: render_hub_html(server_ip), outputs=hub_html
            )
            stop_b.click(do_stop, inputs=svc_dd, outputs=adv_out).then(
                lambda: render_hub_html(server_ip), outputs=hub_html
            )

    return demo


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MoNaVLA 서비스 허브")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    server_ip = get_server_ip()
    print(f"[HUB] Server IP: {server_ip}")
    print(f"[HUB] http://0.0.0.0:{args.port}")

    demo = build_hub(server_ip)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
