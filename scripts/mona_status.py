# -*- coding: utf-8 -*-
# MoNaVLA Ablation 자동 모니터링 및 상태 대시보드 툴 (Mona Status Tracker)

import os
import re
import subprocess
import sys
from datetime import datetime

def get_process_status():
    """현재 백그라운드에서 구동 중인 ablation 쉘 스크립트 및 main.py 프로세스를 탐색합니다."""
    sh_pid = None
    py_pid = None
    py_cmd = None

    try:
        # ps aux 결과를 파이프라인으로 조회
        ps_out = subprocess.check_output(["ps", "aux"]).decode("utf-8")
        for line in ps_out.splitlines():
            if "run_v5_ablations.sh" in line and "grep" not in line:
                parts = line.split()
                sh_pid = parts[1]
            if "third_party/RoboVLMs/main.py" in line and "grep" not in line:
                parts = line.split()
                py_pid = parts[1]
                py_cmd = " ".join(parts[10:])
    except Exception as e:
        print(f"⚠️ 프로세스 조회 중 오류: {e}")

    return sh_pid, py_pid, py_cmd

def get_gpu_status():
    """nvidia-smi 명령어를 통해 실시간 GPU VRAM 및 사용률 정보를 파싱합니다."""
    try:
        smi_out = subprocess.check_output([
            "nvidia-smi", 
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu", 
            "--format=csv,noheader,nounits"
        ]).decode("utf-8").strip()
        
        gpu_info = []
        for line in smi_out.splitlines():
            idx, name, util, mem_used, mem_total, temp = line.split(", ")
            gpu_info.append({
                "index": idx,
                "name": name,
                "util": f"{util}%",
                "vram": f"{mem_used}MB / {mem_total}MB",
                "temp": f"{temp}°C"
            })
        return gpu_info
    except Exception:
        return None

def parse_overall_progress():
    """overall.log를 통해 완료된 실험과 현재 진행 중인 실험을 분석합니다."""
    overall_log = "logs/v5_ablation/overall.log"
    if not os.path.exists(overall_log):
        return [], None

    completed = []
    current = None

    with open(overall_log, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if "완료:" in line:
            exp = line.replace("완료:", "").strip()
            completed.append(exp)
        elif "시작:" in line:
            current = line.replace("시작:", "").strip()

    # 이미 완료 리스트에 있는데 시작으로 남아있다면 대기 중이거나 교체된 것
    if current in completed:
        current = None

    return completed, current

def parse_current_run_metrics(current_exp):
    """현재 구동 중인 실험의 로그파일을 파싱하여 에폭, 스텝, 손실(Loss)을 가져옵니다."""
    if not current_exp:
        return None

    log_path = f"logs/v5_ablation/{current_exp}.log"
    if not os.path.exists(log_path):
        return {"status": "Config 준비됨 (로그 시작 대기)"}

    epoch = "N/A"
    step = "N/A"
    loss = "N/A"
    progress = "fresh"

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            # 메모리 절약을 위해 뒷부분만 읽음
            f.seek(0, 2)
            file_size = f.tell()
            # 마지막 20KB 정도만 읽음
            read_size = min(20480, file_size)
            f.seek(file_size - read_size)
            lines = f.readlines()

        # PyTorch Lightning 로그 포맷 파싱
        # 예: Epoch 1:  90%|█████████ | 3180/3534 [01:10<00:07, 45.34it/s, v_num=1, loss=0.485]
        for line in reversed(lines):
            if "Epoch" in line and "loss=" in line:
                epoch_match = re.search(r"Epoch\s+(\d+)", line)
                loss_match = re.search(r"loss=([\d\.]+)", line)
                step_match = re.search(r"(\d+/\d+)", line)
                
                if epoch_match:
                    epoch = epoch_match.group(1)
                if loss_match:
                    loss = loss_match.group(1)
                if step_match:
                    step = step_match.group(1)
                
                progress = line.strip()
                break
            elif "Loading checkpoint shards" in line:
                progress = "가중치 로딩 중..."
            elif "LOCAL_RANK" in line:
                progress = "학습 초기화 중..."
    except Exception as e:
        progress = f"파싱 중 에러: {e}"

    return {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "progress": progress
    }

def print_dashboard():
    """모니터링 대시보드를 터미널에 깔끔하게 출력합니다."""
    os.system("clear")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("=" * 70)
    print(f" 🤖  [Mona] MoNaVLA Ablation Study Status Dashboard  ({now})")
    print("=" * 70)

    # 1. 프로세스 상태
    sh_pid, py_pid, py_cmd = get_process_status()
    print("\n[1] 시스템 프로세스 상태")
    if sh_pid:
        print(f"  ● Ablation 스케줄러 : RUNNING (PID: {sh_pid})")
    else:
        print("  ○ Ablation 스케줄러 : STOPPED")
        
    if py_pid:
        print(f"  ● 파이썬 학습 프로세스 : RUNNING (PID: {py_pid})")
        # 간단히 명령 축소 출력
        short_cmd = py_cmd[:60] + "..." if len(py_cmd) > 60 else py_cmd
        print(f"    - 실행 명령: {short_cmd}")
    else:
        print("  ○ 파이썬 학습 프로세스 : IDLE")

    # 2. GPU 상태
    gpu_info = get_gpu_status()
    if gpu_info:
        print("\n[2] GPU 리소스 상태")
        for gpu in gpu_info:
            print(f"  - GPU [{gpu['index']}] {gpu['name']} | Util: {gpu['util']} | VRAM: {gpu['vram']} | Temp: {gpu['temp']}")

    # 3. 전체 실험 진행 내역
    completed, current = parse_overall_progress()
    print("\n[3] Ablation 실험 스케줄 진행률")
    
    # 8가지 순차 진행률 계산
    total_experiments = 8
    progress_percent = (len(completed) / total_experiments) * 100
    
    # 진행바 그리기
    bar_len = 20
    filled_len = int(round(bar_len * len(completed) / total_experiments))
    bar = "█" * filled_len + "-" * (bar_len - filled_len)
    
    print(f"  진행 바 : |{bar}| {len(completed)}/{total_experiments} 완료 ({progress_percent:.1f}%)")
    
    print("\n  실험 목록 상태:")
    all_experiments = [
        "v5_ablation_top2_proj_frozen", "v5_ablation_top2_proj_tuned",
        "v5_ablation_top4_proj_frozen", "v5_ablation_top4_proj_tuned",
        "v5_ablation_top6_proj_frozen", "v5_ablation_top6_proj_tuned",
        "v5_ablation_top8_proj_frozen", "v5_ablation_top8_proj_tuned"
    ]
    
    for exp in all_experiments:
        if exp in completed:
            print(f"   [✅ 완료] {exp}")
        elif exp == current:
            print(f"   [🔥 실행] {exp}")
        else:
            print(f"   [⏳ 대기] {exp}")

    # 4. 현재 진행 중인 실험 상세 메트릭
    if current:
        print(f"\n[4] 현재 실험 디테일 ({current})")
        metrics = parse_current_run_metrics(current)
        if metrics:
            if "status" in metrics:
                print(f"  - 상태 : {metrics['status']}")
            else:
                print(f"  - 현재 에폭 : {metrics['epoch']} / 5")
                print(f"  - 스텝 진행 : {metrics['step']}")
                print(f"  - 최신 Loss  : {metrics['loss']}")
                print(f"  - 최신 로그  : {metrics['progress'][:75]}")
    else:
        print("\n[4] 현재 실행 중인 실험 상세 메트릭 정보가 없습니다.")

    print("\n" + "=" * 70)
    print(" 💡 실시간 모니터링 실행: watch -n 10 python3 scripts/mona_status.py")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    print_dashboard()
