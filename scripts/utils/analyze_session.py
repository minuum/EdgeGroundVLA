import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_session_log(json_path: str):
    """
    VLA 주행 세션 로그를 로드하여 Latency 통계, 액션 변환 MAE,
    라벨 분포 등을 계산하고 시각화합니다.
    """
    path = Path(json_path)
    if not path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {json_path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print(f"📊 세션 분석 리포트: {data['session_id']}")
    print(f"🗓️ 시작 시간: {data['timestamp']}")
    print(f"🤖 사용 모델: {data['model_name']}")
    print(f"💬 주행 지시어: {data['instruction']}")
    print(f"📡 추론 모드: {data['instruction_mode']}")
    print("=" * 60)

    history = data.get("history", [])
    if not history:
        print("⚠️ 세션 기록이 비어 있습니다.")
        return

    steps = []
    latencies = []
    labels = []
    raw_actions = []
    snapped_actions = []

    # Step 1은 [0, 0, 0] dummy action이므로 통계에서 제외할 수 있도록 처리
    for item in history:
        step_idx = item["step"]
        # Step 1인 초기화 스텝은 latency가 0이므로 제외
        if step_idx == 1:
            continue
        
        steps.append(step_idx)
        latencies.append(item.get("latency_ms", 0.0))
        labels.append(item.get("predicted_label", "UNKNOWN"))
        raw_actions.append(item.get("raw_action", [0.0, 0.0, 0.0]))
        snapped_actions.append(item.get("action", [0.0, 0.0, 0.0]))

    # 1. Latency 통계 분석
    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    max_lat = np.max(latencies)
    min_lat = np.min(latencies)
    p95_lat = np.percentile(latencies, 95)

    print(f"🔹 [1] 추론 레이턴시(Latency) 정량 메트릭 (총 {len(steps)} 스텝)")
    print(f"  - 평균 Latency: {mean_lat:.2f} ms")
    print(f"  - 최대 Latency: {max_lat:.2f} ms (웜업 포함)")
    print(f"  - 최소 Latency: {min_lat:.2f} ms")
    print(f"  - 95% 백분위수: {p95_lat:.2f} ms")
    print("-" * 60)

    # 2. 액션 스냅핑(Snap) 오차 분석
    raw_arr = np.array(raw_actions)
    snap_arr = np.array(snapped_actions)
    mae = np.mean(np.abs(raw_arr - snap_arr), axis=0)

    print(f"🔹 [2] MoNa-pi 연속 액션 → 이산 안전 액션 변환 오차 (MAE)")
    print(f"  - Linear X (전진속도) 평균 오차: {mae[0]:.4f}")
    print(f"  - Linear Y (좌우속도) 평균 오차: {mae[1]:.4f}")
    print(f"  - Angular Z (회전속도) 평균 오차: {mae[2]:.4f}")
    print("-" * 60)

    # 3. 라벨 분포 분석
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"🔹 [3] 예측 주행 라벨(predicted_label) 분포")
    for lbl, cnt in zip(unique_labels, counts):
        pct = (cnt / len(labels)) * 100
        print(f"  - {lbl}: {cnt}회 ({pct:.1f}%)")
    print("=" * 60)

    # 4. 시각화 그래프 생성 및 저장
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Latency 추이 그래프
    axs[0].plot(steps, latencies, "r-o", linewidth=2, label="Latency (ms)")
    axs[0].axhline(mean_lat, color="blue", linestyle="--", alpha=0.7, label=f"Mean ({mean_lat:.1f}ms)")
    axs[0].set_title("VLA Inference Latency Trend", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Inference Step")
    axs[0].set_ylabel("Latency (ms)")
    axs[0].grid(True, linestyle="--", alpha=0.6)
    axs[0].legend()

    # Action 비교 그래프 (Linear X, Y)
    axs[1].plot(steps, raw_arr[:, 0], "g--", alpha=0.6, label="Raw Linear X")
    axs[1].plot(steps, snap_arr[:, 0], "g-", linewidth=2, label="Snapped Linear X")
    axs[1].plot(steps, raw_arr[:, 1], "b--", alpha=0.6, label="Raw Linear Y")
    axs[1].plot(steps, snap_arr[:, 1], "b-", linewidth=2, label="Snapped Linear Y")
    axs[1].set_title("Raw vs Snapped Control Velocity", fontsize=12, fontweight="bold")
    axs[1].set_xlabel("Inference Step")
    axs[1].set_ylabel("Velocity Value")
    axs[1].grid(True, linestyle="--", alpha=0.6)
    axs[1].legend()

    plt.tight_layout()
    output_img = path.parent / f"{path.stem}_analysis.png"
    plt.savefig(output_img, dpi=150)
    print(f"💾 분석 그래프가 저장되었습니다: {output_img}")
    print("=" * 60)

if __name__ == "__main__":
    # 기본 분석 대상 세션 파일 지정
    target_session = "/home/soda/MoNaVLA/docs/inference_reports/session_20260604_085923.json"
    analyze_session_log(target_session)
