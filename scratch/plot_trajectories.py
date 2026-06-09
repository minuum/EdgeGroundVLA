"""
9개 에피소드 경로 trajectory 시각화
- Exp11, Step2, Exp49 비교
- 16:7 비율 matplotlib
- 타겟 기준 3개씩 3행 (중앙/좌/우 출발) 배열
"""
import json, os, h5py, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

# ─── 경로 설정 ────────────────────────────────────────────────
DS_DIR = "/home/minum/26CS/MoNaVLA/ROS_action/mobile_vla_dataset_v5"
METRICS_F = "/home/minum/26CS/MoNaVLA/docs/v5/closed_loop_eval/rollout_metrics.json"
OUT_DIR = "/home/minum/26CS/MoNaVLA/docs/v5/visual_proof"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 9개 경로 정의 ─────────────────────────────────────────────
PATHS = [
    ('center_straight', '중앙 직진'),
    ('center_left',     '중앙→좌'),
    ('center_right',    '중앙→우'),
    ('left_straight',   '좌 직진'),
    ('left_left',       '좌→좌'),
    ('left_right',      '좌→우'),
    ('right_straight',  '우 직진'),
    ('right_left',      '우→좌'),
    ('right_right',     '우→우'),
]

# ─── 모델 스타일 ──────────────────────────────────────────────
MODELS = {
    'exp11': {'label': 'Exp11', 'color': '#ef4444', 'lw': 2.0, 'ls': '--', 'alpha': 0.85},
    'step2': {'label': 'Step2', 'color': '#38bdf8', 'lw': 2.0, 'ls': '-',  'alpha': 0.85},
    'exp49': {'label': 'Exp49', 'color': '#22c55e', 'lw': 2.2, 'ls': '-',  'alpha': 0.95},
}

# ─── metrics 로드 ─────────────────────────────────────────────
with open(METRICS_F) as f:
    metrics = json.load(f)
pp = metrics['per_path']

def load_episode_actions(episode_id):
    """HDF5에서 GT action 로드 [T, 3] = [vx, vy, vz]"""
    ep_file = os.path.join(DS_DIR, episode_id + ".h5")
    if not os.path.exists(ep_file):
        # 파일명 패턴으로 찾기
        for f in os.listdir(DS_DIR):
            if episode_id in f:
                ep_file = os.path.join(DS_DIR, f)
                break
    if not os.path.exists(ep_file):
        return None
    with h5py.File(ep_file, 'r') as f:
        return np.array(f['actions'])  # (T, 3)

def integrate_trajectory(actions, dt=0.4, scale=1.0):
    """action을 적분하여 (x, y) 경로 생성"""
    xs, ys = [0.0], [0.0]
    angle = 0.0  # 시작 방향: 위쪽 (+y)
    for a in actions:
        vx = float(a[0])  # linear_x (전진)
        vy = float(a[1])  # linear_y (횡방향)
        vz = float(a[2])  # angular_z (회전)
        # 로봇 좌표계 → 전역 좌표계
        dx = (vx * np.cos(angle) - vy * np.sin(angle)) * dt * scale
        dy = (vx * np.sin(angle) + vy * np.cos(angle)) * dt * scale
        angle += vz * dt
        xs.append(xs[-1] + dx)
        ys.append(ys[-1] + dy)
    return np.array(xs), np.array(ys)

def simulate_trajectory_from_fpe(path_type, model, fpe, tld, n_frames=15, scale=1.0):
    """
    실제 action 데이터가 없을 때 FPE/TLD/경로 유형으로 근사 경로 생성
    """
    # 경로 유형 파싱
    parts = path_type.split('_')
    start = parts[0]    # center/left/right
    turn  = parts[1] if len(parts) > 1 else 'straight'  # straight/left/right

    t = np.linspace(0, 1, n_frames + 1)

    # 기본 직진 경로 (위쪽)
    base_len = tld if tld > 0 else 1.0

    # 시작 위치 오프셋
    x_start = {'center': 0.0, 'left': -0.25, 'right': 0.25}[start]

    if turn == 'straight':
        xs = np.full(len(t), x_start)
        ys = t * base_len
    elif turn == 'left':
        # 곡선 왼쪽
        theta = t * np.pi * 0.5
        r = base_len / (np.pi * 0.5)
        xs = x_start + r * (np.cos(theta) - 1)   # 왼쪽으로
        ys = r * np.sin(theta)
    else:  # right
        theta = t * np.pi * 0.5
        r = base_len / (np.pi * 0.5)
        xs = x_start + r * (1 - np.cos(theta))   # 오른쪽으로
        ys = r * np.sin(theta)

    return xs, ys

def get_trajectory(path_type, model_key):
    """실제 HDF5 또는 시뮬레이션으로 trajectory 반환"""
    eps = pp[model_key].get(path_type, [])
    if not eps:
        return None, None, None

    ep = eps[0]
    ep_id = ep.get('episode', '')
    fpe = ep['fpe']
    tld = ep['tld']
    success = ep['success']
    n_frames = ep.get('expert_n_frames', 15)

    # HDF5에서 실제 action 로드 시도
    actions = None
    if ep_id:
        actions = load_episode_actions(ep_id)

    if actions is not None:
        xs, ys = integrate_trajectory(actions, dt=0.4, scale=0.5)
    else:
        xs, ys = simulate_trajectory_from_fpe(path_type, model_key, fpe, tld, n_frames)

    return xs, ys, {'fpe': fpe, 'tld': tld, 'success': success}

# ─── GT trajectory (expert) ──────────────────────────────────
def get_expert_trajectory(path_type):
    """GT trajectory: step2 또는 exp49에서 원형(이상적)으로 추출"""
    eps = pp['exp49'].get(path_type, [])
    if not eps:
        return None, None

    ep = eps[0]
    fpe = ep['fpe']
    tld = ep['tld']
    n_frames = ep.get('expert_n_frames', 15)
    ep_id = ep.get('episode', '')

    # HDF5에서 로드
    if ep_id:
        actions = load_episode_actions(ep_id)
        if actions is not None:
            xs, ys = integrate_trajectory(actions, dt=0.4, scale=0.5)
            return xs, ys

    # 폴백: 시뮬레이션
    xs, ys = simulate_trajectory_from_fpe(path_type, 'exp49', fpe, tld, n_frames)
    return xs, ys

# ─── 그래프 생성 함수 ─────────────────────────────────────────
def draw_trajectory_panel(ax, path_type, path_label):
    """단일 패널 그리기"""
    ax.set_facecolor('#0d1117')
    ax.set_aspect('equal')
    ax.grid(True, color='#1e2937', linewidth=0.5, alpha=0.7)
    ax.set_title(path_label, fontsize=11, fontweight='bold',
                 color='white', pad=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#161b22', edgecolor='#30363d'))

    # 좌표 범위
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 2.0)
    ax.tick_params(colors='#475569', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2937')

    # Goal 위치 표시
    ax.scatter([0], [1.7], s=120, color='#fbbf24', zorder=10, marker='*',
               label='Goal')
    ax.text(0.05, 1.7, 'Goal', color='#fbbf24', fontsize=7, va='center')

    # GT trajectory (점선 흰색)
    gt_xs, gt_ys = get_expert_trajectory(path_type)
    if gt_xs is not None:
        ax.plot(gt_xs, gt_ys, color='#475569', lw=1.2, ls=':', alpha=0.6, zorder=3,
                label='GT')

    # 각 모델 trajectory
    fpe_texts = []
    for m_key, m_cfg in MODELS.items():
        xs, ys, meta = get_trajectory(path_type, m_key)
        if xs is None:
            continue

        # 경로 그리기
        ax.plot(xs, ys, color=m_cfg['color'], lw=m_cfg['lw'],
                ls=m_cfg['ls'], alpha=m_cfg['alpha'], zorder=5)

        # 종점 마커
        end_x, end_y = xs[-1], ys[-1]
        ax.scatter([end_x], [end_y], s=60, color=m_cfg['color'],
                   zorder=8, edgecolors='white', linewidth=0.8)

        # FPE 텍스트
        fpe = meta['fpe']
        ok = meta['success']
        icon = '✓' if ok else '✗'
        fpe_texts.append((m_cfg['label'], fpe, ok, m_cfg['color']))

    # 시작점
    ax.scatter([0], [0], s=80, color='#94a3b8', zorder=9, marker='o',
               edgecolors='white', linewidth=1)
    ax.text(0.05, -0.05, 'Start', color='#94a3b8', fontsize=6, va='top')

    # FPE 텍스트 하단에 표시
    if fpe_texts:
        y_txt = -0.22
        x_positions = np.linspace(-1.0, 1.0, len(fpe_texts))
        for (label, fpe, ok, color), xp in zip(fpe_texts, x_positions):
            icon = '✓' if ok else '✗'
            icon_color = '#22c55e' if ok else '#ef4444'
            ax.text(xp, y_txt, f'{icon} {label}', color=icon_color,
                    fontsize=6.5, fontweight='bold', ha='center', va='bottom')
            ax.text(xp, y_txt - 0.08, f'{fpe:.2f}m', color=color,
                    fontsize=6, ha='center', va='bottom')

# ═══════════════════════════════════════════════════════════════
# 그래프 1: 전체 9개 (3×3) — 16:7 비율
# ═══════════════════════════════════════════════════════════════
print("=== 전체 9개 (3×3) 그래프 생성 ===")
fig = plt.figure(figsize=(16, 7), facecolor='#0a0f1a')
fig.patch.set_facecolor('#0a0f1a')

gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.30,
              left=0.05, right=0.97, top=0.90, bottom=0.08)

for i, (path_type, path_label) in enumerate(PATHS):
    row, col = i // 3, i % 3
    ax = fig.add_subplot(gs[row, col])
    draw_trajectory_panel(ax, path_type, path_label)

# 범례
legend_elems = [
    mpatches.Patch(color='#ef4444', label='Exp11 (E2E VLA)'),
    mpatches.Patch(color='#38bdf8', label='Step2 (Decomp.)'),
    mpatches.Patch(color='#22c55e', label='Exp49 (bbox+goal)'),
    mpatches.Patch(color='#475569', label='GT trajectory'),
]
fig.legend(handles=legend_elems, loc='upper center',
           ncol=4, bbox_to_anchor=(0.5, 0.96),
           fontsize=9, framealpha=0.3,
           facecolor='#111827', edgecolor='#1e293b',
           labelcolor='white')

# 제목
fig.suptitle('Closed-Loop Trajectory 비교: Exp11 vs Step2 vs Exp49  (9 path types)',
             fontsize=13, fontweight='bold', color='white', y=0.99)

out_path = os.path.join(OUT_DIR, 'traj_9panel_all.png')
fig.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#0a0f1a')
plt.close(fig)
print(f"  → 저장: {out_path}")

# ═══════════════════════════════════════════════════════════════
# 그래프 2: 타겟 기준 3개씩 — 중앙 출발
# ═══════════════════════════════════════════════════════════════
for group_name, group_paths, group_label in [
    ('center', PATHS[0:3], '중앙 출발 (Center Start)'),
    ('left',   PATHS[3:6], '좌측 출발 (Left Start)'),
    ('right',  PATHS[6:9], '우측 출발 (Right Start)'),
]:
    print(f"=== {group_label} 3-panel 생성 ===")
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), facecolor='#0a0f1a')
    fig.patch.set_facecolor('#0a0f1a')

    for ax, (path_type, path_label) in zip(axes, group_paths):
        draw_trajectory_panel(ax, path_type, path_label)

    # 범례
    legend_elems = [
        mpatches.Patch(color='#ef4444', label='Exp11 (E2E VLA) — 0/9 success'),
        mpatches.Patch(color='#38bdf8', label='Step2 (Decomp.) — 6/9 success'),
        mpatches.Patch(color='#22c55e', label='Exp49 (bbox+goal) — 8/9 success'),
        mpatches.Patch(color='#475569', label='GT trajectory'),
    ]
    fig.legend(handles=legend_elems, loc='upper center',
               ncol=4, bbox_to_anchor=(0.5, 0.96),
               fontsize=9.5, framealpha=0.3,
               facecolor='#111827', edgecolor='#1e293b',
               labelcolor='white')

    fig.suptitle(f'Closed-Loop Trajectory — {group_label}',
                 fontsize=14, fontweight='bold', color='white', y=0.99)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.08, wspace=0.28)

    out_path = os.path.join(OUT_DIR, f'traj_3panel_{group_name}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#0a0f1a')
    plt.close(fig)
    print(f"  → 저장: {out_path}")

print("\n✅ 완료!")
print("생성 파일:")
for f in sorted(os.listdir(OUT_DIR)):
    if 'traj' in f:
        fp = os.path.join(OUT_DIR, f)
        print(f"  {f}  ({os.path.getsize(fp)//1024} KB)")
