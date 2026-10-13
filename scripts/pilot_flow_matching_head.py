#!/usr/bin/env python3
"""2편 pilot — Flow Matching 연속 액션 헤드 (아키텍처 준비, 실데이터 대기중).

배경:
  1편 MLPActionHead(1560→512→128→8)는 이산 8-class 분류. 2편에서는 같은 프레임
  특징(260dim×6window=1560dim) 입력을 그대로 재사용하고, 출력만 연속 3dim
  액션(lx, ly, az)의 flow-matching 벡터필드 예측으로 바꾼다.

  ⚠️ 중요: 지금 로컬 225ep 캐시(CACHE_V6)는 discrete 모드로 수집되어 실제
  연속 타겟(x1)이 없다 — LUT 고정값(±1.15)만 존재. 진짜 아날로그 연속 액션은
  soda가 `dual`/`continuous` 모드로 재수집해야 얻을 수 있다(핸드오프 문서 참조).
  이 스크립트는 그 데이터가 오기 전까지 아키텍처·학습 루프만 검증하는 용도이며,
  synthetic 데이터로 forward/backward pass만 확인한다. 실제 학습 결과 수치는
  아직 없음 — 지어내지 말 것.

사용법 (실데이터 도착 후):
  X, y1을 real dual-mode 캐시에서 로드하도록 load_real_data()만 교체하면 됨.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FRAME_DIM = 260
WINDOW = 6
ACTION_DIM = 3  # (lx, ly, az)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlowMatchingActionHead(nn.Module):
    """v_theta(x_t, t, frame_feat) -> predicted velocity (x1 - x0), dim=ACTION_DIM.

    frame_feat: 1편과 동일한 1560dim(윈도우 flatten) 조건 입력.
    x_t, t: flow matching 표준 입력(현재 노이즈 섞인 액션, 타임스텝).
    """

    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW, action_dim=ACTION_DIM, hidden=256):
        super().__init__()
        cond_dim = frame_dim * window
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.time_proj = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden + 32 + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim))

    def forward(self, frame_feat, x_t, t):
        # frame_feat: (B, window, frame_dim) 또는 (B, window*frame_dim)
        cond = self.cond_proj(frame_feat.flatten(1))
        tt = self.time_proj(t.view(-1, 1))
        h = torch.cat([cond, tt, x_t], dim=-1)
        return self.head(h)


def flow_matching_loss(model, frame_feat, x1):
    """표준 conditional flow matching loss.

    x0 ~ N(0, I) (노이즈), x1 = 실제 연속 액션 타겟.
    t ~ U[0, 1], x_t = (1-t) x0 + t x1, target v = x1 - x0.
    """
    b = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(b, device=x1.device)
    xt = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1
    target_v = x1 - x0
    pred_v = model(frame_feat, xt, t)
    return ((pred_v - target_v) ** 2).mean()


def load_synthetic_data(n=512):
    """실데이터 도착 전까지 shape/수렴성만 확인하는 synthetic 샘플.

    ⚠️ 결과 수치를 논문/보고서에 쓰면 안 됨 — 아키텍처 sanity check 전용.
    """
    frame_feat = torch.randn(n, WINDOW, FRAME_DIM)
    # 8개 LUT 포인트 중 하나를 살짝 노이즈 섞어 흉내 (진짜 연속분포 아님, 임시)
    lut = torch.tensor([
        [0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [0.0, 1.15, 0.0], [0.0, -1.15, 0.0],
        [1.15, 1.15, 0.0], [1.15, -1.15, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, -0.25],
    ])
    idx = torch.randint(0, 8, (n,))
    x1 = lut[idx] + 0.02 * torch.randn(n, 3)
    return frame_feat, x1


def main():
    print(f"[DEVICE] {DEVICE}")
    print("[DATA] synthetic sanity-check 샘플 사용 (실데이터 없음 — dual 모드 수집 대기중)")
    frame_feat, x1 = load_synthetic_data()
    frame_feat, x1 = frame_feat.to(DEVICE), x1.to(DEVICE)

    model = FlowMatchingActionHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("[SANITY] forward/backward pass 확인 (수렴성 여부만 확인, 실험 결과 아님)")
    for step in range(200):
        opt.zero_grad()
        loss = flow_matching_loss(model, frame_feat, x1)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[MODEL] FlowMatchingActionHead 파라미터 수: {n_params/1e6:.3f}M")
    print("[NEXT] 실데이터(dual 모드) 도착 시 load_synthetic_data()를 real loader로 교체 후 재실행")


if __name__ == "__main__":
    main()
