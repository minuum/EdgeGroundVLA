"""EgVla2FMPolicy 서브골 결측 처리: has_bbox<0.5이면 bbox 창을 전부 0으로 넣는다(드롭아웃 체크포인트 기본). 체크포인트가 없으면 건너뜀."""
from pathlib import Path

import numpy as np
import pytest
import torch

from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy

CK = Path(__file__).resolve().parent.parent / "runs/egvla2_fm/deploy/C_pi0_adaln_drop30_e120.pt"
pytestmark = pytest.mark.skipif(not CK.exists(), reason="deploy checkpoint 없음")


def test_miss_equals_zero_window():
    feat = np.random.RandomState(0).randn(768).astype(np.float32)
    miss = np.array([0.5, 0.6, 0.3, 0.0], np.float32)       # has_bbox=0 (값은 이전 값이 남아 있어도 무시)
    p = EgVla2FMPolicy(CK, k_samples=4, device=torch.device("cpu"))
    assert p.zero_on_miss
    torch.manual_seed(1); a = p.step(miss, feat)
    p2 = EgVla2FMPolicy(CK, k_samples=4, device=torch.device("cpu"), zero_on_miss=False)
    torch.manual_seed(1); b = p2.step(np.zeros(4, np.float32), feat)
    assert a["subgoal_missing"] and not b["subgoal_missing"]
    np.testing.assert_allclose(a["chunk"], b["chunk"], atol=1e-6)


def test_present_not_zeroed():
    feat = np.random.RandomState(0).randn(768).astype(np.float32)
    ok = np.array([0.5, 0.6, 0.3, 1.0], np.float32)
    p = EgVla2FMPolicy(CK, k_samples=4, device=torch.device("cpu"))
    assert not p.step(ok, feat)["subgoal_missing"]
