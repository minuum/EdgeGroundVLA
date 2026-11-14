"""텍스트 조건화 서빙 경로 — 체크포인트 로드·서비스 연동을 모의 추출기로 검증(Florence-2·GPU 불필요)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from robovlm_nav.serve.egvla2_inference_server import EgVla2Service
from robovlm_nav.serve.egvla2_only_logic import EgVla2Controller
from robovlm_nav.serve.egvla2_text_policy import DEFAULT_PHRASE, TextCondPolicy, norm_phrase

ROOT = Path(__file__).resolve().parent.parent
CK = ROOT / "runs/egvla2_fm/textcond/EG4-04_V4_full_s1.pt"


class FakeExt:
    def __init__(self): self.phrase, self.seen = DEFAULT_PHRASE, []
    def set_phrase(self, p): self.phrase = norm_phrase(p); self.seen.append(self.phrase)
    def __call__(self, img): return np.array([0.5, 0.6, 0.05, 1.0], np.float32), np.zeros(1536, np.float32)


def test_norm_phrase():
    assert norm_phrase(None) == DEFAULT_PHRASE and norm_phrase("  ") == DEFAULT_PHRASE and norm_phrase(" flower pot ") == "flower pot"


@pytest.mark.skipif(not CK.exists(), reason="체크포인트 없음")
def test_policy_and_service_roundtrip():
    pol = TextCondPolicy(CK, device=torch.device("cpu"), k_samples=2)
    ext = FakeExt(); svc = EgVla2Service(ext, pol, EgVla2Controller(), meta=dict(ckpt=CK.name, sha256="x", arch=pol.meta["arch"], dtype="fp32", variant=pol.meta.get("variant")))
    r = svc.predict_rgb(np.zeros((8, 8, 3), np.uint8), "go to the gray basket")
    assert r["instruction_used"] == "go to the gray basket" and r["variant"] == "V4_big_vocab" and ext.seen == ["go to the gray basket"]
    r2 = svc.predict_rgb(np.zeros((8, 8, 3), np.uint8), None)
    assert r2["instruction_used"] == DEFAULT_PHRASE


def test_rejects_non_text_checkpoint(tmp_path):
    p = tmp_path / "a.pt"; torch.save(dict(state={}, bdim=12, fdim=768), p)
    with pytest.raises(ValueError):
        TextCondPolicy(p, device=torch.device("cpu"))
