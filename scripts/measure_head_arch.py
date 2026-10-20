#!/usr/bin/env python3
"""실험군 헤드(MLP 회귀 / FM-MLP / FM-FiLM / FM-π0)의 파라미터 수와 헤드 연산 시간(K=8 샘플, Euler 10스텝)을 측정해 JSON으로 저장한다.
실행: .venv_lerobot/bin/python3 scripts/measure_head_arch.py   출력: docs/v5/egvla2_fm/head_arch_stats.json"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robovlm_nav.models.egvla2_fm_heads import ARCH, MLPReg, STEPS_EULER, sample_chunk  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cls = {"MLP_regression": MLPReg, "A_mlp_fm": ARCH["A_mlp_fm"], "B2_film_res_fm": ARCH["B2_film_res_fm"], "C_pi0_adaln": ARCH["C_pi0_adaln"]}
bb, ft = torch.randn(1, 12, device=dev), torch.randn(1, 768, device=dev); out = {}
for n, c in cls.items():
    m = c(12, 768).to(dev).eval(); p = sum(x.numel() for x in m.parameters())
    if n == "MLP_regression":
        with torch.no_grad():
            [m(bb, ft) for _ in range(5)]; torch.cuda.synchronize(); t = time.time(); [m(bb, ft) for _ in range(50)]; torch.cuda.synchronize(); ms = (time.time() - t) / 50 * 1000
    else:
        [sample_chunk(m, bb, ft, 8) for _ in range(3)]; torch.cuda.synchronize(); t = time.time(); [sample_chunk(m, bb, ft, 8) for _ in range(20)]; torch.cuda.synchronize(); ms = (time.time() - t) / 20 * 1000
    out[n] = dict(params=p, head_ms=round(ms, 3), device=str(dev), k=8 if n != "MLP_regression" else 1, steps=STEPS_EULER if n != "MLP_regression" else 1)
    print(n, out[n])
(ROOT / "docs/v5/egvla2_fm/head_arch_stats.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
