"""EGVLA2 서빙 정책: Florence-2 fused 특징(bbox 4-d + 풀링 768-d) → FM 액션헤드 → 속력 규칙(action_cont).

plan: docs/plans/plan_20261016_egvla2_serving_path.md §1~2. 학습과 같은 입력 규칙을 쓴다:
- 입력 창: 최근 WINDOW(3)프레임의 bbox [cx,cy,area,has_bbox]를 이어 붙임(앞이 부족하면 첫 프레임 반복) + 현재 768-d 특징
- 샘플: Euler 10스텝 K개 평균(청크 H=8), 첫 스텝 (lx,ly,az)를 속력 규칙에 전달
- 특징 추출은 `scripts/precompute_v6_florence2_fused.run_batch`를 그대로 쓴다(재구현 금지). 입력은 RGB(h5의 BGR은 학습 시 뒤집어 RGB로 변환됨).
체크포인트 규격: `scripts/train_egvla2_s1.py SAVE_ONLY=1`이 저장하는 dict(arch,state,bdim,fdim,window,H,...).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robovlm_nav.models.egvla2_fm_heads import ARCH, K_SAMPLES, sample_chunk
from robovlm_nav.serve.continuous_speed import to_drive_command

ROOT = Path(__file__).resolve().parents[2]


class FusedF2Extractor:
    """Florence-2 base fp16, phrase grounding('gray basket') + 풀링 특징을 1회 통과로 얻는다."""

    def __init__(self, device: torch.device | None = None):
        sys.path.insert(0, str(ROOT / "scripts"))
        from transformers import AutoModelForCausalLM, AutoProcessor
        import precompute_v6_florence2_fused as P
        self._P = P
        self.device = device or P.DEV
        self.model = AutoModelForCausalLM.from_pretrained(P.MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(P.MODEL, trust_remote_code=True)

    def __call__(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from PIL import Image
        pil = Image.fromarray(np.ascontiguousarray(image_rgb).astype(np.uint8)).convert("RGB")
        bbox, feat = self._P.run_batch(self.model, self.proc, [pil])
        return bbox[0].astype(np.float32), feat[0].astype(np.float32)


class EgVla2FMPolicy:
    def __init__(self, ckpt_path: str | Path, device: torch.device | None = None, *, k_samples: int = K_SAMPLES,
                 speed_source: str = "shadow", s_max: float | None = None, zero_on_miss: bool | None = None, ema_alpha: float = 0.0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.meta = {k: v for k, v in ck.items() if k not in ("state", "files")}
        self.window = int(ck["window"])
        self.model = ARCH[ck["arch"]](ck["bdim"], ck["fdim"]).to(self.device)
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.k, self.speed_source, self.s_max = k_samples, speed_source, s_max
        # 서브골(bbox) 결측(has_bbox<0.5)이면 창 전체를 0으로: 서브골 드롭아웃으로 학습한 체크포인트(subgoal_dropout>0)의 결측 표현과 같다.
        self.zero_on_miss = bool(ck.get("subgoal_dropout", 0) > 0) if zero_on_miss is None else zero_on_miss
        self.ema_alpha, self._ema = float(ema_alpha), None   # 0이면 끔. 첫 스텝 (lx,ly,az)에 지수이동평균(새 값 가중 alpha)
        self._hist: list[np.ndarray] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._hist = []
            self._ema = None

    def _window(self) -> np.ndarray:
        t = len(self._hist) - 1
        return np.concatenate([self._hist[max(0, t - self.window + 1 + i)] for i in range(self.window)])

    @torch.no_grad()
    def step(self, bbox: np.ndarray, feat: np.ndarray) -> dict[str, Any]:
        with self._lock:
            self._hist.append(np.asarray(bbox, np.float32))
            win = self._window()
            miss = bool(self.zero_on_miss and float(np.asarray(bbox, np.float32)[3]) < 0.5)
        if miss:
            win = np.zeros_like(win)
        bb = torch.from_numpy(win)[None].to(self.device)
        ft = torch.from_numpy(np.asarray(feat, np.float32))[None].to(self.device)
        S = sample_chunk(self.model, bb, ft, self.k)[0]        # (k, H, 3)
        chunk = S.mean(0).cpu().numpy()
        first = chunk[0].astype(np.float64)
        if self.ema_alpha > 0:
            self._ema = first if self._ema is None else self.ema_alpha * first + (1 - self.ema_alpha) * self._ema
            first = self._ema
        lx, ly, az = (float(x) for x in first)
        cmd = to_drive_command(lx, ly, az, speed_source=self.speed_source, s_max=self.s_max)
        return dict(chunk=chunk, spread=float(S.std(0).mean().item()), first=(lx, ly, az), action_cont=cmd, arch=self.meta.get("arch"), subgoal_missing=miss)
