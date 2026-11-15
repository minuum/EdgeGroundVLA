"""EGVLA2 텍스트 조건화(V4·V4R) 서빙 경로. plan: docs/plans/plan_20261115_v4_serving.md

설계: 박스는 학습 때와 같이 항상 'gray basket' 문구로 찾고, 사용자 지시어는 문구 특징(융합 인코더의
이미지토큰 뒤 토큰 평균, 768-d)으로만 넣는다. 헤드 입력 = 박스 창(3×4) + [이미지 풀링 768 | 문구 특징 768].
기본 서빙(A3) 경로는 건드리지 않는다. 지시어가 gray basket 계열이 아니면 정지가 정상 동작(학습된 동작)이다."""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robovlm_nav.models.egvla2_fm_heads import K_SAMPLES, PiZeroFM
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy, FusedF2Extractor

DEFAULT_PHRASE = "gray basket"


def norm_phrase(p: str | None) -> str:
    p = (p or "").strip()
    return p if p else DEFAULT_PHRASE      # 대소문자는 학습 문구 그대로 두고 변형하지 않는다(학습 어휘는 소문자)


class FusedF2TextExtractor(FusedF2Extractor):
    """이미지 인코더 1회 → (박스 생성 gray basket 고정) + (지시어 문구 특징). 반환 feat = 1536-d."""

    def __init__(self, device=None, dtype: torch.dtype = torch.float16):
        super().__init__(device, dtype)
        self._tok: dict[str, torch.Tensor] = {}
        self._phrase = DEFAULT_PHRASE            # set_phrase 경로(이 저장소 서버) 전용 기본값. 인자 경로(soda)는 이 값을 바꾸지 않는다.
        self.last_phrase = DEFAULT_PHRASE        # 마지막 호출에 쓰인 문구(기록용). 추론에는 쓰지 않는다.
        self._lock = threading.Lock()

    def set_phrase(self, phrase: str | None) -> None:
        """이 저장소 서버(predict_rgb의 hasattr 분기)용. soda 서버는 호출하지 않고 __call__의 phrase 인자만 쓴다."""
        self._phrase = norm_phrase(phrase)

    @property
    def phrase(self) -> str:
        return self.last_phrase

    def _phrase_ids(self, phrase: str) -> torch.Tensor:
        if phrase not in self._tok:
            from PIL import Image
            P = self._P
            self._tok[phrase] = self.proc(text=[P.TASK + phrase], images=[Image.new("RGB", (64, 64))], return_tensors="pt")["input_ids"].to(self.device)
        return self._tok[phrase]

    @torch.no_grad()
    def __call__(self, image_rgb: np.ndarray, phrase: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """phrase 인자가 있으면 그 문구, 없으면 set_phrase 값(기본 gray basket). 인자 경로는 요청 사이에 상태를 남기지 않는다."""
        from PIL import Image
        P = self._P
        phrase = norm_phrase(phrase) if phrase is not None else self._phrase
        self.last_phrase = phrase
        pil = Image.fromarray(np.ascontiguousarray(image_rgb).astype(np.uint8)).convert("RGB")
        inp = self.proc(text=[P.TASK + P.PHRASE], images=[pil], return_tensors="pt")      # 박스 문구는 gray basket 고정
        pv = inp["pixel_values"].to(self.device, self.dtype)
        ids = inp["input_ids"].to(self.device)
        img = self.model._encode_image(pv)                                                 # 1회
        emb, _ = self.model._merge_input_ids_with_image_features(img, self.model.get_input_embeddings()(ids))
        out = self.model.language_model.generate(input_ids=None, inputs_embeds=emb, max_new_tokens=128, num_beams=3, do_sample=False)
        text = self.proc.batch_decode(out, skip_special_tokens=False)[0]
        parsed = self.proc.post_process_generation(text, task=P.TASK, image_size=(pil.width, pil.height))[P.TASK]
        b = P.florence2_output_to_bbox(parsed.get("bboxes", []), pil.width, pil.height)
        bbox = np.array([b["cx"], b["cy"], b["area"], float(b["has_bbox"])], np.float32)
        vis = img.float().mean(1)[0].cpu().numpy()
        pe = self.model.get_input_embeddings()(self._phrase_ids(phrase))
        pe, att = self.model._merge_input_ids_with_image_features(img, pe)
        enc = self.model.language_model.get_encoder()(inputs_embeds=pe, attention_mask=att).last_hidden_state.float()
        txt = enc[:, img.shape[1]:].mean(1)[0].cpu().numpy()
        return bbox, np.concatenate([vis, txt]).astype(np.float32)


class TextCondPolicy(EgVla2FMPolicy):
    """T4/T7 텍스트 조건화 체크포인트(`textcond/EG4-0x_*.pt`: state·bdim=12·fdim=1536)를 서빙용으로 로드. step()은 부모와 동일."""

    def __init__(self, ckpt_path: str | Path, device: torch.device | None = None, *, k_samples: int = K_SAMPLES,
                 speed_source: str = "shadow", s_max: float | None = None, ema_alpha: float = 0.0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if int(ck.get("bdim", 0)) != 12 or int(ck.get("fdim", 0)) != 1536:
            raise ValueError(f"텍스트 조건화 체크포인트가 아님: bdim={ck.get('bdim')} fdim={ck.get('fdim')}")
        self.meta = {k: v for k, v in ck.items() if k not in ("state", "files")}
        self.meta.setdefault("arch", "PiZeroFM_textcond")
        self.window, self.is_reg = 3, False
        self.model = PiZeroFM(12, 1536).to(self.device)
        self.model.load_state_dict(ck["state"]); self.model.eval()
        self.k, self.speed_source, self.s_max = k_samples, speed_source, s_max
        self.zero_on_miss = True       # 학습(t7 재생)과 동일: has_bbox<0.5이면 창 전체 0
        self.ema_alpha, self._ema = float(ema_alpha), None
        self._hist: list[np.ndarray] = []
        self._lock = threading.Lock()
