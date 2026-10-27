"""EGVLA2 FM 헤드 + 학습 전용 월드 모델식 보조 헤드(plan_20261028_world_model_aux_head.md).

PiZeroFM과 파라미터 이름·forward(a_true=None)가 같아 서빙은 기존 `PiZeroFM`에 `aux.*`를 뺀 state_dict로 로드한다.
보조 헤드는 (행동 스트림 표현, 실제 행동 청크)로 k 프레임 뒤 변화량(Δbbox, Δfeat)을 예측한다. 추론에는 쓰지 않는다.
"""
import torch
import torch.nn as nn

from robovlm_nav.models.egvla2_fm_heads import ADIM, H, HID, PiZeroFM, temb

N_BB = 3          # Δbbox 차원: cx, cy, area(4번째 값은 상수 1.0)


class PiZeroFMWM(PiZeroFM):
    def __init__(s, bdim, fdim, src="h", **kw):
        super().__init__(bdim, fdim, **kw)
        s.src = src                              # "h": 블록 통과 후 행동 스트림, "prefix": bbox·특징 선형 투영(대조)
        s.aux = nn.Sequential(nn.Linear(HID + ADIM * H, 256), nn.GELU(), nn.Linear(256, N_BB + fdim))

    def forward(s, bbox, feat, xt, t, a_true=None):
        prefix = torch.stack([s.sub(bbox), s.vis(feat)], 1)
        h, c = s.inp(xt) + s.pos, s.tm(temb(t))
        for blk in s.blocks:
            h = blk(h, prefix, c)
        out = s.out(h)
        if a_true is None:
            return out
        z = h.mean(1) if s.src == "h" else prefix.mean(1)
        return out, s.aux(torch.cat([z, a_true.flatten(1)], -1))

    def serving_state_dict(s):
        return {k: v for k, v in s.state_dict().items() if not k.startswith("aux.")}
