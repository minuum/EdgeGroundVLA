"""EGVLA2 FM 액션헤드 정의(A MLP-FM / B FiLM-FM / B2 / C π0형 AdaLN-Zero)와 Euler 샘플러.

학습(`scripts/train_egvla2_fm_head.py`, `train_egvla2_s1.py`)과 서빙(`robovlm_nav/serve/egvla2_fm_policy.py`)이 같은 정의를 쓰도록 분리했다
(plan_20261016_egvla2_serving_path.md §2). 구조·초기화는 분리 전과 동일하다.
"""
import numpy as np
import torch
import torch.nn as nn

H, ADIM, WINDOW, HID = 8, 3, 3, 128   # 청크 길이, 액션 차원(lx,ly,az), bbox 창 길이, 은닉
STEPS_EULER = 10
K_SAMPLES = 8


def temb(t, dim=64):
    half = dim // 2
    fr = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * 1000 * fr[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


class MLPFM(nn.Module):                      # 구조 A
    def __init__(s, bdim, fdim):
        super().__init__()
        s.fp = nn.Sequential(nn.LayerNorm(fdim), nn.Linear(fdim, HID), nn.GELU())
        s.net = nn.Sequential(nn.Linear(HID + bdim + 64 + H * ADIM, 256), nn.GELU(), nn.Dropout(0.1),
                              nn.Linear(256, 256), nn.GELU(), nn.Linear(256, H * ADIM))

    def forward(s, bbox, feat, xt, t):
        return s.net(torch.cat([s.fp(feat), bbox, temb(t), xt.flatten(1)], -1)).view(-1, H, ADIM)


class FiLMFM(nn.Module):                     # 구조 B: 서브골(bbox) + flow 시간을 FiLM으로 주입
    def __init__(s, bdim, fdim):
        super().__init__()
        s.fp = nn.Sequential(nn.LayerNorm(fdim), nn.Linear(fdim, HID))
        s.film = nn.Sequential(nn.Linear(bdim + 64, 128), nn.GELU(), nn.Linear(128, 2 * HID))
        s.trunk = nn.Sequential(nn.GELU(), nn.Linear(HID + H * ADIM, 256), nn.GELU(), nn.Dropout(0.1),
                                nn.Linear(256, 256), nn.GELU(), nn.Linear(256, H * ADIM))

    def forward(s, bbox, feat, xt, t):
        g, b = s.film(torch.cat([bbox, temb(t)], -1)).chunk(2, -1)
        h = (1 + g) * s.fp(feat) + b
        return s.trunk(torch.cat([h, xt.flatten(1)], -1)).view(-1, H, ADIM)


class FiLMResFM(nn.Module):                  # 구조 B2: B + 잔차 블록 + 시간 입력 (B 부진 원인 점검용 변형)
    def __init__(s, bdim, fdim):
        super().__init__()
        s.fp = nn.Sequential(nn.LayerNorm(fdim), nn.Linear(fdim, HID))
        s.film = nn.Sequential(nn.Linear(bdim + 64, 128), nn.GELU(), nn.Linear(128, 2 * HID))
        s.inp = nn.Linear(HID + H * ADIM + 64, 256)
        s.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.1)) for _ in range(2))
        s.out = nn.Linear(256, H * ADIM)

    def forward(s, bbox, feat, xt, t):
        te = temb(t)
        g, b = s.film(torch.cat([bbox, te], -1)).chunk(2, -1)
        h = (1 + g) * s.fp(feat) + b
        z = s.inp(torch.cat([h, xt.flatten(1), te], -1))
        for blk in s.blocks:
            z = z + blk(z)
        return s.out(z).view(-1, H, ADIM)


class AdaBlock(nn.Module):
    def __init__(s, d, heads=4):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(d, elementwise_affine=False), nn.LayerNorm(d, elementwise_affine=False)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(s.ada.weight); nn.init.zeros_(s.ada.bias)   # AdaLN-Zero

    def forward(s, h, prefix, c):
        a1, b1, g1, a2, b2, g2 = s.ada(c)[:, None].chunk(6, -1)
        x = s.n1(h) * (1 + a1) + b1
        kv = torch.cat([prefix, x], 1)
        h = h + g1 * s.attn(x, kv, kv, need_weights=False)[0]
        return h + g2 * s.mlp(s.n2(h) * (1 + a2) + b2)


class PiZeroFM(nn.Module):                   # 구조 C: π0형 action expert (경량)
    def __init__(s, bdim, fdim, layers=2, d=HID):
        super().__init__()
        s.sub, s.vis = nn.Linear(bdim, d), nn.Sequential(nn.LayerNorm(fdim), nn.Linear(fdim, d))
        s.inp, s.pos = nn.Linear(ADIM, d), nn.Parameter(torch.zeros(1, H, d))
        s.tm = nn.Sequential(nn.Linear(64, d), nn.GELU(), nn.Linear(d, d))
        s.blocks = nn.ModuleList(AdaBlock(d) for _ in range(layers))
        s.out = nn.Linear(d, ADIM)

    def forward(s, bbox, feat, xt, t):
        prefix = torch.stack([s.sub(bbox), s.vis(feat)], 1)
        h, c = s.inp(xt) + s.pos, s.tm(temb(t))
        for blk in s.blocks:
            h = blk(h, prefix, c)
        return s.out(h)


class MLPReg(nn.Module):                     # 기준선(sanity): 동일 입력 회귀
    def __init__(s, bdim, fdim):
        super().__init__()
        s.net = nn.Sequential(nn.LayerNorm(fdim + bdim), nn.Linear(fdim + bdim, 256), nn.GELU(), nn.Dropout(0.1),
                              nn.Linear(256, H * ADIM))

    def forward(s, bbox, feat):
        return s.net(torch.cat([feat, bbox], -1)).view(-1, H, ADIM)


ARCH = {"A_mlp_fm": MLPFM, "B_film_fm": FiLMFM, "B2_film_res_fm": FiLMResFM, "C_pi0_adaln": PiZeroFM}


@torch.no_grad()
def sample_chunk(m, bbox, feat, k=K_SAMPLES, steps=STEPS_EULER):
    """(B, k, H, ADIM): 같은 조건에서 k개 flow 샘플(Euler). m.eval() 상태로 호출된다."""
    m.eval()
    B = len(bbox)
    bb, ft = bbox.repeat_interleave(k, 0), feat.repeat_interleave(k, 0)
    x = torch.randn(B * k, H, ADIM, device=bbox.device)
    for i in range(steps):
        t = torch.full((B * k,), i / steps, device=bbox.device)
        x = x + m(bb, ft, x, t) / steps
    return x.view(B, k, H, ADIM)
