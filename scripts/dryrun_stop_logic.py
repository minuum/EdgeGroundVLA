#!/usr/bin/env python3
"""
Exp66 STOP 로직 드라이런.
inference_server 없이 PG2Grounder + STOP threshold만 단독 테스트.

사용법:
  cd /home/minum/26CS/MoNaVLA
  .venv/bin/python3 scripts/dryrun_stop_logic.py
"""
import re, sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

# ── STOP 파라미터 (서버와 동일) ─────────────────────────────────────────
GOAL_AREA_THRESHOLD = 0.25
GOAL_CX_TOLERANCE   = 0.35
GOAL_CONSEC_FRAMES  = 2

_LOC_RE = re.compile(r"<loc(\d{4})>")

def parse_pg2(raw: str) -> dict:
    locs = [int(v) / 1023.0 for v in _LOC_RE.findall(raw)]
    if len(locs) >= 4:
        y1, x1, y2, x2 = locs[:4]
        area = max(0.0, (x2 - x1) * (y2 - y1))
        cx   = (x1 + x2) / 2
        cy   = (y1 + y2) / 2
        return {"cx": cx, "cy": cy, "area": area, "has_bbox": True}
    return {"cx": 0.5, "cy": 0.6, "area": 0.0, "has_bbox": False}

def is_stop(g: dict) -> bool:
    return (g["area"] >= GOAL_AREA_THRESHOLD and
            abs(g["cx"] - 0.5) <= GOAL_CX_TOLERANCE)

# ── PG2 로드 ─────────────────────────────────────────────────────────
import os
PG2_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--paligemma2-3b-mix-224"
    "/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
)
PG2_PATH = Path(os.getenv("VLA_PG2_PATH", PG2_CACHE))
if not PG2_PATH.exists():
    print(f"[ERROR] PG2 모델을 찾을 수 없음: {PG2_PATH}")
    print("       VLA_PG2_PATH 환경변수로 경로를 지정하거나 HF 캐시를 확인하세요.")
    sys.exit(1)

from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] PG2 로딩 중... ({device})  {PG2_PATH.name}")
pg2_proc  = PaliGemmaProcessor.from_pretrained(str(PG2_PATH))
pg2_model = PaliGemmaForConditionalGeneration.from_pretrained(
    str(PG2_PATH),
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device).eval()
print("[INFO] PG2 로드 완료\n")

# ── 테스트 이미지 ────────────────────────────────────────────────────
FRAME_DIR = ROOT / "docs/v5/exp66_masking_viz/frame_inspect"
TESTS = [
    # (파일명,             area 예상,   STOP 예상, 설명)
    ("center_area0.79_f12.png", "~0.79", True,  "near-goal, 바스켓 큼 → STOP"),
    ("center_area0.05_f5.png",  "~0.05", False, "far, 바스켓 작음 → 주행 계속"),
    ("left_area0.12.png",       "~0.12", False, "left path, 중간 거리"),
    ("left_area0.17.png",       "~0.17", False, "left path, 조금 가까움"),
    ("right_area0.01.png",      "~0.01", False, "right path, 매우 멀음"),
]

print("=" * 70)
print(f"{'파일':35s}  {'area':>6}  {'cx':>6}  {'bbox?':>5}  {'결과':>16}  {'기대':>8}")
print("-" * 70)

all_pass = True
for fname, area_hint, expect_stop, desc in TESTS:
    path = FRAME_DIR / fname
    if not path.exists():
        print(f"  SKIP (없음): {fname}")
        continue

    img = Image.open(path).convert("RGB")
    prompt = "<image>detect gray basket"
    inputs = pg2_proc(text=prompt, images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = pg2_model.generate(**inputs, max_new_tokens=60)
    raw = pg2_proc.decode(ids[0], skip_special_tokens=False)

    g    = parse_pg2(raw)
    stop = is_stop(g)
    ok   = stop == expect_stop
    all_pass = all_pass and ok

    result_str  = "★ STOP" if stop else "주행 계속"
    expect_str  = "STOP" if expect_stop else "주행"
    status      = "OK" if ok else "FAIL ✗"
    print(f"  {fname:35s}  {g['area']:6.3f}  {g['cx']:6.3f}  {str(g['has_bbox']):>5}  {result_str:>16}  [{status}]")
    if not ok:
        print(f"    ↑ 기대={expect_str}, 실제={result_str}  raw: {raw[-60:]}")

print("=" * 70)
print(f"\n{'전체 테스트 통과 ✓' if all_pass else '실패 항목 있음 — 위 FAIL 행 확인'}")
print(f"\nSTOP 기준: area≥{GOAL_AREA_THRESHOLD}  AND  |cx-0.5|≤{GOAL_CX_TOLERANCE}  (연속 {GOAL_CONSEC_FRAMES}프레임, 서버 기준)")
