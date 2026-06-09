#!/usr/bin/env python3
"""
Exp47 InstructionMLPInference 단독 테스트
서버 전체를 올리지 않고 MLP 클래스만 직접 검증

실행: python3 scripts/test_exp47_mlp.py
"""
import sys, os, time, json, traceback
import numpy as np

from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# ─────────────────────────────────────────────
# FastAPI 의존성 없이 MLP 클래스만 추출
# ─────────────────────────────────────────────
print("=" * 60)
print("Exp47 MLP 단독 테스트")
print("=" * 60)

# torch 확인
try:
    import torch
    print(f"✅ torch {torch.__version__}  CUDA={torch.cuda.is_available()}")
except ImportError:
    print("❌ torch 없음"); sys.exit(1)

# ─────────────────────────────────────────────
# inference_server.py에서 InstructionMLPInference 클래스 소스만 추출
# ─────────────────────────────────────────────
import ast, types

SRC_PATH = str(_project_root / 'robovlm_nav/serve/inference_server.py')
with open(SRC_PATH) as f:
    full_src = f.read()

# 필요한 임포트 + InstructionMLPInference 클래스 + 관련 상수만 추출
MINIMAL_SRC = """
import os, sys, time, json, traceback, logging
import numpy as np
from pathlib import Path
from collections import deque

logger = logging.getLogger("mlp_test")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
"""

# InstructionMLPInference 클래스 소스 추출
tree = ast.parse(full_src)
lines = full_src.split('\n')
cls_src = None
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'InstructionMLPInference':
        start = node.lineno - 1
        end   = node.end_lineno
        cls_src = '\n'.join(lines[start:end])
        break

if cls_src is None:
    print("❌ InstructionMLPInference 클래스를 찾을 수 없음")
    sys.exit(1)

print(f"✅ InstructionMLPInference 클래스 추출 완료 ({cls_src.count(chr(10))+1} lines)")

# 클래스 동적 실행
exec_globals = {}
exec(compile(MINIMAL_SRC + "\n" + cls_src, '<mlp_class>', 'exec'), exec_globals)
InstructionMLPInference = exec_globals['InstructionMLPInference']

# ─────────────────────────────────────────────
# 테스트 1: 인스턴스 생성
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("테스트 1: 인스턴스 생성 (가중치 없음)")
print("─" * 40)

instr_emb_path = 'docs/v5/bbox_nav_exp47/instruction_embeddings.json'
weights_path   = 'docs/v5/bbox_nav_exp47/exp47_mlp.pt'  # 없어도 랜덤 초기화

try:
    t0 = time.time()
    mlp = InstructionMLPInference(
        mlp_weights_path=weights_path,
        instruction_embeddings_path=instr_emb_path,
        device='cpu',
        vision_cache_ttl_sec=1.0,
    )
    print(f"✅ 인스턴스 생성: {(time.time()-t0)*1000:.1f}ms")
    print(f"   d_in={mlp.D_IN}, num_classes={mlp.NUM_CLASSES}")
    print(f"   instruction keys: {list(mlp._instr_embeddings.keys())}")
except Exception as e:
    print(f"❌ 생성 실패: {e}")
    traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────
# 테스트 2: vision cache stale 확인
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("테스트 2: Vision cache stale 확인")
print("─" * 40)

print(f"초기 stale={mlp.is_vision_cache_stale()}, initialized={mlp._vision_cache['initialized']}")

# 더미 vision feature 업데이트 (1024-dim)
dummy_vis = np.random.randn(1024).astype(np.float32)
mlp.update_vision_feature(dummy_vis)
print(f"업데이트 후 stale={mlp.is_vision_cache_stale()}, age={mlp.vision_cache_age_ms():.1f}ms")

# ─────────────────────────────────────────────
# 테스트 3: bbox 업데이트 후 추론
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("테스트 3: bbox 업데이트 + predict (window=8 채울 때까지 반복)")
print("─" * 40)

instructions = list(mlp._instr_embeddings.keys()) or ["navigate to target"]
test_instr = instructions[0]
print(f"테스트 instruction: '{test_instr}'")

latencies = []
for step in range(10):
    # 랜덤 bbox (normalized)
    cx   = np.random.uniform(0.3, 0.7)
    cy   = np.random.uniform(0.4, 0.8)
    area = np.random.uniform(0.05, 0.3)
    mlp.update_bbox(cx=cx, cy=cy, area=area, has_bbox=True)

    t0 = time.time()
    result = mlp.predict(test_instr)
    lat = time.time() - t0
    latencies.append(lat * 1000)

    print(f"  step {step+1:2d} | class={result['class_idx']}({result['class_name']:<15}) "
          f"| action={result['action']} | {lat*1000:.2f}ms "
          f"| matched={result['instruction_matched']}")

print(f"\n📊 평균 추론 레이턴시: {np.mean(latencies):.2f}ms "
      f"(min={np.min(latencies):.2f}, max={np.max(latencies):.2f})")

# ─────────────────────────────────────────────
# 테스트 4: has_bbox=False (타겟 없음)
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("테스트 4: bbox 없음 (has_bbox=False)")
print("─" * 40)

mlp.update_bbox(cx=0.5, cy=0.5, area=0.0, has_bbox=False)
result = mlp.predict(test_instr)
print(f"  class={result['class_idx']}({result['class_name']}) action={result['action']}")

# ─────────────────────────────────────────────
# 테스트 5: reset 후 재추론
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("테스트 5: reset() 후 재추론")
print("─" * 40)

mlp.reset()
print(f"reset 후 bbox_history len={len(mlp._bbox_history)}, vision_initialized={mlp._vision_cache['initialized']}")
mlp.update_bbox(cx=0.5, cy=0.6, area=0.1, has_bbox=True)
result = mlp.predict(test_instr)
print(f"  class={result['class_idx']}({result['class_name']}) action={result['action']}")

# ─────────────────────────────────────────────
# 테스트 6: 모든 instruction key 테스트
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("테스트 6: 모든 instruction key 테스트")
print("─" * 40)

mlp.update_vision_feature(np.random.randn(1024).astype(np.float32))
mlp.update_bbox(cx=0.4, cy=0.6, area=0.15, has_bbox=True)
for key in mlp._instr_embeddings.keys():
    r = mlp.predict(key)
    print(f"  '{key[:40]:<40}' → class={r['class_idx']}({r['class_name']})")

print("\n" + "=" * 60)
print("✅ 모든 테스트 통과!")
print("=" * 60)
