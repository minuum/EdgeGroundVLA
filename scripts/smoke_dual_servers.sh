#!/usr/bin/env bash
# Dual-server smoke test (T1.4)
# 8000 (Exp25 end-to-end) + 8001 (BBox proxy)에 동일 이미지 보내고 응답 비교
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PRIMARY_URL="${PRIMARY_URL:-http://localhost:8000}"
PROXY_URL="${PROXY_URL:-http://localhost:8001}"
API_KEY="${VLA_API_KEY:-}"

if [ -z "$API_KEY" ]; then
    echo "[smoke] VLA_API_KEY not set; /predict 검증은 skip됩니다." >&2
fi

# 테스트 이미지: dataset에서 첫 H5의 첫 프레임 추출
TEST_IMG_DIR="/tmp/vla_smoke_$$"
mkdir -p "$TEST_IMG_DIR"
python3 - <<PY > "$TEST_IMG_DIR/info.json"
import h5py, base64, json, sys, glob
import numpy as np
from PIL import Image
import io

candidates = [
    "/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5",
    "/home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v5",
]
ds = None
for c in candidates:
    eps = sorted(glob.glob(f"{c}/episode_*center_left_path*.h5"))
    if eps:
        ds = eps[0]
        break
if ds is None:
    print(json.dumps({"error":"no episode found"}), file=sys.stderr)
    sys.exit(1)

with h5py.File(ds, "r") as f:
    if "observations" in f and "images" in f["observations"]:
        img = f["observations"]["images"][0]
    else:
        img = f["images"][0]

pil = Image.fromarray(np.array(img))
buf = io.BytesIO()
pil.save(buf, format="JPEG", quality=90)
b64 = base64.b64encode(buf.getvalue()).decode("ascii")
with open("$TEST_IMG_DIR/test.b64", "w") as f:
    f.write(b64)

print(json.dumps({"episode": ds, "frame": 0, "size": pil.size, "b64_len": len(b64)}))
PY

echo "[smoke] Test image:"
cat "$TEST_IMG_DIR/info.json"
echo

# ── 1. Health checks ─────────────────────────────────────
echo "[smoke] === Health: $PRIMARY_URL ==="
curl -sf "$PRIMARY_URL/health" | python3 -m json.tool || echo "[smoke] PRIMARY health FAIL"
echo

echo "[smoke] === Health: $PROXY_URL ==="
curl -sf "$PROXY_URL/health" | python3 -m json.tool || echo "[smoke] PROXY health FAIL"
echo

# ── 2. Predict (key 있을 때만) ───────────────────────────
if [ -n "$API_KEY" ]; then
    B64=$(cat "$TEST_IMG_DIR/test.b64")
    PAYLOAD=$(python3 -c "import json,sys; print(json.dumps({'image':sys.stdin.read(),'instruction':'navigate to gray basket'}))" <<< "$B64")

    echo "[smoke] === Predict: $PRIMARY_URL ==="
    echo "$PAYLOAD" | curl -sf -X POST "$PRIMARY_URL/predict" \
        -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d @- \
        | python3 -m json.tool | head -30 || echo "[smoke] PRIMARY predict FAIL"
    echo

    echo "[smoke] === Predict: $PROXY_URL ==="
    echo "$PAYLOAD" | curl -sf -X POST "$PROXY_URL/predict" \
        -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d @- \
        | python3 -m json.tool | head -30 || echo "[smoke] PROXY predict FAIL"
    echo
fi

# ── 3. 정리 ───────────────────────────────────────────────
rm -rf "$TEST_IMG_DIR"
echo "[smoke] done."
