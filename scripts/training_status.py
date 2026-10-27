"""학습 진행 상태 파일 기록(대시보드 Tab 1이 폴링): docs/v5/egvla2_fm/training_status.json. 실제 학습 루프에서만 호출한다(값을 만들어 내지 않음)."""
import json
import subprocess
import time
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "docs/v5/egvla2_fm/training_status.json"


def gpu():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        name, util = o[0].strip(), o[1].strip()
        used, tot = o[2].strip(), o[3].strip()
        return dict(name=name, util=int(util) if util.isdigit() else None, vram_used_gb=round(float(used) / 1024, 1) if used.replace(".", "").isdigit() else None,
                    vram_total_gb=round(float(tot) / 1024, 1) if tot.replace(".", "").isdigit() else None)
    except Exception:
        return {}


def write(name, state, epoch, epochs, loss=None, val_angle=None, history=None, extra=None):
    d = dict(name=name, state=state, epoch=epoch, epochs=epochs, loss=loss, val_angle=val_angle, history=history or [], gpu=gpu(), updated_at=time.time())
    if extra:
        d.update(extra)
    PATH.write_text(json.dumps(d, ensure_ascii=False))
