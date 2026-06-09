# -*- coding: utf-8 -*-
"""vision LoRA 패치 적용 후 RoboVLMs main.py 실행 (third_party 비수정)."""
import sys, runpy
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts/train_active"))
import _vlora_patch  # noqa: F401  (monkeypatch 적용)
cfg = sys.argv[1]
sys.argv = ["main.py", cfg]
runpy.run_path(str(ROOT / "third_party/RoboVLMs/main.py"), run_name="__main__")
