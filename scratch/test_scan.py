import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/soda/MoNaVLA")

def scan_local_files():
    ckpt_tuples = []
    for root_dir in (PROJECT_ROOT, PROJECT_ROOT / "runs"):
        if not root_dir.exists():
            print(f"Skipping {root_dir}")
            continue
        pattern = "**/*" if root_dir.name == "runs" else "*"
        for path in root_dir.glob(pattern):
            # The original code filters for .ckpt and .pth
            if path.suffix not in {".ckpt", ".pth"} or not path.is_file():
                continue
            try:
                rel = path.relative_to(PROJECT_ROOT)
                display_name = str(rel)
            except ValueError:
                display_name = path.name
            ckpt_tuples.append((display_name, str(path)))

    configs_dir = PROJECT_ROOT / "configs"
    conf_tuples = []
    if configs_dir.exists():
        for path in configs_dir.glob("*.json"):
            conf_tuples.append((path.name, str(path)))

    return sorted(set(ckpt_tuples)), sorted(conf_tuples)

ckpts, confs = scan_local_files()
print("\n--- Checkpoints ---")
for label, path in ckpts:
    print(f"{label} -> {path}")

print("\n--- Configs ---")
for label, path in confs:
    print(f"{label} -> {path}")
