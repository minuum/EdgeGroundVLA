import sys
import os

with open('/home/billy/25-1kp/MoNaVLA/robovlm_nav/trainer/nav_trainer.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'print(f"Validation dataset initialized' in line:
        line = line.replace('is_validation', 'self.is_validation')
    new_lines.append(line)

with open('/home/billy/25-1kp/MoNaVLA/robovlm_nav/trainer/nav_trainer.py', 'w') as f:
    f.writelines(new_lines)
