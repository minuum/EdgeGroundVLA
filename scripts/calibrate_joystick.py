"""
DragonRise 조이스틱 축/버튼 번호 확인 유틸.
실행 후 스틱·버튼을 움직이면 번호가 출력됩니다.
확인이 끝나면 q를 입력해 joystick_config.json을 저장하세요.

사용법:
    python3 scripts/calibrate_joystick.py
"""
import os
import sys
import json
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

try:
    import pygame
except ImportError:
    print("pygame 없음 → pip install pygame")
    sys.exit(1)

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("조이스틱 미연결. USB 확인 후 재실행.")
    sys.exit(1)

js = pygame.joystick.Joystick(0)
js.init()
print(f"\n연결된 장치: {js.get_name()}")
print(f"  축 수: {js.get_numaxes()}  |  버튼 수: {js.get_numbuttons()}  |  Hat 수: {js.get_numhats()}")
print("\n스틱/버튼을 움직이면 번호가 출력됩니다. Ctrl-C 또는 q+Enter로 종료.\n")

DEADZONE = 0.2
prev_axes = [0.0] * js.get_numaxes()
prev_btns = [0] * js.get_numbuttons()

config = {
    "axes": {"left_x": 0, "left_y": 1, "right_x": 2},
    "buttons": {"stop": 0, "undo": 1, "teleop_toggle": 7, "rec_toggle": 6, "discard": 2},
    "device": js.get_name(),
}

try:
    while True:
        pygame.event.pump()

        for i in range(js.get_numaxes()):
            v = js.get_axis(i)
            if abs(v - prev_axes[i]) > DEADZONE and abs(v) > DEADZONE:
                direction = "+" if v > 0 else "-"
                print(f"  Axis {i}: {v:+.3f}  ({direction})")
                prev_axes[i] = v

        for i in range(js.get_numbuttons()):
            v = js.get_button(i)
            if v != prev_btns[i]:
                print(f"  Button {i}: {'DOWN' if v else 'UP'}")
                prev_btns[i] = v

        for i in range(js.get_numhats()):
            v = js.get_hat(i)
            if v != (0, 0):
                print(f"  Hat {i}: {v}")

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

out = Path(__file__).parent / "joystick_config.json"
with open(out, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"\n기본 설정을 저장했습니다: {out}")
print("필요하면 joystick_config.json에서 axes/buttons 번호를 수정하세요.")
pygame.quit()
