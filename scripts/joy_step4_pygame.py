#!/usr/bin/env python3
# Step 4: pygame 읽기 — evdev는 됐는데 pygame이 문제인지 확인
import os, sys, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

try:
    import pygame
except ImportError:
    print("pip install pygame")
    sys.exit(1)

pygame.init()
pygame.joystick.init()

count = pygame.joystick.get_count()
print(f"pygame이 감지한 조이스틱: {count}개")

if count == 0:
    print("❌ pygame에서 조이스틱 미감지")
    sys.exit(1)

js = pygame.joystick.Joystick(0)
js.init()
print(f"✅ {js.get_name()}  {js.get_numaxes()}축 {js.get_numbuttons()}버튼")
print("스틱/버튼 눌러보세요. Ctrl+C 종료")
print("─" * 50)

prev_axes = [js.get_axis(i) for i in range(js.get_numaxes())]
prev_btns = [js.get_button(i) for i in range(js.get_numbuttons())]

while True:
    try:
        pygame.event.pump()
        for i in range(js.get_numaxes()):
            v = js.get_axis(i)
            if abs(v - prev_axes[i]) > 0.05:
                print(f"  AXIS[{i}]  {prev_axes[i]:+.3f} → {v:+.3f}")
                prev_axes[i] = v
        for i in range(js.get_numbuttons()):
            v = js.get_button(i)
            if v != prev_btns[i]:
                print(f"  BTN [{i:2d}]  {'DOWN' if v else 'UP  '}")
                prev_btns[i] = v
        time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n종료")
        break
