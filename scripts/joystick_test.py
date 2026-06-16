#!/usr/bin/env python3
"""독립 조이스틱 테스트 — ROS/Gradio 없이 실행"""
import os, sys, time

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

try:
    import pygame
except ImportError:
    print("pygame 없음 — pip install pygame")
    sys.exit(1)

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("❌ 조이스틱 미감지 — USB 확인")
    sys.exit(1)

js = pygame.joystick.Joystick(0)
js.init()
n_axes   = js.get_numaxes()
n_btns   = js.get_numbuttons()

print(f"✅ 연결됨: {js.get_name()}")
print(f"   {n_axes}축  {n_btns}버튼")
print()
print("스틱/버튼 움직여보세요. Ctrl+C 로 종료.")
print("─" * 60)

prev_axes = [0.0] * n_axes
prev_btns = [0]   * n_btns

while True:
    try:
        pygame.event.pump()

        axes = [js.get_axis(i) for i in range(n_axes)]
        btns = [js.get_button(i) for i in range(n_btns)]

        # 변화 감지
        for i, (p, c) in enumerate(zip(prev_axes, axes)):
            if abs(c - p) > 0.05:
                bar_len = 20
                pos = int((c + 1) / 2 * bar_len)
                pos = max(0, min(bar_len - 1, pos))
                bar = "─" * bar_len
                bar = bar[:pos] + "█" + bar[pos+1:]
                print(f"  AXIS[{i}]  [{bar}]  {c:+.3f}")
                prev_axes[i] = c

        for i, (p, c) in enumerate(zip(prev_btns, btns)):
            if c != p:
                print(f"  BTN [{i:2d}]  {'▼ DOWN' if c else '▲ UP  '}")
                prev_btns[i] = c

        time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n종료")
        break
