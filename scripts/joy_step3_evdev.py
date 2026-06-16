#!/usr/bin/env python3
# Step 3: evdev 직접 읽기 (pygame 없이)
import sys
try:
    import evdev
except ImportError:
    print("pip install evdev")
    sys.exit(1)

devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
joysticks = [d for d in devices if "Controller" in d.name or "joystick" in d.name.lower() or "Gamepad" in d.name]

if not joysticks:
    print("❌ evdev에서 컨트롤러 못찾음")
    print("발견된 장치들:")
    for d in devices:
        print(f"  {d.path}  {d.name}")
    sys.exit(1)

js = joysticks[0]
print(f"✅ {js.name}  ({js.path})")
print("스틱/버튼 눌러보세요. Ctrl+C 종료")
print("─" * 50)

for event in js.read_loop():
    if event.type == evdev.ecodes.EV_ABS:
        name = evdev.ecodes.ABS.get(event.code, event.code)
        print(f"  ABS  {name:15s}  code={event.code}  val={event.value}")
    elif event.type == evdev.ecodes.EV_KEY:
        name = evdev.ecodes.KEY.get(event.code, event.code)
        state = "DOWN" if event.value else "UP  "
        print(f"  KEY  {name:15s}  code={event.code}  {state}")
