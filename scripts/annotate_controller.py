#!/usr/bin/env python3
"""Data Frog 컨트롤러 — Bounding Box 어노테이션.

각 버튼 위치에 컬러 사각 박스 + 번호 태그를 그리고, 하단 범례로 기능 설명.

사용:
  python3 scripts/annotate_controller.py docs/assets/SCR-20260611-obsw.png
"""
import sys, os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT   = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
FONT_B = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"

# ── 이미지 크기 ──────────────────────────────────────────
# 제공 좌표는 [Ymin, Xmin, Ymax, Xmax] 형태 — 1024×1024 기준 추정값
# 700×718 이미지에 맞게 sx=700/1024, sy=718/1024 스케일 적용
SX = 700 / 1024
SY = 718 / 1024

def sc(ymin, xmin, ymax, xmax):
    """1024 기준 좌표 → 700×718 스케일 변환."""
    return (int(xmin*SX), int(ymin*SY), int(xmax*SX), int(ymax*SY))  # x1,y1,x2,y2

# ── 버튼 정의: (num, 이름, 기능, bbox_1024=[Ymin,Xmin,Ymax,Xmax], RGB) ──────
BUTTONS = [
    # 페이스 버튼
    ( 1, "A",       "STOP 기록",              sc(224, 735, 298, 814), ( 60, 210,  80)),
    ( 2, "B",       "UNDO (마지막 취소)",      sc(165, 804, 235, 876), (100, 180, 255)),
    ( 3, "X",       "DISCARD (에피소드 폐기)", sc(163, 665, 236, 737), (255, 155,  60)),
    ( 4, "Y",       "TELEOP 토글",             sc(103, 733, 175, 804), (255,  80,  80)),
    # 숄더 버튼 (사이드뷰 기준 → TOP_SPLIT 이후 y 오프셋 적용 필요)
    # 탑뷰에서도 어깨 상단이 보이므로 탑뷰 상단 좌표로 근사
    ( 5, "L1",      "REC 시작 ▶",             (100, 68, 165,  98), ( 60, 220,  80)),
    ( 6, "R1",      "REC 저장 ■",             (515, 68, 580,  98), ( 60, 220,  80)),
    ( 7, "L2",      "미사용",                  ( 80, 52, 145,  72), (140, 140, 140)),
    ( 8, "R2",      "미사용",                  (535, 52, 600,  72), (140, 140, 140)),
    # 중앙 버튼
    ( 9, "SELECT",  "시나리오 이전 ◀",        sc(183, 427, 241, 483), (215, 215,  50)),
    (10, "START",   "시나리오 다음 ▶",        sc(184, 574, 243, 629), (215, 215,  50)),
    (11, "MODE",    "입력 모드 전환",          sc(254, 503, 292, 555), (160, 160, 160)),
    # D-pad
    (12, "D◀",     "시나리오 이전",           sc(174, 212, 231, 269), (195, 195,  40)),
    (13, "D▶",     "시나리오 다음",           sc(174, 338, 231, 395), (195, 195,  40)),
    (14, "D▲",     "예비 (미정)",             sc(107, 276, 167, 332), (140, 140, 140)),
    (15, "D▼",     "미사용",                  sc(237, 276, 296, 332), (140, 140, 140)),
    # 아날로그 스틱
    (16, "L-Stick", "로봇 이동 (전진/회전)",  sc(258, 334, 412, 492), (255, 255, 255)),
    (17, "R-Stick", "예비",                   sc(258, 564, 412, 720), (140, 140, 140)),
]

# 사이드뷰 bbox — y는 TOP_SPLIT(450) 이후 서브이미지 좌표
TOP_SPLIT = 450
SIDE_BUTTONS = [
    ( 7, "L2",  ( 80, 22, 108, 45), (140, 140, 140)),
    ( 5, "L1",  (112, 36, 145, 68), ( 60, 220,  80)),
    ( 6, "R1",  (510, 36, 544, 68), ( 60, 220,  80)),
    ( 8, "R2",  (545, 22, 575, 45), (140, 140, 140)),
]

LEGEND_H = 220


def draw_bbox(draw, x1, y1, x2, y2, num, rgb, fnt_tag, thick=2):
    """컬러 사각 박스 + 번호 태그 (좌상단 코너)."""
    # 박스
    draw.rectangle([x1, y1, x2, y2], outline=rgb, width=thick)
    # 번호 태그 배경
    tag = str(num)
    bb = fnt_tag.getbbox(tag)
    tw, th = bb[2]-bb[0]+6, bb[3]-bb[1]+4
    draw.rectangle([x1, y1-th-1, x1+tw, y1], fill=rgb)
    draw.text((x1+3, y1-th), tag, font=fnt_tag, fill=(10, 10, 10))


def annotate(input_path, output_path):
    img_cv = cv2.imread(input_path)
    if img_cv is None:
        print(f"❌ 이미지 없음: {input_path}"); return
    h, w = img_cv.shape[:2]
    print(f"  입력: {w}×{h}")

    fnt_tag  = ImageFont.truetype(FONT_B, 11)
    fnt_leg  = ImageFont.truetype(FONT,   13)
    fnt_legb = ImageFont.truetype(FONT_B, 13)
    fnt_head = ImageFont.truetype(FONT_B, 14)

    # ── 탑뷰 bbox ──────────────────────────────────────
    top_pil = Image.fromarray(cv2.cvtColor(img_cv[:TOP_SPLIT], cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(top_pil)
    for num, name, func, (x1,y1,x2,y2), rgb in BUTTONS:
        # y 범위가 탑뷰 안에 있는 것만
        if y1 < TOP_SPLIT and y2 < TOP_SPLIT:
            draw_bbox(draw, x1, y1, x2, y2, num, rgb, fnt_tag)
    img_cv[:TOP_SPLIT] = cv2.cvtColor(np.array(top_pil), cv2.COLOR_BGR2RGB)

    # 구분선
    cv2.line(img_cv, (0, TOP_SPLIT), (w, TOP_SPLIT), (80, 80, 80), 1)

    # ── 사이드뷰 bbox ──────────────────────────────────
    side_pil = Image.fromarray(cv2.cvtColor(img_cv[TOP_SPLIT:], cv2.COLOR_BGR2RGB))
    draw2 = ImageDraw.Draw(side_pil)
    for num, name, (x1,y1,x2,y2), rgb in SIDE_BUTTONS:
        draw_bbox(draw2, x1, y1, x2, y2, num, rgb, fnt_tag)
    img_cv[TOP_SPLIT:] = cv2.cvtColor(np.array(side_pil), cv2.COLOR_RGB2BGR)

    # ── 하단 범례 캔버스 확장 ──────────────────────────
    canvas = np.zeros((h + LEGEND_H, w, 3), dtype=np.uint8)
    canvas[:h] = img_cv
    canvas[h:] = (18, 18, 18)

    full_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw3 = ImageDraw.Draw(full_pil)

    draw3.text((10, h+6), "MoNaVLA  Data Frog Controller — 버튼 기능 매핑",
               font=fnt_head, fill=(200, 200, 200))
    draw3.line([(10, h+24), (w-10, h+24)], fill=(60,60,60), width=1)

    # 2열
    col_w  = w // 2
    row_h  = 18
    y0     = h + 30
    for i, (num, name, func, _, rgb) in enumerate(BUTTONS):
        col = i % 2
        row = i // 2
        x   = 10 if col == 0 else col_w + 8
        y   = y0 + row * row_h
        r   = 8
        # 번호 태그 (작은 사각형)
        bb  = fnt_tag.getbbox(str(num))
        tw  = bb[2]-bb[0]+6
        th  = bb[3]-bb[1]+4
        draw3.rectangle([x, y+1, x+tw, y+th+1], fill=rgb)
        draw3.text((x+3, y+2), str(num), font=fnt_tag, fill=(10,10,10))
        # 이름 + 기능
        draw3.text((x+tw+4, y+1), name, font=fnt_legb, fill=rgb)
        nw = fnt_legb.getbbox(name)[2] - fnt_legb.getbbox(name)[0]
        draw3.text((x+tw+4+nw+4, y+1), func, font=fnt_leg, fill=(190,190,190))

    note_y = y0 + ((len(BUTTONS)+1)//2) * row_h + 6
    draw3.line([(10, note_y), (w-10, note_y)], fill=(50,50,50), width=1)
    draw3.text((10, note_y+4),
               "※ SELECT/D-pad ◀▶ 시나리오 이동 중복  |  L-Stick: 상하=전진, 좌우=회전",
               font=fnt_leg, fill=(110, 110, 110))

    result = cv2.cvtColor(np.array(full_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result)
    print(f"✅ 저장: {output_path}  ({w}×{h+LEGEND_H})")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base)
    inp = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/SCR-20260611-obsw.png"
    out = inp.rsplit(".", 1)[0] + "_annotated.png"
    annotate(inp, out)
