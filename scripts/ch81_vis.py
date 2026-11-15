"""CH81 추가 탭: 실제 화면(11/13 세션 프레임) + 구조도(SVG). 이미지는 base64로 내장(Pages에서 .jpg가 .gitignore 되어도 보이도록).
원본 프레임: docs/v5/figures/objtest/{sid}_{i}.jpg (scripts/build_object_test_page.py 생성), 박스: docs/v5/egvla2_fm/objtest_data.json"""
import base64, io, json, math
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs/v5/figures/objtest"
D = {s["sid"]: s for s in json.loads((ROOT / "docs/v5/egvla2_fm/objtest_data.json").read_text())}
P = lambda t: f'<div style="font-size:0.82rem;line-height:1.8;margin:6px 0">{t}</div>'

PICK = [("20261113_150911", "바구니 · 옆(왼쪽)"), ("20261113_165500", "바구니 · 가운데"), ("20261113_154226", "의자 · 옆(오른쪽)"),
        ("20261113_170121", "의자 · 가운데"), ("20261113_164006", "작은 책상 · 옆(왼쪽)"), ("20261113_164534", "작은 책상 · 옆(왼쪽)"),
        ("20261113_162953", "작은 책상 · 가운데"), ("20261113_151819", "화분 · 가운데"), ("20261113_151741", "화분 · 가운데")]


def _img(sid, i, bb):
    im = Image.open(FIG / f"{sid}_{i}.jpg").convert("RGB"); W, H = im.size
    cx, cy, ar, has = bb[i] if i < len(bb) else (0, 0, 0, 0)
    if has > 0.5:
        s = math.sqrt(max(ar, 1e-6) * W * H) / 2; x, y = cx * W, cy * H; dr = ImageDraw.Draw(im)
        dr.rectangle([x - s, y - s, x + s, y + s], outline=(251, 191, 36), width=2); dr.line([x - 6, y, x + 6, y], fill=(251, 191, 36)); dr.line([x, y - 6, x, y + 6], fill=(251, 191, 36))
    b = io.BytesIO(); im.save(b, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def images_tab(H, st):
    G, R, Y = "#4ade80", "#f87171", "#fbbf24"
    cards = ""
    for sid, lab in PICK:
        s = D[sid]; n = len(list(FIG.glob(f"{sid}_*.jpg"))); idx = sorted({0, n // 2, n - 1})
        ok = s["res"] == "성공"
        frames = "".join(f'<div style="text-align:center"><img src="{_img(sid, i, s["bbox"])}" style="width:100%;border-radius:6px;display:block"><div style="font-size:0.66rem;color:#94a3b8">프레임 {i}/{n - 1}</div></div>' for i in idx)
        cards += (f'<div style="background:#0a1628;border:1px solid #1e3050;border-left:5px solid {G if ok else R};border-radius:10px;padding:10px 12px;margin:10px 0">'
                  f'<div style="font-size:0.84rem;font-weight:800;color:#fcd34d">{lab} {st("성공" if ok else "실패", G if ok else R)} <span style="color:#94a3b8;font-weight:600;font-size:0.74rem">· 세션 {sid} · 지시어 “{s["ins"]}”</span></div>'
                  f'<div style="display:grid;grid-template-columns:repeat({len(idx)},1fr);gap:6px;margin:8px 0">{frames}</div>'
                  f'<div style="font-size:0.74rem;color:#94a3b8;line-height:1.6">첫 박스 중심 cx={s["cx0"]:.2f}·면적 {s["area0"]:.3f}, 박스 검출 {s["vis"] * 100:.0f}% 프레임, 이동 프레임 {s["go"] * 100:.0f}%, 첫 이동 프레임 {s["first"] if s["first"] >= 0 else "없음(출발 안 함)"}, 총 {s["n"]}프레임</div></div>')
    return [H("1. 11/13 실제 로봇 카메라 화면 — 물체·위치별 대표 세션"),
            P("각 세션의 <b>처음·중간·마지막 프레임</b>(원본 1280×720을 축소). 노란 사각형은 그 프레임에서 서버가 기록한 <b>박스 중심(cx, cy)과 면적</b>을 표시한 것이다. "
              "기록에 가로·세로가 없어 면적에서 환산한 <b>정사각형 근사</b>이므로 실제 박스 모양과 다르다. 대표 세션은 각 조합의 첫 세션(실패가 있으면 실패 세션 포함)이며 고른 것이지 무작위 표본이 아니다."),
            cards,
            P("읽는 법: 의자·가운데는 박스가 계속 보이는데도(검출 100%) 이동 프레임이 0%다. 화분은 박스 면적이 0.008로 아주 작다. 바구니 옆·가운데는 첫 프레임부터 움직인다. "
              "프레임 전체 재생과 명령 그래프는 <a href='object_test_1113.html' style='color:#38bdf8'>object_test_1113.html</a>.")]


# ── 구조도(SVG) ──
C = {"a3": "#38bdf8", "txt": "#4ade80", "new": "#fbbf24", "gray": "#94a3b8", "bg": "#0a1628", "line": "#475569"}


def _node(x, y, w, h, t1, t2="", col="#38bdf8", fill=None):
    fill = fill or "#0f1f38"
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{col}" stroke-width="1.6"/>'
    if t2:
        s += f'<text x="{x + w / 2}" y="{y + h / 2 - 3}" text-anchor="middle" fill="#e2e8f0" font-size="11.5" font-weight="700">{t1}</text><text x="{x + w / 2}" y="{y + h / 2 + 13}" text-anchor="middle" fill="#94a3b8" font-size="10">{t2}</text>'
    else:
        s += f'<text x="{x + w / 2}" y="{y + h / 2 + 4}" text-anchor="middle" fill="#e2e8f0" font-size="11.5" font-weight="700">{t1}</text>'
    return s


def _arrow(x1, y1, x2, y2, col="#64748b", dash=False):
    d = ' stroke-dasharray="5 4"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="1.6"{d} marker-end="url(#ah)"/>'


def _svg(w, h, body, title):
    return (f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px;height:auto;background:{C["bg"]};border:1px solid #1e3050;border-radius:10px;margin:8px 0" role="img" aria-label="{title}">'
            '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#64748b"/></marker></defs>'
            f'{body}</svg>')


def diagram_exp():
    b = '<text x="12" y="20" fill="#fcd34d" font-size="13" font-weight="800">① 11/13 실기 실험 경로 (확인됨: soda 서버 코드·로그) — 지시어는 박스 단계에서만 쓰인다</text>'
    ys = 50
    nodes = [("대시보드 입력창", "operator가 문구 타이핑", C["gray"]), ("/predict", "image + instruction", C["a3"]), ("Florence-2 그라운딩", "phrase=지시어 그대로", C["new"]),
             ("박스 (cx, cy, 면적)", "3프레임 이력", C["a3"]), ("A3 헤드", "텍스트 입력 없음", C["a3"]), ("후처리", "near_area 0.25·lost 5", C["a3"]), ("/cmd_vel", "로봇 구동", C["gray"])]
    x = 10
    for i, (a, c2, col) in enumerate(nodes):
        b += _node(x, ys, 122, 58, a, c2, col)
        if i: b += _arrow(x - 16, ys + 29, x, ys + 29)
        x += 138
    b += '<text x="12" y="140" fill="#94a3b8" font-size="10.5">· 스위치 phrase_from_request가 켜져 있고 지시어가 있고 "basket"이 아닐 때만 지시어를 사용, 아니면 "gray basket" 고정 (런타임 전용, 재시작·패밀리 전환 시 꺼짐)</text>'
    b += '<text x="12" y="157" fill="#94a3b8" font-size="10.5">· 이미지 특징은 Florence-2 인코더에서 A3 헤드로 직접 들어간다 (그림 생략). 성패는 operator가 조이스틱으로 직접 표시</text>'
    return _svg(980, 172, b, "11/13 실험 경로")


def diagram_v4():
    b = '<text x="12" y="20" fill="#fcd34d" font-size="13" font-weight="800">② V4/V4R 서빙 설계 (승인됨·오프라인 검증 통과·실기 전) — 박스는 고정, 지시어는 문구 특징으로</text>'
    b += _node(12, 40, 120, 48, "지시어", "예: go to the chair", C["gray"])
    b += _node(12, 128, 120, 48, "카메라 이미지", "JPEG q95", C["gray"])
    b += _node(200, 128, 150, 48, "Florence-2 인코더", "이미지 1회 인코딩", C["a3"])
    b += _node(200, 40, 150, 48, "문구 특징 768-d", "융합 인코더 출력 평균", C["txt"], "#0f2a1f")
    b += _node(420, 128, 160, 48, "박스 창 3×4", "문구 \"gray basket\" 고정", C["a3"])
    b += _node(420, 70, 160, 40, "이미지 풀링 768-d", "", C["a3"])
    b += _node(640, 70, 160, 78, "V4 / V4R 헤드", "입력 = 박스창 12 + [768 | 768]", C["txt"], "#0f2a1f")
    b += _node(850, 86, 118, 46, "후처리", "A3와 동일", C["a3"])
    b += _arrow(132, 64, 200, 64, C["txt"]) + _arrow(132, 152, 200, 152) + _arrow(275, 128, 275, 88, C["txt"]) + _arrow(350, 152, 420, 152)
    b += _arrow(350, 140, 420, 100, C["a3"]) + _arrow(350, 64, 640, 100, C["txt"]) + _arrow(580, 90, 640, 100) + _arrow(580, 152, 640, 130) + _arrow(800, 109, 850, 109)
    b += '<text x="12" y="204" fill="#94a3b8" font-size="10.5">· 지시어가 gray basket 계열이 아니면 문구 특징이 달라져 헤드가 정지를 낸다 (학습된 동작). V4는 학습 박스가 항상 gray basket 그라운딩이어서 박스 단계에 문구를 쓰지 않는다</text>'
    b += '<text x="12" y="221" fill="#94a3b8" font-size="10.5">· V4R = V4 + 박스 훼손 증강(드롭 30·지터 20·미러 10·정상 40). 서빙 구조는 같고 체크포인트만 다르다. 호출 시 지시어는 soda가 phrase_from_request로 넘겨준 값</text>'
    return _svg(980, 234, b, "V4 V4R 서빙 설계")


def diagram_patch():
    b = '<text x="12" y="20" fill="#fcd34d" font-size="13" font-weight="800">③ soda 최소 패치 (승인 대기) — 새 파일 1개 + 분기 7줄, 나머지는 건드리지 않음</text>'
    b += _node(12, 44, 150, 48, "mona_dashboard.py", "수정 없음", C["gray"])
    b += _node(210, 44, 190, 48, "stage2_v2_inference_server.py", "수정 없음 · phrase_from_request", C["gray"])
    b += _node(450, 44, 220, 48, "egvla2_inference_server.py", "load_service_from_env() +7줄", C["new"], "#2a1f06")
    b += _node(740, 8, 230, 40, "A3 경로 (기본)", "VLA_EGVLA2_TEXT 미설정", C["a3"])
    b += _node(740, 70, 230, 48, "egvla2_text_policy.py (새 파일)", "VLA_EGVLA2_TEXT=1 → V4/V4R", C["txt"], "#0f2a1f")
    b += _arrow(162, 68, 210, 68) + _arrow(400, 68, 450, 68) + _arrow(670, 56, 740, 30, C["a3"]) + _arrow(670, 80, 740, 92, C["txt"])
    b += '<text x="12" y="150" fill="#94a3b8" font-size="10.5">· soda의 predict_rgb는 extractor(image) 또는 extractor(image, phrase)로 호출 → 서버 본체 수정 불필요. 새 모듈의 extractor가 phrase를 문구 특징으로만 사용</text>'
    b += '<text x="12" y="167" fill="#94a3b8" font-size="10.5">· 시험 시 phrase_from_request를 켜지 않으면 모든 요청이 "gray basket"이 되어 무효 — 전환·재시작마다 /health로 확인. soda 미커밋 수정 11개 파일은 보존</text>'
    return _svg(980, 182, b, "soda 최소 패치 구조")


def diagrams_tab(H):
    return [H("1. 실험 경로 — 11/13에 실제로 돈 것"), diagram_exp(),
            H("2. 서빙 설계 — V4/V4R이 지시어를 쓰는 방식"), diagram_v4(),
            H("3. soda에 넣을 변경 범위"), diagram_patch(),
            P("범례: <span style='color:#38bdf8'>■</span> A3 기존 경로 · <span style='color:#4ade80'>■</span> 텍스트 조건화(V4/V4R) · <span style='color:#fbbf24'>■</span> 변경·주목 지점 · <span style='color:#94a3b8'>■</span> 입출력·수정 없음. "
              "①은 서버 코드·로그로 확인된 사실, ②는 구현·오프라인 검증까지 끝난 설계, ③은 승인 전 계획이다.")]


# ═══ 탭별 맥락 이미지 (2026-11-15 추가) ═══
import h5py
import numpy as np

CY = (56, 189, 248)      # 서빙 설계 박스(gray basket 고정)색


def _draw(im, bb, i, col):
    W, H_ = im.size
    cx, cy, ar, has = bb[i] if i < len(bb) else (0, 0, 0, 0)
    if has > 0.5:
        s = math.sqrt(max(ar, 1e-6) * W * H_) / 2; x, y = cx * W, cy * H_; dr = ImageDraw.Draw(im)
        dr.rectangle([x - s, y - s, x + s, y + s], outline=col, width=2); dr.line([x - 6, y, x + 6, y], fill=col); dr.line([x, y - 6, x, y + 6], fill=col)


def _b64(im, q=72):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _frame(sid, i, boxes):
    """boxes: [(bbox리스트, 색)] — 같은 프레임에 박스 여러 개를 겹쳐 그린다."""
    im = Image.open(FIG / f"{sid}_{i}.jpg").convert("RGB")
    for bb, col in boxes:
        if bb:
            _draw(im, bb, i, col)
    return _b64(im)


def _cell(src, cap):
    return f'<div style="text-align:center"><img src="{src}" style="width:100%;border-radius:6px;display:block"><div style="font-size:0.66rem;color:#94a3b8;line-height:1.5">{cap}</div></div>'


def _card(title, cells, note, col="#38bdf8"):
    return (f'<div style="background:#0a1628;border:1px solid #1e3050;border-left:5px solid {col};border-radius:10px;padding:10px 12px;margin:10px 0">'
            f'<div style="font-size:0.84rem;font-weight:800;color:#fcd34d">{title}</div>'
            f'<div style="display:grid;grid-template-columns:repeat({len(cells)},1fr);gap:6px;margin:8px 0">{"".join(cells)}</div>'
            f'<div style="font-size:0.76rem;color:#94a3b8;line-height:1.65">{note}</div></div>')


def _a3(s, i):
    a = s["a3"][i] if i < len(s["a3"]) else [0, 0]
    return f"A3 명령 ({a[0]:.2f}, {a[1]:.2f})"


def scene_strip():
    """t1: 같은 가운데 위치, 지시어만 다른 4세션의 첫 프레임 — 지시어가 박스를 고른다."""
    rows = [("20261113_165500", "바구니"), ("20261113_170121", "의자"), ("20261113_162953", "작은 책상"), ("20261113_151741", "화분")]
    cells = []
    for sid, lab in rows:
        s = D[sid]
        cells.append(_cell(_frame(sid, 0, [(s["bbox"], (251, 191, 36))]), f'<b style="color:#e2e8f0">{lab}</b> · “{s["ins"]}”<br>박스 cx {s["cx0"]:.2f} · 면적 {s["area0"]:.3f}<br>결과 {s["res"]}'))
    return _card("같은 가운데(G3) 위치, 지시어만 바꾼 4세션의 첫 프레임", cells,
                 "장면에 바구니·의자·책상·화분이 함께 놓여 있고, <b>지시어에 따라 박스가 다른 물체를 잡는다</b>(노란 박스). 가운데 위치에서 성공은 바구니 5/5, 화분 5/13, 의자 0/5, 책상 0/1이었다. "
                 "박스는 면적에서 환산한 정사각형 근사이며, 세션은 각 물체의 첫 가운데 세션이다.", "#fbbf24")


def fail_cards():
    """t2: 실패 사례별 프레임 + 프레임별 A3 명령."""
    out = ""
    s = D["20261113_170121"]; n = s["n"]; ix = [0, n // 3, 2 * n // 3, n - 1]
    out += _card("의자 · 가운데 · 실패 (세션 20261113_170121)", [_cell(_frame(s["sid"], i, [(s["bbox"], (251, 191, 36))]), f'프레임 {i}/{n - 1}<br>{_a3(s, i)}') for i in ix],
                 f'박스는 처음부터 끝까지 의자 위에 있다(검출 {s["vis"] * 100:.0f}%, 면적 {s["area0"]:.3f}). 그런데 <b>A3 명령은 모든 프레임에서 (0, 0)</b>이라 로봇이 출발하지 않았다. 박스를 놓친 것이 아니다(관측). 헤드가 왜 0을 내는지는 이 데이터로 확정할 수 없다.', "#f87171")
    ok, bad = D["20261113_151819"], D["20261113_151741"]
    cells = [_cell(_frame(ok["sid"], i, [(ok["bbox"], (251, 191, 36))]), f'성공 {ok["sid"][-6:]} · 프레임 {i}<br>{_a3(ok, i)}') for i in (0, ok["n"] - 1)]
    cells += [_cell(_frame(bad["sid"], i, [(bad["bbox"], (251, 191, 36))]), f'실패 {bad["sid"][-6:]} · 프레임 {i}<br>{_a3(bad, i)}') for i in (0, bad["n"] - 1)]
    out += _card("화분 · 가운데 — 성공 세션과 실패 세션 비교", cells,
                 f'두 세션의 첫 박스는 거의 같다(면적 {ok["area0"]:.3f} vs {bad["area0"]:.3f}). 화분은 화면에서 <b>아주 작아</b> 박스가 가늘다. 같은 시작 프레임에서 성패가 갈렸으므로 시작 박스만으로는 구분되지 않는다. '
                 f"실패 세션은 진행 중 박스 중심 cx가 {min(b[0] for b in bad['bbox']):.2f}~{max(b[0] for b in bad['bbox']):.2f}로 크게 움직였고, 성공 세션은 {min(b[0] for b in ok['bbox']):.2f}~{max(b[0] for b in ok['bbox']):.2f}였다(관측).", "#fbbf24")
    d = D["20261113_164534"]; n = d["n"]; ix = [0, 3, n // 2, n - 1]
    out += _card("작은 책상 · 옆 · 실패 1건 (세션 20261113_164534)", [_cell(_frame(d["sid"], i, [(d["bbox"], (251, 191, 36))]), f'프레임 {i}/{n - 1}<br>{_a3(d, i)}') for i in ix],
                 f'같은 조건 19건은 성공했다. 이 세션은 첫 이동 프레임이 {d["first"]}번이고 이동 프레임 비율이 {d["go"] * 100:.0f}%로 낮았다(관측). 실패 원인은 기록만으로 알 수 없다(operator 판정).', "#f87171")
    return out


def box_compare():
    """t3: 같은 프레임에 '기록된 박스(지시어 기반)'와 '서빙 설계 박스(gray basket 고정)'를 겹쳐 본다."""
    out = ""
    for sid, lab, why in [("20261113_165500", "바구니 · 가운데 (두 박스가 겹침)", "지시어가 gray basket이면 soda의 기록 박스와 서빙 설계 박스가 <b>같다</b>. 서빙 경로가 학습 경로와 같은 박스를 재현한다는 근거."),
                          ("20261113_170121", "의자 · 가운데 (두 박스가 다름)", "soda는 지시어(의자)로 박스를 찾았고, 서빙 설계는 지시어와 무관하게 <b>바구니</b>를 잡는다. 이 세션에서 V4R은 모든 프레임에서 정지로 판정했다."),
                          ("20261113_151741", "화분 · 가운데 (두 박스가 다름)", "기록 박스는 화분(작음), 서빙 설계 박스는 바구니. 이 세션에서 V4R의 이동 프레임 비율은 약 7%(10시드 평균).")]:
        s = D[sid]; n = s["n"]; ix = sorted({0, n // 2, n - 1})
        cells = [_cell(_frame(sid, i, [(s["bbox"], (251, 191, 36)), (s["gb_bbox"], CY)]), f'프레임 {i}/{n - 1}') for i in ix]
        out += _card(lab + f" · 지시어 “{s['ins']}”", cells, why + ' <span style="color:#fbbf24">■ 기록된 박스(soda, 지시어 기반)</span> <span style="color:#38bdf8">■ 서빙 설계 박스(gray basket 고정)</span>', "#38bdf8")
    return out


def _crop(sid, i, size=300):
    f = h5py.File(ROOT / f"docs/inference_sessions/session_{sid}.h5", "r")
    img = f["observations/images"][i]; bb = f["grounding/bbox"][i]; f.close()
    im = Image.fromarray(np.ascontiguousarray(img).astype(np.uint8)).convert("RGB"); W, H_ = im.size
    cx, cy, ar = bb[0] * W, bb[1] * H_, float(bb[2]); s = math.sqrt(ar * W * H_) / 2
    dr = ImageDraw.Draw(im); dr.rectangle([cx - s, cy - s, cx + s, cy + s], outline=(251, 191, 36), width=3)
    x0 = int(min(max(cx - size / 2, 0), W - size)); y0 = int(min(max(cy - size / 2, 0), H_ - size))
    return _b64(im.crop((x0, y0, x0 + size, y0 + size)).resize((260, 260)), 80), W, H_, 2 * s


def hardware_cards():
    """t4: 카메라 원본 해상도, 물체 크기 비교(같은 배율 크롭)."""
    cells = []
    for sid, lab in [("20261113_165500", "바구니"), ("20261113_170121", "의자"), ("20261113_162953", "작은 책상"), ("20261113_151741", "화분")]:
        src, W, H_, side = _crop(sid, 0); s = D[sid]
        cells.append(_cell(src, f'<b style="color:#e2e8f0">{lab}</b><br>면적 {s["area0"]:.3f} · 한 변 약 {side:.0f}px'))
    c1 = _card("물체 크기 비교 — 같은 배율(원본 300px 크롭)로 본 첫 프레임의 박스", cells,
               f"원본 프레임은 {W}×{H_}다. 박스 면적은 바구니 0.027, 책상 0.03, 의자 0.06, 화분 <b>0.008</b>. 화분은 학습 때 본 바구니보다 한 변이 약 절반 크기라 "
               "그라운딩·정지 판정이 흔들리기 쉬운 조건으로 보인다(해석, 미검증). 박스는 면적 환산 정사각형 근사.", "#fbbf24")
    s = D["20261113_165500"]
    c2 = _card("로봇 카메라 원본 시점 (바구니 · 가운데 · 첫 프레임)", [_cell(_frame(s["sid"], 0, []), "H5 저장 프레임을 384×216으로 축소, 박스 표시 없음")],
               "<b>확인 필요</b>: H5에 저장된 프레임은 <b>1280×720</b>인데 soda의 카메라 노드 설정(<code>camera_publisher_usb.py</code>)은 <b>640×480 @ 30fps</b>다. "
               "저장 전에 다른 경로로 해상도가 바뀌었는지(대시보드의 별도 캡처 등)는 이번 자료로 알 수 없다. 서빙 입력 해상도와 학습 해상도가 어긋나지 않았는지 동환에게 확인이 필요하다. "
               "카메라 마운트 높이·각도는 코드에 없다.", "#f87171")
    return c1 + c2
