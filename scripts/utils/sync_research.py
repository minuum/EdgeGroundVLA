#!/usr/bin/env python3
"""
Research Status Sync — docs/RESEARCH_STATUS.md → README / index.html / menemory

사용법:
  python3 scripts/utils/sync_research.py               # 실제 적용
  python3 scripts/utils/sync_research.py --diff        # 변경사항만 미리보기
  python3 scripts/utils/sync_research.py --dry-run     # 파일 수정 없이 로그만
  python3 scripts/utils/sync_research.py --validate    # 앵커 존재 여부 검사
  python3 scripts/utils/sync_research.py --propose-menemory  # menemory 제안 출력
  python3 scripts/utils/sync_research.py --add-exp     # 새 실험 행 대화형 추가

mona-sync 로 alias 등록됨.
"""
from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

# ── 경로 ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
STATUS_FILE  = ROOT / "docs" / "RESEARCH_STATUS.md"
README_FILE  = ROOT / "README.md"
INDEX_FILE   = ROOT / "docs" / "index.html"
HISTORY_FILE = ROOT / "docs" / "vla_experiment_history_table.md"
MENEMORY_FILE = ROOT / ".menemory" / "core" / "master_memory.md"

# ── 색상 ────────────────────────────────────────────────────────────────────
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ────────────────────────────────────────────────────────────────────────────
# 파서
# ────────────────────────────────────────────────────────────────────────────

def parse_status(path: Path) -> dict:
    """RESEARCH_STATUS.md의 YAML front-matter + 섹션 블록을 파싱."""
    text = path.read_text(encoding="utf-8")

    # ── YAML front-matter (--- ... ---) ──────────────────────────────────
    meta: dict = {}
    fm_match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if fm_match:
        for line in fm_match.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                meta[key.strip()] = val.strip().strip('"')

    # ── 섹션 블록 <!-- BEGIN:xxx --> ... <!-- END:xxx --> ─────────────────
    sections: dict[str, str] = {}
    for m in re.finditer(r"<!-- BEGIN:(\w+) -->\n(.*?)<!-- END:\1 -->", text, re.DOTALL):
        sections[m.group(1)] = m.group(2).strip()

    return {**meta, "sections": sections}


# ────────────────────────────────────────────────────────────────────────────
# 싱크 엔진
# ────────────────────────────────────────────────────────────────────────────

def replace_anchor(content: str, anchor: str, new_inner: str, *, html: bool = False) -> tuple[str, bool]:
    """
    앵커 패턴:
      Markdown : <!-- SYNC:xxx:start --> ... <!-- SYNC:xxx:end -->
      HTML     : <!-- SYNC:xxx:start --> ... <!-- SYNC:xxx:end -->  (same)
    inner 부분을 new_inner 로 교체. 변경 여부 반환.
    """
    pattern = rf"(<!-- SYNC:{re.escape(anchor)}:start -->)\n.*?(<!-- SYNC:{re.escape(anchor)}:end -->)"
    replacement = rf"\1\n{new_inner}\n\2"
    new_content, n = re.subn(pattern, replacement, content, flags=re.DOTALL)
    return new_content, n > 0


def _section_table_only(text: str) -> str:
    """섹션 텍스트에서 첫 번째 Markdown 표만 추출 (헤더 제외)."""
    lines = text.splitlines()
    table_lines = [l for l in lines if l.strip().startswith("|")]
    return "\n".join(table_lines).strip()


def make_md_results_table(data: dict) -> str:
    """RESEARCH_STATUS 섹션 → README 결과 표 (헤더 없이 표만)."""
    return _section_table_only(data["sections"].get("results_table", ""))


def make_md_checkpoints(data: dict) -> str:
    return _section_table_only(data["sections"].get("checkpoints", ""))


def make_html_metrics_bar(data: dict) -> str:
    def cell(v, lbl, note, color):
        return (
            f'    <div class="metric-cell">\n'
            f'      <div class="metric-value" style="color:{color}">{v}</div>\n'
            f'      <div class="metric-label">{lbl}</div>\n'
            f'      <div class="metric-note">{note}</div>\n'
            f'    </div>'
        )
    lines = [
        '<div class="metrics-bar">',
        '  <div class="metrics-grid">',
        cell(data["hero_metric1_value"], data["hero_metric1_label"], data["hero_metric1_note"], data["hero_metric1_color"]),
        cell(data["hero_metric2_value"], data["hero_metric2_label"], data["hero_metric2_note"], data["hero_metric2_color"]),
        cell(data["hero_metric3_value"], data["hero_metric3_label"], data["hero_metric3_note"], data["hero_metric3_color"]),
        cell(data["hero_metric4_value"], data["hero_metric4_label"], data["hero_metric4_note"], data["hero_metric4_color"]),
        "  </div>",
        "</div>",
    ]
    return "\n".join(lines)


def make_html_results_tbody(data: dict) -> str:
    """결과 표 섹션(Markdown) → HTML <tbody> rows."""
    md_table = data["sections"].get("results_table", "")
    rows = []
    for line in md_table.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("| Method") or set(line.replace("|", "").replace("-", "").replace(" ", "")) == set():
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        method, arch, cl, fpe = cells[0], cells[1], cells[2], cells[3]
        note = cells[4] if len(cells) > 4 else ""

        is_highlight = "★" in method or method.startswith("**")
        method_clean = method.strip("*").strip()
        cl_clean = cl.strip("*").strip()
        fpe_clean = fpe.strip("*").strip()
        arch = arch.strip("*").strip()

        # CL 컬러 클래스
        try:
            cl_num = float(cl_clean.rstrip("%"))
            cl_class = "val-good" if cl_num >= 80 else ("val-mid" if cl_num >= 50 else "val-bad")
        except ValueError:
            cl_class = ""

        row_class = ' class="highlight-row"' if is_highlight else ""
        rows.append(
            f'        <tr{row_class}>\n'
            f'          <td class="tag-model">{method_clean}</td>\n'
            f'          <td>{arch}</td>\n'
            f'          <td class="{cl_class}">{cl_clean}</td>\n'
            f'          <td>{fpe_clean}</td>\n'
            f'          <td>{note}</td>\n'
            f'        </tr>'
        )
    return "\n".join(rows)


def make_html_hero_tagline(data: dict) -> str:
    """Hero tagline 생성.

    ⚠️ 2026-08-16: 이전 버전은 sota_cl을 무조건 "closed-loop success"로 표기했는데,
    exp73부터 SOTA 지표가 시뮬 closed-loop가 아니라 실로봇 성공률로 바뀌어
    측정 축을 잘못 붙이는 문제가 있었다(89/100 배치와 95/100 배치 혼동과 같은 종류).
    이제 sota_metric_label로 축 이름을 명시하고, 부연 문구도 사실 그대로 쓴다.
    """
    cl = data.get("sota_cl", "—")
    note = data.get("sota_note", "")
    metric_label = data.get("sota_metric_label", "real-robot success")
    sub = data.get("hero_tagline_sub", "")
    lines = [
        '  <p class="hero-tagline">',
        '    Open-vocabulary detection + lightweight action head for mobile robot basket navigation.<br>',
        f'    <strong>{cl} {metric_label}</strong>' + (f' ({note})' if note else '') + '<br>',
    ]
    if sub:
        lines.append(f'    {sub}')
    lines.append('  </p>')
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# diff 출력
# ────────────────────────────────────────────────────────────────────────────

def show_diff(label: str, old: str, new: str):
    if old == new:
        return
    print(f"\n{BOLD}{CYAN}── {label} ──{RESET}")
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    for line in old_lines:
        if line not in new_lines:
            print(f"  {RED}- {line}{RESET}")
    for line in new_lines:
        if line not in old_lines:
            print(f"  {GREEN}+ {line}{RESET}")


# ────────────────────────────────────────────────────────────────────────────
# validate
# ────────────────────────────────────────────────────────────────────────────

ANCHORS_README = ["results_table", "checkpoints"]
ANCHORS_INDEX  = ["hero_tagline", "metrics_bar", "results_table_body"]

def validate(verbose: bool = True) -> bool:
    ok = True
    checks = [
        (README_FILE, "<!-- SYNC:{a}:start -->", ANCHORS_README),
        (INDEX_FILE,  "<!-- SYNC:{a}:start -->", ANCHORS_INDEX),
    ]
    for filepath, pattern_tmpl, anchors in checks:
        content = filepath.read_text(encoding="utf-8")
        for a in anchors:
            pattern = pattern_tmpl.format(a=a)
            found = pattern in content
            if verbose:
                sym = f"{GREEN}✓{RESET}" if found else f"{RED}✗{RESET}"
                print(f"  {sym}  {filepath.name:<30} SYNC:{a}")
            if not found:
                ok = False
    # RESEARCH_STATUS 섹션
    data = parse_status(STATUS_FILE)
    for sec in ["results_table", "checkpoints", "key_findings", "experiment_history"]:
        found = sec in data["sections"]
        if verbose:
            sym = f"{GREEN}✓{RESET}" if found else f"{RED}✗{RESET}"
            print(f"  {sym}  RESEARCH_STATUS.md              section:{sec}")
        if not found:
            ok = False
    return ok


# ────────────────────────────────────────────────────────────────────────────
# 실험 추가 (--add-exp)
# ────────────────────────────────────────────────────────────────────────────

def add_experiment_interactive():
    print(f"\n{BOLD}새 실험 추가 → RESEARCH_STATUS.md experiment_history{RESET}")
    exp    = input("  Exp 이름 (예: Exp68): ").strip()
    date   = input("  날짜 (예: 2026-06-17): ").strip()
    arch   = input("  Architecture: ").strip()
    cl     = input("  CL%: ").strip()
    fpe    = input("  FPE (없으면 —): ").strip() or "—"
    val    = input("  val_acc (없으면 —): ").strip() or "—"
    note   = input("  특이사항: ").strip()

    new_row = f"| {exp} | {date} | {arch} | {cl} | {fpe} | {val} | {note} |"

    text = STATUS_FILE.read_text(encoding="utf-8")
    if new_row in text:
        print(f"{YELLOW}이미 존재하는 행: {exp}{RESET}")
        return

    # experiment_history 섹션 마지막 행 뒤에 삽입
    old_marker = "<!-- END:experiment_history -->"
    new_text = text.replace(old_marker, f"{new_row}\n{old_marker}")
    STATUS_FILE.write_text(new_text, encoding="utf-8")
    print(f"{GREEN}✓ 추가됨: {new_row}{RESET}")
    print(f"  → 이제 sync 실행: python3 scripts/utils/sync_research.py")


# ────────────────────────────────────────────────────────────────────────────
# menemory 제안 출력
# ────────────────────────────────────────────────────────────────────────────

def propose_menemory(data: dict):
    proposal = data["sections"].get("menemory_proposal", "")
    sota = data.get("sota_exp", "Exp66")
    cl   = data.get("sota_cl", "96.6%")
    fpe  = data.get("sota_fpe", "0.094 m")
    acc  = data.get("sota_val_acc", "93.5%")
    s1   = data.get("sota_ckpt_s1", "")
    s2   = data.get("sota_ckpt_s2", "")
    updated = data.get("updated", "")

    print(f"\n{BOLD}{CYAN}══ menemory 업데이트 제안 ══════════════════════════════{RESET}")
    print(f"  파일: .menemory/core/master_memory.md")
    print(f"  섹션: '현재 최선 모델' 교체 권장\n")
    print(textwrap.dedent(f"""\
## 현재 최선 모델 ({updated} 기준)

### Decomposition SOTA (실로봇 배포 중)
- **{sota}** — Stage2 v2 FrozenCLIP + MLP + L2-norm + aug
  - closed-loop: **{cl}**, FPE: {fpe}, val_acc: {acc}
  - ckpt S1: `{s1}`
  - ckpt S2: `{s2}`
  - 추론 서버: `robovlm_nav/serve/stage2_v2_inference_server.py --port 8001`

### End-to-end (폐기)
- **Exp11** — Kosmos-2 + LoRA: CL 0%, text attn 구조적 사망 → E2E 경로 완전 종료

### 핵심 음성 결과
- Grounding source irrelevant: HSV/PG2/LoRA 모두 96.6% → LoRA grounding 연구의 action 기여 = 0
- Pipeline (L2+aug)이 유일 결정 변수
    """))
    print(f"{YELLOW}→ 위 내용을 master_memory.md에 반영하려면 직접 편집하거나{RESET}")
    print(f"{YELLOW}  `menemory` 명령으로 업데이트하세요. (CLAUDE.md 정책: 제안만, 직접 쓰지 않음){RESET}")


# ────────────────────────────────────────────────────────────────────────────
# 상태 리포트
# ────────────────────────────────────────────────────────────────────────────

def status_report(data: dict):
    print(f"\n{BOLD}══ Research Status ({data.get('updated', '?')}) ══════════════════════{RESET}")
    print(f"  SOTA: {BOLD}{data.get('sota_exp','?')}{RESET} — {data.get('sota_arch','?')}")
    print(f"  CL:   {GREEN}{data.get('sota_cl','?')}{RESET}  │  FPE: {data.get('sota_fpe','?')}  │  val_acc: {data.get('sota_val_acc','?')}")
    print(f"  E2E:  {RED}{data.get('e2e_cl','?')}{RESET} (Exp11) │  Pipeline gap: {data.get('pipeline_gap','?')}")
    secs = data.get("sections", {})
    print(f"\n  섹션: {', '.join(secs.keys())}")


# ────────────────────────────────────────────────────────────────────────────
# 메인 싱크
# ────────────────────────────────────────────────────────────────────────────

def sync(dry_run: bool = False, show_diff_flag: bool = False):
    if not STATUS_FILE.exists():
        print(f"{RED}✗ {STATUS_FILE} 없음{RESET}")
        sys.exit(1)

    data = parse_status(STATUS_FILE)

    changes: list[tuple[str, Path, str]] = []  # (label, path, new_content)

    # ── README.md ────────────────────────────────────────────────────────
    readme = README_FILE.read_text(encoding="utf-8")
    readme_orig = readme

    # 결과 표
    md_table = make_md_results_table(data)
    readme, found = replace_anchor(readme, "results_table", md_table)
    if not found:
        print(f"{YELLOW}  ⚠ README: SYNC:results_table 앵커 없음{RESET}")

    # 체크포인트
    md_ckpt = make_md_checkpoints(data)
    readme, found = replace_anchor(readme, "checkpoints", md_ckpt)
    if not found:
        print(f"{YELLOW}  ⚠ README: SYNC:checkpoints 앵커 없음{RESET}")

    # 날짜
    readme = re.sub(
        r"<!-- SYNC:updated -->.*?<!-- /SYNC:updated -->",
        f"<!-- SYNC:updated -->{data.get('updated', '')}<!-- /SYNC:updated -->",
        readme,
    )

    if readme != readme_orig:
        changes.append(("README.md", README_FILE, readme))
        if show_diff_flag:
            show_diff("README.md", readme_orig, readme)

    # ── docs/index.html ──────────────────────────────────────────────────
    index = INDEX_FILE.read_text(encoding="utf-8")
    index_orig = index

    # hero tagline
    tagline = make_html_hero_tagline(data)
    index, found = replace_anchor(index, "hero_tagline", tagline, html=True)
    if not found:
        print(f"{YELLOW}  ⚠ index.html: SYNC:hero_tagline 앵커 없음{RESET}")

    # metrics bar
    metrics = make_html_metrics_bar(data)
    index, found = replace_anchor(index, "metrics_bar", metrics, html=True)
    if not found:
        print(f"{YELLOW}  ⚠ index.html: SYNC:metrics_bar 앵커 없음{RESET}")

    # results table body
    tbody_rows = make_html_results_tbody(data)
    index, found = replace_anchor(index, "results_table_body", f"      <tbody>\n{tbody_rows}\n      </tbody>", html=True)
    if not found:
        print(f"{YELLOW}  ⚠ index.html: SYNC:results_table_body 앵커 없음{RESET}")

    if index != index_orig:
        changes.append(("docs/index.html", INDEX_FILE, index))
        if show_diff_flag:
            show_diff("docs/index.html", index_orig, index)

    # ── 결과 출력 ─────────────────────────────────────────────────────────
    if not changes:
        print(f"{GREEN}✓ 모든 파일 최신 상태 — 변경 없음{RESET}")
        return

    print()
    for label, path, new_content in changes:
        if dry_run:
            print(f"  {YELLOW}[dry-run]{RESET} {label} — 변경 예정")
        else:
            path.write_text(new_content, encoding="utf-8")
            print(f"  {GREEN}✓{RESET} {label} 갱신 완료")

    if not dry_run:
        print(f"\n{BOLD}총 {len(changes)}개 파일 업데이트됨{RESET}")
        print(f"  commit: git add docs/RESEARCH_STATUS.md README.md docs/index.html")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Research Status Sync")
    parser.add_argument("--dry-run",          action="store_true", help="파일 수정 없이 로그만")
    parser.add_argument("--diff",             action="store_true", help="변경사항 diff 출력")
    parser.add_argument("--validate",         action="store_true", help="앵커 존재 여부만 검사")
    parser.add_argument("--propose-menemory", action="store_true", help="menemory 업데이트 제안 출력")
    parser.add_argument("--add-exp",          action="store_true", help="새 실험 행 대화형 추가")
    parser.add_argument("--status",           action="store_true", help="현재 SOTA 요약 출력")
    args = parser.parse_args()

    print(f"\n{BOLD}EdgeGround-VLA Research Sync{RESET}  ←  {STATUS_FILE.relative_to(ROOT)}")

    if not STATUS_FILE.exists():
        print(f"{RED}✗ RESEARCH_STATUS.md 없음: {STATUS_FILE}{RESET}")
        sys.exit(1)

    data = parse_status(STATUS_FILE)

    if args.status:
        status_report(data)
        return

    if args.validate:
        print()
        ok = validate()
        if ok:
            print(f"\n{GREEN}✓ 모든 앵커 정상{RESET}")
        else:
            print(f"\n{RED}✗ 일부 앵커 누락 — 위 목록 확인{RESET}")
            sys.exit(1)
        return

    if args.propose_menemory:
        propose_menemory(data)
        return

    if args.add_exp:
        add_experiment_interactive()
        return

    sync(dry_run=args.dry_run, show_diff_flag=args.diff)


if __name__ == "__main__":
    main()
