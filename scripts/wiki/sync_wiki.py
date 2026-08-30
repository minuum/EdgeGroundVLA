#!/usr/bin/env python3
"""wiki-sync 오케스트레이터 — research_story.html에 새 챕터가 추가된 뒤 위키를
안전하게 재동기화한다 (`.claude/skills/wiki-sync/SKILL.md`에서 호출).

순서: parse_research_story.py(chapters.json 갱신) → build_wiki_pages.py(주제
페이지 재생성, 기존 압축 요약은 보존) → render_wiki_html.py(정적 HTML) →
build_archive_index.py(357개 스냅샷 카탈로그, chapters.json과 무관하지만 매번
같이 최신화). 마지막에 "사람/LLM이 추가로 해야 할 일" 목록을 출력한다 —
이 스크립트 자체는 토픽 배정이나 요약 압축처럼 판단이 필요한 일은 절대
자동으로 하지 않는다(Karpathy 방식 원칙: 기계적 파싱/재구성은 스크립트,
"무엇을 아는가"의 압축은 LLM).

사용: .venv/bin/python3 scripts/wiki/sync_wiki.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
WIKI = ROOT / "scripts/wiki"
PY = sys.executable


def run(script, use_venv_markdown=False):
    exe = PY
    venv_py = ROOT / ".venv/bin/python3"
    if use_venv_markdown and venv_py.exists():
        exe = str(venv_py)
    print(f"\n$ {exe} {script}")
    r = subprocess.run([exe, str(WIKI / script)], cwd=str(ROOT), capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"{script} 실패(exit {r.returncode})")
    return r.stdout


def main():
    print("=" * 60)
    print("wiki-sync 시작")
    print("=" * 60)

    out1 = run("parse_research_story.py")
    out2 = run("build_wiki_pages.py")
    run("build_archive_index.py")
    try:
        run("render_wiki_html.py", use_venv_markdown=True)
    except SystemExit as e:
        print(f"\n⚠️ HTML 렌더링 실패({e}) — markdown 패키지 미설치일 수 있음. "
              f"pip install -r scripts/wiki/requirements.txt 후 재시도. "
              f"(markdown 파일 자체는 정상 갱신됐으니 급하지 않으면 넘어가도 됨)")

    print("\n" + "=" * 60)
    print("다음 확인 필요(사람/LLM 판단 필요, 이 스크립트가 자동으로 안 함):")
    print("=" * 60)

    if "재압축 필요" in out2:
        idx = out2.index("재압축 필요")
        print("\n[1] 새 챕터가 추가돼 요약이 낡은 주제:")
        print(out2[idx - 2:].split("어떤 주제")[0].strip())
        print("→ 해당 topic 파일을 Read로 전체 정독 후, '## 압축 요약' 섹션을 다시 쓸 것.")
        print("  (파일이 크면 general-purpose 서브에이전트에 위임 권장 — 이전 세션 패턴 참고)")

    if "안 걸린 챕터" in out2:
        idx = out2.index("안 걸린 챕터")
        print("\n[2] 어떤 주제에도 배정 안 된 새 챕터:")
        print(out2[idx:].strip())
        print("→ 챕터 제목/내용 확인 후 docs/wiki/data/topic_index.json의 chapter_ids에 추가")
        print("  (기존 주제에 안 맞으면 새 주제를 만들어도 됨 — topic_index.json에 새 키 추가)")

    print("\n완료 후: git add docs/wiki/ scripts/wiki/ && git commit && git push")


if __name__ == "__main__":
    main()
