#!/bin/bash
# 리서치 히스토리 로컬호스트 서빙 — GitHub Pages가 docs/**/*.png를 gitignore로 빼서
# 이미지가 안 뜨거나, 아직 push 안 한 로컬 그림/문서를 미리 보고 싶을 때 사용.
#
# 로컬 docs/는 gitignore와 무관하게 모든 png/jpg가 그대로 있으므로, 여기서 서빙하면
# CH64 그림 포함 전부 정상 렌더된다.
#
# 사용:
#   bash scripts/serve_research_story.sh            # 기본 포트 8899
#   bash scripts/serve_research_story.sh 9000        # 포트 지정
# 그 후 브라우저에서:
#   http://localhost:<port>/v5/research_story.html
#   http://localhost:<port>/index.html
set -e
cd "$(dirname "$0")/../docs"
PORT="${1:-8899}"
echo "리서치 히스토리 로컬 서버 시작 → http://localhost:${PORT}/v5/research_story.html"
echo "(CH64 그림은 http://localhost:${PORT}/v5/research_story.html#ch64 에서 확인)"
echo "중지: Ctrl+C"
python3 -m http.server "$PORT"
