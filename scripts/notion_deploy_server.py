#!/usr/bin/env python3
"""MoNaVLA docs 정적 서빙 + 내 Notion 전용 원클릭 배포 통합 서버 (기본 포트 8001).

왜 8001 포트에 통합했는가?
  - UFW 방화벽이 8001/tcp(tailscale0)만 허용하고 별도 포트(8788)를 차단함
  - 동일 출처(Same-Origin: /api/notion/*)로 동작하므로 CORS·방화벽·추가 포트 문제 없음

사용법:
  python3 scripts/notion_deploy_server.py --port 8001
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

NOTION_VERSION = "2022-06-28"
ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
ENV_FILE = ROOT / ".env.notion"
MCP_CONFIG = Path.home() / ".gemini/antigravity/mcp_config.json"


def _read_env_notion() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ENV_FILE.exists():
        return out
    for line in ENV_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _get_server_config() -> tuple[str, str, str]:
    env_map = _read_env_notion()
    default_page = os.environ.get("NOTION_PAGE_ID") or env_map.get("NOTION_PAGE_ID", "")

    if os.environ.get("NOTION_TOKEN"):
        return os.environ["NOTION_TOKEN"].strip(), default_page, "ENV(NOTION_TOKEN)"
    if env_map.get("NOTION_TOKEN"):
        return env_map["NOTION_TOKEN"].strip(), default_page, ".env.notion"
    if MCP_CONFIG.exists():
        raw = MCP_CONFIG.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"(ntn_[A-Za-z0-9_\-]{20,}|secret_[A-Za-z0-9_\-]{20,})", raw)
        if m:
            return m.group(1), default_page, "mcp_config.json"
    return "", default_page, "none"


def _normalize_page_id(raw: str) -> str:
    raw = raw.strip()
    m = re.search(r"([0-9a-fA-F]{32})", raw.replace("-", ""))
    if not m:
        raise ValueError("유효한 Notion Page ID(32자리 hex) 또는 페이지 URL이 아닙니다.")
    h = m.group(1).lower()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def _rich_text(text: str, bold: bool = False, url: str | None = None) -> dict:
    obj: dict = {"type": "text", "text": {"content": text[:1900]}}
    if url:
        obj["text"]["link"] = {"url": url}
    if bold:
        obj["annotations"] = {"bold": True}
    return obj


def _build_blocks(payload: dict) -> list[dict]:
    blocks: list[dict] = []
    # 상단 원라인 요약 콜아웃 + 클릭 가능한 링크
    summary_line = str(
        payload.get("hero_callout")
        or "클라우드·지도 없이 엣지 로봇에서 작동하고, 미학습 물체에도 안전하게 일반화되는 온디바이스 VLA 정책을 연구·구현합니다."
    ).strip()
    blocks.append({
        "object": "block",
        "type": "callout",
        "callout": {
            "icon": {"type": "emoji", "emoji": "🤖"},
            "color": "gray_background",
            "rich_text": [
                _rich_text(summary_line + "\n"),
                _rich_text("Project Page", bold=True, url="https://minuum.github.io/EdgeGroundVLA/"),
                _rich_text("  ·  "),
                _rich_text("GitHub", bold=True, url="https://github.com/minuum"),
                _rich_text("  ·  "),
                _rich_text("minwool0357@gmail.com", bold=True, url="mailto:minwool0357@gmail.com"),
            ],
        },
    })

    for sec in payload.get("sections", []):
        if not sec.get("enabled", True):
            continue
        heading = str(sec.get("heading", "")).strip()
        body = str(sec.get("body", "")).strip()
        img_url = str(sec.get("image_url", "")).strip()

        if heading:
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [_rich_text(heading)]},
            })
        if body:
            for line in [ln.strip() for ln in body.split("\n") if ln.strip()]:
                if line.startswith(("- ", "• ", "* ")):
                    txt = line[2:].strip()
                    if ":" in txt and len(txt.split(":", 1)[0]) <= 18:
                        k, v = txt.split(":", 1)
                        rt = [_rich_text(k + ":", bold=True), _rich_text(v)]
                    else:
                        rt = [_rich_text(txt)]
                    blocks.append({
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": rt},
                    })
                else:
                    blocks.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [_rich_text(line)]},
                    })
        if img_url.startswith("https://"):
            blocks.append({
                "object": "block",
                "type": "image",
                "image": {"type": "external", "external": {"url": img_url}},
            })
        blocks.append({"object": "block", "type": "divider", "divider": {}})

    details = payload.get("details", [])
    if details:
        children = []
        for grp in details:
            gtitle = str(grp.get("title", "")).strip()
            if gtitle:
                children.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {"rich_text": [_rich_text(gtitle)]},
                })
            for item in grp.get("items", []):
                txt = str(item).strip()
                if txt:
                    children.append({
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [_rich_text(txt)]},
                    })
        blocks.append({
            "object": "block",
            "type": "toggle",
            "toggle": {
                "rich_text": [_rich_text("세부 논문 · 프로젝트 · 수상 · 기술 스택 보기", bold=True)],
                "children": children[:95],
            },
        })
    return blocks


def _notion_request(method: str, path: str, token: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"https://api.notion.com/v1{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _list_accessible_pages(token: str) -> list[dict]:
    res = _notion_request(
        "POST",
        "/search",
        token,
        {"filter": {"value": "page", "property": "object"}, "page_size": 25},
    )
    pages = []
    for p in res.get("results", []):
        title = "(제목 없음)"
        for pv in p.get("properties", {}).values():
            if pv.get("type") == "title" and pv.get("title"):
                title = "".join(t.get("plain_text", "") for t in pv["title"])
                break
        pages.append({"id": p.get("id", ""), "title": title, "url": p.get("url", "")})
    return pages


def _clear_page_blocks(token: str, pid: str) -> int:
    """대상 페이지의 기존 최상위 블록을 모두 아카이브(삭제)해 중복을 없앤다."""
    deleted = 0
    cursor = None
    while True:
        q = f"/blocks/{pid}/children?page_size=100" + (f"&start_cursor={cursor}" if cursor else "")
        res = _notion_request("GET", q, token)
        for blk in res.get("results", []):
            bid = blk.get("id")
            if bid:
                _notion_request("DELETE", f"/blocks/{bid}", token)
                deleted += 1
        if not res.get("has_more"):
            break
        cursor = res.get("next_cursor")
    return deleted


def _deploy_to_notion(token: str, page_id: str, payload: dict) -> dict:
    pid = _normalize_page_id(page_id)
    blocks = _build_blocks(payload)
    mode = payload.get("mode", "replace")

    if mode == "new_subpage":
        title = str(payload.get("doc_title") or "Minwoo Lee | Robotics & AI Portfolio").strip()
        created = _notion_request(
            "POST",
            "/pages",
            token,
            {
                "parent": {"page_id": pid},
                "properties": {"title": {"title": [_rich_text(title)]}},
                "children": blocks[:95],
            },
        )
        return {"ok": True, "url": created.get("url", ""), "blocks_added": len(blocks), "mode": mode}

    deleted = 0
    if mode == "replace":
        deleted = _clear_page_blocks(token, pid)

    _notion_request("PATCH", f"/blocks/{pid}/children", token, {"children": blocks[:95]})
    page_info = _notion_request("GET", f"/pages/{pid}", token)
    return {"ok": True, "url": page_info.get("url", ""), "blocks_added": len(blocks), "blocks_deleted": deleted, "mode": mode}


class UnifiedHandler(SimpleHTTPRequestHandler):
    def _send_json(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/api/notion/health", "/health"):
            token, default_page, source = _get_server_config()
            self._send_json(200, {
                "ok": True,
                "token_configured": bool(token),
                "token_source": source,
                "default_page_id": default_page,
            })
            return
        if path in ("/api/notion/pages", "/pages"):
            token, default_page, source = _get_server_config()
            if not token:
                self._send_json(400, {"ok": False, "error": "서버에 NOTION_TOKEN이 설정되지 않았습니다 (.env.notion 확인)."})
                return
            try:
                pages = _list_accessible_pages(token)
                self._send_json(200, {
                    "ok": True,
                    "token_source": source,
                    "default_page_id": default_page,
                    "pages": pages,
                })
            except urllib.error.HTTPError as e:
                self._send_json(502, {
                    "ok": False,
                    "token_source": source,
                    "error": f"Notion 인증 오류 ({e.code}) — {source}의 토큰이 만료되었거나 권한이 없습니다.",
                })
            except Exception as e:
                self._send_json(500, {"ok": False, "error": str(e)})
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in ("/api/notion/deploy", "/deploy"):
            self._send_json(404, {"ok": False, "error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 500_000:
                raise ValueError("요청 본문 크기가 올바르지 않습니다.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            token, default_page, source = _get_server_config()
            if not token:
                raise ValueError("서버에 NOTION_TOKEN이 설정되지 않았습니다 (.env.notion에 설정하세요).")
            page_id = str(payload.get("page_id") or default_page or "").strip()
            if not page_id:
                raise ValueError("배포할 내 Notion 페이지를 선택하세요.")
            result = _deploy_to_notion(token, page_id, payload)
            result["token_source"] = source
            self._send_json(200, result)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:350]
            self._send_json(502, {"ok": False, "error": f"Notion API 오류 ({e.code}): {detail}"})
        except Exception as e:
            self._send_json(400, {"ok": False, "error": str(e)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    handler = functools.partial(UnifiedHandler, directory=str(DOCS_DIR))
    print(f"MoNaVLA Docs + Notion Bridge listening on http://{args.host}:{args.port} (dir={DOCS_DIR})")
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
