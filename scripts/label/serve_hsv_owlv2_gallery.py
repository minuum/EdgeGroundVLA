#!/usr/bin/env python3
"""CH54 프리뷰 후보(H1/H2/OWL-v2 vs PG2) 갤러리 정적 서빙.
Usage: python3 scripts/label/serve_hsv_owlv2_gallery.py
접속: http://localhost:7792
"""
import http.server
import socketserver
from pathlib import Path

DIR = Path("/home/minum/26CS/MoNaVLA/docs/v5/hsv_owlv2_preview_20260704")
PORT = 7792


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIR), **kwargs)


if __name__ == "__main__":
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"브라우저 → http://localhost:{PORT}/gallery.html")
        httpd.serve_forever()
