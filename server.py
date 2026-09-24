#!/usr/bin/env python3
"""
Super dynamic, gorgeous, mobile-first viewer server.
Serves viewer.html + output/plots + output/web
"""
import http.server
import socketserver
import os
from pathlib import Path

ROOT = Path(__file__).parent
PORT = int(os.environ.get("PORT", 8000))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)
    
    def end_headers(self):
        # CORS + mobile-friendly headers
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()
    
    def do_GET(self):
        # serve viewer.html as index
        if self.path == "/" or self.path == "":
            self.path = "/viewer.html"
        return super().do_GET()

if __name__ == "__main__":
    print(f"Serving at http://0.0.0.0:{PORT} — root {ROOT}")
    print(f"Viewer: http://0.0.0.0:{PORT}/viewer.html")
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped")
