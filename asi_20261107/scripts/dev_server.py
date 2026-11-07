"""Local dev server: runs the Lambda handler on http://localhost:8000 with uploads stored on disk.

usage: .venv/bin/python scripts/dev_server.py     (reads META_API_KEY from env or ../env.local)
"""

import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("LOCAL_UPLOAD_DIR", os.path.join(tempfile.gettempdir(), "asi_video_uploads"))
if not os.environ.get("META_API_KEY"):
    env_file = os.path.join(HERE, "..", "..", "env.local")
    if os.path.exists(env_file):
        os.environ["META_API_KEY"] = open(env_file).read().strip()
sys.path.insert(0, os.path.join(HERE, "..", "app"))
from handler import lambda_handler  # noqa: E402

UPLOADS = os.environ["LOCAL_UPLOAD_DIR"]


class Handler(BaseHTTPRequestHandler):
    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _send(self, status, body, headers):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _lambda(self, method):
        body = self._body().decode() if method == "POST" else None
        ev = {"rawPath": self.path.split("?")[0], "body": body, "requestContext": {"http": {"method": method}}}
        res = lambda_handler(ev, None)
        self._send(res["statusCode"], res["body"], res["headers"])

    def do_GET(self):
        self._lambda("GET")

    def do_POST(self):
        self._lambda("POST")

    def do_PUT(self):
        key = self.path.split("?")[0][len("/local-upload/"):]
        if not self.path.startswith("/local-upload/uploads/") or ".." in key:
            return self._send(404, "not found", {})
        dest = os.path.join(UPLOADS, key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(self._body())
        self._send(200, "", {})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"http://localhost:{port}  (uploads in {UPLOADS})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
