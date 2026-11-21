"""Run the Lambda handler locally: python scripts/local_server.py  ->  http://localhost:8000"""

import base64
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from handler import lambda_handler  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    def _handle(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode() if length else None
        event = {"rawPath": self.path.split("?")[0], "body": body, "requestContext": {"http": {"method": method}}}
        res = lambda_handler(event, None)
        data = base64.b64decode(res["body"]) if res.get("isBase64Encoded") else res["body"].encode()
        self.send_response(res["statusCode"])
        for k, v in res["headers"].items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Serving on http://localhost:{port}")
    ThreadingHTTPServer(("", port), Handler).serve_forever()
