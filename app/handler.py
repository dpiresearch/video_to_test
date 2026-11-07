"""Lambda entry point (Function URL): serves the web UI and the /api/analyze endpoint."""

import base64
import json
import os
import traceback

from analyzer import analyze_frames

with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "rb") as f:
    INDEX_HTML = f.read().decode()


def _response(status, body, content_type="application/json"):
    if not isinstance(body, str):
        body = json.dumps(body)
    return {"statusCode": status, "headers": {"Content-Type": content_type}, "body": body}


def lambda_handler(event, context):
    http = event.get("requestContext", {}).get("http", {})
    method, path = http.get("method", "GET"), event.get("rawPath", "/")

    if method == "GET" and path in ("/", "/index.html"):
        return _response(200, INDEX_HTML, "text/html; charset=utf-8")
    if method == "GET" and path == "/api/health":
        return _response(200, {"ok": True})
    if method == "POST" and path == "/api/analyze":
        try:
            raw = event.get("body") or "{}"
            if event.get("isBase64Encoded"):
                raw = base64.b64decode(raw).decode()
            payload = json.loads(raw)
            result = analyze_frames(payload.get("frames", []), payload.get("context", "")[:2000])
            return _response(200, result)
        except ValueError as e:
            return _response(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            return _response(500, {"error": f"{type(e).__name__}: {e}"})
    return _response(404, {"error": "Not found"})
