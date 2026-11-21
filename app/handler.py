"""Lambda entry point (Function URL): serves the web UI and the /api endpoints."""

import base64
import json
import mimetypes
import os
import traceback

import github_save
from analyzer import analyze_frames

mimetypes.add_type("font/ttf", ".ttf")
mimetypes.add_type("text/javascript", ".js")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
    INDEX_HTML = f.read().decode()


def _response(status, body, content_type="application/json"):
    if not isinstance(body, str):
        body = json.dumps(body)
    return {"statusCode": status, "headers": {"Content-Type": content_type}, "body": body}


def _static(path):
    rel = path[len("/static/"):]
    full = os.path.realpath(os.path.join(STATIC_DIR, rel))
    if not full.startswith(os.path.realpath(STATIC_DIR) + os.sep) or not os.path.isfile(full):
        return _response(404, {"error": "Not found"})
    with open(full, "rb") as f:
        data = f.read()
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": mimetypes.guess_type(full)[0] or "application/octet-stream",
            "Cache-Control": "public, max-age=86400",
        },
        "body": base64.b64encode(data).decode(),
        "isBase64Encoded": True,
    }


def _json_body(event):
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode()
    return json.loads(raw)


def lambda_handler(event, context):
    http = event.get("requestContext", {}).get("http", {})
    method, path = http.get("method", "GET"), event.get("rawPath", "/")

    if method == "GET" and path in ("/", "/index.html"):
        return _response(200, INDEX_HTML, "text/html; charset=utf-8")
    if method == "GET" and path.startswith("/static/"):
        return _static(path)
    if method == "GET" and path == "/api/health":
        return _response(200, {"ok": True})
    if method == "GET" and path == "/api/github/status":
        return _response(200, github_save.status())
    if method == "POST" and path == "/api/github/save":
        try:
            return _response(200, github_save.save(_json_body(event)))
        except github_save.SaveError as e:
            return _response(e.status, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            return _response(500, {"error": f"{type(e).__name__}: {e}"})
    if method == "POST" and path == "/api/analyze":
        try:
            payload = _json_body(event)
            result = analyze_frames(payload.get("frames", []), payload.get("context", "")[:2000])
            return _response(200, result)
        except ValueError as e:
            return _response(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            return _response(500, {"error": f"{type(e).__name__}: {e}"})
    return _response(404, {"error": "Not found"})
