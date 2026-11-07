"""Lambda entry point (Function URL). Serves the UI and the JSON API.

GET  /                  web UI
POST /api/upload-url    {filename, content_type}            -> presigned S3 PUT
POST /api/concepts      {frames[], context}                  -> SAM concept prompts (Claude)
POST /api/segment       {key, concepts[], duration}          -> SAM 3.1 tracks (boxes + masks)
POST /api/analyze       {frames[], tracks[], fps, duration}  -> steps + test case (Claude + SAM evidence)
"""

import base64
import json
import mimetypes
import os
import shutil
import tempfile
import traceback

import evidence
import llm
import media
import sam

STATIC = os.path.join(os.path.dirname(__file__), "static")
MAX_CONCEPTS = 8


def _resp(status, body, ctype="application/json", cache=None):
    headers = {"Content-Type": ctype}
    if cache:
        headers["Cache-Control"] = cache
    return {"statusCode": status, "headers": headers, "body": body if isinstance(body, str) else json.dumps(body)}


def _static(name):
    path = os.path.realpath(os.path.join(STATIC, name))
    if not path.startswith(os.path.realpath(STATIC)) or not os.path.isfile(path):
        return _resp(404, {"error": "Not found"})
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, encoding="utf-8") as f:
        return _resp(200, f.read(), f"{ctype}; charset=utf-8", "no-cache")


def api_upload_url(p):
    return media.presign_upload(str(p.get("filename", "video.mp4"))[:200], str(p.get("content_type", ""))[:100])


def api_concepts(p):
    frames = p.get("frames") or []
    if not frames:
        raise ValueError("No frames provided")
    return llm.choose_concepts(frames, str(p.get("context", ""))[:2000])


def api_segment(p):
    concepts = []
    for c in p.get("concepts") or []:
        c = " ".join(str(c).split())[:60]
        if c and c.lower() not in (x.lower() for x in concepts):
            concepts.append(c)
    if not concepts:
        raise ValueError("No concepts provided")
    concepts = concepts[:MAX_CONCEPTS]
    workdir = tempfile.mkdtemp(dir="/tmp")
    file_id = None
    try:
        src = media.download(str(p.get("key", "")), workdir)
        path, info, transcoded = media.normalize(src, workdir)
        file_id = sam.upload_video(path, "video" + os.path.splitext(path)[1], "video/quicktime"
                                   if path.lower().endswith(".mov") else "video/mp4")
        out = sam.segment_video(file_id, concepts)
    finally:
        if file_id:
            sam.delete_file(file_id)
        shutil.rmtree(workdir, ignore_errors=True)
    duration = info["duration"] or float(p.get("duration") or 0)
    out["duration"] = duration
    out["fps"] = (out["frames"] / duration) if duration and out["frames"] else 30.0
    out["concepts"] = concepts
    out["video"] = {**info, "transcoded": transcoded}
    return out


def api_analyze(p):
    frames = p.get("frames") or []
    tracks = p.get("tracks") or []
    fps = float(p.get("fps") or 30)
    duration = float(p.get("duration") or 0)
    text, facts = evidence.summarize(tracks, fps, duration, [float(f["t"]) for f in frames])
    result = llm.analyze(frames, text, str(p.get("context", ""))[:2000])
    result["sam_evidence"] = text
    result["track_facts"] = facts
    return result


ROUTES = {"/api/upload-url": api_upload_url, "/api/concepts": api_concepts,
          "/api/segment": api_segment, "/api/analyze": api_analyze}


def lambda_handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("rawPath", "/")
    if method == "GET":
        if path in ("/", "/index.html"):
            return _static("index.html")
        if path == "/api/health":
            body = {"ok": True}
            if "deep=1" in (event.get("rawQueryString") or ""):
                try:
                    body["meta_models"] = sam.list_models()
                except Exception as e:
                    body["meta_error"] = str(e)
            return _resp(200, body)
        if path.startswith("/static/"):
            return _static(path[len("/static/"):])
        return _resp(404, {"error": "Not found"})
    if method == "POST" and path in ROUTES:
        try:
            raw = event.get("body") or "{}"
            if event.get("isBase64Encoded"):
                raw = base64.b64decode(raw).decode()
            return _resp(200, ROUTES[path](json.loads(raw)))
        except ValueError as e:
            return _resp(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            return _resp(500, {"error": f"{type(e).__name__}: {e}"})
    return _resp(405, {"error": "Method not allowed"})
