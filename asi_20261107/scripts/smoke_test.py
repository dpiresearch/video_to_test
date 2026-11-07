"""End-to-end API smoke test against a running server (local or deployed), mimicking the browser flow.

usage: python scripts/smoke_test.py BASE_URL VIDEO [--frames 8]
Frames are extracted with local ffmpeg (no SAM boxes drawn: the UI does that, the server doesn't need it).
"""

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import tempfile
import time
import urllib.request


def call(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=8)
    a = ap.parse_args()
    base = a.base.rstrip("/")
    t0 = time.time()
    lap = lambda msg: print(f"[{time.time() - t0:5.1f}s] {msg}")  # noqa: E731

    print(urllib.request.urlopen(base + "/api/health").read().decode())
    ctype = mimetypes.guess_type(a.video)[0] or "video/mp4"
    up = call(base, "/api/upload-url", {"filename": os.path.basename(a.video), "content_type": ctype})
    url = up["url"] if up["url"].startswith("http") else base + up["url"]
    with open(a.video, "rb") as f:
        urllib.request.urlopen(urllib.request.Request(url, f.read(), {"Content-Type": up["content_type"]}, method="PUT"))
    lap(f"uploaded -> {up['key']}")

    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.video],
                               capture_output=True, text=True).stdout.strip())
    work = tempfile.mkdtemp()
    frames = []
    for i in range(a.frames):
        t = dur * (i + 0.5) / a.frames
        out = os.path.join(work, f"{i}.jpg")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", a.video, "-frames:v", "1",
                        "-vf", "scale=768:-2", "-q:v", "4", out], check=True)
        frames.append({"t": round(t, 3), "data": base64.b64encode(open(out, "rb").read()).decode()})

    c = call(base, "/api/concepts", {"frames": frames})
    lap(f"concepts: {c}")
    seg = call(base, "/api/segment", {"key": up["key"], "concepts": c["concepts"], "duration": dur})
    size = len(json.dumps(seg))
    lap(f"segment: {len(seg['tracks'])} tracks, frames={seg['frames']} fps={seg['fps']:.2f} "
        f"found={seg.get('concepts_found')} errors={seg['errors']} video={seg['video']} response={size / 1e6:.2f}MB")
    tracks = [{k: t[k] for k in ("track_id", "label", "concept", "boxes")} for t in seg["tracks"]]
    res = call(base, "/api/analyze", {"frames": frames, "tracks": tracks, "fps": seg["fps"], "duration": seg["duration"]})
    lap(f"analyze: {res['title']}  ({len(res['steps'])} steps, {len(res['test_case']['procedure'])} procedure steps)")
    for s in res["steps"]:
        print(f"   {s['start']:5.2f}-{s['end']:5.2f}s  {s['action'][:110]}  {s.get('track_ids')}")


if __name__ == "__main__":
    main()
