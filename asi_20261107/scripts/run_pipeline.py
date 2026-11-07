"""Run the full pipeline on a local video file (no browser / S3) and write <out>/<name>.analysis.json + .md

usage: python scripts/run_pipeline.py VIDEO [--out examples] [--frames 16] [--context "..."] [--concepts a,b,c]
Needs: META_API_KEY (or ../env.local), AWS credentials for Bedrock, ffmpeg on PATH, Pillow.
"""

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))

if not os.environ.get("META_API_KEY"):
    env_file = os.path.join(HERE, "..", "..", "env.local")
    if os.path.exists(env_file):
        os.environ["META_API_KEY"] = open(env_file).read().strip()

import evidence  # noqa: E402
import llm  # noqa: E402
import media  # noqa: E402
import sam  # noqa: E402
from report import to_markdown  # noqa: E402

COLORS = ["#3b82f6", "#f97316", "#22c55e", "#ec4899", "#a855f7", "#14b8a6", "#eab308", "#ef4444"]  # same as UI


def grab_frames(path, duration, n, workdir):
    """Evenly spaced 768px JPEG frames, as the web UI samples them."""
    frames = []
    for i in range(n):
        t = duration * (i + 0.5) / n
        out = os.path.join(workdir, f"f{i}.jpg")
        subprocess.run([media.ffmpeg_bin(), "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
                        "-vf", "scale=768:-2", "-q:v", "4", out], check=True)
        frames.append({"t": round(t, 3), "data": base64.b64encode(open(out, "rb").read()).decode(), "path": out})
    return frames


def annotate(frames, tracks, fps):
    """Draw SAM boxes + track tags on the frames (the web UI does the same on a canvas)."""
    from PIL import Image, ImageDraw

    out = []
    for fr in frames:
        img = Image.open(fr["path"]).convert("RGB")
        d = ImageDraw.Draw(img)
        W, H = img.size
        f = round(fr["t"] * fps)
        for k, tr in enumerate(tracks):
            near = [r for r in tr["boxes"] if abs(r[0] - f) <= 3]
            if not near:
                continue
            b = min(near, key=lambda r: abs(r[0] - f))
            c = COLORS[k % len(COLORS)]
            x1, y1, x2, y2 = b[1] * W, b[2] * H, b[3] * W, b[4] * H
            d.rectangle([x1, y1, x2, y2], outline=c, width=2)
            tw = d.textlength(tr["track_id"]) + 6
            ty = y1 - 13 if y1 >= 13 else y1
            d.rectangle([x1, ty, x1 + tw, ty + 13], fill=c)
            d.text((x1 + 3, ty + 1), tr["track_id"], fill="white")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        out.append({"t": fr["t"], "data": base64.b64encode(buf.getvalue()).decode()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "examples"))
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--context", default="")
    ap.add_argument("--concepts", default="")
    a = ap.parse_args()

    work = tempfile.mkdtemp()
    t0 = time.time()
    path, info, transcoded = media.normalize(a.video, work)
    print(f"video: {info} transcoded={transcoded}")
    raw = grab_frames(path, info["duration"], a.frames, work)

    if a.concepts:
        concepts = {"scene": "", "concepts": [c.strip() for c in a.concepts.split(",") if c.strip()]}
    else:
        concepts = llm.choose_concepts(raw, a.context)
    print(f"[{time.time() - t0:.0f}s] concepts: {concepts}")

    file_id = sam.upload_video(path)
    try:
        seg = sam.segment_video(file_id, concepts["concepts"])
    finally:
        sam.delete_file(file_id)
    fps = seg["frames"] / info["duration"] if seg["frames"] else 30.0
    print(f"[{time.time() - t0:.0f}s] SAM: {len(seg['tracks'])} tracks, {seg['frames']} frames, errors={seg['errors']}")

    annotated = annotate(raw, seg["tracks"], fps)
    text, facts = evidence.summarize(seg["tracks"], fps, info["duration"], [f["t"] for f in annotated])
    print(text)
    result = llm.analyze(annotated, text, a.context)
    result.update(sam_evidence=text, track_facts=facts, concepts=concepts,
                  tracks=[{k: t[k] for k in ("track_id", "label", "concept")} for t in seg["tracks"]])
    print(f"[{time.time() - t0:.0f}s] analysis done: {result['title']}")

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, os.path.splitext(os.path.basename(a.video))[0])
    with open(stem + ".analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(stem + ".test_case.md", "w") as f:
        f.write(to_markdown(result))
    print(f"wrote {stem}.analysis.json and .test_case.md")


if __name__ == "__main__":
    main()
