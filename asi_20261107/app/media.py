"""Video storage (S3) and normalization (ffmpeg) so any browser upload/recording is something SAM accepts."""

import os
import re
import shutil
import subprocess
import uuid

import boto3
from botocore.config import Config

BUCKET = os.environ.get("UPLOAD_BUCKET", "")
LOCAL_DIR = os.environ.get("LOCAL_UPLOAD_DIR", "")  # dev mode: scripts/dev_server.py accepts the PUT itself
REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_DIM = 1920  # larger videos are downscaled (huge masks, slow tracking)
SAM_CODECS = {"h264", "hevc", "mpeg4"}

_s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version="s3v4"))


def ffmpeg_bin():
    for cand in (os.environ.get("FFMPEG"), "/opt/bin/ffmpeg", shutil.which("ffmpeg")):
        if cand and os.path.exists(cand):
            return cand
    raise RuntimeError("ffmpeg not found")


def presign_upload(filename, content_type):
    ext = "." + (re.sub(r"[^a-z0-9]", "", os.path.splitext(filename or "")[1].lower())[:5] or "bin")
    key = f"uploads/{uuid.uuid4().hex}{ext}"
    if LOCAL_DIR:
        return {"key": key, "url": f"/local-upload/{key}", "content_type": content_type or "application/octet-stream"}
    if not BUCKET:
        raise RuntimeError("UPLOAD_BUCKET is not configured")
    url = _s3.generate_presigned_url(
        "put_object", Params={"Bucket": BUCKET, "Key": key, "ContentType": content_type or "application/octet-stream"},
        ExpiresIn=900,
    )
    return {"key": key, "url": url, "content_type": content_type or "application/octet-stream"}


def download(key, workdir):
    if not key.startswith("uploads/") or ".." in key:
        raise ValueError("Invalid upload key")
    path = os.path.join(workdir, "input" + os.path.splitext(key)[1])
    if LOCAL_DIR:
        shutil.copy(os.path.join(LOCAL_DIR, key), path)
    else:
        _s3.download_file(BUCKET, key, path)
    return path


def probe(path):
    out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    info = {"container": "", "codec": "", "width": 0, "height": 0, "duration": 0.0, "rotated": False}
    m = re.search(r"Input #0, ([\w,]+)", out)
    info["container"] = m.group(1) if m else ""
    m = re.search(r"Stream #\S+.*?Video: (\w+).*?, (\d{2,5})x(\d{2,5})", out)
    if m:
        info["codec"], info["width"], info["height"] = m.group(1), int(m.group(2)), int(m.group(3))
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    info["rotated"] = bool(re.search(r"rotat\w* of -?(90|180|270)|rotate\s*:\s*(90|180|270)", out))
    return info


def normalize(path, workdir):
    """Return (path, info, transcoded) with an H.264/HEVC MP4/MOV no larger than MAX_DIM, upright."""
    info = probe(path)
    ok = ("mp4" in info["container"] or "mov" in info["container"]) and info["codec"] in SAM_CODECS \
        and max(info["width"], info["height"]) <= MAX_DIM and not info["rotated"]
    if ok:
        return path, info, False
    out = os.path.join(workdir, "normalized.mp4")
    scale = f"scale='trunc(min(1,{MAX_DIM}/max(iw,ih))*iw/2)*2':'trunc(min(1,{MAX_DIM}/max(iw,ih))*ih/2)*2'"
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-vf", scale, "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(f"Could not convert video: {res.stderr[-400:]}")
    return out, probe(out), True
