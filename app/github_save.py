"""Commit a generated test case (Markdown, PDF, JSON, key frames) to a GitHub repo as a single commit.

Configuration lives in a Secrets Manager secret (name in GITHUB_SECRET_ID) holding JSON:
  {"token": "github_pat_...", "repo": "owner/name", "branch": "main",
   "path_prefix": "test-cases", "save_passcode": "..."}
"""

import base64
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.request

import boto3

GITHUB_API = os.environ.get("GITHUB_API", "https://api.github.com")
SECRET_ID = os.environ.get("GITHUB_SECRET_ID", "video-to-test/github")
_CACHE = {"at": 0, "config": None}


class SaveError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def get_config():
    """Load (and cache for 5 min) the GitHub config. Returns None if not configured."""
    if time.time() - _CACHE["at"] < 300:
        return _CACHE["config"]
    try:
        raw = boto3.client("secretsmanager").get_secret_value(SecretId=SECRET_ID)["SecretString"]
        config = json.loads(raw)
        config = config if config.get("token") and config.get("repo") else None
    except Exception as e:  # missing secret / no permission -> treat as not configured
        print(f"GitHub config unavailable: {type(e).__name__}: {e}")
        config = None
    _CACHE.update(at=time.time(), config=config)
    return config


def status():
    config = get_config()
    if not config:
        return {"configured": False}
    return {
        "configured": True,
        "repo": config["repo"],
        "branch": config.get("branch") or "(default)",
        "path_prefix": config.get("path_prefix", "test-cases"),
        "passcode_required": bool(config.get("save_passcode")),
    }


def _gh(config, method, path, body=None):
    req = urllib.request.Request(
        f"{GITHUB_API}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {config['token']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "video-to-test",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            return json.loads(res.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        hint = {401: "token is invalid or expired", 403: "token lacks Contents: write on this repo",
                404: "repo/branch not found or token has no access", 409: "branch is empty or conflicted",
                422: "GitHub rejected the request"}.get(e.code, "")
        raise SaveError(f"GitHub {method} {path} failed: {e.code} {hint} {detail}".strip(), 502)


def _slug(s, limit=60):
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:limit].strip("-") or "test-case"


def save(payload):
    """payload: {"passcode", "test_id", "title", "files": [{"path", "content", "encoding": "utf-8"|"base64"}],
    "message"?} -> {"commit_url", "folder_url", "files"}"""
    config = get_config()
    if not config:
        raise SaveError("GitHub saving is not configured on the server", 503)
    expected = config.get("save_passcode")
    if expected and not hmac.compare_digest(str(payload.get("passcode", "")), expected):
        raise SaveError("Incorrect save passcode", 403)
    files = payload.get("files") or []
    if not files:
        raise SaveError("No files to save")

    repo = config["repo"]
    prefix = config.get("path_prefix", "test-cases").strip("/")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    folder = "/".join(p for p in [prefix, f"{_slug(payload.get('test_id'), 30)}-{_slug(payload.get('title'), 40)}-{stamp}"] if p)

    branch = config.get("branch") or _gh(config, "GET", f"/repos/{repo}")["default_branch"]
    head = _gh(config, "GET", f"/repos/{repo}/git/ref/heads/{branch}")["object"]["sha"]
    base_tree = _gh(config, "GET", f"/repos/{repo}/git/commits/{head}")["tree"]["sha"]

    tree = []
    for f in files:
        rel = f["path"].lstrip("/")
        if ".." in rel.split("/"):
            raise SaveError(f"Invalid path: {rel}")
        encoding = "base64" if f.get("encoding") == "base64" else "utf-8"
        blob = _gh(config, "POST", f"/repos/{repo}/git/blobs", {"content": f["content"], "encoding": encoding})
        tree.append({"path": f"{folder}/{rel}", "mode": "100644", "type": "blob", "sha": blob["sha"]})

    new_tree = _gh(config, "POST", f"/repos/{repo}/git/trees", {"base_tree": base_tree, "tree": tree})["sha"]
    message = payload.get("message") or f"Add test case {payload.get('test_id') or ''}: {payload.get('title') or ''}".strip()
    commit = _gh(config, "POST", f"/repos/{repo}/git/commits", {"message": message, "tree": new_tree, "parents": [head]})
    _gh(config, "PATCH", f"/repos/{repo}/git/refs/heads/{branch}", {"sha": commit["sha"]})

    return {
        "commit_url": commit.get("html_url") or f"https://github.com/{repo}/commit/{commit['sha']}",
        "folder_url": f"https://github.com/{repo}/tree/{branch}/{folder}",
        "files": [t["path"] for t in tree],
    }
