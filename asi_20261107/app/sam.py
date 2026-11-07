"""Meta SAM 3.1 client: upload a video, segment it with concept prompts, and normalize the tracks.

API reference: https://dev.meta.ai/docs/sam/segmenting  (POST /v1/responses, model "sam-3.1")
Masks are passed through untouched; the browser decodes them with Meta's @meta-sam/parser.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout


API_BASE = os.environ.get("META_API_BASE", "https://api.meta.ai/v1")
SAM_MODEL = os.environ.get("SAM_MODEL", "sam-3.1")
MASK_BUDGET_CHARS = int(os.environ.get("MASK_BUDGET_CHARS", 3_000_000))  # keeps responses under Lambda's 6 MB
SAM_DEADLINE_S = int(os.environ.get("SAM_DEADLINE_S", 420))  # whole segmentation stage; Lambda max is 900
BOX_BUDGET = int(os.environ.get("BOX_BUDGET", 30_000))  # box rows (~45 bytes each)


def _api_key():
    key = os.environ.get("META_API_KEY", "").strip()
    if not key:
        raise RuntimeError("META_API_KEY is not configured")
    return key


def _request(method, path, body=None, headers=None, timeout=120):
    req = urllib.request.Request(f"{API_BASE}{path}", data=body, method=method)
    req.add_header("Authorization", f"Bearer {_api_key()}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise RuntimeError(f"Meta API {method} {path} failed ({e.code}): {detail}") from None


def upload_video(path, filename="video.mp4", content_type="video/mp4"):
    """Upload a video through the Files API and return its file id."""
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    body = b"".join([
        f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        data,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    with _request("POST", "/files", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=300) as r:
        return json.load(r)["id"]


def list_models():
    with _request("GET", "/models", timeout=30) as r:
        return [m["id"] for m in json.load(r).get("data", [])]


def delete_file(file_id):
    try:
        _request("DELETE", f"/files/{file_id}", timeout=30).close()
    except Exception:
        pass  # best effort


def _stream_text(file_id, concept, attempts=3):
    """Run one concept prompt; retry transient API errors (seen: 429/5xx and a spurious 404 model_not_found)."""
    for i in range(attempts):
        try:
            return _stream_once(file_id, concept)
        except RuntimeError as e:
            msg = str(e)
            transient = any(code in msg for code in ("(404)", "(429)", "(500)", "(502)", "(503)", "(504)"))
            if not transient or i == attempts - 1:
                raise
            time.sleep(2 * (i + 1))


def _stream_once(file_id, concept):
    """Run one concept prompt over the video; return (raw output text, frames processed)."""
    payload = {
        "model": SAM_MODEL,
        "stream": True,
        "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": concept},
                {"type": "input_video", "file_id": file_id},
            ],
        }],
    }
    chunks, frames, t0, first = [], None, time.time(), None
    with _request("POST", "/responses", json.dumps(payload).encode(), {"Content-Type": "application/json"}, timeout=180) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: {"):
                continue
            event = json.loads(line[6:])
            kind = event.get("type")
            if kind == "response.output_text.delta":
                first = first or time.time() - t0
                chunks.append(event["delta"])
            elif kind == "response.completed":
                usage = (event.get("response") or {}).get("usage") or {}
                frames = usage.get("video_frames_processed")
            elif kind in ("response.failed", "error"):
                err = (event.get("response") or {}).get("error") or event.get("error") or event
                raise RuntimeError(f"SAM failed for '{concept}': {err}")
    print(f"SAM '{concept}': {time.time() - t0:.1f}s total, first record {first or 0:.1f}s, "
          f"{len(chunks)} frames with output, {frames} frames processed", flush=True)
    return "".join(chunks), frames


_LINE_RE = re.compile(r"<(\d+)f>(.*)")
_RECORD_RE = re.compile(r"([^<|>]+?)<\|box;([^|]*)\|>(?:<\|mask;([^|]*)\|>)?")


def _fields(body):
    out = {}
    for part in body.split(";"):
        k, _, v = part.partition("=")
        out[k.strip()] = v
    return out


def _parse_tracks(concept, text):
    """Parse SAM 3.1 video output lines into normalized tracks.

    Line format (https://dev.meta.ai/docs/sam/reading-segmentation):
      <Nf>id<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H[;c=..]|><|mask;x=0;y=0[;c=..];data=H,W,<enc>payload|> ...
    Box corners are inclusive pixels; the mask raster covers the box. This mirrors meta_sam_parser's record
    semantics, but skips its per-mask decode/re-encode validation (~0.6 s/mask in pure Python). The browser
    decodes and validates the masks with Meta's @meta-sam/parser JS package instead.
    """
    tracks = {}
    size = None
    for line in text.splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        f = int(m.group(1))
        for oid, box_body, mask_body in _RECORD_RE.findall(m.group(2)):
            b = _fields(box_body)
            try:
                x1, y1, x2, y2, w, h = (int(float(b[k])) for k in ("x1", "y1", "x2", "y2", "w", "h"))
            except (KeyError, ValueError):
                continue
            if w <= 0 or h <= 0 or x2 < x1 or y2 < y1:
                continue
            size = size or (w, h)
            conf = float(b.get("c") or 0)
            nb = [round(x1 / w, 4), round(y1 / h, 4), round((x2 + 1) / w, 4), round((y2 + 1) / h, 4)]
            oid = oid.strip()
            t = tracks.setdefault(oid, {"concept": concept, "object_id": oid, "boxes": {}, "masks": {}})
            t["boxes"][f] = [*nb, round(conf, 3)]
            if mask_body:
                data = _fields(mask_body).get("data", "")
                parts = data.split(",", 2)
                if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2]:
                    payload = parts[2]
                    t["masks"][f] = {"bounds": nb, "mask": {
                        "encoding": "lossless" if payload.startswith("~") else "one_bit", "payload": payload,
                        "width": int(parts[1]), "height": int(parts[0])}}
    return list(tracks.values()), (size or (1, 1))


def segment_video(file_id, concepts):
    """Segment every concept in parallel. Returns {"tracks": [...], "frames": n, "size": [w, h], "errors": [...]}."""
    tracks, errors, frames, size = [], [], None, None
    empty = []

    deadline = time.time() + SAM_DEADLINE_S

    def run(batch):
        nonlocal frames, size
        pool = ThreadPoolExecutor(max_workers=max(1, len(batch)))
        try:
            futures = {c: pool.submit(_stream_text, file_id, c) for c in batch}
            for concept, fut in futures.items():
                try:
                    text, n = fut.result(timeout=max(1.0, deadline - time.time()))
                    frames = frames or n
                    found, sz = _parse_tracks(concept, text)
                    if found:
                        size = size or sz
                    else:
                        empty.append(concept)
                    tracks.extend(found)
                except FuturesTimeout:
                    errors.append(f"{concept}: SAM did not finish within {SAM_DEADLINE_S}s, skipped")
                except Exception as e:  # one failed concept should not sink the whole run
                    errors.append(f"{concept}: {e}")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    run(concepts)
    # SAM is literal: "telescopic antenna" may miss where "antenna" hits. Retry empty phrases with the head noun.
    retry = {}
    for c in empty:
        head = c.split()[-1]
        if " " in c and head not in concepts and head not in retry.values():
            retry[c] = head
    if retry and deadline - time.time() > 60:
        empty.clear()
        run(list(retry.values()))
        concepts = concepts + [c for c in retry.values() if c not in empty]

    # Stable labels and ordering: by concept order, then first appearance.
    order = {c: i for i, c in enumerate(concepts)}
    tracks.sort(key=lambda t: (order[t["concept"]], min(t["boxes"]) if t["boxes"] else 0))
    counts = {}
    for i, t in enumerate(tracks):
        counts[t["concept"]] = counts.get(t["concept"], 0) + 1
        t["label"] = f'{t["concept"]} #{counts[t["concept"]]}'
        t["track_id"] = f"T{i + 1}"

    _thin_masks(tracks)
    _thin_boxes(tracks)
    for t in tracks:  # JSON-friendly sorted lists
        t["boxes"] = [[f, *b] for f, b in sorted(t["boxes"].items())]
        t["masks"] = [{"f": f, **m} for f, m in sorted(t["masks"].items())]
    if not frames:
        frames = max((t["boxes"][-1][0] + 1 for t in tracks if t["boxes"]), default=0)
    found = sorted({t["concept"] for t in tracks}, key=concepts.index)
    return {"tracks": tracks, "frames": frames, "size": list(size or (0, 0)), "errors": errors,
            "concepts_found": found, "concepts_tried": concepts}


def _thin_masks(tracks):
    """Subsample mask frames uniformly so the total payload stays inside the response budget."""
    total = sum(len(m["mask"]["payload"]) for t in tracks for m in t["masks"].values())
    if total <= MASK_BUDGET_CHARS:
        return
    stride = -(-total // MASK_BUDGET_CHARS)  # ceil
    for t in tracks:
        keys = sorted(t["masks"])
        keep = set(keys[::stride]) | ({keys[-1]} if keys else set())
        t["masks"] = {k: v for k, v in t["masks"].items() if k in keep}


def _thin_boxes(tracks):
    total = sum(len(t["boxes"]) for t in tracks)
    if total <= BOX_BUDGET:
        return
    stride = -(-total // BOX_BUDGET)
    for t in tracks:  # same global frames for every track so hand/object overlaps still line up
        t["boxes"] = {k: v for k, v in t["boxes"].items() if k % stride == 0}
