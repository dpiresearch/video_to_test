"""Turn SAM tracks (normalized boxes per frame) into a compact, time-stamped evidence summary.

The summary gives the LLM grounded facts it cannot reliably get from a handful of still frames:
when each object is visible, when it moves (and which way), when it grows/shrinks (approach/zoom),
and when a hand overlaps an object (likely manipulation).
"""

import math
from statistics import median

ACTOR_WORDS = ("hand", "finger", "arm", "person", "glove", "operator")
GAP_S = 0.4          # visibility gap that splits an interval
MOVE_SPEED = 0.12    # frame-widths / second counted as "moving"
MIN_EVENT_S = 0.25   # shortest motion / contact event reported
CONTACT_RATIO = 0.25  # intersection / smaller box area
MIN_SIZE_AREA = 0.01  # objects smaller than 1% of the frame skip size-change analysis


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap_ratio(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min(_area(a), _area(b)) or 1e-9
    return ix * iy / small


def _where(b):
    cx, cy = _center(b)
    h = "left" if cx < 0.34 else "right" if cx > 0.66 else "center"
    v = "top" if cy < 0.34 else "bottom" if cy > 0.66 else "middle"
    return "center" if (h, v) == ("center", "middle") else f"{v}-{h}"


def _intervals(times, gap=GAP_S):
    out = []
    for t in times:
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return out


def _runs(flags, times, min_len=MIN_EVENT_S):
    """Contiguous True runs in a boolean series -> [(t0, t1, i0, i1)]."""
    runs, start = [], None
    for i, on in enumerate(flags + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if times[i - 1] - times[start] >= min_len:
                runs.append((times[start], times[i - 1], start, i - 1))
            start = None
    return runs


def _direction(dx, dy):
    parts = []
    if abs(dx) > 0.02:
        parts.append("right" if dx > 0 else "left")
    if abs(dy) > 0.02:
        parts.append("down" if dy > 0 else "up")
    return "-".join(parts) or "in place"


def _smooth(values, k=2):
    return [sum(values[max(0, i - k):i + k + 1]) / len(values[max(0, i - k):i + k + 1]) for i in range(len(values))]


def _track_series(track, fps):
    rows = [r for r in track["boxes"] if r[5] == 0 or r[5] >= 0.3]  # drop very low-confidence detections
    return [r[0] / fps for r in rows], [r[1:5] for r in rows]


def _motion_events(times, boxes):
    if len(times) < 3:
        return []
    cx = _smooth([_center(b)[0] for b in boxes])
    cy = _smooth([_center(b)[1] for b in boxes])
    speed = [0.0]
    for i in range(1, len(times)):
        dt = max(times[i] - times[i - 1], 1e-3)
        speed.append(math.hypot(cx[i] - cx[i - 1], cy[i] - cy[i - 1]) / dt)
    events = []
    for t0, t1, i0, i1 in _runs([s > MOVE_SPEED for s in speed], times):
        a, b = max(i0 - 1, 0), i1
        dx, dy = cx[b] - cx[a], cy[b] - cy[a]
        dist = math.hypot(dx, dy)
        if dist < 0.04:
            continue
        events.append(f"{t0:.2f}-{t1:.2f}s moves {_direction(dx, dy)} ~{dist * 100:.0f}% of frame")
    # Size change (approach / recede / zoom)
    area = _smooth([_area(b) for b in boxes], 3)
    if median(area) < MIN_SIZE_AREA:
        return events  # box jitter on tiny objects is mostly noise
    step = max(1, round(len(times) / max(times[-1] - times[0], 1e-3) * 0.6))  # ~0.6 s windows
    i = 0
    while i + step < len(times):
        j = i + step
        ratio = area[j] / max(area[i], 1e-6)
        if ratio > 1.4 or ratio < 0.7:
            events.append(f"{times[i]:.2f}-{times[j]:.2f}s apparent size {'grows' if ratio > 1 else 'shrinks'}"
                          f" x{ratio:.1f} (moved toward/away from camera, or partly hidden/revealed)")
        i = j
    return events


def _global_motion(series):
    """Median centroid shift of all objects between consecutive frames -> camera pan estimate."""
    by_frame = {}
    for times, boxes in series:
        for t, b in zip(times, boxes):
            by_frame.setdefault(round(t, 3), []).append(_center(b))
    if len(series) < 2:
        return []
    keys = sorted(by_frame)
    flags, moves = [], []
    per_object = [dict(zip((round(t, 3) for t in ts), bs)) for ts, bs in series]
    for k0, k1 in zip(keys, keys[1:]):
        d = [(_center(o[k1])[0] - _center(o[k0])[0], _center(o[k1])[1] - _center(o[k0])[1])
             for o in per_object if k0 in o and k1 in o]
        if len(d) < 2:
            flags.append(False), moves.append((0, 0))
            continue
        mx, my = median(x for x, _ in d), median(y for _, y in d)
        agree = sum(1 for x, y in d if x * mx + y * my > 0) / len(d)
        dt = max(k1 - k0, 1e-3)
        flags.append(math.hypot(mx, my) / dt > MOVE_SPEED and agree > 0.7)
        moves.append((mx, my))
    events = []
    for t0, t1, i0, i1 in _runs(flags, keys[1:]):
        dx, dy = sum(m[0] for m in moves[i0:i1 + 1]), sum(m[1] for m in moves[i0:i1 + 1])
        events.append(f"{t0:.2f}-{t1:.2f}s whole scene shifts {_direction(dx, dy)} (camera moves "
                      f"{_direction(-dx, -dy)})")
    return events


def _contacts(actor, target, fps):
    a = {r[0]: r[1:5] for r in actor["boxes"]}
    b = {r[0]: r[1:5] for r in target["boxes"]}
    shared = sorted(set(a) & set(b))
    if not shared:
        return []
    times = [f / fps for f in shared]
    flags = [_overlap_ratio(a[f], b[f]) >= CONTACT_RATIO for f in shared]
    out = []
    for t0, t1, _, _ in _runs(flags, times):
        out.append(f"{t0:.2f}-{t1:.2f}s {actor['track_id']} ({actor['label']}) overlaps {target['track_id']} "
                   f"({target['label']})")
    return out


def summarize(tracks, fps, duration, frame_times=()):
    """Return (summary_text, per_track_facts) for the LLM and the UI."""
    fps = fps or 30.0
    lines = [f"SAM 3.1 tracked {len(tracks)} object instance(s) over {duration:.2f}s (~{fps:.1f} fps). "
             "Coordinates are fractions of the frame (0..1)."]
    facts, series, timeline = [], [], []
    for t in tracks:
        times, boxes = _track_series(t, fps)
        if not times:
            continue
        series.append((times, boxes))
        vis = _intervals(times)
        covered = sum(b - a for a, b in vis) + len(vis) / fps
        med = [median(b[i] for b in boxes) for i in range(4)]
        conf = [r[5] for r in t["boxes"] if r[5]]
        motion = _motion_events(times, boxes)
        fact = {
            "track_id": t["track_id"], "label": t["label"], "concept": t["concept"],
            "visible": [[round(a, 2), round(b, 2)] for a, b in vis],
            "coverage": round(min(1.0, covered / max(duration, 1e-3)), 2),
            "typical_position": _where(med), "typical_size_pct": round(_area(med) * 100, 1),
            "mean_confidence": round(sum(conf) / len(conf), 2) if conf else None,
            "events": motion,
        }
        facts.append(fact)
        lines.append(
            f"- {t['track_id']} \"{t['label']}\": visible "
            + ", ".join(f"{a:.2f}-{b:.2f}s" for a, b in vis)
            + f" ({fact['coverage'] * 100:.0f}% of video); usually {fact['typical_position']}, "
            f"~{fact['typical_size_pct']}% of frame area"
            + (f"; confidence {fact['mean_confidence']}" if conf else "")
        )
        tag = f"{t['track_id']} ({t['label']})"
        for a, b in vis:
            if a > 0.3:
                timeline.append((a, f"{tag} appears"))
            if b < duration - 0.3:
                timeline.append((b, f"{tag} disappears / leaves view"))
        for e in motion:
            timeline.append((float(e.split("-")[0]), f"{tag} {e.split('s ', 1)[1]} ({e.split(' ')[0]})"))

    for e in _global_motion(series):
        timeline.append((float(e.split("-")[0]), f"camera: {e.split('s ', 1)[1]} ({e.split(' ')[0]})"))

    actors = [t for t in tracks if any(w in t["concept"].lower() for w in ACTOR_WORDS)]
    others = [t for t in tracks if t not in actors]
    for a in actors:
        for o in others:
            for c in _contacts(a, o, fps):
                timeline.append((float(c.split("-")[0]), f"contact: {c.split('s ', 1)[1]} ({c.split(' ')[0]})"))

    lines.append("Event timeline (chronological, from every video frame):")
    lines += [f"    {t:6.2f}s  {msg}" for t, msg in sorted(timeline)] or ["    (no appearance, motion or contact events)"]
    if not actors:
        lines.append("No hands/operators were detected by SAM (actions may happen off-screen or be electronic).")
    elif not any("contact:" in m for _, m in timeline):
        lines.append("Hands were tracked but never overlapped another tracked object.")

    if frame_times:
        lines.append("Tracks visible in each key frame (boxes are drawn on the images):")
        prev = set()
        for i, ft in enumerate(frame_times):
            f = round(ft * fps)
            tol = max(1, fps * 0.1)
            seen = [t["track_id"] for t in tracks if any(abs(r[0] - f) <= tol for r in t["boxes"])]
            cur = set(seen)
            delta = []
            if i and cur - prev:
                delta.append("new: " + ", ".join(x for x in seen if x not in prev))
            if i and prev - cur:
                delta.append("gone: " + ", ".join(sorted(prev - cur)))
            lines.append(f"    frame {i} (t={ft:.2f}s): " + (", ".join(seen) or "none")
                         + (f"  [{'; '.join(delta)}]" if delta else ""))
            prev = cur
    return "\n".join(lines), facts
