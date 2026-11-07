"""Markdown rendering of an analysis result (used by the CLI; the web UI has an equivalent in JS)."""


def _bullets(items):
    return "\n".join(f"- {i}" for i in items) if items else "- (none)"


def to_markdown(r):
    tc = r.get("test_case", {})
    out = [f"# {tc.get('id', 'TC')}: {tc.get('title') or r.get('title', '')}", "",
           f"**Objective:** {tc.get('objective') or r.get('objective', '')}", "",
           "## Summary of the recorded run", "", r.get("summary", ""), "",
           "## Equipment", "", _bullets(f"**{e['name']}**: {e['role']}" for e in r.get("equipment", [])), "",
           "## Observed action steps (from video)", "",
           "| # | Time | Action | Observation | SAM tracks |", "|---|---|---|---|---|"]
    for i, s in enumerate(r.get("steps", []), 1):
        out.append(f"| {i} | {s['start']:.1f}-{s['end']:.1f}s | {s['action']} | {s['observation']} | "
                   f"{', '.join(s.get('track_ids', []))} |")
    out += ["", "## Test procedure", "", "### Preconditions", "", _bullets(tc.get("preconditions", [])), "",
            "### Setup", "", _bullets(tc.get("setup", [])), "", "### Steps", "",
            "| Step | Action | Expected result | Video ref |", "|---|---|---|---|"]
    for p in tc.get("procedure", []):
        out.append(f"| {p['step']} | {p['action']} | {p['expected_result']} | {p.get('video_ref', '')} |")
    out += ["", "### Pass criteria", "", _bullets(tc.get("pass_criteria", [])), "",
            "### Fail criteria", "", _bullets(tc.get("fail_criteria", [])), "",
            "### Teardown", "", _bullets(tc.get("teardown", [])), "",
            "## Assumptions", "", _bullets(r.get("assumptions", [])), ""]
    if r.get("sam_evidence"):
        out += ["## SAM 3.1 evidence", "", "```", r["sam_evidence"], "```", ""]
    return "\n".join(out)
