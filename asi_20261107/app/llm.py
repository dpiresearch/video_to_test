"""Claude on Amazon Bedrock: (1) pick SAM concept prompts, (2) turn frames + SAM evidence into steps and a test case."""

import base64
import json
import os
import re

import boto3
from botocore.config import Config

MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-opus-5-5")
FAST_MODEL_ID = os.environ.get("FAST_MODEL_ID", "us.anthropic.claude-haiku-5-5")
MAX_IMAGES = 20

_client = boto3.client(
    "bedrock-runtime",
    region_name=os.environ.get("BEDROCK_REGION", os.environ.get("AWS_REGION", "us-east-1")),
    config=Config(read_timeout=600, retries={"max_attempts": 2}),
)


def _image_blocks(frames, label="Frame"):
    blocks = []
    for i, fr in enumerate(frames):
        data = fr["data"].split(",", 1)[1] if fr["data"].startswith("data:") else fr["data"]
        blocks.append({"text": f"{label} {i} at t={float(fr['t']):.2f}s"})
        blocks.append({"image": {"format": "jpeg", "source": {"bytes": base64.b64decode(data)}}})
    return blocks


def _call_tool(model, system, content, tool, max_tokens):
    # Opus 5.5 on Bedrock rejects forced toolChoice, so we ask for the tool and fall back to parsing JSON text.
    resp = _client.converse(
        modelId=model,
        system=[{"text": system}],
        messages=[{"role": "user", "content": content}],
        toolConfig={"tools": [tool]},
        inferenceConfig={"maxTokens": max_tokens},
    )
    texts = []
    for block in resp["output"]["message"]["content"]:
        if "toolUse" in block:
            return block["toolUse"]["input"], resp.get("usage", {})
        if "text" in block:
            texts.append(block["text"])
    m = re.search(r"\{.*\}", "\n".join(texts), re.S)
    if m:
        return json.loads(m.group(0)), resp.get("usage", {})
    raise RuntimeError("Model did not return structured output")


# ---------------------------------------------------------------- concepts

CONCEPT_SYSTEM = """You choose segmentation prompts for Meta's Segment Anything Model 3.1 (SAM), which finds and \
tracks every instance of a short noun phrase through a video. You see a few frames from a video of someone \
performing a test.

Pick up to {n} concepts that matter for understanding the test procedure: the device under test, instruments and \
their displays/screens, tools, cables/connectors, controls being operated (buttons, knobs, switches), indicator \
lights, and any person or hands acting on things. Rules for each concept:
- a concrete, common noun phrase of 1-2 words, e.g. "circuit board", "screen", "screwdriver", "cable" \
(SAM matches literally: prefer the everyday name over technical jargon)
- one concept per entry; no lists, verbs, questions, negation or spatial phrases
- generic visual words SAM can match, not brand names or part numbers
- do NOT include "hand" (it is always added)
Order them by importance. Call the choose_concepts tool."""

CONCEPT_TOOL = {"toolSpec": {
    "name": "choose_concepts",
    "description": "Record the SAM concept prompts.",
    "inputSchema": {"json": {
        "type": "object",
        "properties": {
            "scene": {"type": "string", "description": "One sentence: what the video appears to show"},
            "concepts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["scene", "concepts"],
    }},
}}


def choose_concepts(frames, context="", max_concepts=5):
    content = [{"text": f"User context: {context}"}] if context else []
    content += _image_blocks(frames[:8])
    out, _ = _call_tool(FAST_MODEL_ID, CONCEPT_SYSTEM.format(n=max_concepts), content, CONCEPT_TOOL, 800)
    seen, concepts = set(), []
    for c in out.get("concepts", []):
        c = re.sub(r"[^\w\s\-]", "", str(c)).strip().lower()
        if c and c not in seen and c not in ("hand", "hands"):
            seen.add(c)
            concepts.append(" ".join(c.split()[:4]))
    return {"scene": out.get("scene", ""), "concepts": ["hand"] + concepts[:max_concepts]}


# ---------------------------------------------------------------- analysis

ANALYSIS_SYSTEM = """You are a senior test engineer turning a recorded test run into documentation.

You receive:
1. Key frames sampled from the video in time order. Thin coloured boxes with tags like "T3" mark objects that \
Meta's SAM 3.1 segmented and tracked through the whole video.
2. A SAM evidence summary: per-track visibility intervals, motion, size changes, camera motion and hand/object \
overlaps computed from every frame of the video (not just the key frames). Timestamps there are reliable.

Do this:
- Identify the equipment, the device under test and the setup. Read on-screen text, labels, displays and readings \
when legible.
- Split the video into discrete, recognisable action steps. Use the SAM evidence to set accurate start/end times \
(e.g. a hand overlapping a switch, an object appearing, a display changing) and cite the track IDs involved.
- Work out what the test is verifying.
- Write a formal, repeatable test case procedure someone else could execute: preconditions, setup, numbered steps \
with expected results, pass/fail criteria, teardown.

Stay faithful to what is visible. Treat scene-wide motion as camera movement. When something cannot be determined, \
put it under assumptions rather than inventing it. Call the record_test_analysis tool."""

_STR_LIST = {"type": "array", "items": {"type": "string"}}

ANALYSIS_TOOL = {"toolSpec": {
    "name": "record_test_analysis",
    "description": "Record the action-step breakdown and the test case procedure.",
    "inputSchema": {"json": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string", "description": "2-4 sentences on what happens in the video"},
            "objective": {"type": "string", "description": "What the test verifies"},
            "objects": {
                "type": "array",
                "description": "Tracked SAM objects worth naming, with their role",
                "items": {"type": "object", "properties": {
                    "track_id": {"type": "string"},
                    "name": {"type": "string", "description": "Specific name, e.g. 'tinySA spectrum analyzer display'"},
                    "role": {"type": "string", "description": "DUT, instrument, stimulus, tool, operator..."},
                }, "required": ["track_id", "name", "role"]},
            },
            "equipment": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "name": {"type": "string"}, "role": {"type": "string"},
                }, "required": ["name", "role"]},
            },
            "steps": {
                "type": "array",
                "description": "Chronological action steps observed",
                "items": {"type": "object", "properties": {
                    "start": {"type": "number", "description": "seconds"},
                    "end": {"type": "number", "description": "seconds"},
                    "action": {"type": "string", "description": "What the operator / system does"},
                    "observation": {"type": "string", "description": "Visible result or system response"},
                    "track_ids": {"type": "array", "items": {"type": "string"}},
                    "sam_evidence": {"type": "string", "description": "Which SAM evidence supports this step"},
                    "key_frame": {"type": "integer", "description": "Index of the key frame that best shows it"},
                }, "required": ["start", "end", "action", "observation", "track_ids", "key_frame"]},
            },
            "test_case": {"type": "object", "properties": {
                "id": {"type": "string", "description": "e.g. TC-RF-001"},
                "title": {"type": "string"},
                "objective": {"type": "string"},
                "preconditions": _STR_LIST,
                "setup": _STR_LIST,
                "procedure": {"type": "array", "items": {"type": "object", "properties": {
                    "step": {"type": "integer"},
                    "action": {"type": "string"},
                    "expected_result": {"type": "string"},
                    "video_ref": {"type": "string", "description": "Time range in the video, e.g. '2.8-3.8s'"},
                }, "required": ["step", "action", "expected_result"]}},
                "pass_criteria": _STR_LIST,
                "fail_criteria": _STR_LIST,
                "teardown": _STR_LIST,
            }, "required": ["id", "title", "objective", "preconditions", "setup", "procedure",
                            "pass_criteria", "fail_criteria", "teardown"]},
            "assumptions": _STR_LIST,
        },
        "required": ["title", "summary", "objective", "objects", "equipment", "steps", "test_case", "assumptions"],
    }},
}}


def analyze(frames, evidence_text, context=""):
    if not frames:
        raise ValueError("No frames provided")
    if len(frames) > MAX_IMAGES:
        raise ValueError(f"At most {MAX_IMAGES} frames allowed")
    content = [{"text": f"{len(frames)} key frames follow, in chronological order."}]
    if context:
        content.append({"text": f"Context from the user: {context}"})
    content += _image_blocks(frames, "Key frame")
    content.append({"text": "SAM 3.1 evidence summary:\n" + evidence_text})
    content.append({"text": "Analyze the test and call record_test_analysis."})
    result, usage = _call_tool(MODEL_ID, ANALYSIS_SYSTEM, content, ANALYSIS_TOOL, 10000)
    result["model"] = MODEL_ID
    result["usage"] = usage
    return result
