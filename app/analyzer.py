"""Turn a sequence of video frames into action steps and a test case procedure using Claude on Bedrock."""

import base64
import json
import os

import boto3
from botocore.config import Config

MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-opus-5-5")
MAX_FRAMES = 20  # Bedrock Converse image limit per request

_bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.environ.get("BEDROCK_REGION", os.environ.get("AWS_REGION", "us-east-1")),
    config=Config(read_timeout=600, retries={"max_attempts": 2}),
)

SYSTEM_PROMPT = """You are a senior test engineer. You are shown a chronological sequence of frames sampled \
from a video of someone performing a test (hardware, software, lab, or manual QA). Each frame is labelled with \
its timestamp in seconds.

Your job:
1. Identify the equipment, devices, software, and setup visible in the video. Read on-screen text, labels, \
displays, and instrument readings (e.g. frequencies, values, LED states) whenever legible.
2. Break the video down into discrete, recognizable action steps, each with a time range, what the operator \
did, and what was observed as a result (state changes, readings, indicator lights, UI changes).
3. Infer the purpose of the test: what is being verified?
4. Write a formal, repeatable test case procedure that someone else could follow to perform the same test, \
including preconditions, setup, numbered procedure steps with expected results, and pass/fail criteria.

Be concrete and specific to what is actually visible. Where something cannot be determined from the frames, \
say so in assumptions rather than inventing details. Always respond by calling the record_analysis tool."""

ANALYSIS_TOOL = {
    "toolSpec": {
        "name": "record_analysis",
        "description": "Record the step breakdown and test case procedure derived from the video.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title of the test being performed"},
                    "summary": {"type": "string", "description": "2-4 sentence summary of what happens in the video"},
                    "test_objective": {"type": "string", "description": "What the test is trying to verify"},
                    "equipment": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "role": {"type": "string", "description": "Role in the test (DUT, instrument, stimulus...)"},
                            },
                            "required": ["name", "role"],
                        },
                    },
                    "steps": {
                        "type": "array",
                        "description": "Chronological action steps observed in the video",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_time": {"type": "number", "description": "Seconds"},
                                "end_time": {"type": "number", "description": "Seconds"},
                                "action": {"type": "string", "description": "What the operator does"},
                                "observation": {"type": "string", "description": "Visible result / system response"},
                                "key_frame_index": {"type": "integer", "description": "0-based index of the frame that best shows this step"},
                            },
                            "required": ["start_time", "end_time", "action", "observation", "key_frame_index"],
                        },
                    },
                    "test_case": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "e.g. TC-RF-001"},
                            "title": {"type": "string"},
                            "objective": {"type": "string"},
                            "preconditions": {"type": "array", "items": {"type": "string"}},
                            "setup": {"type": "array", "items": {"type": "string"}},
                            "procedure": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step": {"type": "integer"},
                                        "action": {"type": "string"},
                                        "expected_result": {"type": "string"},
                                    },
                                    "required": ["step", "action", "expected_result"],
                                },
                            },
                            "pass_criteria": {"type": "array", "items": {"type": "string"}},
                            "fail_criteria": {"type": "array", "items": {"type": "string"}},
                            "teardown": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "title", "objective", "preconditions", "setup", "procedure", "pass_criteria", "fail_criteria", "teardown"],
                    },
                    "assumptions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Things inferred or not determinable from the video",
                    },
                },
                "required": ["title", "summary", "test_objective", "equipment", "steps", "test_case", "assumptions"],
            }
        },
    }
}


def analyze_frames(frames, context=""):
    """frames: list of {"t": seconds, "data": base64 JPEG (optionally a data: URL)}."""
    if not frames:
        raise ValueError("No frames provided")
    if len(frames) > MAX_FRAMES:
        raise ValueError(f"At most {MAX_FRAMES} frames allowed")

    content = [{"text": f"The video is sampled into {len(frames)} frames, in chronological order."}]
    if context:
        content.append({"text": f"Context provided by the user: {context}"})
    for i, frame in enumerate(frames):
        data = frame["data"]
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        content.append({"text": f"Frame {i} at t={float(frame['t']):.2f}s"})
        content.append({"image": {"format": "jpeg", "source": {"bytes": base64.b64decode(data)}}})
    content.append({"text": "Analyze the video and call record_analysis."})

    response = _bedrock.converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": content}],
        toolConfig={"tools": [ANALYSIS_TOOL]},
        inferenceConfig={"maxTokens": 8000},
    )
    for block in response["output"]["message"]["content"]:
        if "toolUse" in block:
            result = block["toolUse"]["input"]
            result["model"] = MODEL_ID
            result["usage"] = response.get("usage", {})
            return result
    raise RuntimeError("Model did not return a structured analysis")


if __name__ == "__main__":
    # Local test: python analyzer.py frame_dir  (frames named <seconds>.jpg)
    import sys

    frame_dir = sys.argv[1]
    files = sorted(os.listdir(frame_dir), key=lambda f: float(f.rsplit(".", 1)[0]))
    frames = [
        {"t": float(f.rsplit(".", 1)[0]), "data": base64.b64encode(open(os.path.join(frame_dir, f), "rb").read()).decode()}
        for f in files
    ]
    print(json.dumps(analyze_frames(frames), indent=2))
