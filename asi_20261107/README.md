# Video → Test Procedure (Meta SAM 3.1 + Claude)

Record a live video in the browser or upload a video file of someone performing a test. The app segments and
tracks the important objects through **every frame** with Meta's **Segment Anything Model 3.1**, breaks the video
into timed, recognizable action steps grounded in those tracks, and writes a formal, repeatable **test case
procedure** (preconditions, setup, numbered steps with expected results, pass/fail criteria, teardown).

**Live:** https://ewl6j4w2czgdo52hqp7xkaeibu0ujqsa.lambda-url.us-east-1.on.aws/

Example output for `../data/IMG_0054.MP4`: [`examples/`](examples/) (JSON, Markdown test case, UI screenshot).

## Pipeline

```
browser ── upload / MediaRecorder ──► S3 (presigned PUT)
   │  samples N key frames (canvas)
   ├─► POST /api/concepts   Claude Haiku 5.5 picks SAM noun-phrase prompts from 8 frames (+ "hand", always)
   ├─► POST /api/segment    Lambda: S3 → ffmpeg normalize (only if needed) → Meta Files API
   │                        → SAM 3.1 per concept, in parallel → normalized tracks: box per frame + masks
   │  draws SAM boxes + "T#" tags onto the key frames
   └─► POST /api/analyze    Lambda: tracks → evidence timeline (visibility, motion, camera pan, hand↔object
                            overlap) → Claude Opus 5.5 with annotated frames + evidence → steps + test case
browser renders: video with live SAM mask/box overlay, per-track timeline lanes, steps, test case, MD/JSON export
```

Why SAM matters here: a handful of still frames can't tell you *when* something happened. SAM's frame-by-frame
tracks give the model exact appearance/disappearance times (e.g. an indicator LED turning on at 3.85 s), motion,
and hand contact. Each step cites the track IDs and SAM evidence it relies on.

## Layout

```
app/
  handler.py     Lambda (Function URL) router: UI + JSON API
  sam.py         Meta SAM 3.1 client: Files upload, streaming /v1/responses, fast output tokenizer, payload budgets
  evidence.py    Tracks → chronological evidence timeline for the LLM
  llm.py         Bedrock Converse: concept selection + analysis tool schema
  media.py       S3 presign/download, ffmpeg probe + conditional transcode (WebM/VP9, >1920px, rotated → H.264)
  report.py      Markdown rendering (CLI)
  static/        index.html, app.js (recording, sampling, overlay via @meta-sam/parser), styles.css
scripts/
  deploy.py      Idempotent AWS deploy (bucket, ffmpeg layer, IAM role, Lambda, public Function URL)
  dev_server.py  Run the handler locally on :8000 (uploads stored on disk, no S3 needed)
  run_pipeline.py  Whole pipeline from the CLI for a local file → examples/
  smoke_test.py  API end-to-end test against any base URL
  ui_test.py     Playwright + Chrome test of upload and live-record (fake camera) flows
```

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # ffmpeg must be on PATH
.venv/bin/python scripts/dev_server.py                                # http://localhost:8000
.venv/bin/python scripts/run_pipeline.py ../data/IMG_0054.MP4         # CLI → examples/
```

The Meta key comes from `META_API_KEY` or `../env.local`; Bedrock uses your default AWS credentials.

## Deploy

```bash
.venv/bin/pip download imageio-ffmpeg --platform manylinux2014_x86_64 --only-binary=:all: --no-deps -d /tmp/ffw
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
  .venv/bin/python scripts/deploy.py --ffmpeg-wheel /tmp/ffw/imageio_ffmpeg-*.whl
```

Creates/updates (all prefixed `asi-video-to-test`): S3 bucket (private, PUT-only CORS, uploads expire after
1 day), an ffmpeg Lambda layer, an IAM role (Bedrock invoke + S3 on `uploads/*`), a Python 3.12 Lambda
(3 GB, 15 min) and a public Function URL. The Meta key is stored as a Lambda environment variable.
`GET /api/health?deep=1` checks the Meta API from inside Lambda.

## Notes and limits

- **SAM latency varies** (typically 15–40 s for a short clip, sometimes minutes). The segmentation stage has a
  420 s deadline (`SAM_DEADLINE_S`); concepts that don't finish are skipped and reported, the rest are kept.
- SAM matches noun phrases literally. If a multi-word concept finds nothing it is retried with its head noun
  ("telescopic antenna" → "antenna"). You can also type your own concepts under *Options*.
- Masks are decoded in the browser with Meta's `@meta-sam/parser` (loaded from jsDelivr). The server tokenizes
  SAM output itself, because the official Python parser re-validates every mask in pure Python (~0.6 s/mask).
  The tokenizer is checked against `meta-sam-parser`: identical boxes, bounds and payloads on the sample video.
- Responses stay under Lambda's 6 MB limit by subsampling masks (and boxes, on long videos) evenly.
- Audio is ignored. The Function URL is public with no auth, so anyone with the link can spend Meta/Bedrock quota.
- Live recording prefers MP4/H.264 (Chrome, Safari). Browsers that only record WebM are transcoded server-side.
