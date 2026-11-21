# Video → Test Case

Record a live video (browser camera) or upload a video file of a test being performed. The app breaks the
video into recognizable action steps and generates a formal test case procedure (preconditions, setup,
numbered steps with expected results, pass/fail criteria, teardown).

**Live:** https://dl5waxovovkgftzfuwl3ppglse0eeupr.lambda-url.us-east-1.on.aws/

## How it works

1. The browser samples 4–20 evenly spaced frames from the video (canvas, 768px JPEG). Recording uses
   `getUserMedia` + `MediaRecorder`, so it works on desktop and phone browsers over HTTPS.
2. Frames + timestamps are POSTed to `/api/analyze` on a single Python Lambda (Function URL).
3. The Lambda sends them to Claude Opus 5.5 on Amazon Bedrock with a `record_analysis` tool schema and
   returns the structured result: equipment, timed steps (with key frame), and the test case.
4. The UI renders steps with thumbnails (click to seek the video) and exports PDF / Markdown / JSON.
5. **Download PDF** builds a printable procedure in the browser (jsPDF + DejaVu fonts, vendored in
   `app/static/vendor`): procedure table with Pass/Fail + Notes columns, criteria, key-frame thumbnails,
   and an execution sign-off block.
6. **Save to GitHub** commits one folder per test case — `README.md`, the PDF, `analysis.json`, and
   `frames/step-N.jpg` — in a single commit via the Git Data API.

```
app/
  handler.py        Lambda entry: serves UI, routes /api/analyze
  analyzer.py       Bedrock prompt + tool schema
  github_save.py    Commit files to GitHub (config from Secrets Manager)
  static/index.html Single-page UI (upload / record / results / export)
scripts/
  deploy.sh         Idempotent deploy: IAM role, Lambda, Function URL
  configure_github.py  Store GitHub repo/token/passcode in Secrets Manager
  local_server.py   Run the handler locally on :8000
examples/           Output for data/IMG_0054.MP4
```

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install boto3
.venv/bin/python scripts/local_server.py     # uses your default AWS profile for Bedrock
# CLI-only: .venv/bin/python app/analyzer.py <dir of frames named <seconds>.jpg>
```

## Deploy

```bash
PYTHON=.venv/bin/python ./scripts/deploy.sh   # needs IAM + Lambda permissions
```

Env overrides: `AWS_REGION` (default `us-east-1`), `MODEL_ID` (default `us.anthropic.claude-opus-5-5`).

## GitHub saving

1. Create a **fine-grained personal access token** (GitHub → Settings → Developer settings → Fine-grained
   tokens): *Repository access* = only the target repo; *Permissions → Contents* = **Read and write**
   (Metadata: read is added automatically). Pick an expiration.
2. Store it (prompts for repo, token, branch, folder, and a save passcode):
   ```bash
   .venv/bin/python scripts/configure_github.py
   ```
   This writes the Secrets Manager secret `video-to-test/github`; the Lambda role can only read that secret.
3. Users enter the save passcode in the app the first time (remembered in their browser).

Rotate the token by rerunning the script. If the branch is protected, point it at another branch.

## Notes / limits

- Analysis is visual only (audio is not transcribed). Use the optional context box to give hints.
- Function URL is public (no auth); anyone with the URL can trigger Bedrock calls.
- Browsers must be able to decode the uploaded codec (HEVC .MOV works in Safari / Chrome on macOS).
