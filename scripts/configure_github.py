"""Store the GitHub settings used by "Save to GitHub" in AWS Secrets Manager.

Usage: python scripts/configure_github.py   (prompts; the token is read without echo)
Needs AWS credentials allowed to create/update the secret (secretsmanager:CreateSecret/PutSecretValue).
"""

import getpass
import json
import os
import urllib.error
import urllib.request

import boto3

SECRET_ID = os.environ.get("GITHUB_SECRET_ID", "video-to-test/github")
REGION = os.environ.get("AWS_REGION", "us-east-1")


def gh(token, path):
    req = urllib.request.Request(f"https://api.github.com{path}", headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": "video-to-test"})
    with urllib.request.urlopen(req, timeout=20) as res:
        return json.loads(res.read())


def main():
    repo = input("GitHub repo (owner/name): ").strip()
    token = getpass.getpass("Fine-grained token (Contents: read & write on that repo): ").strip()
    try:
        info = gh(token, f"/repos/{repo}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Could not access {repo} with that token: HTTP {e.code}")
    if not info.get("permissions", {}).get("push"):
        print("WARNING: token can read the repo but GitHub reports no push permission.")
    branch = input(f"Branch to commit to [{info['default_branch']}]: ").strip() or info["default_branch"]
    prefix = input("Folder in repo [test-cases]: ").strip() or "test-cases"
    passcode = getpass.getpass("Save passcode users must enter in the app (blank = none, NOT recommended): ").strip()

    value = json.dumps({"token": token, "repo": repo, "branch": branch, "path_prefix": prefix, "save_passcode": passcode})
    sm = boto3.client("secretsmanager", region_name=REGION)
    try:
        sm.put_secret_value(SecretId=SECRET_ID, SecretString=value)
        print(f"Updated secret {SECRET_ID}")
    except sm.exceptions.ResourceNotFoundException:
        sm.create_secret(Name=SECRET_ID, SecretString=value, Description="GitHub settings for video-to-test")
        print(f"Created secret {SECRET_ID}")
    print("Done. The app picks up changes within 5 minutes (or on next cold start).")


if __name__ == "__main__":
    main()
