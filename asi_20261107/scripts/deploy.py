"""Idempotent AWS deploy: S3 upload bucket, ffmpeg layer, IAM role, Lambda + public Function URL.

usage: python scripts/deploy.py [--env-file ../env.local] [--region us-east-1]
Uses whatever AWS credentials are active (env vars / profile). The Meta API key is read from --env-file
(or META_API_KEY) and stored as a Lambda environment variable; it is never written into the repo.
"""

import argparse
import hashlib
import io
import json
import os
import time
import zipfile

import boto3
from botocore.exceptions import ClientError

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP = "asi-video-to-test"
LAYER = f"{APP}-ffmpeg"
ROLE = f"{APP}-role"
RUNTIME = "python3.12"
APP_FILES = ["handler.py", "sam.py", "llm.py", "evidence.py", "media.py", "report.py"]


def log(msg):
    print(f"==> {msg}", flush=True)


def build_app_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in APP_FILES:
            z.write(os.path.join(ROOT, "app", f), f)
        for f in sorted(os.listdir(os.path.join(ROOT, "app", "static"))):
            z.write(os.path.join(ROOT, "app", "static", f), f"static/{f}")
    return buf.getvalue()


def ensure_bucket(s3, bucket, region):
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        log(f"creating bucket {bucket}")
        kw = {} if region == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": region}}
        s3.create_bucket(Bucket=bucket, **kw)
    s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_cors(Bucket=bucket, CORSConfiguration={"CORSRules": [{
        "AllowedMethods": ["PUT"], "AllowedOrigins": ["*"], "AllowedHeaders": ["*"], "MaxAgeSeconds": 3600}]})
    s3.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration={"Rules": [{
        "ID": "expire-uploads", "Status": "Enabled", "Filter": {"Prefix": "uploads/"},
        "Expiration": {"Days": 1}, "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}}]})


def ensure_ffmpeg_layer(lam, s3, bucket, wheel):
    """Layer with /opt/bin/ffmpeg taken from the imageio-ffmpeg manylinux wheel (static build incl. libx264)."""
    with zipfile.ZipFile(wheel) as wz:
        name = next(n for n in wz.namelist() if "/binaries/ffmpeg-linux" in n)
        binary = wz.read(name)
    digest = hashlib.sha256(binary).hexdigest()[:16]
    for v in lam.list_layer_versions(LayerName=LAYER).get("LayerVersions", []):
        if digest in (v.get("Description") or ""):
            return v["LayerVersionArn"]
    log(f"publishing ffmpeg layer ({len(binary) / 1e6:.0f} MB binary)")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("bin/ffmpeg")
        info.external_attr = 0o755 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, binary)
    key = f"layers/{LAYER}-{digest}.zip"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    v = lam.publish_layer_version(LayerName=LAYER, Description=f"ffmpeg static {digest}",
                                  Content={"S3Bucket": bucket, "S3Key": key}, CompatibleRuntimes=[RUNTIME],
                                  CompatibleArchitectures=["x86_64"])
    return v["LayerVersionArn"]


def ensure_role(iam, account, bucket):
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                                                       "Action": "sts:AssumeRole"}]}
    created = False
    try:
        arn = iam.get_role(RoleName=ROLE)["Role"]["Arn"]
    except ClientError:
        log(f"creating IAM role {ROLE}")
        arn = iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        iam.attach_role_policy(RoleName=ROLE, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
        created = True
    iam.put_role_policy(RoleName=ROLE, PolicyName="app", PolicyDocument=json.dumps({
        "Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
             "Resource": ["arn:aws:bedrock:*::foundation-model/anthropic.*",
                          f"arn:aws:bedrock:*:{account}:inference-profile/*"]},
            {"Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"],
             "Resource": f"arn:aws:s3:::{bucket}/uploads/*"},
        ]}))
    if created:
        time.sleep(12)  # IAM propagation
    return arn


def ensure_function(lam, role_arn, layer_arn, env, code):
    common = dict(Runtime=RUNTIME, Handler="handler.lambda_handler", Role=role_arn, Timeout=900, MemorySize=3008,
                  EphemeralStorage={"Size": 2048}, Layers=[layer_arn], Environment={"Variables": env})
    try:
        lam.get_function(FunctionName=APP)
        log("updating Lambda code + config")
        lam.update_function_code(FunctionName=APP, ZipFile=code)
        lam.get_waiter("function_updated_v2").wait(FunctionName=APP)
        lam.update_function_configuration(FunctionName=APP, **common)
    except lam.exceptions.ResourceNotFoundException:
        log("creating Lambda function")
        for attempt in range(6):
            try:
                lam.create_function(FunctionName=APP, Code={"ZipFile": code}, Architectures=["x86_64"],
                                    Description="Video -> test procedure (Meta SAM 3.1 + Claude)", **common)
                break
            except ClientError as e:
                if "role" not in str(e).lower() or attempt == 5:
                    raise
                time.sleep(8)  # role not assumable yet
    lam.get_waiter("function_updated_v2").wait(FunctionName=APP)


def ensure_url(lam):
    try:
        url = lam.get_function_url_config(FunctionName=APP)["FunctionUrl"]
    except lam.exceptions.ResourceNotFoundException:
        url = lam.create_function_url_config(FunctionName=APP, AuthType="NONE")["FunctionUrl"]
    try:
        policy = lam.get_policy(FunctionName=APP)["Policy"]
    except lam.exceptions.ResourceNotFoundException:
        policy = ""
    if "public-url" not in policy:
        lam.add_permission(FunctionName=APP, StatementId="public-url", Action="lambda:InvokeFunctionUrl",
                           Principal="*", FunctionUrlAuthType="NONE")
    if "public-invoke" not in policy:
        lam.add_permission(FunctionName=APP, StatementId="public-invoke", Action="lambda:InvokeFunction",
                           Principal="*", InvokedViaFunctionUrl=True)
    return url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--env-file", default=os.path.join(ROOT, "..", "env.local"))
    ap.add_argument("--ffmpeg-wheel", default=os.environ.get("FFMPEG_WHEEL", ""),
                    help="imageio_ffmpeg manylinux x86_64 wheel (pip download imageio-ffmpeg --platform "
                         "manylinux2014_x86_64 --only-binary=:all: --no-deps)")
    ap.add_argument("--model", default=os.environ.get("MODEL_ID", "us.anthropic.claude-opus-5-5"))
    ap.add_argument("--fast-model", default=os.environ.get("FAST_MODEL_ID", "us.anthropic.claude-haiku-5-5"))
    a = ap.parse_args()

    key = os.environ.get("META_API_KEY") or open(a.env_file).read().strip()
    session = boto3.Session(region_name=a.region)
    account = session.client("sts").get_caller_identity()["Account"]
    bucket = f"{APP}-{account}-{a.region}"
    s3, lam, iam = session.client("s3"), session.client("lambda"), session.client("iam")

    log(f"account {account}, region {a.region}")
    ensure_bucket(s3, bucket, a.region)
    layer = ensure_ffmpeg_layer(lam, s3, bucket, a.ffmpeg_wheel)
    role = ensure_role(iam, account, bucket)
    env = {"META_API_KEY": key, "UPLOAD_BUCKET": bucket, "MODEL_ID": a.model, "FAST_MODEL_ID": a.fast_model,
           "BEDROCK_REGION": a.region}
    ensure_function(lam, role, layer, env, build_app_zip())
    url = ensure_url(lam)
    print(f"\nDeployed: {url}")


if __name__ == "__main__":
    main()
