#!/usr/bin/env bash
# Deploy (or update) the app as a Lambda behind an HTTPS Function URL.
# Uses whatever AWS credentials are active in the environment. Idempotent.
set -euo pipefail

APP=video-to-test
REGION=${AWS_REGION:-us-east-1}
MODEL_ID=${MODEL_ID:-us.anthropic.claude-opus-5-5}
ROLE=${APP}-lambda-role
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export AWS_DEFAULT_REGION=$REGION AWS_PAGER=""

echo "==> Packaging"
BUILD=$(mktemp -d)
(cd "$ROOT/app" && zip -qr "$BUILD/app.zip" handler.py analyzer.py static)

echo "==> IAM role"
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{
    "Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  CREATED_ROLE=1
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name bedrock-invoke --policy-document '{
  "Version":"2012-10-17","Statement":[{"Effect":"Allow",
  "Action":["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream"],
  "Resource":["arn:aws:bedrock:*::foundation-model/anthropic.*","arn:aws:bedrock:*:'"$ACCOUNT"':inference-profile/*"]}]}'
ROLE_ARN=$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)
[[ -n "${CREATED_ROLE:-}" ]] && { echo "    waiting for role propagation"; sleep 12; }

echo "==> Lambda function"
ENV="Variables={MODEL_ID=$MODEL_ID,BEDROCK_REGION=$REGION}"
if aws lambda get-function --function-name "$APP" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$APP" --zip-file "fileb://$BUILD/app.zip" >/dev/null
  aws lambda wait function-updated --function-name "$APP"
  aws lambda update-function-configuration --function-name "$APP" --timeout 300 --memory-size 512 --environment "$ENV" >/dev/null
else
  for i in 1 2 3 4 5; do
    aws lambda create-function --function-name "$APP" --runtime python3.12 --handler handler.lambda_handler \
      --role "$ROLE_ARN" --zip-file "fileb://$BUILD/app.zip" --timeout 300 --memory-size 512 \
      --environment "$ENV" >/dev/null && break
    echo "    role not ready yet, retrying..."; sleep 8
  done
fi
aws lambda wait function-updated --function-name "$APP"

echo "==> Function URL"
if ! aws lambda get-function-url-config --function-name "$APP" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$APP" --auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$APP" --statement-id public-url --action lambda:InvokeFunctionUrl \
    --principal '*' --function-url-auth-type NONE >/dev/null
fi
# Function URLs also require lambda:InvokeFunction scoped to URL invocations (needs a recent boto3/CLI)
if ! aws lambda get-policy --function-name "$APP" --query Policy --output text | grep -q public-invoke; then
  "${PYTHON:-python3}" -c "import boto3; boto3.client('lambda', region_name='$REGION').add_permission(
    FunctionName='$APP', StatementId='public-invoke', Action='lambda:InvokeFunction', Principal='*', InvokedViaFunctionUrl=True)"
fi
URL=$(aws lambda get-function-url-config --function-name "$APP" --query FunctionUrl --output text)
rm -rf "$BUILD"
echo
echo "Deployed: $URL"
