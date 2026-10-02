#!/usr/bin/env bash
# Create the SageMaker execution role (idempotent) and attach its inline policy.
# Run with:  AWS_PROFILE=stanford_gpu bash iam/create_role.sh
set -euo pipefail

ROLE_NAME="${ROLE_NAME:-sagemaker-stanford-exec}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "role $ROLE_NAME already exists"
else
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "file://$HERE/execution-role-trust.json"
  echo "created role $ROLE_NAME"
fi

aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "sagemaker-stanford-exec-policy" \
  --policy-document "file://$HERE/execution-role-policy.json"

echo "attached inline policy"

aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text
