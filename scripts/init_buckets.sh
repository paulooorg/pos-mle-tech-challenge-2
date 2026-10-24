#!/bin/sh
# Cria, de forma idempotente, os buckets usados pela infra local (remote do DVC e
# artefatos do MLflow). Roda no serviço "init" do docker-compose (imagem amazon/aws-cli).
set -eu

: "${S3_ENDPOINT_URL:?defina S3_ENDPOINT_URL}"
BUCKETS="${BUCKETS:-dvc-remote mlflow-artifacts}"

echo "Aguardando o S3 em ${S3_ENDPOINT_URL}..."
until aws --endpoint-url "$S3_ENDPOINT_URL" s3api list-buckets >/dev/null 2>&1; do
  sleep 2
done

for bucket in $BUCKETS; do
  if aws --endpoint-url "$S3_ENDPOINT_URL" s3api head-bucket --bucket "$bucket" >/dev/null 2>&1; then
    echo "Bucket já existe: $bucket"
  else
    aws --endpoint-url "$S3_ENDPOINT_URL" s3 mb "s3://$bucket"
  fi
done
