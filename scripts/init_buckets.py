"""Cria, de forma idempotente, os buckets da infraestrutura local (S3 compatível).

Roda no serviço `init` do docker-compose, dentro da imagem do projeto (que já traz
boto3 e a pasta `scripts/`), então não depende de nenhuma pasta compartilhada com o
Docker. Variáveis de ambiente:

    S3_ENDPOINT_URL  endpoint do S3 (ex.: http://rustfs:9000)
    BUCKETS          nomes separados por espaço (padrão: dvc-remote mlflow-artifacts)
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
"""

import os
import time
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

DEFAULT_BUCKETS = "dvc-remote mlflow-artifacts"
WAIT_SECONDS = 120
POLL_INTERVAL_SECONDS = 2
MISSING_BUCKET_CODES = frozenset({"404", "NoSuchBucket", "NotFound"})


def wait_until_ready(
    client: Any, timeout: float = WAIT_SECONDS, interval: float = POLL_INTERVAL_SECONDS
) -> None:
    """Espera o S3 responder, tentando de novo até o limite de tempo.

    Raises:
        BotoCoreError | ClientError: se o S3 não responder dentro do limite.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            client.list_buckets()
            return
        except (BotoCoreError, ClientError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(interval)


def ensure_bucket(client: Any, name: str) -> bool:
    """Cria o bucket se ele ainda não existir.

    Returns:
        True se o bucket foi criado, False se já existia.

    Raises:
        ClientError: para qualquer erro diferente de "bucket inexistente".
    """
    try:
        client.head_bucket(Bucket=name)
        return False
    except ClientError as error:
        if error.response["Error"]["Code"] not in MISSING_BUCKET_CODES:
            raise
    client.create_bucket(Bucket=name)
    return True


def main() -> None:
    """Ponto de entrada: espera o S3 e garante todos os buckets."""
    endpoint_url = os.environ["S3_ENDPOINT_URL"]
    names = os.environ.get("BUCKETS", DEFAULT_BUCKETS).split()
    client = boto3.client("s3", endpoint_url=endpoint_url)
    print(f"Aguardando o S3 em {endpoint_url}...", flush=True)
    wait_until_ready(client)
    for name in names:
        created = ensure_bucket(client, name)
        print(f"Bucket {'criado' if created else 'já existe'}: {name}", flush=True)


if __name__ == "__main__":
    main()
