"""Testes do script que cria os buckets da infraestrutura local."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "init_buckets.py"


def _client_error(code: str, operation: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


class FakeS3:
    """Cliente S3 falso: guarda os buckets existentes e registra as criações."""

    def __init__(self, existing: set[str] | None = None) -> None:
        self.buckets = set(existing or ())
        self.created: list[str] = []

    def head_bucket(self, Bucket: str) -> None:
        if Bucket not in self.buckets:
            raise _client_error("404", "HeadBucket")

    def create_bucket(self, Bucket: str) -> None:
        self.buckets.add(Bucket)
        self.created.append(Bucket)


@pytest.fixture(scope="module")
def init_buckets() -> ModuleType:
    """Carrega o script (fora do pacote) como módulo."""
    spec = importlib.util.spec_from_file_location("init_buckets", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_bucket_is_created(init_buckets: ModuleType) -> None:
    client = FakeS3()

    assert init_buckets.ensure_bucket(client, "dvc-remote") is True
    assert client.created == ["dvc-remote"]


def test_existing_bucket_is_left_alone(init_buckets: ModuleType) -> None:
    client = FakeS3({"dvc-remote"})

    assert init_buckets.ensure_bucket(client, "dvc-remote") is False
    assert client.created == []


def test_running_twice_creates_each_bucket_once(init_buckets: ModuleType) -> None:
    client = FakeS3()

    init_buckets.ensure_bucket(client, "mlflow-artifacts")
    init_buckets.ensure_bucket(client, "mlflow-artifacts")

    assert client.created == ["mlflow-artifacts"]


def test_unexpected_errors_are_not_swallowed(init_buckets: ModuleType) -> None:
    class Denied(FakeS3):
        def head_bucket(self, Bucket: str) -> None:
            raise _client_error("403", "HeadBucket")

    with pytest.raises(ClientError):
        init_buckets.ensure_bucket(Denied(), "dvc-remote")


def test_wait_retries_until_the_s3_answers(init_buckets: ModuleType) -> None:
    class SlowStart:
        calls = 0

        def list_buckets(self) -> dict:
            self.calls += 1
            if self.calls < 3:
                raise EndpointConnectionError(endpoint_url="http://rustfs:9000")
            return {}

    client = SlowStart()

    init_buckets.wait_until_ready(client, timeout=5, interval=0)

    assert client.calls == 3


def test_wait_gives_up_after_the_timeout(init_buckets: ModuleType) -> None:
    class NeverReady:
        def list_buckets(self) -> dict:
            raise EndpointConnectionError(endpoint_url="http://rustfs:9000")

    with pytest.raises(EndpointConnectionError):
        init_buckets.wait_until_ready(NeverReady(), timeout=0, interval=0)
