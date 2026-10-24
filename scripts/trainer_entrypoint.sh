#!/bin/sh
# Entrypoint do trainer: ajusta o remote do DVC para o ambiente do container,
# baixa os dados da versão do commit (dvc pull) e executa o pipeline (dvc repro).
set -eu

# No .dvc/config o endpoint aponta para localhost (uso na máquina do desenvolvedor);
# dentro do compose o S3 é o serviço "rustfs". O override fica em .dvc/config.local.
dvc remote modify --local rustfs endpointurl "${DVC_S3_ENDPOINT_URL:-http://rustfs:9000}"

dvc pull
exec dvc repro "$@"
