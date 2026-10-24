#!/bin/sh
# Entrypoint do trainer: ajusta o remote do DVC para o ambiente do container,
# baixa os dados da versão do commit (dvc pull) e treina (dvc repro --force).
#
# O trainer é um job de treino: sempre executa o pipeline (--force), registra uma
# nova versão no MLflow e deixa o portão de promoção decidir se ela vira champion.
# Já `make repro`, no host, respeita o cache do DVC (e atualiza dvc.lock/metrics.json).
set -eu

# No .dvc/config o endpoint aponta para localhost (uso na máquina do desenvolvedor);
# dentro do compose o S3 é o serviço "rustfs". O override fica em .dvc/config.local.
dvc remote modify --local rustfs endpointurl "${DVC_S3_ENDPOINT_URL:-http://rustfs:9000}"

dvc pull
exec dvc repro --force "$@"
