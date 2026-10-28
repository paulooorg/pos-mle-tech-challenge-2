# pos-mle-tech-challenge-2

FIAP Pós MLE — **Tech Challenge Fase 2**: pipeline de engenharia de ML para prever a **propensão de compra** de usuários de e-commerce a partir do comportamento de navegação (dataset *Online Shoppers Purchasing Intention*, UCI).

O foco é a engenharia de ML, não a complexidade do modelo: código limpo, ambiente reproduzível (`uv`), dados e pipeline versionados (DVC), experimentos e Model Registry (MLflow) e execução em containers (Docker).

> **Quer reproduzir tudo do zero?** Rode `./reproduzir.sh` (veja o [guia de reprodução](docs/reproducao.md) para o que cada etapa faz).

## Visão geral

```
data/source/ (dataset completo, no git)
      │  scripts/release_data.py v1|v2   simula a chegada de dado novo
      ▼
data/raw/*.csv ── dvc add ──▶ ponteiro .dvc (git)  +  dado no bucket S3 (RustFS)
      │
      ▼  dvc repro  (dvc.yaml + params.yaml)
[preprocess]  remove duplicatas, split estratificado ──▶ data/processed/{train,test}.csv
      ▼
[train]  validação cruzada, teste, MLflow Tracking, Model Registry e portão de promoção
      ▼
[inference]  API que serve o modelo registrado (champion ou uma versão fixada)  ← extra
```

- **Git** guarda código, `dvc.yaml`, `dvc.lock`, `params.yaml` e os ponteiros `.dvc`.
- **DVC** versiona os dados e reexecuta só o que mudou (dado, código ou parâmetros).
- **MLflow** registra cada run (parâmetros, métricas, figuras, modelo) e as versões do modelo; o alias `champion` aponta para o modelo em uso.
- **Docker Compose** sobe a infraestrutura local (S3 compatível, servidor MLflow) e o `trainer`, um job que baixa o dado, treina e aplica o portão de promoção.

## Resultados

Modelo: `HistGradientBoostingClassifier` (Scikit-Learn), escolhido em `notebooks/02_modeling.ipynb`. Métrica de seleção: **PR-AUC** (`average_precision`), adequada ao desbalanceamento (~15% de compras). Um classificador trivial tem PR-AUC de 0,156.

| Dado | Sessões | PR-AUC (validação cruzada) | PR-AUC (teste) | ROC-AUC (teste) |
|---|---|---|---|---|
| v1 (Feb a Oct) | 7.605 | 0,7917 | 0,7705 | 0,9565 |
| v2 (completo, com Nov e Dec) | 12.330 | 0,7534 | 0,7574 | 0,9378 |

Ao chegar a v2, o modelo treinado com a v1 (champion) é **reavaliado no teste novo** (PR-AUC 0,7414) e o modelo novo (0,7574) o supera e é promovido.

## Início rápido

Pré-requisitos: macOS ou Linux (no Windows, WSL2), [uv](https://docs.astral.sh/uv/), `make`, Docker Compose com 4 CPUs e 4 GiB de RAM.

```bash
git clone <url-do-repositorio>
cd pos-mle-tech-challenge-2
./reproduzir.sh
```

Um único comando reproduz tudo: instala o ambiente, roda os testes, executa o pipeline DVC, sobe a infraestrutura (S3 local e MLflow) e treina no container, incluindo o cenário de dado novo com promoção do modelo, e, como extra, sobe a API de inferência trocando de versão do modelo. Cada etapa é conferida (`OK` ou `FALHOU`) e, no final, a infraestrutura continua de pé para você explorar o MLflow em http://localhost:5001. O script **começa resetando o estado local** e pede confirmação (`--sim` para confirmar sem terminal). Duração: 15 a 25 minutos.

O [guia de reprodução](docs/reproducao.md) explica o que cada etapa faz e traz os comandos para executá-las à mão.

### Alternativa sem o script (passo a passo)

O fluxo do `reproduzir.sh` que passa pelo MLflow do Compose (sem a execução local do pipeline), executado à mão a partir da raiz do repositório. O comentário de cada comando fica na linha acima, para você poder copiar e colar o bloco inteiro no terminal.

```bash
# Instala as dependências nas versões exatas do uv.lock
make install

# Lint (ruff) e testes (pytest)
make check

# Credenciais do S3 local (RustFS) para o remote do DVC; ficam em .dvc/config.local, fora do git
uv run dvc remote modify --local rustfs access_key_id devadmin
uv run dvc remote modify --local rustfs secret_access_key devadmin-secret

# Libera o dado v1 (Feb a Oct) em data/raw/ e registra no DVC: cria o ponteiro .dvc (um hash)
make data-v1

# Sobe o RustFS e o servidor MLflow, cria os buckets e envia o dado v1 ao bucket (dvc push)
make up

# Confere o estado dos serviços: rustfs e mlflow saudáveis; init concluído (Exited 0)
docker compose ps -a

# 1º treino no container: a versão 1 vira champion (primeiro modelo)
make train

# 2º treino, com o mesmo dado: empate, o champion não muda (versão 2 não promovida)
make train

# Dado novo (v2, dataset completo com Nov e Dec): atualiza o ponteiro do DVC
make data-v2

# Envia o dado novo ao bucket
uv run dvc push

# Retreino: a versão 3 supera o champion e é promovida
make train

# Extra: sobe a API de inferência servindo a versão 1 (http://localhost:8000/docs)
make deploy-inference MODEL=1

# Troca a API para o champion (recria o container com outro MODEL_VERSION)
make deploy-inference

# Volta o ponteiro do dado para a v1 (estado commitado)
make data-v1

# Opcional: reseta o estado local (volumes, dados gerados, MLflow local, cache e credenciais do DVC); pede confirmação
make clean
```

O `make train` executa o `docker compose run --build --rm trainer`, e é o `scripts/trainer_entrypoint.sh` (o `CMD` da imagem) que faz, dentro do container, o `dvc pull` do dado de entrada e o `dvc repro --force`. Por isso não há um `dvc repro` solto nos treinos do container. Depois do `make train`, o MLflow em http://localhost:5001 mostra as versões e o alias `champion`.

## Comandos

| Comando | O que faz |
|---|---|
| `make install` | instala as dependências (`uv sync`) |
| `make check` | `ruff check`, `ruff format --check` e `pytest` |
| `make repro` | executa o pipeline DVC no host (respeita o cache; atualiza `dvc.lock` e `metrics.json`) |
| `make data-v1` / `make data-v2` | libera uma versão do dado e registra no DVC (não faz commit nem treina) |
| `make up` | reconstrói a imagem, sobe RustFS e MLflow, cria os buckets (`docker compose up --build --wait`) e faz `dvc push` |
| `make train` | roda o `trainer` no container (reconstrói a imagem; sempre treina) |
| `make deploy-inference MODEL=…` | sobe ou atualiza a API de inferência (porta 8000): `MODEL=champion` (padrão) ou o número de uma versão (`MODEL=1`); precisa de `make up` e `make train` antes |
| `make clean` | reseta o estado local (volumes, dados gerados, MLflow local, cache e credenciais do DVC) para repetir o guia; pede confirmação e não toca em arquivos do git |

Outros comandos úteis, sem atalho:

```bash
uv run dvc dag                  # desenho do pipeline
uv run dvc status               # pipeline e dados em dia?
uv run dvc metrics show         # métricas do último run
uv run dvc metrics diff         # compara com o commit anterior
docker compose ps -a            # estado dos serviços
docker compose logs -f mlflow   # logs do servidor MLflow
docker compose down             # para os serviços (mantém os volumes)
docker compose down -v          # para e apaga os volumes (bucket e banco do MLflow)
uv run jupyter lab --notebook-dir=notebooks
```

## Estrutura do repositório

```
src/pos_mle_tech_challenge_2/
  config.py, params.py          # .env (infra) e params.yaml (experimento)
  data/                         # schema, loader, cleaner, splitter
  features/preprocessor.py      # ColumnTransformer (padronização + one-hot)
  training/                     # model_factory, evaluator, tracker (MLflow), registry,
                                # promotion (portão), reporting (figuras), lineage, trainer
  pipelines/                    # preprocess.py e train.py: estágios do DVC
  serving/                      # API de inferência: schemas, model_loader, predictor, api
scripts/                        # release_data.py, init_buckets.py, trainer_entrypoint.sh
tests/                          # testes automatizados (pytest)
notebooks/                      # EDA e escolha do modelo (fora do pipeline)
data/source/                    # dataset completo (commitado, semente dos scripts)
data/raw/, data/processed/      # versionados pelo DVC (fora do git)
dvc.yaml, dvc.lock, params.yaml # pipeline, seus hashes e os parâmetros
Dockerfile                      # imagem do trainer (também serve o MLflow)
Dockerfile.inference            # imagem enxuta da inferência (só o grupo `serving`)
docker-compose.yml              # infraestrutura local: RustFS, MLflow, trainer, inference
docs/                           # enunciado do desafio e guia de reprodução
reproduzir.sh                   # reproduz tudo com um comando
```

## Decisões técnicas

- **Modelo e métrica:** `HistGradientBoostingClassifier` com `class_weight="balanced"`. Em validação cruzada, após ajuste, ele empata com XGBoost e Random Forest (PR-AUC ≈ 0,753; desvio entre dobras ≈ 0,02); o desempate foi de engenharia: é nativo do Scikit-Learn, sem dependência extra. Threshold de decisão 0,73 (maximiza o F1 nas predições fora da amostra do treino).
- **Protocolo:** remove duplicatas antes do split, split 80/20 estratificado, validação cruzada estratificada de 5 dobras no treino, seed 42 em todo o pipeline. O pré-processamento é treinado dentro do `Pipeline` (sem vazamento do teste).
- **Portão de promoção:** cada run registra uma versão no Model Registry; o alias `champion` só se move se o modelo novo superar o champion atual **reavaliado no mesmo conjunto de teste**. Um empate mantém o champion.
- **Linhagem:** cada run e cada versão do modelo levam o hash do dado (`data_md5`, do ponteiro DVC) e o commit do código (`git_commit`).
- **Serialização do modelo:** `skops` com o tipo interno do HistGradientBoosting liberado de forma explícita (mais seguro que `cloudpickle`, que executa código ao carregar).
- **Trainer como job:** o `make train` sempre treina e deixa o portão decidir; o `make repro` no host respeita o cache do DVC.
- **Dado novo simulado:** o dataset é cortado por mês (v1 sem Nov e Dec; v2 completo). Os meses novos concentram mais volume e mudam o comportamento (Nov converte 25%).

## Inferência (extra)

Uma API (FastAPI) serve o modelo registrado no MLflow. **Qual modelo** é uma variável de ambiente, `MODEL_VERSION`: `champion` (padrão, segue o alias do Registry) ou o número de uma versão (por exemplo `1`). Trocar de modelo é um **deploy**: recriar o container com outro valor.

```bash
make deploy-inference                # serve o champion
make deploy-inference MODEL=1        # fixa a versão 1
```

| Endpoint | Função |
|---|---|
| `GET /health` | pronto? Informa a versão carregada (`503` se o modelo não carregou) |
| `GET /model` | nome, versão, `run_id`, threshold, hash do dado, commit e PR-AUC do modelo |
| `POST /predict` | lote de sessões: probabilidade de compra e classe de cada uma |
| `GET /docs` | Swagger, com 4 exemplos prontos |

A classe vem do **threshold da versão** (0,73), gravado como tag no Registry pelo trainer. Se `MODEL_VERSION` pedir uma versão inexistente, o serviço sobe, mas responde `503` com o motivo, e o deploy acusa `unhealthy`. Os exemplos do Swagger incluem um caso de fronteira em que versões diferentes do modelo discordam (o resultado muda ao trocar de versão).

## Do projeto à produção

O que fizemos mapeia, em escala menor, para o que se vê no mercado:

| Aqui | Em produção (por exemplo, Kubernetes) |
|---|---|
| `make train`: container que treina e termina | `Job` ou `CronJob`, disparado por Argo, Airflow ou Kubeflow |
| `make deploy-inference`: container sempre ligado | `Deployment` com réplicas atrás de um `Service`, com autoscaling |
| Dado novo: `make data-v2` + commit do ponteiro | ingestão e validação, que registram a versão do dado (ponteiro do DVC ou partição/snapshot do data lake) e disparam o treino |
| Gatilho do treino: um comando seu | agenda, evento de dado novo, alerta de drift ou chamada de API |
| Portão: nova versão só vira champion se superar a atual no mesmo teste | o mesmo, mais validações online (canary, shadow) antes de assumir todo o tráfego |
| `MODEL_VERSION` define o modelo no deploy | versão fixada na configuração do deploy (GitOps), com rollback por reversão de commit |

Fica de fora, por escopo: gatilho automático de treino, validação de dados na entrada, monitoramento de drift e de métricas online, CI/CD, canary, gerenciamento de segredos (as credenciais aqui são só de desenvolvimento) e MLflow com banco de produção (aqui é SQLite).

## Configuração

- **`.env`** (modelo em `.env.example`): infraestrutura (URI do MLflow, nomes, caminhos, credenciais de desenvolvimento do S3 local). O `.env` não vai para o git.
- **`params.yaml`**: seed, split, validação cruzada, métrica principal, threshold e hiperparâmetros do modelo.

## Limitações

- **`PageValues` domina o modelo** (importância por permutação ≈ 0,55; a segunda feature, `Month`, ≈ 0,05). A variável é calculada a partir de páginas visitadas *antes de uma transação*, então pode embutir parte da própria compra (possível vazamento parcial).
- O conjunto de teste é pequeno (~380 compras na v2): diferenças de ~0,01 em PR-AUC entre modelos não são significativas.
- Carregar artefatos do MLflow do compose **pela máquina do desenvolvedor** não funciona de forma prática (o servidor entrega URLs internas do S3). Treino, portão e consulta de métricas e versões não são afetados.
- A imagem do trainer tem ~2,5 GB (MLflow completo, DVC com S3 e Scikit-Learn); a da inferência, ~1,3 GB (só o necessário para carregar e servir o modelo).

## Notebooks

- [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb): qualidade dos dados, target, distribuições, categóricas, correlações.
- [`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb): baseline, candidatos, validação cruzada, ajuste, threshold e importância das features.

## Vídeo (método STAR)

- Slides usados no vídeo: [`docs/apresentacao.pdf`](docs/apresentacao.pdf)
- Vídeo (método STAR): [`docs/video-star.mp4`](docs/video-star.mp4)
