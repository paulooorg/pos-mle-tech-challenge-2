# Guia de reprodução

Reproduz, do zero, tudo o que o Tech Challenge pede: ambiente com `uv`, testes, pipeline DVC, containerização com Docker e experimentos no MLflow (Tracking e Model Registry).

## Um comando

```bash
git clone <url-do-repositorio>
cd pos-mle-tech-challenge-2
./reproduzir.sh
```

O script executa as 9 etapas descritas abaixo, mostra cada comando que roda (`+ comando`), confere os resultados esperados (`OK` ou `FALHOU`) e, no final, **deixa a infraestrutura de pé** para você explorar o MLflow e o RustFS, perguntando se deseja limpar o ambiente. Duração: 15 a 25 minutos (12 minutos num teste com cache de dependências e de imagens; a primeira vez baixa mais coisa). A saída completa fica em `reproducao.log`.

> **O script começa resetando o estado local** e pede confirmação antes. Isso descarta os volumes do Docker deste projeto, os dados gerados, o MLflow local, o cache e as credenciais do DVC, e **mudanças não commitadas** em `dvc.lock`, `metrics.json` e `data/raw/*.dvc`. Sem terminal interativo, use `./reproduzir.sh --sim`.

### Requisitos

| Item | Observação |
|---|---|
| Sistema | macOS ou Linux; no Windows, use WSL2 |
| Git, `make`, [uv](https://docs.astral.sh/uv/) | o uv instala o Python 3.12 e as dependências (testado com uv 0.12.3) |
| Docker com Compose v2 | **4 CPUs e 4 GiB de RAM**: com menos memória o MLflow é morto (código 137) |
| Portas livres | `9000` e `9001` (RustFS, S3 local) e `5001` (MLflow) |
| Rede | para baixar as dependências e as imagens Docker |

No macOS com Colima: `colima start --cpu 4 --memory 4`. No Docker Desktop: Settings > Resources.

## O que cada etapa faz

Os comandos abaixo são os que o script executa; você pode rodá-los à mão.

### 1. Verificar o ambiente
Confere `git`, `make`, `uv` e `docker`, se o daemon do Docker está rodando e se ele tem memória suficiente. Falha cedo, com a correção, em vez de um erro obscuro no meio.

### 2. Resetar o estado local
`make clean` e `git restore dvc.lock metrics.json data/raw/online_shoppers_intention.csv.dvc`. Garante o mesmo ponto de partida de uma máquina nova, e por isso os números esperados são sempre os mesmos. Depois confere se as portas estão livres.

### 3. Instalar (`make install`)
`uv sync` instala exatamente as versões do `uv.lock` (produção e desenvolvimento).

### 4. Lint, formatação e testes (`make check`)
`ruff check`, `ruff format --check` e `pytest`. **Verificação:** Ruff sem erros e todos os testes passando.

### 5. Dado v1 e pipeline DVC no host
```bash
uv run dvc remote modify --local rustfs access_key_id devadmin
uv run dvc remote modify --local rustfs secret_access_key devadmin-secret
make data-v1     # copia a v1 (Feb a Oct, 7.605 sessões) de data/source/ para data/raw/ e roda dvc add
make repro       # dvc repro: preprocess e train, com MLflow local (SQLite)
```
O dataset completo está em `data/source/`; o pipeline lê `data/raw/`, versionado pelo DVC (o dado em si não vai para o git, só o ponteiro `.dvc`). As credenciais ficam em `.dvc/config.local`, fora do git.
**Verificações:** o ponteiro do dado v1 é idêntico ao commitado, a primeira versão é promovida a champion e a PR-AUC no teste é 0,7705 (os mesmos números em qualquer máquina: seed fixa e dado versionado). Em seguida o script restaura `dvc.lock` e `metrics.json`, que o `repro` altera a cada run.

### 6. Infraestrutura (`make up`)
Sobe o RustFS (S3 compatível, usado como remote do DVC e como armazenamento de artefatos do MLflow), cria os buckets (`init`, que roda uma vez e termina), sobe o servidor MLflow e envia o dado ao bucket (`dvc push`).
**Verificação:** `rustfs` e `mlflow` saudáveis e `init` com `Exited (0)`.

### 7. Treino no container, duas vezes (`make train`)
O `trainer` é um job: baixa o dado (`dvc pull`), executa o pipeline (`dvc repro --force`), registra o run e a versão do modelo no MLflow do compose e aplica o **portão de promoção**: o modelo só vira `champion` se superar o champion atual **reavaliado no mesmo conjunto de teste**.

| Execução | Mensagem do trainer | Significado |
|---|---|---|
| 1ª | `Versão 1: True (primeiro modelo)` | vira champion |
| 2ª | `Versão 2: False (não supera o champion)` | mesmo dado e parâmetros: empate, o champion não muda |

### 8. Dado novo (v2): retreino e promoção
```bash
make data-v2        # dataset completo (12.330 sessões, com Nov e Dec) e atualiza o ponteiro do DVC
uv run dvc push     # envia o dado novo ao bucket
make train          # o trainer baixa o dado v2 e retreina
```
**Verificações:** `Versão 3: True (supera o champion)` e o alias `champion` do Registry apontando para a versão 3. O champion (treinado com a v1) é reavaliado no teste novo (PR-AUC 0,7414), o modelo novo faz 0,7574 e é promovido. A tag `data_md5` mostra qual dado gerou cada versão.

> O `make train` reconstrói a imagem a cada execução (`--build`): a imagem congela o commit e os ponteiros do dado do momento do build.

### 9. Voltar ao estado original
`make data-v1`. **Verificação:** `dvc.lock`, `metrics.json` e o ponteiro do dado iguais ao commit (outras alterações suas no repositório não interferem).

## Explorando o resultado

| Serviço | Endereço |
|---|---|
| MLflow (UI) | http://localhost:5001 |
| Console do RustFS | http://localhost:9001 (usuário `devadmin`, senha `devadmin-secret`) |

No MLflow:
- **Experiments > purchase-propensity:** um run por execução, com parâmetros, métricas (`cv_*` e `test_*`), figuras (`plots/precision_recall.png`, `plots/confusion_matrix.png`), o modelo e as tags `data_md5`, `git_commit` e `promotion`.
- **Models > purchase-propensity-classifier:** as versões 1, 2 e 3, com o alias `champion` na versão 3.

A métrica que decide a promoção é `test_average_precision`, comparada com `champion_test_average_precision` (o champion reavaliado no mesmo teste). Os números de runs com dados diferentes **não são comparáveis entre si**, porque cada teste tem sessões diferentes.

Para encerrar: `docker compose down` (mantém os dados dos volumes), `docker compose down -v` ou `make clean` (apaga tudo; depois `git restore dvc.lock metrics.json` se necessário).

## Onde cada critério do desafio aparece

| Critério | Onde ver |
|---|---|
| Clean code e estrutura | `src/pos_mle_tech_challenge_2/` (módulos curtos, type hints, POO) e etapa 4 |
| Reprodutibilidade | `pyproject.toml` + `uv.lock`, `.env.example`, seed e parâmetros em `params.yaml`; este script |
| Docker | `Dockerfile` e `docker-compose.yml` (etapas 6 a 8) |
| DVC + pipeline | `dvc.yaml`, `dvc.lock`, ponteiro `data/raw/*.dvc`, `uv run dvc dag` (etapa 5) |
| Modelagem clássica | `HistGradientBoostingClassifier` (Scikit-Learn); EDA e escolha em `notebooks/` |
| MLflow + Registry | runs, métricas e modelo `champion` na UI (etapas 7 e 8) |

## Problemas comuns

| Sintoma | Causa e solução |
|---|---|
| O script para no preflight dizendo que o Docker não roda | inicie o Docker (Docker Desktop, ou `colima start --cpu 4 --memory 4`) |
| `mlflow` reinicia ou o build é interrompido (código 137) | falta de memória no Docker: use 4 GiB ou mais |
| Porta em uso | libere 9000, 9001 ou 5001. O MLflow usa a **5001** de propósito: a 5000 costuma estar ocupada pelo AirPlay no macOS |
| `dvc push` ou `dvc pull` com erro de credenciais | rode as duas linhas `dvc remote modify --local` da etapa 5 |
| O trainer pede o hash de outro dado no `dvc pull` | a imagem está desatualizada: use `make train` (já reconstrói) |
| Os CSVs de `data/processed/` aparecem no `git status` depois de um `dvc pull` | `uv run dvc repro --force preprocess` recria o `.gitignore` da pasta |
| `load_model` do champion pela sua máquina demora ou não conclui | o MLflow do compose entrega artefatos por URLs internas (`rustfs:9000`); carregue o modelo de dentro do compose. Versões, métricas e a UI funcionam normalmente |
| Windows: `no such file or directory` ao rodar o entrypoint | quebras de linha CRLF nos scripts. O `.gitattributes` do repositório já força LF; reclone o repositório depois de configurá-lo, e rode em WSL2 |
