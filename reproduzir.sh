#!/usr/bin/env bash
# Reproduz, de ponta a ponta, o que o Tech Challenge pede: ambiente com uv, testes,
# pipeline DVC, infraestrutura em Docker (S3 local + MLflow) e treino com Model Registry,
# mais um extra: a API de inferência servindo o modelo registrado.
#
# Uso:   ./reproduzir.sh          pergunta antes de resetar o estado local
#        ./reproduzir.sh --sim    confirma o reset de antemão (para execução sem terminal)
#
# Cada etapa mostra o comando executado; a saída completa vai para reproducao.log.
# Detalhes de cada etapa: docs/reproducao.md.

set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

LOG_FILE="reproducao.log"
LAST_OUT="$(mktemp)"
TOTAL_STEPS=10
MIN_DOCKER_MEMORY_BYTES=3758096384 # 3,5 GiB (uma VM de 4 GiB reporta um pouco menos)
REQUIRED_PORTS="9000 9001 5001 8000"
INFERENCE_URL="http://localhost:8000"
CHAMPION_VERSION=""
RAW_POINTER="data/raw/online_shoppers_intention.csv.dvc"
TRACKED_OUTPUTS="dvc.lock metrics.json ${RAW_POINTER}"
EXPECTED_V1_AP="0.7705"
AP_TOLERANCE="0.005"
RESULTS=("")
ASSUME_YES=0
STEP_START=0

if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; RED=$'\033[31m'
  YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; GREEN=""; RED=""; YELLOW=""; RESET=""
fi

trap 'rm -f "$LAST_OUT"' EXIT

# --- utilidades -------------------------------------------------------------

say() { printf '%s\n' "$*"; }

fail() {
  printf '\n%sERRO:%s %s\n' "$RED" "$RESET" "$1" >&2
  [ -n "${2:-}" ] && printf '%s\n' "$2" >&2
  exit 1
}

step() {
  STEP_START=$(date +%s)
  printf '\n%s[%s/%s] %s%s\n' "$BOLD" "$1" "$TOTAL_STEPS" "$2" "$RESET"
}

end_step() {
  printf '%s    (%ss)%s\n' "$DIM" "$(( $(date +%s) - STEP_START ))" "$RESET"
}

# Executa um comando e devolve o código de saída (usado na etapa extra, que não aborta).
try_run() {
  printf '%s  + %s%s\n' "$DIM" "$*" "$RESET"
  printf '\n$ %s\n' "$*" >> "$LOG_FILE"
  "$@" > "$LAST_OUT" 2>&1
  local status=$?
  cat "$LAST_OUT" >> "$LOG_FILE"
  if [ "$status" -ne 0 ]; then
    printf '%s--- últimas linhas da saída ---%s\n' "$YELLOW" "$RESET"
    tail -n 20 "$LAST_OUT"
  fi
  return "$status"
}

# Executa um comando e interrompe o script se ele falhar.
run() {
  try_run "$@"
  local status=$?
  [ "$status" -eq 0 ] || fail "o comando falhou (código $status): $*" "Log completo em ${LOG_FILE}."
}

# Registra o resultado de uma verificação (OK ou FALHOU), sem interromper a execução.
record() {
  if [ "$1" = "ok" ]; then
    printf '  %sOK%s      %s\n' "$GREEN" "$RESET" "$2"
    RESULTS+=("ok|$2")
  else
    printf '  %sFALHOU%s  %s\n' "$RED" "$RESET" "$2"
    RESULTS+=("fail|$2")
  fi
}

expect_output() {
  if grep -qF -- "$1" "$LAST_OUT"; then record ok "$2"; else record fail "$2"; fi
}

# True se os arquivos informados estão iguais ao commit atual (outras alterações no
# repositório, como mudanças suas ainda não commitadas, não interferem).
files_match_commit() {
  [ -z "$(git status --porcelain --untracked-files=no -- "$@")" ]
}

port_in_use() {
  (exec 3<> "/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

metrics_field() {
  uv run python -c "import json; print(json.load(open('metrics.json'))['$1'])" 2> /dev/null
}

# --- etapas -----------------------------------------------------------------

preflight() {
  step 1 "Verificando o ambiente"
  local missing=""
  for tool in git make uv docker; do
    command -v "$tool" > /dev/null 2>&1 || missing="$missing $tool"
  done
  [ -z "$missing" ] || fail "ferramentas ausentes:$missing" \
    "Instale-as e rode de novo (uv: https://docs.astral.sh/uv/)."
  docker info > /dev/null 2>&1 || fail "o Docker não está rodando." \
    "Inicie o Docker (Docker Desktop, ou 'colima start --cpu 4 --memory 4')."
  docker compose version > /dev/null 2>&1 || fail "o Docker Compose v2 não foi encontrado."
  local memory
  memory=$(docker info --format '{{.MemTotal}}' 2> /dev/null || echo 0)
  [ "$memory" -ge "$MIN_DOCKER_MEMORY_BYTES" ] || fail \
    "o Docker tem pouca memória ($(awk -v b="$memory" 'BEGIN { printf "%.1f", b / 1073741824 }') GiB); são necessários ~4 GiB." \
    "Aumente nas configurações do Docker (Colima: 'colima start --cpu 4 --memory 4')."
  git rev-parse --git-dir > /dev/null 2>&1 || fail "rode dentro do repositório clonado."
  say "  uv: $(uv --version) | $(docker compose version | head -n 1)"
  say "  memória do Docker: $(awk -v b="$memory" 'BEGIN { printf "%.1f", b / 1073741824 }') GiB"
  end_step
}

confirm_reset() {
  step 2 "Resetando o estado local (como numa máquina nova)"
  say "${YELLOW}ATENÇÃO:${RESET} esta etapa descarta, sem backup:"
  say "  - volumes do Docker deste projeto (bucket S3 e banco do MLflow do compose);"
  say "  - dados gerados, MLflow local (mlflow.db, mlruns), cache e credenciais do DVC;"
  say "  - mudanças NÃO commitadas em: ${TRACKED_OUTPUTS}"
  if [ "$ASSUME_YES" -eq 1 ]; then
    say "  (--sim: confirmado)"
  elif [ -t 0 ]; then
    local answer
    read -r -p "Prosseguir? [s/N] " answer
    [ "$answer" = "s" ] || { say "Cancelado. Nada foi alterado."; exit 0; }
  else
    fail "sem terminal interativo para confirmar." "Use ./reproduzir.sh --sim para confirmar o reset."
  fi
}

reset_state() {
  : > "$LOG_FILE"
  run bash -c "printf 's\n' | make clean"
  # shellcheck disable=SC2086
  run git restore $TRACKED_OUTPUTS
  local busy=""
  for port in $REQUIRED_PORTS; do
    ! port_in_use "$port" || busy="$busy $port"
  done
  [ -z "$busy" ] || fail "portas em uso:$busy" \
    "Libere essas portas (RustFS: 9000 e 9001; MLflow: 5001; inferência: 8000) e rode de novo."
  say "  portas livres: ${REQUIRED_PORTS}"
  end_step
}

install_and_test() {
  step 3 "Instalando as dependências (uv sync)"
  run make install
  end_step

  step 4 "Lint, formatação e testes (leva alguns minutos)"
  run make check
  local passed
  passed=$(grep -oE '[0-9]+ passed' "$LAST_OUT" | head -n 1)
  record ok "ruff e pytest passam (${passed:-testes ok})"
  end_step
}

host_pipeline() {
  step 5 "Dado v1 e pipeline DVC no host (MLflow local)"
  run uv run dvc remote modify --local rustfs access_key_id devadmin
  run uv run dvc remote modify --local rustfs secret_access_key devadmin-secret
  run make data-v1
  if files_match_commit "$RAW_POINTER"; then
    record ok "ponteiro do dado v1 idêntico ao commitado"
  else
    record fail "ponteiro do dado v1 idêntico ao commitado"
  fi
  run make repro
  local promoted score
  promoted=$(metrics_field promoted)
  score=$(metrics_field test_average_precision)
  [ "$promoted" = "True" ] && record ok "primeira versão promovida a champion" \
    || record fail "primeira versão promovida a champion"
  if awk -v a="$score" -v e="$EXPECTED_V1_AP" -v t="$AP_TOLERANCE" \
    'BEGIN { d = a - e; if (d < 0) d = -d; exit !(d <= t) }'; then
    record ok "PR-AUC no teste (v1) = ${score} (esperado ${EXPECTED_V1_AP})"
  else
    record fail "PR-AUC no teste (v1) = ${score} (esperado ${EXPECTED_V1_AP})"
  fi
  # shellcheck disable=SC2086
  run git restore dvc.lock metrics.json
  end_step
}

infrastructure() {
  step 6 "Subindo a infraestrutura (RustFS, MLflow) e enviando o dado"
  run make up
  local services
  services=$(docker compose ps -a --format '{{.Service}}: {{.Status}}')
  printf '%s\n' "$services" | sed 's/^/    /'
  if printf '%s\n' "$services" | grep -q 'rustfs: .*(healthy)' \
    && printf '%s\n' "$services" | grep -q 'mlflow: .*(healthy)' \
    && printf '%s\n' "$services" | grep -q 'init: Exited (0)'; then
    record ok "RustFS e MLflow saudáveis; buckets criados"
  else
    record fail "RustFS e MLflow saudáveis; buckets criados"
  fi
  end_step
}

train_v1_twice() {
  step 7 "Treino no container com o dado v1 (duas vezes)"
  run make train
  expect_output "Versão 1: True (primeiro modelo)" "1ª execução: versão 1 promovida (primeiro modelo)"
  run make train
  expect_output "Versão 2: False (não supera o champion)" "2ª execução: empate, o champion não muda"
  end_step
}

train_v2() {
  step 8 "Dado novo (v2): retreino e promoção"
  run make data-v2
  run uv run dvc push
  run make train
  expect_output "Versão 3: True (supera o champion)" "dado v2: versão 3 supera o champion e é promovida"
  local champion
  champion=$(MLFLOW_DISABLE_AGENT_HINT=1 uv run python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5001')
model = mlflow.MlflowClient().get_registered_model('purchase-propensity-classifier')
print(model.aliases.get('champion'))" 2> /dev/null | tail -n 1)
  CHAMPION_VERSION="$champion"
  [ "$champion" = "3" ] && record ok "alias champion do Registry aponta para a versão 3" \
    || record fail "alias champion do Registry aponta para a versão 3 (obtido: ${champion:-nenhum})"
  end_step
}

restore_v1() {
  step 9 "Voltando ao estado original do repositório"
  run make data-v1
  # shellcheck disable=SC2086
  if files_match_commit $TRACKED_OUTPUTS; then
    record ok "dvc.lock, metrics.json e ponteiro do dado iguais ao commit"
  else
    record fail "dvc.lock, metrics.json e ponteiro do dado iguais ao commit"
  fi
  end_step
}

# Chama a API de inferência: imprime o corpo da resposta e devolve 1 se o status não for 2xx.
api() {
  uv run python - "$INFERENCE_URL" "$@" <<'PY'
import sys
import urllib.error
import urllib.request

base, method, path = sys.argv[1:4]
request = urllib.request.Request(base + path, method=method)
try:
    with urllib.request.urlopen(request, timeout=20) as response:
        print(response.read().decode())
except urllib.error.HTTPError as error:
    print(error.read().decode())
    sys.exit(1)
PY
}

# Versão do modelo que a API está servindo (vazio se não houver resposta válida).
served_version() {
  api GET /model 2> /dev/null \
    | uv run python -c "import json, sys; print(json.load(sys.stdin)['version'])" 2> /dev/null
}

# Envia os exemplos do Swagger ao /predict e imprime uma linha por exemplo.
show_predictions() {
  uv run python - "$INFERENCE_URL" <<'PY'
import json
import sys
import urllib.request

from pos_mle_tech_challenge_2.serving.examples import EXAMPLE_SESSIONS

payload = json.dumps({"sessions": list(EXAMPLE_SESSIONS.values())}).encode()
request = urllib.request.Request(
    sys.argv[1] + "/predict", data=payload, method="POST",
    headers={"Content-Type": "application/json"},
)
body = json.load(urllib.request.urlopen(request, timeout=20))
if len(body["predictions"]) != len(EXAMPLE_SESSIONS):
    sys.exit("resposta com número inesperado de predições")
print(f"    modelo versão {body['model_version']}, threshold {body['threshold']}")
for name, item in zip(EXAMPLE_SESSIONS, body["predictions"]):
    print(f"    {name:24s} probabilidade {item['probability']:.4f} -> classe {item['predicted_class']}")
PY
}

want_switch() {
  if [ "$ASSUME_YES" -eq 0 ] && [ -t 0 ]; then
    local answer
    read -r -p "Trocar a inferência para o champion agora? [s/N] " answer
    [ "$answer" = "s" ]
  else
    say "  (sem terminal ou --sim: trocando para o champion automaticamente)"
  fi
}

deploy_and_check() {
  # deploy_and_check <MODEL> <versão esperada> <descrição>
  if try_run make deploy-inference MODEL="$1" && [ "$(served_version)" = "$2" ]; then
    record ok "$3"
    if show_predictions 2>&1 | tee -a "$LOG_FILE"; then
      record ok "POST /predict responde para os 3 exemplos (versão $2)"
    else
      record fail "POST /predict responde para os 3 exemplos (versão $2)"
    fi
  else
    record fail "$3"
  fi
}

inference_demo() {
  step 10 "Extra: inferência servindo o modelo do Registry"
  deploy_and_check 1 1 "inferência no ar com a versão 1 fixada (MODEL=1)"
  say "  Explore a API (Swagger com exemplos): ${INFERENCE_URL}/docs"
  if want_switch; then
    deploy_and_check champion "$CHAMPION_VERSION" \
      "troca para o champion: a inferência passou a servir a versão ${CHAMPION_VERSION}"
  else
    say "  Troca ignorada. Para trocar depois: make deploy-inference  (champion) ou MODEL=<versão>"
  fi
  end_step
}

summary() {
  local failures=0 entry
  printf '\n%sResumo das verificações%s\n' "$BOLD" "$RESET"
  for entry in "${RESULTS[@]}"; do
    [ -n "$entry" ] || continue
    if [ "${entry%%|*}" = "ok" ]; then
      printf '  %sOK%s      %s\n' "$GREEN" "$RESET" "${entry#*|}"
    else
      printf '  %sFALHOU%s  %s\n' "$RED" "$RESET" "${entry#*|}"
      failures=$((failures + 1))
    fi
  done
  printf '\n%sPara explorar o resultado (a infraestrutura continua rodando):%s\n' "$BOLD" "$RESET"
  say "  MLflow (runs, métricas, modelos e alias champion):  http://localhost:5001"
  say "  Console do RustFS (bucket dvc-remote):              http://localhost:9001  (devadmin / devadmin-secret)"
  say "  API de inferência (Swagger):                         ${INFERENCE_URL}/docs"
  say "  Log completo:                                        ${LOG_FILE}"
  return "$failures"
}

offer_cleanup() {
  if [ -t 0 ]; then
    local answer
    read -r -p "Limpar o ambiente agora? [s/N] " answer
    if [ "$answer" = "s" ]; then
      run bash -c "printf 's\n' | make clean"
      # shellcheck disable=SC2086
      run git restore $TRACKED_OUTPUTS
      say "Ambiente limpo."
      return
    fi
  fi
  say "Ambiente mantido. Para limpar depois: make clean (e: git restore ${TRACKED_OUTPUTS})."
}

# --- execução ---------------------------------------------------------------

for arg in "$@"; do
  case "$arg" in
    --sim) ASSUME_YES=1 ;;
    -h | --ajuda) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) fail "opção desconhecida: $arg (use --sim ou --ajuda)" ;;
  esac
done

START_TIME=$(date +%s)
preflight
confirm_reset
reset_state
install_and_test
host_pipeline
infrastructure
train_v1_twice
train_v2
restore_v1
inference_demo
summary
failures=$?
printf '\nTempo total: %s min\n' "$(( ($(date +%s) - START_TIME) / 60 ))"
offer_cleanup
exit "$failures"
