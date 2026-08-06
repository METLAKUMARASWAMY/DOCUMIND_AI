#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_VENV="${DOCUMIND_AI_VENV:-${PROJECT_ROOT}/.venv-ai}"
PYTHON_COMMAND="${DOCUMIND_AI_PYTHON:-}"
SUMMARY_PATH="${PROJECT_ROOT}/var/ai-core/verification.json"
LOG_PATH="${PROJECT_ROOT}/var/ai-core/bootstrap.log"
INSTALL_DEPENDENCIES=true
START_SERVICES=true
INSTALL_PADDLE=true
ALLOW_PADDLE_MODEL_DOWNLOADS=false
ALLOW_BGE_DOWNLOADS=false
ALLOW_LOCAL_LLM_DOWNLOADS=false
LOCAL_LLM_CHECKPOINT=""
STRICT=false
STAGE_FAILURES=0

usage() {
  sed -n '2,55p' "$0" | sed -n 's/^# //p'
}

# Bootstrap the practical DocuMind AI core without printing credentials.
#
# Usage: scripts/bootstrap_ai_core.sh [options]
#
# Options:
#   --venv PATH                       AI virtual environment (default: .venv-ai)
#   --python COMMAND                  Python 3.11/3.12 executable
#   --summary PATH                    Verification JSON destination
#   --skip-install                    Audit the existing environment only
#   --skip-services                   Do not start Qdrant or Neo4j
#   --no-paddle                       Do not install the Paddle runtime/packages
#   --allow-paddle-model-downloads    Permit PP-OCR/PP-Structure asset downloads
#   --allow-bge-downloads             Permit the two named BGE checkpoint downloads
#   --local-llm-checkpoint ID_OR_PATH Enable this explicitly selected local model
#   --allow-local-llm-downloads       Permit download of the selected local model
#   --strict                          Return non-zero when a configured core probe fails
#   --help                            Show this help

while (($#)); do
  case "$1" in
    --venv)
      AI_VENV="$2"
      shift 2
      ;;
    --python)
      PYTHON_COMMAND="$2"
      shift 2
      ;;
    --summary)
      SUMMARY_PATH="$2"
      shift 2
      ;;
    --skip-install)
      INSTALL_DEPENDENCIES=false
      shift
      ;;
    --skip-services)
      START_SERVICES=false
      shift
      ;;
    --no-paddle)
      INSTALL_PADDLE=false
      shift
      ;;
    --allow-paddle-model-downloads)
      ALLOW_PADDLE_MODEL_DOWNLOADS=true
      shift
      ;;
    --allow-bge-downloads)
      ALLOW_BGE_DOWNLOADS=true
      shift
      ;;
    --local-llm-checkpoint)
      LOCAL_LLM_CHECKPOINT="$2"
      shift 2
      ;;
    --allow-local-llm-downloads)
      ALLOW_LOCAL_LLM_DOWNLOADS=true
      shift
      ;;
    --strict)
      STRICT=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "$SUMMARY_PATH")" "$(dirname "$LOG_PATH")"
: >"$LOG_PATH"

log() {
  printf '[ai-core] %s\n' "$*" | tee -a "$LOG_PATH"
}

run_stage() {
  local label="$1"
  shift
  log "Starting ${label}."
  "$@" >>"$LOG_PATH" 2>&1
  local command_exit=$?
  if [[ "$command_exit" -eq 0 ]]; then
    log "Completed ${label}."
    return 0
  fi
  STAGE_FAILURES=$((STAGE_FAILURES + 1))
  log "Failed ${label} (exit ${command_exit}); continuing with independent stages."
  return "$command_exit"
}

select_python() {
  if [[ -n "$PYTHON_COMMAND" ]]; then
    command -v "$PYTHON_COMMAND"
    return
  fi
  local candidate
  for candidate in python3.11 python3.12; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return
    fi
  done
  return 1
}

platform_name="$(uname -s)"
machine_name="$(uname -m)"
memory_bytes=""
if [[ "$platform_name" == "Darwin" ]]; then
  memory_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
elif command -v getconf >/dev/null 2>&1; then
  page_size="$(getconf PAGE_SIZE 2>/dev/null || true)"
  page_count="$(getconf _PHYS_PAGES 2>/dev/null || true)"
  if [[ "$page_size" =~ ^[0-9]+$ && "$page_count" =~ ^[0-9]+$ ]]; then
    memory_bytes="$((page_size * page_count))"
  fi
fi
log "Detected ${platform_name}/${machine_name}; memory bytes=${memory_bytes:-unknown}."

if [[ ! -x "${AI_VENV}/bin/python" ]]; then
  selected_python="$(select_python || true)"
  if [[ -z "$selected_python" ]]; then
    log "Python 3.11 or 3.12 is required for the broadest Paddle/BGE wheel compatibility."
    exit 2
  fi
  log "Creating isolated AI environment with $($selected_python --version 2>&1)."
  run_stage "AI virtual environment creation" "$selected_python" -m venv "$AI_VENV" || exit 2
fi
AI_PYTHON="${AI_VENV}/bin/python"

PADDLE_WHEEL_AVAILABLE=false
if [[ "$INSTALL_PADDLE" == true ]]; then
  if "$AI_PYTHON" -m pip install --dry-run --only-binary=:all: \
    'paddlepaddle>=3.2,<3.4' >>"$LOG_PATH" 2>&1; then
    PADDLE_WHEEL_AVAILABLE=true
  else
    log "No compatible PaddlePaddle binary wheel was found for this Python/platform."
  fi
fi

if [[ "$INSTALL_DEPENDENCIES" == true ]]; then
  run_stage "packaging-tool upgrade" \
    "$AI_PYTHON" -m pip install --upgrade pip setuptools wheel || true
  if [[ "$PADDLE_WHEEL_AVAILABLE" == true ]] && run_stage "AI core dependency group" \
    "$AI_PYTHON" -m pip install -r "${PROJECT_ROOT}/backend/requirements/ai-core.txt"; then
    :
  else
    run_stage "base runtime fallback install" \
      "$AI_PYTHON" -m pip install -r "${PROJECT_ROOT}/backend/requirements/base.txt" || true
    run_stage "semantic runtime fallback install" \
      "$AI_PYTHON" -m pip install \
      'torch>=2.4,<3' 'transformers>=4.49,<5' 'sentence-transformers>=3.4,<6' \
      'FlagEmbedding>=1.3.5,<2' 'accelerate>=1.2,<2' 'sentencepiece>=0.2,<1' || true
    if [[ "$PADDLE_WHEEL_AVAILABLE" == true ]]; then
      run_stage "Paddle CPU fallback install" \
        "$AI_PYTHON" -m pip install 'paddlepaddle>=3.2,<3.4' 'paddleocr>=3,<4' || true
    fi
  fi
  run_stage "editable backend install" "$AI_PYTHON" -m pip install -e "${PROJECT_ROOT}/backend" || true
fi

if [[ "$START_SERVICES" == true ]]; then
  if command -v docker >/dev/null 2>&1; then
    if ! docker info >/dev/null 2>&1 && [[ "$platform_name" == "Darwin" ]]; then
      log "Docker daemon is unavailable; requesting Docker Desktop startup."
      open -a Docker >>"$LOG_PATH" 2>&1 || true
      for _ in {1..20}; do
        docker info >/dev/null 2>&1 && break
        sleep 2
      done
    fi
    if docker info >/dev/null 2>&1; then
      run_stage "Qdrant and Neo4j containers" \
        docker compose -f "${PROJECT_ROOT}/docker-compose.yml" \
        --project-directory "$PROJECT_ROOT" up -d qdrant neo4j || true
    else
      STAGE_FAILURES=$((STAGE_FAILURES + 1))
      log "Docker daemon remains unreachable; service smokes will report the blocker."
    fi
  else
    STAGE_FAILURES=$((STAGE_FAILURES + 1))
    log "Docker CLI is not installed; service smokes will report the blocker."
  fi
fi

# Resolve compose's NEO4J_AUTH locally when dedicated driver variables are not
# already present. Values remain process-local and are never written to logs.
if [[ -z "${NEO4J_USER:-}" || -z "${NEO4J_PASSWORD:-}" ]] && [[ -f "${PROJECT_ROOT}/.env" ]]; then
  neo4j_auth="$(sed -n 's/^NEO4J_AUTH=//p' "${PROJECT_ROOT}/.env" | head -n 1)"
  neo4j_auth="${neo4j_auth%\"}"
  neo4j_auth="${neo4j_auth#\"}"
  if [[ "$neo4j_auth" == */* ]]; then
    export NEO4J_USER="${neo4j_auth%%/*}"
    export NEO4J_PASSWORD="${neo4j_auth#*/}"
  fi
fi

export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export NEO4J_URI="${NEO4J_URI:-bolt://127.0.0.1:7687}"
export ENABLE_QDRANT=true
export ENABLE_NEO4J=true

if [[ "$ALLOW_PADDLE_MODEL_DOWNLOADS" == true && "$PADDLE_WHEEL_AVAILABLE" == true ]]; then
  export ENABLE_HEAVY_MODELS=true
  export ENABLE_PADDLEOCR=true
  export ENABLE_PADDLEOCR_STRUCTURE=true
  export PADDLEOCR_DEVICE=cpu
fi
if [[ "$ALLOW_BGE_DOWNLOADS" == true ]]; then
  export ENABLE_HEAVY_MODELS=true
  export ENABLE_BGE_M3=true
  export BGE_ALLOW_DOWNLOAD=true
fi
if [[ -n "$LOCAL_LLM_CHECKPOINT" ]]; then
  if [[ "$ALLOW_LOCAL_LLM_DOWNLOADS" != true && ! -e "$LOCAL_LLM_CHECKPOINT" ]]; then
    cache_slug="models--${LOCAL_LLM_CHECKPOINT//\//--}"
    if [[ ! -d "${HOME}/.cache/huggingface/hub/${cache_slug}" ]]; then
      log "Selected local LLM is neither a local path nor cached; add --allow-local-llm-downloads."
      STAGE_FAILURES=$((STAGE_FAILURES + 1))
    else
      export ENABLE_HEAVY_MODELS=true
      export ENABLE_QWEN_LOCAL=true
      export QWEN_CHECKPOINT="$LOCAL_LLM_CHECKPOINT"
    fi
  else
    export ENABLE_HEAVY_MODELS=true
    export ENABLE_QWEN_LOCAL=true
    export QWEN_CHECKPOINT="$LOCAL_LLM_CHECKPOINT"
    [[ "$ALLOW_LOCAL_LLM_DOWNLOADS" == true ]] && export QWEN_ALLOW_DOWNLOAD=true
  fi
fi

VERIFY_ARGUMENTS=(--run-smoke --output "$SUMMARY_PATH")
[[ "$STRICT" == true ]] && VERIFY_ARGUMENTS+=(--strict)
log "Running dependency, checkpoint, service, and inference verification."
verify_status=0
if "$AI_PYTHON" "${PROJECT_ROOT}/scripts/verify_ai_core.py" "${VERIFY_ARGUMENTS[@]}" \
  >>"$LOG_PATH" 2>&1; then
  verify_status=0
else
  verify_status=$?
  STAGE_FAILURES=$((STAGE_FAILURES + 1))
fi

if [[ -f "$SUMMARY_PATH" ]]; then
  "$AI_PYTHON" - "$SUMMARY_PATH" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
smoke = payload.get("smoke", {})
services = payload.get("services", {})
print(
    "AI core summary: "
    f"smoke passed={smoke.get('passed', 0)} "
    f"failed={smoke.get('failed', 0)} blocked={smoke.get('blocked', 0)}; "
    f"Qdrant reachable={services.get('qdrant', {}).get('reachable', False)}; "
    f"Neo4j reachable={services.get('neo4j', {}).get('reachable', False)}"
)
print(f"Full verification: {sys.argv[1]}")
PY
else
  log "Verification did not produce a summary file; inspect ${LOG_PATH}."
fi

log "Bootstrap finished with ${STAGE_FAILURES} failed stage(s); log=${LOG_PATH}."
if [[ "$STRICT" == true ]]; then
  exit "$verify_status"
fi
exit 0
