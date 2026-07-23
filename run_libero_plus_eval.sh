#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  ./run_libero_plus_eval.sh [VERSION] [STEP] [TASK]
  ./run_libero_plus_eval.sh --version VERSION --step STEP --task TASK [--trials TRIALS] [--port PORT] [--client-host HOST] [--replan-steps STEPS]
  ./run_libero_plus_eval.sh -v VERSION -s STEP -t TASK [-n TRIALS] [-p PORT] [-H HOST] [-r STEPS]

Examples:
  ./run_libero_plus_eval.sh
  ./run_libero_plus_eval.sh wxv31.0 30000 libero_10
  ./run_libero_plus_eval.sh --version wxv31.0 --step 30000 --task libero_10 --trials 1 --port 8000 --replan-steps 10

Defaults:
  VERSION=wxv31.0
  STEP=30000
  TASK=libero_10
  TRIALS=1
  PORT=8000
  CLIENT_HOST=127.0.0.1
  REPLAN_STEPS=10

Environment overrides are also supported:
  VERSION=... STEP=... TASK=... TRIALS=... PORT=... CLIENT_HOST=... REPLAN_STEPS=... ./run_libero_plus_eval.sh
EOF
}

VERSION="${VERSION:-wxv31.0}"
STEP="${STEP:-30000}"
TASK="${TASK:-libero_10}"
TRIALS="${TRIALS:-1}"
PORT="${PORT:-8000}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version|-v)
      VERSION="$2"
      shift 2
      ;;
    --step|-s)
      STEP="$2"
      shift 2
      ;;
    --task|-t)
      TASK="$2"
      shift 2
      ;;
    --trials|-n)
      TRIALS="$2"
      shift 2
      ;;
    --port|-p)
      PORT="$2"
      shift 2
      ;;
    --client-host|-H)
      CLIENT_HOST="$2"
      shift 2
      ;;
    --replan-steps|-r)
      REPLAN_STEPS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
  VERSION="${POSITIONAL_ARGS[0]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
  STEP="${POSITIONAL_ARGS[1]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -ge 3 ]]; then
  TASK="${POSITIONAL_ARGS[2]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -gt 3 ]]; then
  echo "Too many positional arguments." >&2
  usage >&2
  exit 1
fi

EVAL_LOG="${EVAL_LOG:-logs/${VERSION}_liberoplus_${TASK}_output.log}"

mkdir -p logs

server_pid=""
tmp_server_log="$(mktemp /tmp/run_libero_eval_server.XXXXXX.log)"

cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    echo "Stopping policy server (PID ${server_pid})..."
    kill "${server_pid}" 2>/dev/null || true

    for _ in {1..10}; do
      if ! kill -0 "${server_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done

    if kill -0 "${server_pid}" 2>/dev/null; then
      echo "Policy server did not exit after SIGTERM; sending SIGKILL..."
      kill -9 "${server_pid}" 2>/dev/null || true
    fi

    wait "${server_pid}" 2>/dev/null || true
  fi

  rm -f "${tmp_server_log}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "Starting LIBERO evaluation with:"
echo "  VERSION=${VERSION}"
echo "  STEP=${STEP}"
echo "  TASK=${TASK}"
echo "  TRIALS=${TRIALS}"
echo "  PORT=${PORT}"
echo "  CLIENT_HOST=${CLIENT_HOST}"
echo "  REPLAN_STEPS=${REPLAN_STEPS}"
echo "  EVAL_LOG=${EVAL_LOG}"
echo "Foreground job: LIBERO evaluation client"
echo "Background job: policy server"

(
  source .venv/bin/activate
  exec env VERSION="${VERSION}" STEP="${STEP}" uv run --no-sync scripts/serve_policy.py \
    --port "${PORT}" \
    policy:checkpoint \
    --policy.config="${VERSION}" \
    --policy.dir="checkpoints/${VERSION}/experiments/${STEP}"
) >"${tmp_server_log}" 2>&1 &
server_pid=$!

echo "Policy server started in background (PID ${server_pid}). Waiting for port ${PORT}..."

for _ in {1..600}; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Policy server exited unexpectedly. Recent server log:"
    tail -n 50 "${tmp_server_log}" || true
    exit 1
  fi

  if bash -lc ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    echo "Policy server is ready."
    break
  fi

  sleep 2
done

if ! bash -lc ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
  echo "Timed out waiting for the policy server on port ${PORT}. Recent server log:"
  tail -n 50 "${tmp_server_log}" || true
  exit 1
fi

echo "Running LIBERO evaluation..."
(
  source examples/libero/.venv/bin/activate
  export LIBERO_CONFIG_PATH="$PWD/examples/libero"  # "$PWD/examples/libero/config.yaml"
  export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${ROOT_DIR}/third_party/libero_plus"
  python examples/libero/main.py \
    --args.host "${CLIENT_HOST}" \
    --args.task-suite-name "${TASK}" \
    --args.num-trials-per-task "${TRIALS}" \
    --args.port "${PORT}" \
    --args.replan-steps "${REPLAN_STEPS}"
) >"${EVAL_LOG}" 2>&1

echo "Evaluation finished. Log saved to ${EVAL_LOG}"
