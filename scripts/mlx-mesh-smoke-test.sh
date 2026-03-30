#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/mlx-mesh-smoke-test.sh \
    --ssh-host-a HOST_A --ssh-host-b HOST_B \
    --ip-a IP_A --ip-b IP_B \
    --model MODEL_PATH_ON_BOTH_MACS \
    --remote-mesh-bin /path/to/mesh-llm \
    --remote-bin-dir /path/to/llama.cpp/build/bin

Optional:
  --remote-api-port PORT         default: 19337
  --remote-console-port PORT     default: 13131
  --remote-bind-port-a PORT      default: 47010
  --remote-bind-port-b PORT      default: 47011
  --remote-mlx-server-bin PATH   optional
  --connections-per-rank N       default: 1
  --max-tokens N                 default: 32
  --prompt TEXT                  default: Reply with the single word ok.
  --startup-timeout-secs N       default: 180
  --request-timeout-secs N       default: 90
  --pipeline                     sets MESH_LLM_MLX_PIPELINE=1
EOF
}

SSH_HOST_A=""
SSH_HOST_B=""
IP_A=""
IP_B=""
MODEL=""
REMOTE_MESH_BIN=""
REMOTE_BIN_DIR=""
REMOTE_API_PORT="19337"
REMOTE_CONSOLE_PORT="13131"
REMOTE_BIND_PORT_A="47010"
REMOTE_BIND_PORT_B="47011"
REMOTE_MLX_SERVER_BIN=""
CONNECTIONS_PER_RANK="1"
MAX_TOKENS="32"
PROMPT="Reply with the single word ok."
STARTUP_TIMEOUT_SECS="180"
REQUEST_TIMEOUT_SECS="90"
PIPELINE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-host-a) SSH_HOST_A="$2"; shift 2 ;;
    --ssh-host-b) SSH_HOST_B="$2"; shift 2 ;;
    --ip-a) IP_A="$2"; shift 2 ;;
    --ip-b) IP_B="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --remote-mesh-bin) REMOTE_MESH_BIN="$2"; shift 2 ;;
    --remote-bin-dir) REMOTE_BIN_DIR="$2"; shift 2 ;;
    --remote-api-port) REMOTE_API_PORT="$2"; shift 2 ;;
    --remote-console-port) REMOTE_CONSOLE_PORT="$2"; shift 2 ;;
    --remote-bind-port-a) REMOTE_BIND_PORT_A="$2"; shift 2 ;;
    --remote-bind-port-b) REMOTE_BIND_PORT_B="$2"; shift 2 ;;
    --remote-mlx-server-bin) REMOTE_MLX_SERVER_BIN="$2"; shift 2 ;;
    --connections-per-rank) CONNECTIONS_PER_RANK="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --startup-timeout-secs) STARTUP_TIMEOUT_SECS="$2"; shift 2 ;;
    --request-timeout-secs) REQUEST_TIMEOUT_SECS="$2"; shift 2 ;;
    --pipeline) PIPELINE="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$SSH_HOST_A" || -z "$SSH_HOST_B" || -z "$IP_A" || -z "$IP_B" || -z "$MODEL" || -z "$REMOTE_MESH_BIN" || -z "$REMOTE_BIN_DIR" ]]; then
  usage
  exit 1
fi

command -v ssh >/dev/null || { echo "ssh not found" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl not found" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found" >&2; exit 1; }

shell_quote() {
  python3 - "$1" <<'PY'
import shlex, sys
print(shlex.quote(sys.argv[1]))
PY
}

remote_tempfile() {
  local host="$1"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$host" mktemp
}

remote_read() {
  local host="$1"
  local path="$2"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$host" "cat $(shell_quote "$path")"
}

start_mesh_node() {
  local host="$1"
  local log_path="$2"
  local bind_port="$3"
  local join_token="${4:-}"
  local remote_cmd="export MESH_LLM_MLX_CONNECTIONS_PER_RANK=$CONNECTIONS_PER_RANK;"
  if [[ "$PIPELINE" == "1" ]]; then
    remote_cmd+=" export MESH_LLM_MLX_PIPELINE=1;"
  fi
  if [[ -n "$REMOTE_MLX_SERVER_BIN" ]]; then
    remote_cmd+=" export MESH_LLM_MLX_SERVER_BIN=$(shell_quote "$REMOTE_MLX_SERVER_BIN");"
  fi
  remote_cmd+=" nohup $(shell_quote "$REMOTE_MESH_BIN") --model $(shell_quote "$MODEL") --bin-dir $(shell_quote "$REMOTE_BIN_DIR") --port $REMOTE_API_PORT --console $REMOTE_CONSOLE_PORT --listen-all --bind-port $bind_port --split"
  if [[ -n "$join_token" ]]; then
    remote_cmd+=" --join $(shell_quote "$join_token")"
  fi
  remote_cmd+=" > $(shell_quote "$log_path") 2>&1 < /dev/null & echo \$!"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$host" "bash -lc $(shell_quote "$remote_cmd")"
}

wait_for_invite() {
  local host="$1"
  local log_path="$2"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECS))
  while (( SECONDS < deadline )); do
    local token
    token="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$host" "grep -m1 '^Invite:' $(shell_quote "$log_path") | sed 's/^Invite: //'" 2>/dev/null || true)"
    if [[ -n "$token" ]]; then
      printf '%s' "$token"
      return 0
    fi
    sleep 1
  done
  return 1
}

tail_logs() {
  local host="$1"
  local log_path="$2"
  echo "--- $host:$log_path ---" >&2
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$host" "tail -n 120 $(shell_quote "$log_path")" >&2 || true
}

cleanup() {
  set +e
  [[ -n "${PID_A:-}" ]] && ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$SSH_HOST_A" "kill $PID_A >/dev/null 2>&1 || true; pkill -f mesh-llm >/dev/null 2>&1 || true; pkill -f mlx_lm.server >/dev/null 2>&1 || true"
  [[ -n "${PID_B:-}" ]] && ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$SSH_HOST_B" "kill $PID_B >/dev/null 2>&1 || true; pkill -f mesh-llm >/dev/null 2>&1 || true; pkill -f mlx_lm.server >/dev/null 2>&1 || true"
}
trap cleanup EXIT

LOG_A="$(remote_tempfile "$SSH_HOST_A")"
LOG_B="$(remote_tempfile "$SSH_HOST_B")"
PID_A="$(start_mesh_node "$SSH_HOST_A" "$LOG_A" "$REMOTE_BIND_PORT_A")"
INVITE="$(wait_for_invite "$SSH_HOST_A" "$LOG_A")" || {
  tail_logs "$SSH_HOST_A" "$LOG_A"
  echo "Timed out waiting for invite token from node A" >&2
  exit 1
}
PID_B="$(start_mesh_node "$SSH_HOST_B" "$LOG_B" "$REMOTE_BIND_PORT_B" "$INVITE")"

MODEL_NAME="$(basename "$MODEL")"

wait_for_models() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECS))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://$IP_A:$REMOTE_API_PORT/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_models || {
  tail_logs "$SSH_HOST_A" "$LOG_A"
  tail_logs "$SSH_HOST_B" "$LOG_B"
  echo "Timed out waiting for mesh MLX /v1/models on $IP_A:$REMOTE_API_PORT" >&2
  exit 1
}

CHAT_PAYLOAD="$(python3 - "$MODEL_NAME" "$PROMPT" "$MAX_TOKENS" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": sys.argv[2]}],
    "max_tokens": int(sys.argv[3]),
}))
PY
)"

CHAT_RESPONSE="$(curl --fail --silent --show-error --max-time "$REQUEST_TIMEOUT_SECS" \
  -H 'Content-Type: application/json' \
  -d "$CHAT_PAYLOAD" \
  "http://$IP_A:$REMOTE_API_PORT/v1/chat/completions")"

CHAT_CONTENT="$(python3 - "$CHAT_RESPONSE" <<'PY'
import json, sys
body = json.loads(sys.argv[1])
print((body.get("choices") or [{}])[0].get("message", {}).get("content", "").strip())
PY
)"

if [[ -z "$CHAT_CONTENT" ]]; then
  tail_logs "$SSH_HOST_A" "$LOG_A"
  tail_logs "$SSH_HOST_B" "$LOG_B"
  echo "MLX mesh smoke test returned empty chat content" >&2
  exit 1
fi

echo "✅ MLX mesh smoke test passed: rank 0 served HTTP on $IP_A:$REMOTE_API_PORT"
