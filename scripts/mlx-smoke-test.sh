#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/mlx-smoke-test.sh \
    --ssh-host-a HOST_A --ssh-host-b HOST_B \
    --ip-a IP_A --ip-b IP_B \
    --model MODEL

Optional:
  --remote-python PATH          default: python3
  --remote-http-port PORT       default: 18080
  --starting-port PORT          default: 47000
  --connections-per-ip N        default: 1
  --max-tokens N                default: 32
  --prompt TEXT                 default: Reply with the single word ok.
  --startup-timeout-secs N      default: 120
  --request-timeout-secs N      default: 90
  --pipeline                    use MLX pipeline mode
  --env KEY=VALUE               repeatable extra env var
EOF
}

SSH_HOST_A=""
SSH_HOST_B=""
IP_A=""
IP_B=""
MODEL=""
REMOTE_PYTHON="python3"
REMOTE_HTTP_PORT="18080"
STARTING_PORT="47000"
CONNECTIONS_PER_IP="1"
MAX_TOKENS="32"
PROMPT="Reply with the single word ok."
STARTUP_TIMEOUT_SECS="120"
REQUEST_TIMEOUT_SECS="90"
PIPELINE="0"
EXTRA_ENVS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-host-a) SSH_HOST_A="$2"; shift 2 ;;
    --ssh-host-b) SSH_HOST_B="$2"; shift 2 ;;
    --ip-a) IP_A="$2"; shift 2 ;;
    --ip-b) IP_B="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --remote-python) REMOTE_PYTHON="$2"; shift 2 ;;
    --remote-http-port) REMOTE_HTTP_PORT="$2"; shift 2 ;;
    --starting-port) STARTING_PORT="$2"; shift 2 ;;
    --connections-per-ip) CONNECTIONS_PER_IP="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --startup-timeout-secs) STARTUP_TIMEOUT_SECS="$2"; shift 2 ;;
    --request-timeout-secs) REQUEST_TIMEOUT_SECS="$2"; shift 2 ;;
    --pipeline) PIPELINE="1"; shift ;;
    --env) EXTRA_ENVS+=("$2"); shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$SSH_HOST_A" || -z "$SSH_HOST_B" || -z "$IP_A" || -z "$IP_B" || -z "$MODEL" ]]; then
  usage
  exit 1
fi

if ! [[ "$CONNECTIONS_PER_IP" =~ ^[0-9]+$ ]] || [[ "$CONNECTIONS_PER_IP" -lt 1 ]]; then
  echo "--connections-per-ip must be >= 1" >&2
  exit 1
fi

command -v ssh >/dev/null || { echo "ssh not found" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl not found" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found" >&2; exit 1; }

json_quote() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

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

write_remote_file() {
  local host="$1"
  local path="$2"
  local contents="$3"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$host" "bash -lc 'cat > $(shell_quote "$path")'" <<<"$contents"
}

remove_remote_file() {
  local host="$1"
  local path="$2"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$host" rm -f "$path" >/dev/null 2>&1 || true
}

start_rank() {
  local rank="$1"
  local host="$2"
  local hostfile_path="$3"
  local log_path="$4"
  local remote_cmd="export MLX_RANK=$rank; export MLX_HOSTFILE=$(shell_quote "$hostfile_path"); export MLX_RING_VERBOSE=1;"
  local env_var
  for env_var in "${EXTRA_ENVS[@]}"; do
    remote_cmd+=" export $env_var;"
  done
  remote_cmd+=" exec $(shell_quote "$REMOTE_PYTHON") -m mlx_lm.server --model $(shell_quote "$MODEL") --host 0.0.0.0 --port $REMOTE_HTTP_PORT --max-tokens $MAX_TOKENS"
  if [[ "$PIPELINE" == "1" ]]; then
    remote_cmd+=" --pipeline"
  fi
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$host" "bash -lc $(
    shell_quote "$remote_cmd"
  )" >"$log_path" 2>&1 &
  echo $!
}

cleanup() {
  set +e
  if [[ -n "${RANK0_PID:-}" ]]; then kill "$RANK0_PID" >/dev/null 2>&1 || true; wait "$RANK0_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "${RANK1_PID:-}" ]]; then kill "$RANK1_PID" >/dev/null 2>&1 || true; wait "$RANK1_PID" >/dev/null 2>&1 || true; fi
  remove_remote_file "$SSH_HOST_A" "$RANK0_HOSTFILE"
  remove_remote_file "$SSH_HOST_B" "$RANK1_HOSTFILE"
}
trap cleanup EXIT

HOSTFILE="$(python3 - "$IP_A" "$IP_B" "$STARTING_PORT" "$CONNECTIONS_PER_IP" <<'PY'
import json, sys
ip_a, ip_b, start, count = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
port = start
hosts = []
for ip in (ip_a, ip_b):
    entries = []
    for _ in range(count):
        entries.append(f"{ip}:{port}")
        port += 1
    hosts.append(entries)
print(json.dumps(hosts))
PY
)"

RANK0_HOSTFILE="$(remote_tempfile "$SSH_HOST_A")"
RANK1_HOSTFILE="$(remote_tempfile "$SSH_HOST_B")"
write_remote_file "$SSH_HOST_A" "$RANK0_HOSTFILE" "$HOSTFILE"
write_remote_file "$SSH_HOST_B" "$RANK1_HOSTFILE" "$HOSTFILE"

LOG_PREFIX="${TMPDIR:-/tmp}/mesh-llm-mlx-smoke-$(date +%s)"
RANK0_LOG="${LOG_PREFIX}-rank0.log"
RANK1_LOG="${LOG_PREFIX}-rank1.log"

RANK0_PID="$(start_rank 0 "$SSH_HOST_A" "$RANK0_HOSTFILE" "$RANK0_LOG")"
RANK1_PID="$(start_rank 1 "$SSH_HOST_B" "$RANK1_HOSTFILE" "$RANK1_LOG")"

wait_for_health() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECS))
  local url="http://$IP_A:$REMOTE_HTTP_PORT/health"
  while (( SECONDS < deadline )); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for rank 0 health at $url" >&2
  return 1
}

wait_for_health

curl --fail --silent --show-error --max-time "$REQUEST_TIMEOUT_SECS" \
  "http://$IP_A:$REMOTE_HTTP_PORT/v1/models" >/dev/null

CHAT_PAYLOAD="$(python3 - "$PROMPT" "$MAX_TOKENS" <<'PY'
import json, sys
print(json.dumps({
    "model": "default_model",
    "messages": [{"role": "user", "content": sys.argv[1]}],
    "max_tokens": int(sys.argv[2]),
}))
PY
)"

CHAT_RESPONSE="$(curl --fail --silent --show-error --max-time "$REQUEST_TIMEOUT_SECS" \
  -H 'Content-Type: application/json' \
  -d "$CHAT_PAYLOAD" \
  "http://$IP_A:$REMOTE_HTTP_PORT/v1/chat/completions")"

CHAT_CONTENT="$(python3 - "$CHAT_RESPONSE" <<'PY'
import json, sys
body = json.loads(sys.argv[1])
print((body.get("choices") or [{}])[0].get("message", {}).get("content", "").strip())
PY
)"

if [[ -z "$CHAT_CONTENT" ]]; then
  echo "MLX smoke test returned empty chat content" >&2
  exit 1
fi

echo "✅ MLX smoke test passed: rank 0 served HTTP on $IP_A:$REMOTE_HTTP_PORT"
