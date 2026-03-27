#!/usr/bin/env bash
# run-rocm-docker-build.sh — build mesh-llm for ROCm inside a Docker container
#
# This must run on a Linux host with:
# - Docker access
# - ROCm-capable AMD GPU device nodes available to containers
# - /dev/kfd and /dev/dri present
#
# Usage:
#   scripts/run-rocm-docker-build.sh
#   scripts/run-rocm-docker-build.sh --rocm-arch "gfx942"
#   scripts/run-rocm-docker-build.sh --image rocm/dev-ubuntu-22.04:7.0-complete
#   scripts/run-rocm-docker-build.sh --build-only --platform linux/amd64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE="rocm/dev-ubuntu-24.04:7.0-complete"
ROCM_ARCH=""
KEEP_CONTAINER=0
SHELL_AFTER=0
BUILD_ONLY=0
PLATFORM=""

usage() {
    cat <<'EOF'
Usage: scripts/run-rocm-docker-build.sh [options]

Options:
  --image IMAGE         Docker image to use
  --rocm-arch GFX_LIST  Explicit AMDGPU target list, e.g. "gfx942" or "gfx90a;gfx942"
  --build-only          Skip ROCm device checks and compile only
  --platform PLATFORM   Docker platform override, e.g. linux/amd64
  --keep-container      Do not auto-remove the container
  --shell-after         Drop into an interactive shell after the build
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            IMAGE="${2:-}"
            shift 2
            ;;
        --rocm-arch)
            ROCM_ARCH="${2:-}"
            shift 2
            ;;
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --platform)
            PLATFORM="${2:-}"
            shift 2
            ;;
        --keep-container)
            KEEP_CONTAINER=1
            shift
            ;;
        --shell-after)
            SHELL_AFTER=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

die() {
    echo "Error: $*" >&2
    exit 1
}

command -v docker >/dev/null 2>&1 || die "Docker is required."
docker info >/dev/null 2>&1 || die "Docker daemon is not reachable."

HOST_OS="$(uname -s)"
if [[ "$BUILD_ONLY" -eq 0 ]]; then
    [[ "$HOST_OS" == "Linux" ]] || die "This script only works on Linux hosts unless you pass --build-only."
    [[ -e /dev/kfd ]] || die "/dev/kfd is missing. This host is not exposing ROCm GPU devices."
    [[ -d /dev/dri ]] || die "/dev/dri is missing. This host is not exposing GPU render devices."
fi

if [[ -z "$PLATFORM" && "$BUILD_ONLY" -eq 1 && "$HOST_OS" != "Linux" ]]; then
    PLATFORM="linux/amd64"
fi

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
CONTAINER_NAME="mesh-llm-rocm-build-$(date +%s)"
REMOVE_FLAG=(--rm)
if [[ "$KEEP_CONTAINER" -eq 1 ]]; then
    REMOVE_FLAG=()
fi

read -r -d '' CONTAINER_SCRIPT <<'EOF' || true
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  ca-certificates \
  cargo \
  cmake \
  curl \
  git \
  build-essential \
  nodejs \
  npm \
  pkg-config

cd /workspace/mesh-llm

BUILD_ARGS=(--backend rocm)
if [[ "${BUILD_ONLY:-0}" == "1" ]]; then
  if [[ -n "${ROCM_ARCH:-}" ]]; then
    scripts/build-linux-amd.sh "$ROCM_ARCH"
  else
    scripts/build-linux-amd.sh
  fi
else
  if [[ -n "${ROCM_ARCH:-}" ]]; then
    BUILD_ARGS+=(--rocm-arch "$ROCM_ARCH")
  fi
  scripts/build-linux.sh "${BUILD_ARGS[@]}"
fi

for path in \
  /workspace/mesh-llm/llama.cpp \
  /workspace/mesh-llm/mesh-llm/target \
  /workspace/mesh-llm/mesh-llm/ui/node_modules \
  /workspace/mesh-llm/mesh-llm/ui/dist
do
  if [[ -e "$path" ]]; then
    chown -R "$HOST_UID:$HOST_GID" "$path"
  fi
done

if [[ "${SHELL_AFTER:-0}" == "1" ]]; then
  exec bash -l
fi
EOF

echo "Starting ROCm Docker build with image: $IMAGE"
if [[ -n "$ROCM_ARCH" ]]; then
    echo "Using explicit AMDGPU targets: $ROCM_ARCH"
fi
if [[ "$BUILD_ONLY" -eq 1 ]]; then
    echo "Mode: build-only (no ROCm device access required)"
fi

DOCKER_ARGS=()
if [[ -n "$PLATFORM" ]]; then
    DOCKER_ARGS+=(--platform "$PLATFORM")
fi
if [[ "$BUILD_ONLY" -eq 0 ]]; then
    DOCKER_ARGS+=(
        --privileged
        --network=host
        --device=/dev/kfd
        --device=/dev/dri
        --group-add video
        --cap-add=SYS_PTRACE
        --security-opt seccomp=unconfined
        --ipc=host
        --shm-size 16G
    )
fi

if [[ -t 0 && -t 1 ]]; then
    docker run -it \
        "${REMOVE_FLAG[@]}" \
        "${DOCKER_ARGS[@]}" \
        --name "$CONTAINER_NAME" \
        -e HOST_UID="$HOST_UID" \
        -e HOST_GID="$HOST_GID" \
        -e ROCM_ARCH="$ROCM_ARCH" \
        -e SHELL_AFTER="$SHELL_AFTER" \
        -e BUILD_ONLY="$BUILD_ONLY" \
        -v "$REPO_ROOT:/workspace/mesh-llm" \
        -w /workspace/mesh-llm \
        "$IMAGE" \
        bash -lc "$CONTAINER_SCRIPT"
else
    docker run \
        "${REMOVE_FLAG[@]}" \
        "${DOCKER_ARGS[@]}" \
        --name "$CONTAINER_NAME" \
        -e HOST_UID="$HOST_UID" \
        -e HOST_GID="$HOST_GID" \
        -e ROCM_ARCH="$ROCM_ARCH" \
        -e SHELL_AFTER="$SHELL_AFTER" \
        -e BUILD_ONLY="$BUILD_ONLY" \
        -v "$REPO_ROOT:/workspace/mesh-llm" \
        -w /workspace/mesh-llm \
        "$IMAGE" \
        bash -lc "$CONTAINER_SCRIPT"
fi
