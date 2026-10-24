#!/bin/bash
# Start the long-lived 'supermimic' container with one GPU and the repository mounted.
# Usage: [VISER_PORT=8080] docker/run_container.sh [GPU_INDEX] [MEMORY_LIMIT]
# VISER_PORT is published so tools/viser_play.py (inside the container) is reachable from the host browser.
set -euo pipefail
GPU=${1:-0}
MEM=${2:-128g}
PORT=${VISER_PORT:-8080}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
docker run -d --name supermimic --gpus "\"device=${GPU}\"" --memory="$MEM" --shm-size=16g --user root -p "${PORT}:8080" \
  -e HOME=/root -e WARP_CACHE_PATH=/workspace/supermimic/.cache/warp -e MPLCONFIGDIR=/tmp/mpl \
  -e TZ="${TZ:-$(readlink /etc/localtime | sed 's#.*/zoneinfo/##')}" \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/supermimic/.cache/inductor \
  -v "$REPO":/workspace/supermimic -w /workspace/supermimic supermimic:latest sleep infinity
