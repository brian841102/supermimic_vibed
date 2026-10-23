#!/bin/bash
# Clone Isaac Lab at the pinned commit and build the kit-less image + the SuperMimic image.
# Optional: DOCKER_MIRROR=docker.m.daocloud.io tools/setup.sh  pulls the Docker Hub images (Dockerfile
# syntax frontend, ubuntu:24.04) through a mirror when Docker Hub is unreachable.
# BUILD_RETRIES (default 3) retries the kit-less build, whose `uv sync` fetches git dependencies from GitHub.
set -euo pipefail
cd "$(dirname "$0")/.."
ISAACLAB_COMMIT=46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
if [ ! -d IsaacLab ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git
fi
git -C IsaacLab checkout "$ISAACLAB_COMMIT"
MIRROR_ARGS=()
if [ -n "${DOCKER_MIRROR:-}" ]; then
  MIRROR_ARGS=(--build-arg "BUILDKIT_SYNTAX=${DOCKER_MIRROR}/docker/dockerfile:1"
               --build-arg "BASE_IMAGE_ARG=${DOCKER_MIRROR}/library/ubuntu:24.04")
fi
for i in $(seq 1 "${BUILD_RETRIES:-3}"); do
  (cd IsaacLab && docker build -f docker/Dockerfile.kitless "${MIRROR_ARGS[@]}" -t isaac-lab-kitless:main-46d6cc3 .) && break
  [ "$i" = "${BUILD_RETRIES:-3}" ] && exit 1
  echo "kit-less build failed (attempt $i), retrying..."
done
docker build -t supermimic:latest docker/
