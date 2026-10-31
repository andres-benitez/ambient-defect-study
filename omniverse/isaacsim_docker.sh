#!/usr/bin/env bash
# Run a Python script inside the Isaac Sim 6.2.0 container with the repo
# mounted at /workspace. Usage:
#   omniverse/isaacsim_docker.sh omniverse/build_scene.py --config config/experiment.yaml ...
# Environment overrides:
#   ISAACSIM_IMAGE   docker image (default nvcr.io/nvidia/isaac-sim:6.2.0)
#   ISAACSIM_CACHE   host dir for shader/extension caches (default ~/.cache/ambient-defect-study/isaacsim)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${ISAACSIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.2.0}"
CACHE="${ISAACSIM_CACHE:-$HOME/.cache/ambient-defect-study/isaacsim}"
mkdir -p "$CACHE"/{kitcache,cache,computecache,logs,data,documents,nv}
chmod -R a+rwX "$CACHE" 2>/dev/null || true
# The container user (uid 1234) must be able to write into out/.
mkdir -p "$REPO/out"; chmod a+rwX "$REPO/out"
SCRIPT="$1"; shift
exec docker run --rm --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  -e PYTHONUNBUFFERED=1 \
  -v "$REPO":/workspace:rw \
  -v "$CACHE/kitcache":/isaac-sim/kit/cache:rw \
  -v "$CACHE/cache":/isaac-sim/.cache:rw \
  -v "$CACHE/computecache":/isaac-sim/.nv/ComputeCache:rw \
  -v "$CACHE/logs":/isaac-sim/.nvidia-omniverse/logs:rw \
  -v "$CACHE/data":/isaac-sim/.local/share/ov/data:rw \
  -v "$CACHE/documents":/isaac-sim/Documents:rw \
  -w /workspace \
  --entrypoint /isaac-sim/python.sh \
  "$IMAGE" "/workspace/$SCRIPT" "$@"
