#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_URL="https://github.com/nanomaoli/llm_reproducibility.git"
UPSTREAM_COMMIT="f5728b570f404e6f6465a900aa15ec49f65a14bb"
PATCH_FILE="upstream.patch"

UPSTREAM_PATHS=(
    evals
    prompt_util
    eval_main.py
    eval_layercast.py
    patch_vllm.py
)

UNUSED_PATHS=(
    evals/batch
    evals/common
    evals/models
    evals/ray_configs
    evals/labeled_numina_difficulty
    evals/cli.py
    evals/inference_and_check.py
    evals/util/cli_util.py
    evals/util/metrics.py
    evals/util/response.py
)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -f "$PATCH_FILE" ]; then
    echo "error: $PATCH_FILE not found in $ROOT" >&2
    exit 1
fi

command -v git >/dev/null 2>&1 || { echo "error: git is required" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Fetching ${UPSTREAM_URL} at ${UPSTREAM_COMMIT:0:7}"
git init -q "$WORK"
git -C "$WORK" remote add origin "$UPSTREAM_URL"
if ! git -C "$WORK" fetch -q --depth 1 origin "$UPSTREAM_COMMIT" 2>/dev/null; then
    echo "  server refused a single-commit fetch, cloning full history"
    git -C "$WORK" fetch -q origin
fi
git -C "$WORK" checkout -q "$UPSTREAM_COMMIT"

FETCHED="$(git -C "$WORK" rev-parse HEAD)"
if [ "$FETCHED" != "$UPSTREAM_COMMIT" ]; then
    echo "error: expected commit $UPSTREAM_COMMIT but fetched $FETCHED" >&2
    exit 1
fi
echo "  verified $FETCHED"

echo "Installing upstream paths"
for path in "${UPSTREAM_PATHS[@]}"; do
    rm -rf "${ROOT:?}/${path:?}"
    cp -R "$WORK/$path" "$ROOT/$path"
done

echo "Removing upstream paths this work does not use"
for path in "${UNUSED_PATHS[@]}"; do
    rm -rf "${ROOT:?}/${path:?}"
done

echo "Applying $PATCH_FILE"
if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    git apply --whitespace=nowarn "$PATCH_FILE"
else
    patch -p1 --silent < "$PATCH_FILE"
fi

echo "Done: $(find "${UPSTREAM_PATHS[@]}" -type f | wc -l) files in place."
