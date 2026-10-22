#!/usr/bin/env bash
# ==============================================================================
# Ultra-fast Sync between Local Mac and Prajna Cluster
# ==============================================================================
# Code push: transfers ONLY code & config files (instant < 2 seconds).
# Data push: transfers data/ directory when explicitly requested.
# ==============================================================================

set -eo pipefail

REMOTE_HOST="Prajna"
REMOTE_PATH="/home/compiling-ganesh/24m0829/long_document/wiki_pipeline"
LOCAL_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 [--push | --pull | --push-data | --pull-data]"
    echo "  --push:      Push python code, scripts, and config.json (instant < 2s)"
    echo "  --pull:      Pull python code, scripts, and configs from cluster"
    echo "  --push-data: Push data/ directory to cluster"
    echo "  --pull-data: Pull data/ directory from cluster"
    exit 1
}

if [[ $# -eq 0 ]]; then
    usage
fi

case "$1" in
    --push)
        echo "=========================================================="
        echo "Pushing code & config -> ${REMOTE_HOST}:${REMOTE_PATH} ..."
        echo "=========================================================="
        rsync -av --progress \
            --include="*.py" \
            --include="*.sh" \
            --include="*.json" \
            --include="*.md" \
            --include="utils/***" \
            --exclude="data/***" \
            --exclude=".git/***" \
            --exclude="__pycache__/***" \
            --exclude="*.pyc" \
            --exclude="staging/***" \
            --exclude="cache/***" \
            --exclude="*" \
            "${LOCAL_PATH}/" "${REMOTE_HOST}:${REMOTE_PATH}/"
        echo "Push complete!"
        ;;
    --pull)
        echo "=========================================================="
        echo "Pulling code & config from ${REMOTE_HOST}:${REMOTE_PATH} ..."
        echo "=========================================================="
        rsync -av --progress \
            --include="*.py" \
            --include="*.sh" \
            --include="*.json" \
            --include="*.md" \
            --include="utils/***" \
            --exclude="data/***" \
            --exclude=".git/***" \
            --exclude="__pycache__/***" \
            --exclude="*.pyc" \
            --exclude="staging/***" \
            --exclude="cache/***" \
            --exclude="*" \
            "${REMOTE_HOST}:${REMOTE_PATH}/" "${LOCAL_PATH}/"
        echo "Pull complete!"
        ;;
    --push-data)
        echo "=========================================================="
        echo "Pushing data/ directory -> ${REMOTE_HOST}:${REMOTE_PATH}/data/ ..."
        echo "=========================================================="
        rsync -av --progress "${LOCAL_PATH}/data/" "${REMOTE_HOST}:${REMOTE_PATH}/data/"
        echo "Data push complete!"
        ;;
    --pull-data)
        echo "=========================================================="
        echo "Pulling data/ directory from ${REMOTE_HOST}:${REMOTE_PATH}/data/ ..."
        echo "=========================================================="
        rsync -av --progress "${REMOTE_HOST}:${REMOTE_PATH}/data/" "${LOCAL_PATH}/data/"
        echo "Data pull complete!"
        ;;
    *)
        usage
        ;;
esac
