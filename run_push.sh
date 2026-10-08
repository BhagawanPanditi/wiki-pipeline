#!/usr/bin/env bash
# run_push.sh: Push pipeline updates to Prajna server
set -euo pipefail

REMOTE_HOST="Prajna"
REMOTE_DIR="/home/compiling-ganesh/24m0829/long_document/wiki_pipeline/"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"

echo "========================================================"
echo " Syncing wiki_pipeline to ${REMOTE_HOST}:${REMOTE_DIR}"
echo "========================================================"

# Exclude local caches, virtual environments, and large log dumps if desired
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'concepts.log' \
    "${LOCAL_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}"

echo ""
echo " Sync complete!"
