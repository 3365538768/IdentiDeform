#!/usr/bin/env bash
# Download HF snapshots with retries (flaky network). Usage: download_hf_retry.sh REPO [REPO...]
cd "$(dirname "$0")/.."
for repo in "$@"; do
  for attempt in $(seq 1 30); do
    if .venv/bin/python -c "
import sys; from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], local_dir='weights/hf/' + sys.argv[1].replace('/', '__'), max_workers=8)
print('done', p)" "$repo"; then echo "$(date -Is) $repo ok"; break; fi
    echo "$(date -Is) $repo attempt $attempt failed, retrying"; sleep $((attempt < 6 ? 10 * attempt : 60))
  done
done
