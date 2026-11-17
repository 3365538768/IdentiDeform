#!/usr/bin/env bash
# Keeps the R/V2 job queue alive (tmux session "fieldbench"); r_queue.py holds runs/rq.lock, so a second
# instance exits at once. Running jobs live in their own sessions and survive restarts of this loop.
cd "$(dirname "$0")/.."
while true; do
  .venv/bin/python scripts/r_queue.py >> runs/rq_queue.log 2>&1
  echo "$(date -Is) r_queue exited ($?), restarting in 30 s" >> runs/rq_queue.log
  sleep 30
done
