#!/usr/bin/env bash
# Keep the read-only dashboard alive: restart on exit.
cd "$(dirname "$0")/.."
while true; do
  .venv/bin/python monitor_r/server.py --port "${PORT:-18897}" >> runs/monitor_r.log 2>&1
  echo "$(date -Is) server exited ($?), restarting" >> runs/monitor_r.log; sleep 3
done
