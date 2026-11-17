#!/usr/bin/env bash
# Refresh docs/v3b/C1_REFINE_INTERIM.json every 90 min (CPU only) until docs/v3b/STOP_INTERIM exists or 11-14 12:00 UTC.
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test || exit 1
source scripts/r1_env.sh
while [ ! -e docs/v3b/STOP_INTERIM ] && [ "$(date -u +%s)" -lt "$(date -u -d '2026-11-14 12:00' +%s)" ]; do
  date -u
  timeout 900 .venv/bin/python scripts/v3b/c1_report.py --output docs/v3b/C1_REFINE_INTERIM.json > runs/v3b_logs/interim_last.txt 2>&1 \
    && echo ok || echo failed
  sleep 5400
done
