#!/bin/bash
# Stop a v3g PAC-NeRF run whose dynamic-stage iteration exceeds LIMIT s (the official NaN -> dt-halving retry loop makes
# each retry 2x slower; on torus_v3t0 a single iteration exceeded 1 h). Usage: pn_watchdog.sh <name> [limit_s]
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test; name=$1; LIMIT=${2:-2700}
log=$R/docs/v3g/logs/pn_$name.log
while true; do
  sleep 300
  tail -1 $log | /usr/bin/grep -q V3G_END && exit 0
  /usr/bin/grep -q "dynamic Train" $log || continue
  last=$(/usr/bin/grep -E "^[0-9]+ (Time elaspsed|======= dynamic)" $log | tail -1 | cut -d" " -f1)
  now=$(date +%s)
  if [ $((now - last)) -gt $LIMIT ]; then
    for p in $(pgrep -f "pacnerf/bin/python train.py"); do
      tr "\0" " " < /proc/$p/cmdline | /usr/bin/grep -q "pn_cfgs/$name.py" && kill $p && echo "$(date -u +%FT%TZ) watchdog stopped $name pid $p: dynamic iteration > ${LIMIT}s" >> $R/docs/v3g/logs/pn_retries.log
    done
    exit 0
  fi
done
