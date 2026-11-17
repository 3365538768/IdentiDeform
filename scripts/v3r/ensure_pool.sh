#!/usr/bin/env bash
# V3r: (re)start the a800-4 nohup pool for docs/v3r/jobs_a4c.json if it is not running and work is pending. Run ON a800-4.
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test; cd $R
PER=${1:-2}
if ps aux | grep -q "[p]ool.py docs/v3r/jobs_a4c.json"; then echo running; exit 0; fi
pend=$(python3 -c "
import json,os
j=json.load(open('docs/v3r/jobs_a4c.json'))
print(sum(not os.path.exists('runs/v3r_done/'+x['id']) for x in j))")
[ "$pend" -gt 0 ] || { echo nothing; exit 0; }
rm -f docs/v3r/POOL_STOP
setsid nohup .venv/bin/python scripts/v3e/pool.py docs/v3r/jobs_a4c.json --tag v3r --per-gpu $PER >> docs/v3r/logs/pool_a4c.log 2>&1 < /dev/null & disown
echo "started per-gpu $PER pending $pend"
