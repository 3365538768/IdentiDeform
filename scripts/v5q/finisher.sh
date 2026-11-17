#!/bin/bash
# V5q finisher (nohup on a800-4): PLAN.json v5q progress every 10 min; when every task is done, write the report.
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
while true; do
  .venv/bin/python scripts/v5q/plan_update.py --auto > /dev/null 2>&1
  left=$(python3 -c "
import json,os;T=json.load(open(\"docs/v5q/tasks.json\"))[\"tasks\"];print(sum(1 for t in T if not t.get(\"cut\") and not os.path.exists(\"runs/v5q_done/\"+t[\"id\"])))")
  if [ "$left" = "0" ]; then
    .venv/bin/python scripts/v5q/report.py > docs/v5q/logs/report.log 2>&1 && .venv/bin/python scripts/v5q/write_zh.py > /dev/null 2>&1 && \
    .venv/bin/python scripts/v5q/plan_update.py --auto --status done --detail "全部完成；结果见 docs/v5q/RESULTS.json / RESULTS_ZH.md（补充实验）" > /dev/null 2>&1
    echo "finished $(date -u)"; break
  fi
  sleep 600
done
