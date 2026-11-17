import json, os, subprocess
from pathlib import Path
R = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test'); os.chdir(R)
done = set(os.listdir('runs/v3j_done')); logs = R / 'docs/v3j/logs'
a3 = json.loads(Path('docs/v3j/jobs_a3.json').read_text())
un = [j for j in a3 if j['id'] not in done and not (logs / f"{j['id']}.log").exists()]
take = un[-2:]; ids = {j['id'] for j in take}
Path('docs/v3j/jobs_a3.json').write_text(json.dumps([j for j in a3 if j['id'] not in ids], indent=1))
Path('docs/v3j/jobs_a3g4.json').write_text(json.dumps(take, indent=1)); print('moved', [j['id'] for j in take], 'unstarted', len(un))
subprocess.Popen('setsid nohup .venv/bin/python scripts/v3j/pool.py docs/v3j/jobs_a3g4.json --gpus 4 --slots 8 --per-gpu 8 '
                 '--stop-file docs/v3j/POOL_STOP_a3g4 > docs/v3j/logs/pool_a3g4.log 2>&1 < /dev/null &', shell=True, cwd=R)
