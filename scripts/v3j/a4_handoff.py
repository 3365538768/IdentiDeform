"""Wait until the v5r pools on a800-4 have exited, then move v3j jobs that a800-3 has not started
(no log file yet) from docs/v3j/jobs_a3.json to docs/v3j/jobs_a4.json and start a v3j pool on a800-4."""
import json, os, subprocess, time
from pathlib import Path
R = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test'); os.chdir(R)
while subprocess.run(['pgrep', '-f', 'scripts/v3e/[p]ool.py docs/v5r/'], capture_output=True).stdout.strip():
    time.sleep(30)
print(time.strftime('%H:%M:%S'), 'v5r pools gone', flush=True)
done = set(os.listdir('runs/v3j_done')); logs = R / 'docs/v3j/logs'
a3 = json.loads(Path('docs/v3j/jobs_a3.json').read_text())
unstarted = [j for j in a3 if j['id'] not in done and not (logs / f"{j['id']}.log").exists()]
take = unstarted[len(unstarted) // 3:]  # a800-3 keeps the first third, it is next in its queue
ids = {j['id'] for j in take}
Path('docs/v3j/jobs_a3.json').write_text(json.dumps([j for j in a3 if j['id'] not in ids], indent=1))
Path('docs/v3j/jobs_a4.json').write_text(json.dumps(take, indent=1))
print('moved', len(take), 'of', len(unstarted), flush=True)
subprocess.Popen('setsid nohup .venv/bin/python scripts/v3j/pool.py docs/v3j/jobs_a4.json --gpus 0,1,2,3,4,5,6,7 '
                 '--slots 56 --per-gpu 8 --stop-file docs/v3j/POOL_STOP_a4 > docs/v3j/logs/pool_a4.log 2>&1 &',
                 shell=True, cwd=R)
