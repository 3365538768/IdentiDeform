"""Local scheduler for V3i on a800-4 (the a800-2 r_queue was full). At most MAX_PER_GPU of these jobs per GPU,
launch only when the GPU has > MEM_NEED_MIB free. Failed jobs are retried (resuming from last.pt) up to 2 times.
State: docs/v3i/launch_state.json. Run with nohup; a lock file prevents duplicates.
"""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
GPUS = [int(g) for g in os.environ.get('V3I_GPUS', '0,1,2,3,4,5,6,7').split(',')]
MAX_PER_GPU, MEM_NEED_MIB, RETRIES = 2, 24000, 2
STATE = ROOT / 'docs/v3i/launch_state.json'

def free_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=60).stdout
    return {int(i): int(m) for i, m in (l.split(',') for l in out.strip().splitlines())}

def finished(job):
    try:
        return json.loads((ROOT / 'runs' / job['id'] / 'status.json').read_text()).get('status') in {'training_finished', 'early_stopped'}
    except (OSError, ValueError):
        return False

def alive(pid):
    try:  # reap our own finished children first (a zombie still answers kill(pid, 0))
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False

def main():
    lock = open(ROOT / 'runs/v3i_logs/launch.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit('already running')
    while True:
        jobs = json.loads((ROOT / 'docs/v3i/jobs.json').read_text())['jobs']
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        state.pop('_updated', None)
        for j in jobs:
            s = state.setdefault(j['id'], {'status': 'pending', 'attempts': 0})
            if s['status'] == 'running' and not alive(s['pid']):
                s['status'] = 'done' if finished(j) else 'failed'; s['ended'] = time.time()
            if s['status'] in ('pending', 'failed') and finished(j):
                s['status'] = 'done'
            if s['status'] == 'failed' and s['attempts'] <= RETRIES and time.time() - s['ended'] > 60:
                s['status'] = 'pending'
        free = free_mib()
        for j in jobs:
            s = state[j['id']]
            if s['status'] != 'pending':
                continue
            used = {g: sum(1 for k, v in state.items() if k != '_updated' and v['status'] == 'running' and v.get('gpu') == g) for g in GPUS}
            options = [g for g in GPUS if used[g] < MAX_PER_GPU and free.get(g, 0) > MEM_NEED_MIB]
            if not options:
                break
            g = min(options, key=lambda g: (used[g], -free[g]))
            log = ROOT / 'runs/v3i_logs' / f"{j['id']}.log"
            cmd = (f"cd {ROOT} && source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES={g} && "
                   f".venv/bin/python scripts/v3i/train_views.py --config {j['config']}")
            pid = subprocess.Popen(['bash', '-c', cmd], stdout=log.open('a'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True).pid
            s.update(status='running', gpu=g, pid=pid, started=time.time(), attempts=s['attempts'] + 1)
            free[g] -= MEM_NEED_MIB
        state['_updated'] = time.time()
        tmp = STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=1)); os.replace(tmp, STATE)
        if all(state[j['id']]['status'] == 'done' or (state[j['id']]['status'] == 'failed' and state[j['id']]['attempts'] > RETRIES) for j in jobs):
            print('all jobs finished'); return
        time.sleep(30)

if __name__ == '__main__':
    main()
