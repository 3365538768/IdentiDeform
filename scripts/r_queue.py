"""GPU job queue for R/V2-line experiments on A800-2.

Reads every docs/*/queue.json (so later phases can be appended), launches pending jobs on GPUs
with free slots and memory, and records state in docs/rq_state.json for the dashboard.
Job kinds: 'train' (fieldbridge.train --config) or 'cmd' (shell command run from the project root).
A job may list 'after': [ids] that must finish first. Honors docs/OPTIMIZATION_CONTROL.json pause.
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'docs/rq_state.json'
ENV = f'source {ROOT}/scripts/r1_env.sh'
# 2026-11-10: GPUs 0-3 were lent to other experiments (rsi-arena) and are free again; all 8 allowed.
ALLOWED_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]


def read(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write(path, value):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(value, indent=2)); os.replace(tmp, path)


def gpu_free_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=30).stdout
    return {int(i): int(m) for i, m in (line.split(',') for line in out.strip().splitlines())}


def host_available_gib():
    """Reclaimable memory (tmpfs/shm pages cannot be reclaimed without swap, so they are subtracted)."""
    info = {l.split(':')[0]: int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()}
    return (info['MemFree'] + info.get('Buffers', 0) + info.get('Cached', 0) - info.get('Shmem', 0)) / 2**20


# Memory guard (after an earlier whole-machine hang): no launches below LAUNCH_GIB of reclaimable host
# memory; below KILL_GIB the most recently started job is terminated (it is retried/resumed later).
LAUNCH_GIB, KILL_GIB = 200, 64


def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f'/proc/{pid}/cmdline').exists()
    except OSError:
        return False


def finished(job):
    if job['kind'] == 'train':
        status = read(ROOT / read(ROOT / job['config'], {}).get('train', {}).get('output', '') / 'status.json', {})
        return status.get('status') in {'training_finished', 'early_stopped'}
    return (ROOT / 'runs/rq_done' / job['id']).exists()


def launch(job, gpu):
    log = ROOT / 'runs/rq_logs' / f"{job['id']}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / 'runs/rq_done').mkdir(exist_ok=True)
    if job['kind'] == 'train':
        cfg = read(ROOT / job['config'], {})
        resume = ROOT / cfg['train']['output'] / 'last.pt'
        body = f".venv/bin/python -m fieldbridge.train --config {job['config']}"
        if resume.exists():  # restart after a crash/pause continues from the epoch checkpoint
            patched = ROOT / 'runs/rq_logs' / f"{job['id']}.resume.json"
            cfg['train']['resume'] = str(resume); patched.write_text(json.dumps(cfg, indent=2))
            body = f'.venv/bin/python -m fieldbridge.train --config {patched}'
    else:
        body = f"( {job['cmd']} ) && touch runs/rq_done/{job['id']}"
    command = f'cd {ROOT} && {ENV} && export CUDA_VISIBLE_DEVICES={gpu} && {body}'
    process = subprocess.Popen(['bash', '-c', command], stdout=log.open('a'), stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True)
    return process.pid


def main():
    lock = open(ROOT / 'runs/rq.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit('queue already running')
    while True:
        state = read(STATE, {'jobs': {}})
        queues = sorted(ROOT.glob('docs/*/queue.json'))
        jobs = [dict(job, phase=read(q, {}).get('phase', q.parent.name), max_per_gpu=read(q, {}).get('max_per_gpu', 2),
                     gpus=[g for g in read(q, {}).get('gpus', ALLOWED_GPUS) if g in ALLOWED_GPUS]) for q in queues for job in read(q, {}).get('jobs', [])]
        paused = read(ROOT / 'docs/OPTIMIZATION_CONTROL.json', {}).get('desired') == 'paused'
        free = gpu_free_mib()
        mem_gib = host_available_gib()
        if mem_gib < KILL_GIB:
            newest = max((v for v in state['jobs'].values() if v['status'] == 'running'), key=lambda v: v['started_unix'], default=None)
            if newest:
                try:
                    os.killpg(newest['pid'], 15)
                except OSError:
                    pass
                state.setdefault('memory_guard_kills', []).append({'unix': time.time(), 'pid': newest['pid'], 'available_gib': mem_gib})
        low_memory = mem_gib < LAUNCH_GIB
        for job in jobs:
            s = state['jobs'].setdefault(job['id'], {'status': 'pending', 'attempts': 0})
            s.update(phase=job['phase'], group=job.get('group'), seed=job.get('seed'), config=job.get('config'))
            if s['status'] == 'running' and not alive(s['pid']):
                s['status'] = 'done' if finished(job) else 'failed'; s['ended_unix'] = time.time()
            if s['status'] == 'pending' and finished(job):
                s['status'] = 'done'
        running = [s for s in state['jobs'].values() if s['status'] == 'running']
        for job in jobs:
            s = state['jobs'][job['id']]
            if s['status'] == 'failed' and s['attempts'] < job.get('retries', 1) + 1 and time.time() - s.get('ended_unix', 0) > 60:
                s['status'] = 'pending'
            if paused or low_memory or s['status'] != 'pending':
                continue
            if any(state['jobs'].get(d, {}).get('status') != 'done' for d in job.get('after', [])):
                continue
            need = job.get('mem_mib', 16000)
            slots = {g: sum(r.get('gpu') == g for r in running) for g in job['gpus']}
            options = [g for g in job['gpus'] if slots[g] < job['max_per_gpu'] and free.get(g, 0) > need + 4000]
            if not options:
                continue
            gpu = min(options, key=lambda g: (slots[g], -free[g]))
            s.update(status='running', gpu=gpu, pid=launch(job, gpu), started_unix=time.time(), attempts=s['attempts'] + 1)
            running.append(s); free[gpu] -= need
        state.update(updated_unix=time.time(), paused=paused, queue_pid=os.getpid(), host_available_gib=mem_gib, low_memory=low_memory)
        write(STATE, state)
        time.sleep(20)


if __name__ == '__main__':
    main()
