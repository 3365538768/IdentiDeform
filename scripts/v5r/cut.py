"""V5r deadline cut (coordinator priority: 1 = GT + 3 main twins on the V2 grid, 2 = extended depths, 3 = vision 3-seed twin).
  python3 scripts/v5r/cut.py plan            -> docs/v5r/CUT.json (which jobs to stop) + removes them from all job lists / the queue
  python3 scripts/v5r/cut.py kill            -> run ON EACH HOST: kills this host's processes of the cut jobs (wrapper, squeeze_run,
                                                and the MPM child identified by its cache directory), nothing else
Cut: every prior_vis3 job; every GT job whose V2-grid depths are done but that still has >= 2 extended depths to go.
State files stay (resumable); interrupted in-flight sims are simply lost.
"""
import json
import os
import signal
import sys
from pathlib import Path

R = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test')
EXT = ['0.400', '0.480', '0.560', '0.640']
CUT_LAWMEAN = os.environ.get('V5R_CUT_LAWMEAN') == '1'  # 05:20 second cut: class-mean jobs with <= 2 sims done


def plan():
    cut = []
    for tw in sorted((R / 'data/v5r/runs').glob('*/*/twin.json')):
        m = tw.parent.name; i = tw.parent.parent.name; key = json.loads(tw.read_text())['cache']
        if (tw.parent / 'state.json').exists():
            continue  # finished
        try:
            runs = json.loads((tw.parent.parent / 'cache' / key / 'state.json').read_text())['runs']
        except (OSError, ValueError):
            runs = {}
        if m == 'prior_vis3' or (m == 'gt' and sum(k in runs for k in EXT) <= 2) or \
           (CUT_LAWMEAN and m == 'lawmean_pred' and len(runs) <= 2):
            cut.append({'index': i, 'method': m, 'cache': key, 'sims_done': len(runs)})
    ids = {f"v5r_{c['method']}_{c['index']}" for c in cut}
    for f in R.glob('docs/v5r/jobs_*.json'):
        if 'WITHDRAWN' in f.name:
            continue
        L = json.loads(f.read_text()); f.write_text(json.dumps([j for j in L if j['id'] not in ids], indent=1))
    q = json.loads((R / 'docs/v5r/queue.json').read_text()); q['jobs'] = [j for j in q['jobs'] if j['id'] not in ids]
    (R / 'docs/v5r/queue.json').write_text(json.dumps(q, indent=1))
    old = json.loads((R / 'docs/v5r/CUT.json').read_text())['jobs'] if (R / 'docs/v5r/CUT.json').exists() else []
    seen = {(c['index'], c['method']) for c in old}
    cut = old + [c for c in cut if (c['index'], c['method']) not in seen]
    (R / 'docs/v5r/CUT.json').write_text(json.dumps({'reason': __doc__.splitlines()[0], 'jobs': cut}, indent=1))
    print(len(cut), 'cut:', {m: sum(c['method'] == m for c in cut) for m in ('gt', 'prior_vis3', 'lawmean_pred')})


def kill():
    cut = json.loads((R / 'docs/v5r/CUT.json').read_text())['jobs']
    keys = [f"cache/{c['cache']}/" for c in cut]
    sigs = [f"--index {int(c['index'])} --method {c['method']} " for c in cut]
    n = 0
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            cmd = (Path('/proc') / pid / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace') + ' '
        except OSError:
            continue
        if 'v5r' not in cmd or 'cut.py' in cmd:
            continue
        if any(k in cmd for k in keys) or any(s in cmd for s in sigs):
            try:
                os.kill(int(pid), signal.SIGKILL); n += 1
            except OSError:
                pass
    print(os.uname().nodename, 'killed', n)


if __name__ == '__main__':
    {'plan': plan, 'kill': kill}[sys.argv[1]]()
