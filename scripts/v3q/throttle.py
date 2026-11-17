"""V3q throttle (run ON the host): keep the oldest K v3q workers per GPU running and pause (SIGSTOP), resume (SIGCONT)
or terminate (SIGTERM, resumable from state.json) the others' whole process groups.
  python3 scripts/v3q/throttle.py --host a4 --keep 1 --action stop
  python3 scripts/v3q/throttle.py --host a4 --action cont        (resume all paused)
Logged to docs/v3q/DEVIATIONS.md and docs/v3q/throttle_<host>.json."""
import argparse, json, os, signal, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def workers():
    out = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'scripts/v3q/worker.py' not in cmd or cmd.startswith('bash'):
                continue
            env = dict(kv.split('=', 1) for kv in (p / 'environ').read_bytes().decode(errors='replace').split('\0') if '=' in kv)
            st = (p / 'stat').read_text().split()
            out.append({'pid': int(p.name), 'pgid': os.getpgid(int(p.name)), 'gpu': env.get('CUDA_VISIBLE_DEVICES'), 'start': int(st[21]),
                        'state': st[2]})
        except (OSError, ValueError):
            continue
    return out


def task_of(pgid):
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if os.getpgid(int(p.name)) == pgid:
                    cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
                    if 'c1_refine.py' in cmd and not cmd.startswith('bash'):
                        return cmd.split('c1_refine.py', 1)[1].strip()
            except OSError:
                pass
    return None


def main():
    a = argparse.ArgumentParser(); a.add_argument('--host', required=True); a.add_argument('--keep', type=int, default=1)
    a.add_argument('--action', choices=('stop', 'cont', 'term'), required=True)
    a.add_argument('--match', default=None, help='only workers whose c1_refine args contain this (e.g. lawmean_best3); --keep ignored'); args = a.parse_args()
    rec = ROOT / f'docs/v3q/throttle_{args.host}.json'
    R = json.loads(rec.read_text()) if rec.exists() else {'paused': []}
    now = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    acted = []
    if args.action == 'cont':
        for g in R['paused']:
            try:
                os.killpg(g['pgid'], signal.SIGCONT); acted.append(g)
            except OSError:
                pass
        R['paused'] = []
    else:
        by = {}
        for w in workers():
            by.setdefault(w['gpu'], []).append(w)
        for gpu, ws in sorted(by.items()):
            ws.sort(key=lambda w: w['start'])
            for w in (ws if args.match else ws[args.keep:]):
                w['task'] = task_of(w['pgid'])
                if args.match and (w['task'] is None or args.match not in w['task']):
                    continue
                os.killpg(w['pgid'], signal.SIGSTOP if args.action == 'stop' else signal.SIGTERM)
                if args.action == 'stop':
                    R['paused'].append(w)
                acted.append(w)
    rec.write_text(json.dumps(R, indent=1))
    with (ROOT / 'docs/v3q/DEVIATIONS.md').open('a') as f:
        f.write(f"- {now} [{args.host}] {args.action} {len(acted)} v3q searches (keep {args.keep}/GPU for stop/term): "
                f"{', '.join(str(w.get('task')) for w in acted)[:3000]}\n")
    print(args.action, len(acted))


if __name__ == '__main__':
    main()
