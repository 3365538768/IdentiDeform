"""Read-only progress dashboard for R/V2-line experiments, served from A800-2.

  .venv/bin/python monitor_r/server.py --port 18897
  # client: ssh -N -L 18897:127.0.0.1:18897 a800-2  ->  http://localhost:18897
Only reads project files and nvidia-smi; it never starts or stops jobs.
"""
import argparse
import json
import shutil
import statistics
import subprocess
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
_cache = {}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def jsonl(path, limit=None):
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-limit:] if limit else lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def pearson(a, b):
    if len(a) < 3:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    sa = sum((x - ma) ** 2 for x in a) ** .5; sb = sum((y - mb) ** 2 for y in b) ** .5
    return None if not sa or not sb else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def prediction_stats(directory):
    """Law accuracy and within-law E/nu correlation for one validation epoch (cached by mtime)."""
    path = directory / 'predictions.jsonl'
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return None
    if key in _cache:
        return _cache[key]
    rows = jsonl(path)
    if not rows:
        return None
    result = {'law_accuracy': sum(r['predicted_law_id'] == r['law_id'] for r in rows) / len(rows)}
    for j, name in [(0, 'E'), (1, 'nu')]:
        rs = []
        for law in sorted({r['law_id'] for r in rows}):
            sel = [r for r in rows if r['law_id'] == law and r['target_mask'][j]]
            value = pearson([r['prediction_z'][j] for r in sel], [r['target_z'][j] for r in sel])
            if value is not None:
                rs.append(value)
        result[f'within_law_r_{name}'] = statistics.fmean(rs) if rs else None
    _cache[key] = result
    return result


def gpus():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    keys = ['index', 'util', 'mem_used', 'mem_total', 'temp', 'power']
    return [dict(zip(keys, [float(x) for x in line.split(',')])) for line in out.strip().splitlines()]


def remote_gpus():
    """GPU snapshots written by scripts/gpu_status_writer.sh on other hosts (shared filesystem)."""
    out = []
    for f in sorted((ROOT / '.cache').glob('gpu_*.json')):
        d = read(f, None)
        if d:
            d['age_s'] = time.time() - d.get('updated', 0)
            out.append(d)
    return out


def sample_progress(spec):
    """Generation jobs: count finished sample dirs (<dir>/<index>/*/sample_pack.npz) of this shard."""
    root = ROOT / spec['dir']; n = spec.get('num_shards', 1); k = spec.get('shard', 0)
    done = sum(1 for d in root.glob('[0-9]*') if int(d.name) % n == k and any(d.glob('*/sample_pack.npz'))) if root.exists() else 0
    total = len(range(k, spec['total'], n))
    return done, total


def queue_specs():
    return {j['id']: j for q in ROOT.glob('docs/*/queue.json') for j in read(q, {}).get('jobs', [])}


def job_view(job_id, s, spec=None):
    if spec and spec.get('progress'):
        done, total = sample_progress(spec['progress'])
        rate = None
        if s.get('status') == 'running' and s.get('started_unix') and done:
            rate = (time.time() - s['started_unix']) / max(done - spec['progress'].get('done_at_start', 0), 1)
        return {'id': job_id, 'phase': s.get('phase'), 'group': s.get('group'), 'seed': s.get('seed'),
                'status': s.get('status'), 'gpu': s.get('gpu'), 'attempts': s.get('attempts'),
                'started_unix': s.get('started_unix'), 'epoch': done, 'epochs': total, 'unit': '样本',
                'train_phase': 'generation', 'val_curve': [], 'loss_curve': [], 'best_val_mae': None,
                'eta_seconds_upper': None, 'updated_unix': time.time()}
    cfg = read(ROOT / s.get('config', ''), {}) if s.get('config') else {}
    out = ROOT / cfg.get('train', {}).get('output', f'runs/{job_id}')
    timing = jsonl(out / 'timing.jsonl')
    train = jsonl(out / 'train.jsonl', 400)
    status = read(out / 'status.json', {})
    epochs = cfg.get('train', {}).get('epochs')
    best = None; best_epoch = None
    for t in timing:
        if t.get('val_mae') is not None and t.get('phase') == 'joint_training' and (best is None or t['val_mae'] < best):
            best, best_epoch = t['val_mae'], t['epoch']
    stats = None
    if best_epoch is not None:
        stats = prediction_stats(out / f'val_epoch_{best_epoch - 1:04d}')
    step = train[-1] if train else {}
    cycle = [t['cycle_seconds'] for t in timing[-3:] if t.get('cycle_seconds')]
    remaining = None
    if cycle and epochs and s.get('status') == 'running':
        remaining = statistics.fmean(cycle) * max(epochs - len(timing), 0)
    stride = max(1, len(train) // 150)
    return {'id': job_id, 'phase': s.get('phase'), 'group': s.get('group'), 'seed': s.get('seed'),
            'status': s.get('status'), 'gpu': s.get('gpu'), 'attempts': s.get('attempts'),
            'started_unix': s.get('started_unix'), 'train_status': status.get('status'),
            'stop_reason': status.get('reason'), 'epoch': len(timing), 'epochs': epochs,
            'train_phase': step.get('phase'), 'step': step.get('step'),
            'val_curve': [[t['epoch'], t.get('val_mae'), t.get('phase')] for t in timing],
            'loss_curve': [[r['step'], r.get('parameter')] for r in train[::stride]],
            'best_val_mae': best, 'best_epoch': best_epoch, 'eta_seconds_upper': remaining,
            'updated_unix': step.get('updated_unix'), **(stats or {})}


def snapshot():
    state = read(ROOT / 'docs/rq_state.json', {'jobs': {}})
    specs = queue_specs()
    jobs = [job_view(k, v, specs.get(k)) for k, v in state['jobs'].items()]
    groups = {}
    for j in jobs:
        if j['best_val_mae'] is not None:
            g = groups.setdefault(f"{j['phase']}:{j['group']}", {'phase': j['phase'], 'group': j['group'], 'runs': []})
            g['runs'].append(j)
    summary = []
    for g in groups.values():
        def agg(key):
            values = [r[key] for r in g['runs'] if r.get(key) is not None]
            return [statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0., len(values)] if values else None
        summary.append({'phase': g['phase'], 'group': g['group'], 'mae': agg('best_val_mae'),
                        'law_accuracy': agg('law_accuracy'), 'within_law_r_E': agg('within_law_r_E'),
                        'within_law_r_nu': agg('within_law_r_nu'),
                        'finished': sum(r['status'] == 'done' for r in g['runs'])})
    caches = [read(p, {}) | {'name': f'{p.parent.name}/{p.stem}'} for p in sorted(ROOT.glob('docs/*/cache_*.json'))]
    disk = shutil.disk_usage(ROOT)
    return {'now': time.time(), 'queue_updated_unix': state.get('updated_unix'), 'paused': state.get('paused'),
            'gpus': gpus(), 'remote_gpus': remote_gpus(), 'extra_progress': read(ROOT / '.cache/progress_extra.json', {}), 'jobs': jobs, 'summary': summary, 'caches': caches,
            'disk': {'free_gb': disk.free / 1e9, 'total_gb': disk.total / 1e9},
            'notes': read(ROOT / 'docs/notes_r.json', [])}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split('?')[0].rstrip('/') or '/'
        if path.endswith('/api/state'):
            body = json.dumps(snapshot()).encode(); kind = 'application/json'
        elif path.endswith('/favicon.ico'):
            self.send_error(404); return
        elif '/review_v3' in path:  # static review pages (docs/review_v3/: index.html + videos)
            base = (ROOT / 'docs/review_v3').resolve(); sub = path.split('/review_v3', 1)[1].lstrip('/') or 'index.html'
            f = (base / sub).resolve()
            if not f.is_relative_to(base) or not f.is_file():
                self.send_error(404); return
            body = f.read_bytes(); kind = {'.mp4': 'video/mp4', '.html': 'text/html; charset=utf-8', '.jpg': 'image/jpeg', '.png': 'image/png', '.json': 'application/json'}.get(f.suffix, 'application/octet-stream')
        elif path.endswith('/rebuttal'):  # static rebuttal evidence page (also pushed to GitHub)
            body = (ROOT / 'docs/rebuttal/index.html').read_bytes(); kind = 'text/html; charset=utf-8'
        else:  # any other path (incl. code-server /absproxy/<port>/ prefixes) serves the page
            body = (HERE / 'index.html').read_bytes(); kind = 'text/html; charset=utf-8'
        self.send_response(200); self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_POST(self):  # only the review page writes: object marks -> docs/review_v3/object_marks.json
        path = self.path.split('?')[0].rstrip('/')
        n = int(self.headers.get('Content-Length', 0))
        if not path.endswith('/review_v3/api/object_marks') or n > 2_000_000:
            self.send_error(404); return
        try:
            data = json.loads(self.rfile.read(n))
            assert isinstance(data, dict)
        except Exception:
            self.send_error(400); return
        f = ROOT / 'docs/review_v3/object_marks.json'
        f.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        body = json.dumps({'saved': len(data)}).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--port', type=int, default=18897)
    p.add_argument('--host', default='127.0.0.1'); a = p.parse_args()
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
