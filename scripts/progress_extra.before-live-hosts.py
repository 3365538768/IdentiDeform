"""Aggregate progress of the v3d–v3i side experiments (queue jobs + nohup pools on a800-4) for monitor_r.
Writes .cache/progress_extra.json; keeps a short history to estimate remaining time."""
import json, glob, os, re, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.cache/progress_extra.json'

def rd(p, d=None):
    try: return json.loads(Path(p).read_text())
    except Exception: return d

def pool_stats(job_files, log_glob):
    ids = []
    for f in job_files:
        d = rd(f, [])
        d = d.get('jobs', d) if isinstance(d, dict) else d
        ids += [j['id'] for j in d if isinstance(j, dict) and 'id' in j]
    ids = list(dict.fromkeys(ids))
    started, ok, bad = set(), set(), set()
    for lf in glob.glob(str(ROOT / log_glob)):
        for line in open(lf, errors='ignore'):
            m = re.search(r' (start|exit) (\S+)(?: (-?\d+))?', line)
            if not m: continue
            if m.group(1) == 'start': started.add(m.group(2))
            elif m.group(3) == '0': ok.add(m.group(2))
            else: bad.add(m.group(2))
    done = len([i for i in ids if i in ok]); failed = len([i for i in ids if i in bad and i not in ok])
    running = len([i for i in ids if i in started and i not in ok and i not in bad])
    return dict(total=len(ids), done=done, running=running, failed=failed, pending=max(len(ids) - done - running - failed, 0))

def main():
    st = rd(ROOT / 'docs/rq_state.json', {'jobs': {}})['jobs']
    q = [v for v in st.values() if v.get('phase') == 'v3d']
    phases = {
        'v3d': dict(label='C1 test 复现（30 个 test 样本）', unit='任务', total=len(q), done=sum(v['status'] == 'done' for v in q),
                    running=sum(v['status'] == 'running' for v in q), failed=sum(v['status'] == 'failed' for v in q),
                    pending=sum(v['status'] == 'pending' for v in q), host='a800-2', deliverable='docs/v3d/C1_TEST_RESULTS.json'),
        'v3e': dict(label='夹爪数字孪生任务', unit='仿真', host='a800-2 + a800-4', deliverable='docs/v3e/GRIPPER_TASK.json',
                    **pool_stats(sorted(glob.glob(str(ROOT / 'docs/v3e/jobs*.json'))), 'docs/v3e/logs/pool*.log')),
        'v3h': dict(label='先验质量与修正加速', unit='搜索', host='a800-4 GPU 4–7', deliverable='docs/v3h/PRIOR_QUALITY_ZH.md',
                    **pool_stats([ROOT / 'docs/v3h/jobs.json'], 'docs/v3h/logs/pool*.log')),
    }
    ls = rd(ROOT / 'docs/v3i/launch_state.json', {}) or {}
    js = (rd(ROOT / 'docs/v3i/jobs.json', {}) or {}).get('jobs', [])
    stt = [ls.get(j['id'], {}).get('status', 'pending') for j in js]
    phases['v3i'] = dict(label='数据量曲线 + 视角数消融', unit='训练', host='a800-4', deliverable='docs/v3i/ABLATIONS.json',
                         total=len(js), done=stt.count('done'), running=stt.count('running'), failed=stt.count('failed'),
                         pending=len(js) - stt.count('done') - stt.count('running') - stt.count('failed'))
    pn = [rd(f, {}) for f in glob.glob(str(ROOT / 'docs/v3g/pn_results/*_v*.json')) if not f.endswith('_rs.json')]
    pn_done = sum(str(d.get('rc')) == '0' for d in pn)
    sg = rd(ROOT / 'docs/v3g/SPRINGGAUS_OFFICIAL.json', {}) or {}
    sg_done = sum(1 for v in (sg.get('scenes') or {}).values() if v.get('status') == 'completed')
    pt_done = 1 if (ROOT / 'docs/v3g/PHYSTWIN_OFFICIAL.json').exists() else 0
    phases['v3g'] = dict(label='官方方法对齐（PAC-NeRF 12 版本 / Spring-Gaus 3 场景 / PhysTwin 1 场景）', unit='运行', host='a800-2（可迁 a800-4）',
                         deliverable='docs/v3g/OFFICIAL_BASELINES.json', total=12 + 3 + 1, done=pn_done + sg_done + pt_done,
                         running=sum(d.get('rc') is None for d in pn), failed=0, pending=0)
    phases['v3g']['pending'] = max(phases['v3g']['total'] - phases['v3g']['done'] - phases['v3g']['running'], 0)
    for ph, label, host, deliv in [('v3j', '更强无信息基线 + 放开硬化/ν 的修正', 'a800-4', 'docs/v3j/RESULTS.json'),
                                   ('v3k', 'Spring-Gaus 目标内修正 / 重建几何 / 失配 / 校准', 'a800-2 队列', 'docs/v3k/RESULTS.json'),
                                   ('v3l', 'Spring-Gaus/PhysTwin 按其指标对齐（Chamfer、PSNR/SSIM）', 'a800-2 队列', 'docs/v3l/RESULTS.json')]:
        q = [v for k, v in st.items() if v.get('phase') == ph or k.startswith(ph + '_')]
        if q:
            d = dict(total=len(q), done=sum(v['status'] == 'done' for v in q), running=sum(v['status'] == 'running' for v in q),
                     failed=sum(v['status'] == 'failed' for v in q), pending=sum(v['status'] == 'pending' for v in q))
        else:
            d = pool_stats(sorted(glob.glob(str(ROOT / f'docs/{ph}/jobs*.json'))), f'docs/{ph}/logs/pool*.log')
        phases[ph] = dict(label=label, unit='任务', host=host, deliverable=deliv, **d)
    # V3 data generation: count finished samples (result.json exit 0) against the targets
    tgt = {'val': 283, 'test': 300, 'train': 4000}; ok = bad = 0
    for sp in tgt:
        for f in glob.glob(str(ROOT / f'data/v3_main/{sp}/*/result.json')):
            e = (rd(f, {}) or {}).get('exit'); ok += e == 0; bad += e not in (0, None)
    run4 = pool_stats([], 'docs/v3n/logs/pool_a4.log')
    qn = [v for k, v in st.items() if k.startswith('v3n_gen_')]
    running = sum(v['status'] == 'running' for v in qn)
    try:
        lines = open(ROOT / 'docs/v3n/logs/pool_a4.log').read().splitlines()
        running += len({l.split(' start ')[1].rsplit(' gpu', 1)[0] for l in lines if ' start ' in l}) - len({l.split(' exit ')[1].rsplit(' ', 3)[0] for l in lines if ' exit ' in l})
    except Exception:
        pass
    total = sum(tgt.values())
    phases['v3n'] = dict(label='V3 数据生成（接触式动作；val 283 / test 300 / train 4000）', unit='仿真', host='a800-4 池（偶数）+ a800-2 队列（奇数），各 8 卡',
                         deliverable='data/v3_main', total=total, done=ok, running=max(running, 0), failed=bad, pending=max(total - ok - bad - max(running, 0), 0))
    for s_ in (rd(ROOT / 'docs/v3_pipeline/PLAN.json', {}) or {}).get('stages', []):
        tot = int(s_.get('total') or 0); dn = int(s_.get('done') or 0)
        phases[s_['id']] = dict(label=s_['label'], unit=s_.get('unit', '步'), host=s_.get('host', ''), deliverable=s_.get('deliverable', ''),
                                   total=tot, done=dn, running=int(s_.get('status') == 'running'), failed=int(s_.get('status') == 'failed'),
                                   pending=max(tot - dn, 0), status=s_.get('status'))
    figs = glob.glob(str(ROOT / 'docs/paper_figs/*.pdf'))
    phases['figs'] = dict(label='论文图（docs/paper_figs）', unit='张 PDF', host='a800-2', deliverable='docs/paper_figs/SOURCES.md',
                          total=max(len(figs), 7), done=len(figs), running=0, failed=0, pending=max(7 - len(figs), 0))
    now = time.time(); old = rd(OUT, {}) or {}; hist = old.get('history', {})
    for k, p in phases.items():
        h = [x for x in hist.get(k, []) if now - x[0] < 3 * 3600] + [[now, p['done']]]
        hist[k] = h[-200:]
        p['deliverable_exists'] = (ROOT / p['deliverable']).exists()
        span = [x for x in h if now - x[0] < 5400]
        rate = (span[-1][1] - span[0][1]) / max(span[-1][0] - span[0][0], 1) if len(span) > 1 else 0
        remain = p['total'] - p['done']
        p['rate_per_h'] = rate * 3600
        p['eta_unix'] = now + remain / rate if rate > 0 and remain > 0 else (now if remain <= 0 else None)
    tmp = OUT.with_suffix('.tmp'); tmp.write_text(json.dumps({'updated': now, 'phases': phases, 'history': hist})); os.replace(tmp, OUT)

if __name__ == '__main__':
    main()
