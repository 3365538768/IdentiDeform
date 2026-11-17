"""V5g: reorder the NOT-started jobs of each pool list sample-major (sample order of docs/v5r/SAMPLES.json; within a sample GT, vt,
vision, class mean) so that whole samples complete. Started jobs keep their place at the front (never relaunched)."""
import json, re
R = '/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test/'
order = {i: k for k, i in enumerate(json.load(open(R + 'docs/v5r/SAMPLES.json'))['indices'])}
arm = {'gt': 0, 'prior_vt': 1, 'prior_vision': 2, 'lawmean_pred': 3}
for lst, logs in (('jobs_main.json', ['pool_main.log']), ('jobs_a3b.json', ['pool_a3b.log'])):
    started = set()
    for lg in logs:
        started |= set(re.findall(r"start (\S+) gpu", open(R + 'docs/v5g/logs/' + lg).read()))
    L = json.load(open(R + 'docs/v5g/' + lst))
    def key(j):
        p = j['id'].split('_'); i = int(p[-3]); m = '_'.join(p[1:-3])
        return (order[i], arm[m], j['id'])
    a = [j for j in L if j['id'] in started]; b = sorted([j for j in L if j['id'] not in started], key=key)
    json.dump(a + b, open(R + 'docs/v5g/' + lst, 'w'), indent=1); print(lst, 'started', len(a), 'reordered', len(b))
