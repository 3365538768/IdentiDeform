"""V5g: move every sample with no started job on the a800-4 pool (all its jobs, all arms) to an a800-3 pool list."""
import json, re
R = '/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test/'
started = set(re.findall(r"start (\S+) gpu", open(R + 'docs/v5g/logs/pool_main.log').read()))
L = json.load(open(R + 'docs/v5g/jobs_main.json'))
smp = lambda j: j['id'].split('_')[-3]
busy = {smp(j) for j in L if j['id'] in started}
a3 = [j for j in L if smp(j) not in busy]; a4 = [j for j in L if smp(j) in busy]
json.dump(a4, open(R + 'docs/v5g/jobs_main.json', 'w'), indent=1); json.dump(a3, open(R + 'docs/v5g/jobs_a3.json', 'w'), indent=1)
print('a4 samples', len(busy), 'jobs', len(a4), '| a3 jobs', len(a3))
