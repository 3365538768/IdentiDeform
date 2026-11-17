"""V5g a800-3 GPU re-allocation (07:31): the running a800-3 pool keeps only its started jobs (launches nothing new); every
not-started job goes to docs/v5g/jobs_a3b.json for a new pool restricted to GPUs 0,1,3,4."""
import json, re
R = '/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test/'
log = open(R + 'docs/v5g/logs/pool_a3.log').read()
started = re.findall(r"start (\S+) gpu (\d+)", log); exited = set(re.findall(r"exit (\S+) ", log))
st = {j for j, _ in started}
L = json.load(open(R + 'docs/v5g/jobs_a3.json'))
json.dump([j for j in L if j['id'] in st], open(R + 'docs/v5g/jobs_a3.json', 'w'), indent=1)
json.dump([j for j in L if j['id'] not in st], open(R + 'docs/v5g/jobs_a3b.json', 'w'), indent=1)
run567 = [j for j, g in started if g in ('5', '6', '7') and j not in exited]
print('started', len(st), 'moved to a3b', len(L) - len(st), 'running on GPUs 5/6/7:', len(run567))
