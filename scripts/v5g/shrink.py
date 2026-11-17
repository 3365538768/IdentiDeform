"""V5g deviation (08:45 UTC): drop grip levels 0.06 and 0.16 (both positions) from every NOT-started job -> 8 candidates
{0.03, 0.10, 0.25, 0.40} x {end, centre}. Analysis uses these 8 candidates for every sample (samples done earlier have all 12)."""
import json, re
R = '/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test/'
DROP = ('_0.06', '_0.16')
for lst, logs in (('jobs_main.json', ['pool_main.log']), ('jobs_a3b.json', ['pool_a3b.log']), ('jobs_a3c.json', ['pool_a3c.log'])):
    started = set()
    for lg in logs:
        started |= set(re.findall(r"start (\S+) gpu", open(R + 'docs/v5g/logs/' + lg).read()))
    L = json.load(open(R + 'docs/v5g/' + lst))
    keep = [j for j in L if j['id'] in started or not j['id'].endswith(DROP)]
    json.dump(keep, open(R + 'docs/v5g/' + lst, 'w'), indent=1); print(lst, len(L), '->', len(keep))
