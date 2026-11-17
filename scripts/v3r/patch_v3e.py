"""V3r: parameterize scripts/v3e (V2 defaults unchanged) and port the stray-particle clamp into the squeeze path."""
import shutil
from pathlib import Path

R = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test')
bak = R / '.cache/keep/v3e_before_v3r'; bak.mkdir(parents=True, exist_ok=True)
for n in ('squeeze_main', 'squeeze_sim', 'squeeze_run', 'plan_eval', 'pool'):
    if not (bak / f'{n}.py').exists():
        shutil.copy2(R / f'scripts/v3e/{n}.py', bak / f'{n}.py')


def sub(p, old, new):
    p = R / p; t = p.read_text()
    assert t.count(old) == 1, (p, old, t.count(old))
    p.write_text(t.replace(old, new))


sub('scripts/v3e/squeeze_main.py', "su.build_press_boundary_conditions = squeeze\n",
    "if os.environ.get('FB_SQUEEZE_CLAMP') == '1':  # V3r: port of the V3 stray-particle fix (scripts/v2/contact_bc.py)\n"
    "    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v2'))\n"
    "    import contact_bc\n"
    "    contact_bc._clamp_particles()  # particles outside the grid interior are moved in (p2g index out of bounds)\n"
    "su.build_press_boundary_conditions = squeeze\n")
sub('scripts/v3e/squeeze_sim.py', "GRAVITY = 0.0",
    "DATA = os.environ.get('FB_GRIP_DATA', 'data/v2_main/val')  # V3r: data/v3_main/val\n"
    "TWINS = os.environ.get('FB_GRIP_TWINS', 'v2')               # V3r: v3 (scripts/v3r/twins.py)\n"
    "GRAVITY = 0.0")
sub('scripts/v3e/squeeze_sim.py', "g = json.loads((ROOT / f'data/v2_main/val/{index:06d}/config.json').read_text())",
    "g = json.loads((ROOT / f'{DATA}/{index:06d}/config.json').read_text())")
sub('scripts/v3e/squeeze_sim.py', '    """Material twin of each method (object geometry is the observed one for every method)."""\n',
    '    """Material twin of each method (object geometry is the observed one for every method)."""\n'
    "    if TWINS == 'v3':\n"
    "        sys.path.insert(0, str(ROOT / 'scripts/v3r'))\n"
    "        import twins\n"
    "        return twins.twin_params(index, method)\n")
sub('scripts/v3e/squeeze_sim.py', "env = dict(os.environ); env.pop('FB_ACTION', None); env['FB_SQUEEZE_D'] = repr(float(d))",
    "env = dict(os.environ); env.pop('FB_ACTION', None); env.pop('FB_BC_FIX', None); env['FB_SQUEEZE_D'] = repr(float(d))")
sub('scripts/v3e/squeeze_run.py', "gt = json.loads((ROOT / f'data/v2_main/val/{args.index:06d}/gt.json').read_text())",
    "gt = json.loads((ROOT / f'{S.DATA}/{args.index:06d}/gt.json').read_text())")
sub('scripts/v3e/plan_eval.py', "gt = json.loads((ROOT / f'data/v2_main/val/{int(sd.name):06d}/gt.json').read_text())",
    "gt = json.loads((ROOT / f'{S.DATA}/{int(sd.name):06d}/gt.json').read_text())")
sub('scripts/v3e/plan_eval.py', "a.add_argument('--out', default='docs/v3e/GRIPPER_TASK.json'); args = a.parse_args()",
    "a.add_argument('--out', default='docs/v3e/GRIPPER_TASK.json'); a.add_argument('--samples', default='docs/v3e/SAMPLES.json'); args = a.parse_args()")
sub('scripts/v3e/plan_eval.py', "only = set(json.loads((ROOT / 'docs/v3e/SAMPLES.json').read_text())['wave1'])",
    "only = set(json.loads((ROOT / args.samples).read_text())['wave1'])")
sub('scripts/v3e/pool.py', "a.add_argument('--retries', type=int, default=2); args = a.parse_args()",
    "a.add_argument('--retries', type=int, default=2); a.add_argument('--tag', default='v3e'); args = a.parse_args()")
sub('scripts/v3e/pool.py', "done_dir = ROOT / 'runs/v3e_done'", "done_dir = ROOT / f'runs/{args.tag}_done'")
sub('scripts/v3e/pool.py', "logs = ROOT / 'docs/v3e/logs'", "logs = ROOT / f'docs/{args.tag}/logs'")
sub('scripts/v3e/pool.py', "if (ROOT / 'docs/v3e/POOL_STOP').exists():", "if (ROOT / f'docs/{args.tag}/POOL_STOP').exists():")
sub('scripts/v3e/pool.py', "&& touch runs/v3e_done/{j['id']}", "&& touch runs/{args.tag}_done/{j['id']}")
print('patched')
