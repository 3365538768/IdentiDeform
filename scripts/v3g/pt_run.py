"""v3g: run one official PhysTwin stage headless, unmodified algorithm.
Deviations (visualisation only): visualize_pc -> no-op (no display on server), wandb disabled,
diff_gaussian_rasterization / pynput / pyrender import-stubbed (only used by GS rendering / keyboard playground,
not by the CMA-ES or first-order spring-mass optimisation)."""
import json, os, sys, time, types, runpy, argparse
REPO = '/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test/external/v3g_phystwin'
ap = argparse.ArgumentParser()
ap.add_argument('stage', choices=['cma', 'train', 'infer', 'eval_chamfer', 'eval_track'])
ap.add_argument('--case', default='single_push_sloth')
ap.add_argument('--timing', required=True)
a, rest = ap.parse_known_args()

def stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m
class _Unavailable:
    def __init__(self, *a, **k): raise RuntimeError('v3g stub: GS rasterizer not installed (not needed for optimisation)')
stub('diff_gaussian_rasterization', GaussianRasterizationSettings=_Unavailable, GaussianRasterizer=_Unavailable, SparseGaussianAdam=_Unavailable)
stub('pyrender'); stub('simple_knn'); stub('simple_knn._C', distCUDA2=_Unavailable)
kb = types.SimpleNamespace(Listener=_Unavailable, Key=types.SimpleNamespace(), KeyCode=types.SimpleNamespace())
stub('pynput', keyboard=kb); stub('pynput.keyboard', **vars(kb))
sys.path.insert(0, REPO)
sys.argv = [a.stage] + rest
t0 = time.time()
if a.stage in ('cma', 'train', 'infer'):
    import qqtt.utils.visualize as V
    import qqtt.data.real_data as RD, qqtt.engine.cma_optimize_warp as C, qqtt.engine.trainer_warp as T
    noop = lambda *x, **k: None
    for mod in (V, RD, C, T):
        mod.visualize_pc = noop
    import qqtt.utils as U; U.visualize_pc = noop
    T.wandb.Video = lambda *x, **k: None  # no video file is written (visualisation disabled)
    script = {'cma': 'optimize_cma.py', 'train': 'train_warp.py', 'infer': 'inference_warp.py'}[a.stage]
else:
    script = {'eval_chamfer': 'evaluate_chamfer.py', 'eval_track': 'evaluate_track.py'}[a.stage]
status = 'ok'; err = None
try:
    runpy.run_path(os.path.join(REPO, script), run_name='__main__')
except SystemExit as e:
    if e.code not in (0, None): status, err = 'failed', f'SystemExit {e.code}'
except BaseException as e:
    status, err = 'failed', f'{type(e).__name__}: {e}'
    import traceback; traceback.print_exc(file=sys.__stderr__)
dt = time.time() - t0
rec = dict(stage=a.stage, script=script, case=a.case, argv=rest, status=status, error=err, wall_s=dt,
           cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), end_unix=time.time())
try:
    import torch; rec['gpu'] = torch.cuda.get_device_name(0); rec['peak_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
except Exception: pass
with open(a.timing, 'a') as f: f.write(json.dumps(rec) + '\n')
print('V3G_STAGE_DONE', json.dumps(rec), file=sys.__stdout__, flush=True)
sys.exit(0 if status == 'ok' else 1)
