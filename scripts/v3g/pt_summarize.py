import json, csv, time
R = '/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test'
W = R + '/runs/v3g_phystwin'
stages = [json.loads(l) for l in open(W + '/timing.jsonl')]
stages = [s for s in stages if s['status'] == 'ok']
ch = list(csv.DictReader(open(W + '/results/final_results.csv')))[0]
tr = list(csv.DictReader(open(W + '/results/final_track.csv')))[0]
man = json.load(open(R + '/data/external_datasets/v3g_phystwin/single_push_sloth/v3g_download_manifest.json'))
st = {s['stage']: round(s['wall_s'], 1) for s in stages}
out = dict(
  method='PhysTwin (official, github.com/Jianghanxiao/PhysTwin @81c71879, ICCV 2025)', status='completed (physics stages); appearance stage not run',
  case='single_push_sloth (data/different_types, released processed data, HF Jianghanxiao/PhysTwin rev ' + man['revision'] + ')',
  input='3 RealSense RGB-D views, 30 fps, 68 frames (train 0-47, test 47-68); hand-push interaction; processed point clouds + CoTracker 3D tracks + TRELLIS shape prior (released, not recomputed)',
  split=json.load(open(R + '/data/external_datasets/v3g_phystwin/single_push_sloth/split.json')),
  gpu='NVIDIA A800-SXM4-80GB (GPU 3, shared with other jobs at 100% util; timings are therefore upper bounds vs. an idle GPU)',
  env='uv venv .engine-envs/v3g_phystwin: python 3.10, torch 2.4.0+cu121, warp-lang 1.7.0, pytorch3d (cu121/pt240 wheel)',
  stages={
    'zero_order_cma_es': dict(script='optimize_cma.py', iterations=20, function_evals=220, wall_s=st['cma'],
        loss_first_iter=7.334990409815319e-04, loss_best=4.785453881160483e-04, note='official default max_iter=20, configs/real.yaml'),
    'first_order_train': dict(script='train_warp.py', iterations=200, wall_s=st['train'], loss_iter0=4.611491234873424e-04,
        loss_best=4.3936557279306254e-04, best_epoch=120, loss_iter199=4.54172500529081e-04,
        note='official 200 iters (configs/real.yaml); best checkpoint chosen on training loss at vis_interval=20 as in official code'),
    'inference': dict(script='inference_warp.py', wall_s=st['infer']),
    'evaluation': dict(scripts=['evaluate_chamfer.py', 'evaluate_track.py'], wall_s=round(st['eval_chamfer'] + st['eval_track'], 1)),
    'gaussian_appearance': dict(status='not run', reason='needs diff-gaussian-rasterization/simple-knn CUDA builds + export_gaussian_data (raw colour/depth frames, ~0.6 GB more data); appearance is irrelevant to physical-parameter cost; out of scope at verification scale'),
  },
  total_physics_wall_s=round(st['cma'] + st['train'], 1),
  total_physics_wall_min=round((st['cma'] + st['train']) / 60, 1),
  metrics_official_eval=dict(
    chamfer_L1_single_dir_train=float(ch['Train Chamfer Error']), chamfer_train_frames=int(ch['Train Frame Num']),
    chamfer_L1_single_dir_test=float(ch['Test Chamfer Error']), chamfer_test_frames=int(ch['Test Frame Num']),
    track_err_train_m=float(tr['Train Track Error']), track_err_test_m=float(tr['Test Track Error']),
    units='metres (world frame of the processed data); test = future-frame prediction, frames 47-68'),
  peak_gpu_memory='not measured by torch (Warp allocates outside the torch allocator); not separately measured',
  deviations_from_official=[
    'visualisation disabled: qqtt visualize_pc replaced by a no-op and wandb.Video stubbed (no display on server; README notes runs are faster with visualisation disabled)',
    'wandb disabled (WANDB_MODE=disabled)',
    'import-only stubs for diff_gaussian_rasterization, simple_knn, pynput, pyrender (used only by GS rendering / keyboard playground / visualisation, never called in the optimisation path)',
    'no source file of the official repo was modified; wrapper scripts/v3g/pt_run.py runs the official scripts via runpy',
  ],
  security=dict(data='downloaded by HTTP range from the official HF data.zip into a new dir data/external_datasets/v3g_phystwin; CRC verified; pickles scanned with pickletools before use: only numpy ndarray/_reconstruct/dtype globals', manifest='data/external_datasets/v3g_phystwin/single_push_sloth/v3g_download_manifest.json'),
  applicability_to_our_model='Not applicable: PhysTwin consumes multi-view RGB-D + hand-controller tracks of a real interaction and fits a per-scene spring-mass model (per-spring stiffness, no E/nu/sigma_y); there is no GT material parameter. Our feed-forward model takes rendered RGB video of a FieldBench-style loading and outputs continuum parameters. We report only cost (about 31 min of GPU optimisation per scene on a shared A800 vs. one forward pass for ours) and qualitative fit; no accuracy comparison is claimed.',
  files=dict(scripts=['scripts/v3g/pt_env.sh', 'scripts/v3g/pt_install.sh', 'scripts/v3g/pt_download.py', 'scripts/v3g/pt_run.py', 'scripts/v3g/pt_pipeline.sh', 'scripts/v3g/pt_summarize.py'],
    logs=['docs/v3g/logs/pt_pipeline_cma.log', 'docs/v3g/logs/pt_pipeline.log', 'docs/v3g/logs/pt_install.log'], run_dir='runs/v3g_phystwin', timing='runs/v3g_phystwin/timing.jsonl'),
  stages_raw=stages, generated_unix=time.time())
json.dump(out, open(R + '/docs/v3g/PHYSTWIN_OFFICIAL.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('total_physics_wall_min', 'metrics_official_eval')}, indent=1))
