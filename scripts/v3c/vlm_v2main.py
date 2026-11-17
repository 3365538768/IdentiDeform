"""V3c: Qwen3-VL-8B zero-shot on V2 main val, R5 protocol (scripts/vlm_baseline.py) adapted to V2.

Identical to R5: model, 3 views x 8 uniformly spaced frames from the sample pack, interleaved
camera/time captions, greedy decoding (max_new_tokens=160), first-JSON parsing, fallbacks
(unparseable field -> train mean z=0; unparseable material -> law 0). The prompt template is R5's
with ONE change: the parameter-range sentence states V2 main's train-split ranges (R5 stated V1's).
Scoring is in V2's specific space: E/rho and sigma_y/rho are formed from the VLM's own E, rho and
yield answers (needs both; otherwise z=0 fallback, counted). No adaptation, no V2 training data
other than the range sentence.
  .venv/bin/python scripts/v3c/vlm_v2main.py --shard 0 --num-shards 4 --output runs/v3c_vlm_qwen3vl8b_val
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import vlm_baseline as r5
from fieldbridge.data import ParameterCodec, specific_parameters

R5_RANGES = "Parameter ranges in this dataset: Young's modulus E from 1e3 to 1.2e7 Pa; Poisson's ratio nu from 0.24 to 0.46; density rho from 100 to 10000 kg/m^3; yield stress from 1e2 to 1e5 Pa (only metal and plasticine)."


def g(x):
    return f'{x:.2g}'


def v2_prompt(train):
    rng = {k: (min(v), max(v)) for k in ['E', 'nu', 'rho', 'sigma_y']
           for v in [[r['parameters'][k] for r in train if r['parameters'][k] is not None]]}
    ranges = (f"Parameter ranges in this dataset: Young's modulus E from {g(rng['E'][0])} to {g(rng['E'][1])} Pa; "
              f"Poisson's ratio nu from {rng['nu'][0]:.2f} to {rng['nu'][1]:.2f}; density rho from {g(rng['rho'][0])} to {g(rng['rho'][1])} kg/m^3; "
              f"yield stress from {g(rng['sigma_y'][0])} to {g(rng['sigma_y'][1])} Pa (only metal and plasticine).")
    if R5_RANGES not in r5.PROMPT:
        raise ValueError('R5 prompt changed; refusing to adapt')
    return r5.PROMPT.replace(R5_RANGES, ranges), rng


def main():
    p = argparse.ArgumentParser(); p.add_argument('--split', default='val'); p.add_argument('--output', required=True)
    p.add_argument('--manifest', default='data/manifests/v2main.jsonl')
    p.add_argument('--stats', default='data/manifests/v2main.specific.stats.json')
    p.add_argument('--views', type=int, default=3); p.add_argument('--frames', type=int, default=8)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--num-shards', type=int, default=1); a = p.parse_args()
    all_rows = [json.loads(l) for l in open(a.manifest)]
    prompt, ranges = v2_prompt([r for r in all_rows if r['split'] == 'train'])
    rows = [r for r in all_rows if r['split'] == a.split]
    rows = [r for r in rows if int(hashlib.sha256(r['sample_id'].encode()).hexdigest(), 16) % a.num_shards == a.shard]
    codec = ParameterCodec(json.loads(Path(a.stats).read_text()))
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    pred_path = out / f'predictions.shard{a.shard}of{a.num_shards}.jsonl'
    done = {json.loads(l)['sample_id'] for l in pred_path.read_text().splitlines() if l.strip()} if pred_path.exists() else set()
    processor = AutoProcessor.from_pretrained(r5.MODEL)
    model = AutoModelForImageTextToText.from_pretrained(r5.MODEL, dtype=torch.bfloat16).cuda().eval()
    (out / 'protocol.json').write_text(json.dumps({'model': r5.MODEL, 'views': a.views, 'frames_per_view': a.frames,
        'decoding': 'greedy, max_new_tokens=160', 'prompt_template': prompt, 'train_ranges': ranges,
        'difference_from_r5': 'range sentence uses V2 main train ranges', 'adaptation': 'none (zero-shot)',
        'scoring': 'specific space: E/(rho/1000), sigma_y/(rho/1000) from VLM answers; standardized with V2 train stats',
        'fallback': 'unparseable field (or missing rho for specific quantities) -> train mean (z=0); unparseable material -> law 0'}, indent=2))
    started = time.time()
    for n, row in enumerate(r for r in rows if r['sample_id'] not in done):
        views, seconds = r5.frames(row, a.views, a.frames)
        content = []
        for v, sequence in enumerate(views):
            for image, t in sequence:
                content += [{'type': 'text', 'text': f'camera {v + 1}, t={t:.2f}s'}, {'type': 'image', 'image': image}]
        content.append({'type': 'text', 'text': prompt.format(views=len(views), frames=a.frames, seconds=seconds, action=row['action'])})
        inputs = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        text = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        parsed = r5.parse(text) or {}
        target, mask = codec.encode(specific_parameters(row['parameters']))
        est = {k: parsed.get(k) for k in ['E', 'nu', 'rho', 'sigma_y']}
        if est['nu'] is not None:
            est['nu'] = min(max(est['nu'], -0.99), 0.499)
        for k in ['E', 'rho', 'sigma_y']:
            if est[k] is not None and est[k] < 0:
                est[k] = None
        spec = dict(est)
        if est['rho'] is not None and est['rho'] > 0:
            for k in ['E', 'sigma_y']:
                spec[k] = est[k] / (est['rho'] / 1000.) if est[k] is not None else None
        else:
            spec['E'] = spec['sigma_y'] = None
        from fieldbridge.data import transform_parameters
        y, valid = transform_parameters(spec)
        y, valid = torch.from_numpy(y), torch.from_numpy(valid)
        z = torch.where(valid, (y - codec.mean) / codec.std, 0.)
        record = {'sample_id': row['sample_id'], 'object_id': row['object_id'], 'source': row.get('source', ''),
                  'action': row['action'], 'law_id': row['law_id'],
                  'predicted_law_id': parsed.get('law') if parsed.get('law') is not None else 0,
                  'prediction_z': z.tolist(), 'logvar': [0., 0., 0., 0.], 'target_z': target.tolist(),
                  'target_mask': mask.tolist(), 'parsed': bool(parsed), 'z_fallback': (~valid).tolist(),
                  'raw_estimate': est, 'raw_answer': text}
        with pred_path.open('a') as f:
            f.write(json.dumps(record) + '\n')
        if n % 10 == 0:
            print(json.dumps({'done': len(done) + n + 1, 'total': len(rows), 'sec': time.time() - started}), flush=True)


if __name__ == '__main__':
    main()
