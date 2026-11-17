"""Zero-shot VLM material estimator (Qwen3-VL-8B-Instruct) on DeformFieldBench validation/test.

Protocol (recorded in every output): 3 views x 8 uniformly spaced frames (224 px) from the sample
pack, given as interleaved images with view/time captions; greedy decoding; the prompt states the
three material families and the dataset's parameter ranges; the first JSON object in the answer
is parsed. Unparseable answers fall back to the train-set mean (z=0) and are counted. No
fine-tuning or DeformFieldBench training data is used. Output matches evaluation predictions.jsonl.

  .venv/bin/python scripts/vlm_baseline.py --split val --shard 0 --num-shards 2 --output runs/r5_vlm_qwen3vl8b
"""
import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from fieldbridge.data import ParameterCodec

MODEL = 'weights/hf/Qwen__Qwen3-VL-8B-Instruct'
LAWS = {'jelly': 0, 'metal': 1, 'plasticine': 2}
PROMPT = """These images show one deformable object simulated with the Material Point Method, seen from {views} cameras at {frames} time steps over {seconds:.1f} seconds (captions give camera and time). The object is subjected to a '{action}' interaction.

Estimate its material. The material family is one of:
- jelly: elastic (fixed-corotated), no yield stress
- metal: elastoplastic (von Mises), has yield stress
- plasticine: elastoplastic with softening, has yield stress
Parameter ranges in this dataset: Young's modulus E from 1e3 to 1.2e7 Pa; Poisson's ratio nu from 0.24 to 0.46; density rho from 100 to 10000 kg/m^3; yield stress from 1e2 to 1e5 Pa (only metal and plasticine).

Answer with only one JSON object: {{"material": "jelly|metal|plasticine", "E": <Pa>, "nu": <number>, "rho": <kg/m^3>, "yield_stress": <Pa or null>}}"""


def frames(row, views, count):
    with np.load(row['pack'], allow_pickle=False) as p:
        rgb = p['rgb']
    idx = np.linspace(0, rgb.shape[1] - 1, count).round().astype(int)
    times = row.get('timestamps_seconds') or list(range(rgb.shape[1]))
    return [[(Image.fromarray(rgb[v, i].transpose(1, 2, 0)), float(times[i])) for i in idx] for v in range(min(views, rgb.shape[0]))], float(times[-1])


def parse(text):
    match = re.search(r'\{.*?\}', text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    out = {'law': LAWS.get(str(data.get('material', '')).strip().lower())}
    for key, name in [('E', 'E'), ('nu', 'nu'), ('rho', 'rho'), ('yield_stress', 'sigma_y')]:
        value = data.get(key)
        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        out[name] = value if value is not None and math.isfinite(value) else None
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--split', default='val'); p.add_argument('--output', required=True)
    p.add_argument('--manifest', default='data/manifests/deformfield_v1.jsonl')
    p.add_argument('--stats', default='data/manifests/deformfield_v1.stats.json')
    p.add_argument('--views', type=int, default=3); p.add_argument('--frames', type=int, default=8)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--num-shards', type=int, default=1); a = p.parse_args()
    from fieldbridge.paths import path_map, relocate_path
    mapping = path_map()
    rows = [json.loads(l) for l in open(a.manifest)]
    rows = [dict(r, pack=relocate_path(r['pack'], mapping)) for r in rows if r['split'] == a.split]
    rows = [r for r in rows if int(hashlib.sha256(r['sample_id'].encode()).hexdigest(), 16) % a.num_shards == a.shard]
    codec = ParameterCodec(json.loads(Path(a.stats).read_text()))
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    pred_path = out / f'predictions.shard{a.shard}of{a.num_shards}.jsonl'
    done = {json.loads(l)['sample_id'] for l in pred_path.read_text().splitlines()} if pred_path.exists() else set()
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    (out / 'protocol.json').write_text(json.dumps({'model': MODEL, 'views': a.views, 'frames_per_view': a.frames,
        'decoding': 'greedy, max_new_tokens=160', 'prompt_template': PROMPT, 'adaptation': 'none (zero-shot)',
        'fallback': 'unparseable field -> train mean (z=0); unparseable material -> law 0'}, indent=2))
    started = time.time()
    for n, row in enumerate(r for r in rows if r['sample_id'] not in done):
        views, seconds = frames(row, a.views, a.frames)
        content = []
        for v, sequence in enumerate(views):
            for image, t in sequence:
                content += [{'type': 'text', 'text': f'camera {v + 1}, t={t:.2f}s'}, {'type': 'image', 'image': image}]
        content.append({'type': 'text', 'text': PROMPT.format(views=len(views), frames=a.frames, seconds=seconds, action=row['action'])})
        messages = [{'role': 'user', 'content': content}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        text = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        parsed = parse(text) or {}
        target, mask = codec.encode(row['parameters'])
        estimate = {k: parsed.get(k) for k in ['E', 'nu', 'rho', 'sigma_y']}
        if estimate['nu'] is not None:
            estimate['nu'] = min(max(estimate['nu'], -0.99), 0.499)
        z, valid = codec.encode({k: (v if v is not None and (k == 'nu' or v >= 0) else None) for k, v in estimate.items()})
        z = torch.where(valid, z, torch.zeros_like(z))
        record = {'sample_id': row['sample_id'], 'object_id': row['object_id'], 'source': row.get('source', ''),
                  'action': row['action'], 'law_id': row['law_id'],
                  'predicted_law_id': parsed.get('law') if parsed.get('law') is not None else 0,
                  'prediction_z': z.tolist(), 'logvar': [0., 0., 0., 0.], 'target_z': target.tolist(),
                  'target_mask': mask.tolist(), 'parsed': bool(parsed), 'raw_answer': text}
        with pred_path.open('a') as f:
            f.write(json.dumps(record) + '\n')
        if n % 10 == 0:
            print(json.dumps({'done': len(done) + n + 1, 'total': len(rows), 'sec': time.time() - started}), flush=True)


if __name__ == '__main__':
    main()
