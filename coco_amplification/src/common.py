"""Shared helpers for the COCO 2-object spatial-prompt amplification experiment.

This file intentionally preserves the notebook-derived numerical logic. The
script-facing package imports these routines through smaller modules in
``coco_amplification.src``.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from collections import Counter, defaultdict
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from coco_amplification.src import coco_cache


DTYPE_ARTIFACT_DIR = {'fp32': 'FP32', 'bf16': 'BF16'}
PROGRESS_KWARGS = {
    'ncols': 80,
    'dynamic_ncols': False,
    'leave': False,
    'ascii': True,
    'mininterval': 1.0,
    'bar_format': '{l_bar}{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
}


def progress(iterable, **kwargs):
    opts = dict(PROGRESS_KWARGS)
    opts.update(kwargs)
    return tqdm(iterable, **opts)


def torch_dtype_from_name(name):
    if name == 'fp32':
        return torch.float32
    if name == 'bf16':
        return torch.bfloat16
    raise ValueError(f'bad dtype {name!r}')


def dtype_artifact_dir(name):
    return DTYPE_ARTIFACT_DIR[name]


# ---------------------------------------------------------------------------
# Axis constants + caption parsing  (qwen COCO cells 5 / 10b)
# ---------------------------------------------------------------------------
AXIS_BY_DIRECTION = {'left': 'horizontal', 'right': 'horizontal',
                     'above': 'vertical', 'below': 'vertical'}
VALID_BY_AXIS = {'horizontal': {'left', 'right'}, 'vertical': {'above', 'below'}}
LABELS_BY_AXIS = {'horizontal': ('left', 'right'), 'vertical': ('above', 'below')}


def parse_spatial_caption(cap):
    """Parse a COCO caption into ``(subj, direction, obj, axis)`` or ``None``."""
    m = re.match(r'A photo of (.+?) to the (left|right) of (.+)$', cap)
    if m:
        subj, direction, obj = m.groups()
        return subj, direction, obj, AXIS_BY_DIRECTION[direction]
    m = re.match(r'A photo of (.+?) (above|below)(?: of)? (.+)$', cap)
    if m:
        subj, direction, obj = m.groups()
        return subj, direction, obj, AXIS_BY_DIRECTION[direction]
    return None


def build_spatial_data(qa):
    """Filter the QA list to spatial left/right/above/below entries."""
    spatial_data = []
    for pair_idx, (img_id, correct, distractor) in enumerate(qa):
        parsed = parse_spatial_caption(correct)
        if parsed is None:
            continue
        subj, truth, obj, axis = parsed
        spatial_data.append({'image_id': img_id, 'pair_idx': pair_idx, 'correct': correct,
                             'distractor': distractor, 'subj': subj, 'obj': obj,
                             'truth': truth, 'axis': axis})
    return spatial_data


def strip_article(s):
    return re.sub(r'^(a|an|the)\s+', '', s.strip(), flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# COCO data download  (qwen COCO cell 3)
# ---------------------------------------------------------------------------
def download_coco_data(coco_dir, *,
                       val2017_url='http://images.cocodataset.org/zips/val2017.zip'):
    """Load ``coco_qa_two_obj.json`` and ensure ``val2017/`` exists under ``coco_dir``.

    The QA file is bundled next to the package. ``val2017/`` images are fetched
    into the shared COCO cache if missing.
    """
    os.makedirs(coco_dir, exist_ok=True)
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'coco_qa_two_obj.json')
    img_dir = str(coco_cache.ensure_val2017(coco_dir))

    if not os.path.exists(json_path):
        raise FileNotFoundError(f'{json_path} missing.')
    with open(json_path) as f:
        qa = json.load(f)
    print(f'JSON entries: {len(qa)}  |  val2017 images: {len(os.listdir(img_dir))}')
    return qa, img_dir


# ---------------------------------------------------------------------------
# COCO GT bboxes + Grounding-DINO  (qwen COCO cells 16 / 17 / 18)
# ---------------------------------------------------------------------------
def load_coco_gt_bboxes(coco_dir):
    """Download/parse ``instances_val2017.json``; return ``(coco_bboxes, coco_img_size)``."""
    ann_path = str(coco_cache.ensure_annotations(coco_dir))
    with open(ann_path) as f:
        coco_ann = json.load(f)
    coco_cat_name = {c['id']: c['name'] for c in coco_ann['categories']}
    coco_img_size = {im['id']: (im['width'], im['height']) for im in coco_ann['images']}
    coco_bboxes = defaultdict(list)
    for a in coco_ann['annotations']:
        cat = coco_cat_name[a['category_id']]
        x, y, w, h = a['bbox']
        coco_bboxes[a['image_id']].append((cat, [float(x), float(y), float(x + w), float(y + h)], w * h))
    print(f'  {len(coco_cat_name)} categories  |  {len(coco_img_size)} images  |  '
          f'{sum(len(v) for v in coco_bboxes.values())} bboxes')
    return coco_bboxes, coco_img_size


def make_find_coco_bbox(coco_bboxes):
    def find_coco_bbox(image_id, label):
        image_id = int(image_id)
        if image_id not in coco_bboxes:
            return None
        norm = label.strip().lower()
        candidates = [(box, cat, area) for cat, box, area in coco_bboxes[image_id] if cat.lower() == norm]
        if not candidates:
            return None
        box, cat, _ = max(candidates, key=lambda t: t[2])
        return box, cat
    return find_coco_bbox


def load_grounding_dino(device='cpu', *, model_id='IDEA-Research/grounding-dino-base', threshold=0.15):
    """Load Grounding-DINO and return a ``detect_bbox(img_path, label) -> (box|None, img_size)``."""
    from transformers import AutoProcessor as DetectorProcessor, AutoModelForZeroShotObjectDetection
    from PIL import Image as PILImage

    dproc = DetectorProcessor.from_pretrained(model_id)
    dmodel = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    print(f'Grounding DINO loaded on {device}')

    def detect_bbox(img_path, label, thr=threshold):
        img = PILImage.open(img_path).convert('RGB')
        inputs = dproc(images=img, text=label + '.', return_tensors='pt').to(device)
        with torch.inference_mode():
            outputs = dmodel(**inputs)
        res = dproc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=thr, target_sizes=[img.size[::-1]])[0]
        if len(res['boxes']) == 0:
            return None, img.size
        best_idx = res['scores'].argmax()
        return res['boxes'][best_idx].cpu().tolist(), img.size
    return detect_bbox


def make_get_bbox(find_coco_bbox, coco_img_size, detect_bbox=None):
    """COCO-GT-first bbox lookup, with optional Grounding-DINO fallback.

    Returns ``get_bbox(image_id, label, img_path=None) -> (box|None, size, source)`` where
    ``source`` is ``'coco' | 'dino' | 'failed'``.
    """
    from PIL import Image as PILImage

    def get_bbox(image_id, label, img_path=None):
        image_id = int(image_id)
        coco_match = find_coco_bbox(image_id, label)
        if coco_match is not None:
            box, _ = coco_match
            size = coco_img_size.get(image_id) or (PILImage.open(img_path).size if img_path else None)
            return box, size, 'coco'
        if img_path is None:
            return None, coco_img_size.get(image_id), 'failed'
        if detect_bbox is None:
            return None, coco_img_size.get(image_id), 'failed'
        box, size = detect_bbox(img_path, label)
        return box, size, ('dino' if box is not None else 'failed')
    return get_bbox


# ---------------------------------------------------------------------------
# Strict-both-listings baseline eval + partition  (qwen COCO cells 11b / 13 / 14)
# ---------------------------------------------------------------------------
def run_coco_baseline_eval(spatial_data, *, predict_fn, cache_path, schema_version, model_name,
                           force_recompute=False, dataset_tag='coco_qa_two_obj_left_right_above_below',
                           metadata=None):
    """Strict-both-listings baseline over horizontal/vertical axes.

    ``predict_fn(item, listing) -> str`` is the per-model alpha=0 prediction (lowercased word).
    Caches ``results`` to ``cache_path`` (plain pickle) keyed by schema/n_items/model_name.
    Returns ``results = {'lr': [...], 'rl': [...]}``.
    """
    results = None
    eval_items = list(spatial_data)

    if os.path.exists(cache_path) and not force_recompute:
        with open(cache_path, 'rb') as f:
            loaded = pickle.load(f)
        metadata_ok = True
        if metadata is not None:
            observed = loaded.get('metadata', {}) if isinstance(loaded, dict) else {}
            metadata_ok = all(observed.get(k) == v for k, v in metadata.items())
        if (isinstance(loaded, dict) and 'results' in loaded
                and loaded.get('schema_version') == schema_version
                and loaded.get('n_items') == len(eval_items)
                and loaded.get('model_name') == model_name
                and metadata_ok):
            results = loaded['results']
            print(f'Cache hit: {cache_path}  (n_items={len(eval_items)})')

    if results is None:
        print('Computing COCO spatial-prompt baseline from scratch.')
        results = {'lr': [], 'rl': []}
        for entry in progress(eval_items, desc='baseline'):
            item = {'image_id': entry['image_id'], 'pair_idx': entry.get('pair_idx'),
                    'correct': entry['correct'], 'distractor': entry['distractor'],
                    'subj': entry['subj'], 'obj': entry['obj'], 'truth': entry['truth'], 'axis': entry['axis']}
            for listing in ['lr', 'rl']:
                pred = predict_fn(item, listing)
                results[listing].append({**item, 'pred': pred, 'ok': (pred == entry['truth'])})
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({'results': results, 'model_name': model_name, 'dataset': dataset_tag,
                         'n_items': len(eval_items), 'schema_version': schema_version,
                         'metadata': dict(metadata or {})}, f)
        print(f'Saved: {cache_path}')

    pairs = list(zip(results['lr'], results['rl']))
    n = len(pairs)
    strict_ok = sum(1 for a, b in pairs if a['ok'] and b['ok'])
    total_invalid = sum(1 for a, b in pairs for r in (a, b) if r['pred'] not in VALID_BY_AXIS[r['axis']])
    print('\n' + '=' * 60)
    print('BASELINE - COCO SPATIAL PROMPT (strict both listings)')
    print('=' * 60)
    print(f'N items:        {n}')
    print(f'Invalid preds:  {total_invalid}/{2*n} ({100*total_invalid/max(2*n,1):.1f}%)')
    print(f'STRICT overall: {strict_ok}/{n} ({100*strict_ok/max(n,1):.1f}%)')
    for axis in ['horizontal', 'vertical']:
        ap = [(a, b) for a, b in pairs if a['axis'] == axis]
        ok = sum(1 for a, b in ap if a['ok'] and b['ok'])
        print(f'STRICT {axis:10s}: {ok}/{len(ap)} ({100*ok/max(len(ap),1):.1f}%)')
    print('=' * 60)
    return results


def clean_results(results):
    """Drop pairs with any out-of-axis prediction; return ``clean = {'lr','rl'}``."""
    clean = {'lr': [], 'rl': []}
    dropped = []
    for a, b in zip(results['lr'], results['rl']):
        if a['pred'] in VALID_BY_AXIS[a['axis']] and b['pred'] in VALID_BY_AXIS[b['axis']]:
            clean['lr'].append(a); clean['rl'].append(b)
        else:
            dropped.append((a['image_id'], a['axis'], a['pred'], b['pred']))
    n_orig, n_clean = len(results['lr']), len(clean['lr'])
    both_ok = sum(1 for a, b in zip(clean['lr'], clean['rl']) if a['ok'] and b['ok'])
    print('\n' + '=' * 60)
    print('CLEAN BASELINE')
    print('=' * 60)
    print(f'Dropped invalid items: {n_orig - n_clean}')
    print(f'N clean:               {n_clean}/{n_orig}')
    print(f'STRICT clean:          {both_ok}/{n_clean} ({100*both_ok/max(n_clean,1):.1f}%)')
    print('=' * 60)
    return clean


def partition_results(clean):
    """Partition clean pairs into correct / fail_lr_only / fail_rl_only / fail_both and print."""
    partition = {'golden': [], 'fail_lr_only': [], 'fail_rl_only': [], 'fail_both': []}
    for a, b in zip(clean['lr'], clean['rl']):
        assert a['image_id'] == b['image_id'] and a['axis'] == b['axis']
        record = {'image_id': a['image_id'], 'pair_idx': a.get('pair_idx'),
                  'correct': a['correct'], 'distractor': a['distractor'],
                  'subj': a['subj'], 'obj': a['obj'], 'truth': a['truth'], 'axis': a['axis'],
                  'pred_lr': a['pred'], 'pred_rl': b['pred'],
                  'ok_lr': a['ok'], 'ok_rl': b['ok'], 'failed_listings': []}
        if a['ok'] and b['ok']:
            partition['golden'].append(record)
        elif not a['ok'] and not b['ok']:
            record['failed_listings'] = ['lr', 'rl']; partition['fail_both'].append(record)
        elif not a['ok']:
            record['failed_listings'] = ['lr']; partition['fail_lr_only'].append(record)
        else:
            record['failed_listings'] = ['rl']; partition['fail_rl_only'].append(record)
    n_golden = len(partition['golden'])
    n_fail = sum(len(partition[c]) for c in ['fail_lr_only', 'fail_rl_only', 'fail_both'])
    print('\n' + '=' * 60)
    print('PARTITION')
    print('=' * 60)
    print(f'Correct:      {n_golden}')
    print(f'Fail lr only: {len(partition["fail_lr_only"])}')
    print(f'Fail rl only: {len(partition["fail_rl_only"])}')
    print(f'Fail both:    {len(partition["fail_both"])}')
    print(f'Total:        {n_golden + n_fail}')
    print('\nBy axis:')
    for axis in ['horizontal', 'vertical']:
        counts = {k: sum(1 for r in v if r['axis'] == axis) for k, v in partition.items()}
        print(f'  {axis:10s}: correct={counts["golden"]}, fail_lr={counts["fail_lr_only"]}, '
              f'fail_rl={counts["fail_rl_only"]}, fail_both={counts["fail_both"]}')
    print('=' * 60)
    return partition


# ---------------------------------------------------------------------------
# Axis-separated baseline controls  (qwen COCO cell 12)
# ---------------------------------------------------------------------------
def run_axis_controls(results, *, predict_fn, axes=('horizontal', 'vertical'), n_per_axis=60):
    """Axis-separated baseline controls: image-shuffle should be ~chance; object-swap should flip.

    ``predict_fn(image_id, subj, obj, axis, listing) -> str`` is the per-model alpha=0
    prediction (lowercased word). Returns ``control_summary`` keyed by axis.
    """
    import random
    from collections import Counter
    rng = random.Random()
    all_image_ids = [r['image_id'] for r in results['lr']]

    def opposite_truth(truth):
        return {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above'}[truth]

    def run_axis_control(axis):
        axis_items = [r for r in results['lr'] if r['axis'] == axis]
        n = min(n_per_axis, len(axis_items))
        control_items = rng.sample(axis_items, n)

        shuffle_correct = 0
        swap_opposite = 0
        valid_shuffle = 0
        valid_swap = 0
        truth_counts = Counter(r['truth'] for r in control_items)

        for r in progress(control_items, desc=f'{axis} controls'):
            truth = r['truth']
            opposite = opposite_truth(truth)

            wrong_img = rng.choice([img_id for img_id in all_image_ids if img_id != r['image_id']])
            p_shuffle = predict_fn(wrong_img, r['subj'], r['obj'], axis, 'lr')
            if p_shuffle in VALID_BY_AXIS[axis]:
                valid_shuffle += 1
                shuffle_correct += int(p_shuffle == truth)

            p_swap = predict_fn(r['image_id'], r['obj'], r['subj'], axis, 'lr')
            if p_swap in VALID_BY_AXIS[axis]:
                valid_swap += 1
                swap_opposite += int(p_swap == opposite)

        return {
            'n': n,
            'truth_counts': truth_counts,
            'shuffle_correct': shuffle_correct,
            'valid_shuffle': valid_shuffle,
            'swap_opposite': swap_opposite,
            'valid_swap': valid_swap,
        }

    control_summary = {axis: run_axis_control(axis) for axis in axes}

    print('\n' + '=' * 60)
    print('AXIS-SEPARATED BASELINE CONTROLS')
    print('=' * 60)
    print('Randomness: unseeded')
    for axis, s in control_summary.items():
        print(f'\n{axis.upper()} (n={s["n"]})')
        print(f'  truth counts:  {dict(s["truth_counts"])}')
        print(f'  Image shuffle: {s["shuffle_correct"]}/{s["valid_shuffle"]} '
              f'({100*s["shuffle_correct"]/max(s["valid_shuffle"],1):.1f}%) should be near chance')
        print(f'  Object swap:   {s["swap_opposite"]}/{s["valid_swap"]} '
              f'({100*s["swap_opposite"]/max(s["valid_swap"],1):.1f}%) should flip')
    print('=' * 60)
    return control_summary


# ---------------------------------------------------------------------------
# Per-axis linear probe training  (qwen COCO cell 21)
# ---------------------------------------------------------------------------
def train_axis_probes(probe_data, *, axes=('horizontal', 'vertical'), device='cuda',
                      epochs=200, lr=1e-3, weight_decay=1e-4):
    """Train a per-axis 2-class linear probe on ``probe_data[axis] = {'X','y','meta'}``.

    Rows come in (subj, obj) pairs, so the 95/5 split is done at the *item* level. Returns
    ``(W_by_axis, b_by_axis, stats)`` with detached tensors on ``device``.
    """
    W_by_axis, b_by_axis, stats = {}, {}, {}
    for axis in axes:
        X_axis = probe_data[axis]['X']
        y_axis = probe_data[axis]['y']
        meta_axis = probe_data[axis]['meta']
        assert len(meta_axis) >= 2, f'Need at least two {axis} probe items to train/test split.'
        X_t = torch.from_numpy(X_axis).to(device)
        y_t = torch.from_numpy(y_axis).to(device)
        n_items = len(meta_axis)
        perm = torch.randperm(n_items).tolist()
        split = max(1, int(0.95 * n_items))
        if split >= n_items:
            split = n_items - 1
        train_items, test_items = perm[:split], perm[split:]
        train_rows = [2 * i for i in train_items] + [2 * i + 1 for i in train_items]
        test_rows = [2 * i for i in test_items] + [2 * i + 1 for i in test_items]
        train_idx = torch.tensor(train_rows, device=device)
        test_idx = torch.tensor(test_rows, device=device)
        train_X, test_X = X_t[train_idx], X_t[test_idx]
        train_y, test_y = y_t[train_idx], y_t[test_idx]
        print('\n' + '=' * 60)
        print(f'TRAINING {axis.upper()} PROBE')
        print('=' * 60)
        print(f'Train: {tuple(train_X.shape)}  Test: {tuple(test_X.shape)}')

        D = train_X.shape[1]
        W = torch.randn(D, 2, device=device) / (D ** 0.5)
        b = torch.randn(2, device=device) / (D ** 0.5)
        W.requires_grad_(True); b.requires_grad_(True)
        opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)
        for ep in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(train_X @ W + b, train_y)
            loss.backward()
            opt.step()
            if ep % 20 == 0:
                print(f'  epoch {ep:3d}  loss={loss.item():.4f}')
        with torch.no_grad():
            train_acc = ((train_X @ W + b).argmax(dim=-1) == train_y).float().mean().item() * 100
            test_acc = ((test_X @ W + b).argmax(dim=-1) == test_y).float().mean().item() * 100
        print(f'  Final train acc: {train_acc:.1f}%  |  test acc: {test_acc:.1f}%')
        W_by_axis[axis] = W.detach().clone()
        b_by_axis[axis] = b.detach().clone()
        stats[axis] = {'train_acc': train_acc, 'test_acc': test_acc, 'D': D,
                       'train_items': train_items, 'test_items': test_items}
    return W_by_axis, b_by_axis, stats


# ---------------------------------------------------------------------------
# Amplification sweep + aggregation  (qwen COCO cells 25a / 25b)
# ---------------------------------------------------------------------------
FAIL_CATS = ['fail_lr_only', 'fail_rl_only', 'fail_both']


def coco_artifact_metadata(*, model_name, dtype, prompt_mode, artifact_type,
                           dataset='coco_qa_two_obj_left_right_above_below',
                           bbox_mode='coco_gt_plus_grounding_dino'):
    """Small metadata payload used to prevent stale public COCO artifacts."""
    metadata = {
        'model_name': model_name,
        'dtype': dtype,
        'dataset': dataset,
        'prompt_mode': prompt_mode,
        'bbox_mode': bbox_mode,
        'artifact_type': artifact_type,
    }
    return metadata


def metadata_path(path):
    return f'{path}.metadata.json'


def metadata_matches(path, expected):
    """Return True only when the sidecar exists and all expected fields match."""
    meta_path = metadata_path(path)
    if not os.path.exists(path) or not os.path.exists(meta_path):
        return False
    with open(meta_path, encoding='utf-8') as f:
        observed = json.load(f)
    return all(observed.get(k) == v for k, v in expected.items())


def write_artifact_metadata(path, metadata):
    with open(metadata_path(path), 'w', encoding='utf-8') as f:
        json.dump(dict(metadata), f, indent=2, sort_keys=True)


def run_coco_amp_sweep(partition, W_by_axis, *, predict_with_amp_fn, clear_hooks_fn,
                       axes=('horizontal', 'vertical'), alphas=range(16)):
    """Sweep alpha over failed items; a fail is "fixed" iff every failed listing predicts truth.

    ``predict_with_amp_fn(item, listing, W, alpha) -> str`` and ``clear_hooks_fn()`` are the
    per-model amp callbacks. Returns ``(per_alpha, per_alpha_axis, per_item_log)``.
    """
    alphas = list(alphas)
    per_alpha = {a: {c: 0 for c in FAIL_CATS} for a in alphas}
    per_alpha_axis = {a: {axis: 0 for axis in axes} for a in alphas}
    per_item_log = []
    for cat in FAIL_CATS:
        for item in progress(partition[cat], desc=cat):
            axis = item['axis']
            W = W_by_axis[axis]
            entry = {'category': cat, 'axis': axis, 'image_id': item['image_id'], 'truth': item['truth'],
                     'failed_listings': item['failed_listings'], 'fixed_at': {}}
            for alpha in alphas:
                clear_hooks_fn()
                all_correct = True
                for listing in item['failed_listings']:
                    if predict_with_amp_fn(item, listing, W, alpha) != item['truth']:
                        all_correct = False
                entry['fixed_at'][alpha] = all_correct
                if all_correct:
                    per_alpha[alpha][cat] += 1
                    per_alpha_axis[alpha][axis] += 1
            per_item_log.append(entry)
    return per_alpha, per_alpha_axis, per_item_log


def compute_oracle(per_item_log, *, amp_alphas, axes=('horizontal', 'vertical')):
    """Per-item best-alpha (oracle): item counts if ANY alpha>0 fixes it."""
    oracle_per_cat = {c: 0 for c in FAIL_CATS}
    oracle_per_axis = {axis: 0 for axis in axes}
    first_switch = {c: [] for c in FAIL_CATS}
    for entry in per_item_log:
        ok_alphas = [a for a in amp_alphas if entry['fixed_at'].get(a)]
        if ok_alphas:
            oracle_per_cat[entry['category']] += 1
            oracle_per_axis[entry['axis']] += 1
            first_switch[entry['category']].append(min(ok_alphas))
    return oracle_per_cat, oracle_per_axis, first_switch


def make_random_W(W_probe_by_axis, *, axes=('horizontal', 'vertical'), device='cuda',
                  dtype=torch.bfloat16):
    """A fresh random direction per axis (negative control)."""
    rng = np.random.RandomState()
    W_random_by_axis = {}
    for axis in axes:
        D = W_probe_by_axis[axis].shape[0]
        W_np = rng.randn(D, 2).astype(np.float32)
        W_np /= np.linalg.norm(W_np, axis=0, keepdims=True)
        W_random_by_axis[axis] = torch.from_numpy(W_np).to(device, dtype=dtype)
    return W_random_by_axis


def print_amp_summary(title, *, partition, per_alpha, per_alpha_axis, per_item_log,
                      amp_alphas, axes=('horizontal', 'vertical')):
    """Print best-global-alpha + oracle fix rates, overall and per axis. Returns key numbers."""
    n_golden = len(partition['golden'])
    n_fail_total = sum(len(partition[c]) for c in FAIL_CATS)
    n_total = n_golden + n_fail_total
    best_a = max(amp_alphas, key=lambda a: sum(per_alpha[a].values()))
    best_total = sum(per_alpha[best_a].values())
    best_axis = per_alpha_axis[best_a]
    oracle, oracle_axis, _ = compute_oracle(per_item_log, amp_alphas=amp_alphas, axes=axes)
    oracle_total = sum(oracle.values())

    print('\n' + '=' * 60)
    print(title)
    print('=' * 60)
    print(f'Baseline strict acc:  {n_golden}/{n_total} ({100*n_golden/max(n_total,1):.1f}%)')
    print(f'Best alpha:           {best_a}')
    print(f'Best-alpha acc:       {n_golden + best_total}/{n_total} ({100*(n_golden + best_total)/max(n_total,1):.1f}%)')
    print(f'Best-alpha fix rate:  {best_total}/{n_fail_total} ({100*best_total/max(n_fail_total,1):.1f}%)')
    print(f'Oracle acc:           {n_golden + oracle_total}/{n_total} ({100*(n_golden + oracle_total)/max(n_total,1):.1f}%)')
    print(f'Oracle fix rate:      {oracle_total}/{n_fail_total} ({100*oracle_total/max(n_fail_total,1):.1f}%)')
    print('By axis:')
    for axis in axes:
        fail_axis = sum(1 for c in FAIL_CATS for item in partition[c] if item['axis'] == axis)
        golden_axis = sum(1 for item in partition['golden'] if item['axis'] == axis)
        total_axis = fail_axis + golden_axis
        print(f'  {axis:10s} best: {golden_axis + best_axis[axis]}/{total_axis} '
              f'({100*(golden_axis + best_axis[axis])/max(total_axis,1):.1f}%), '
              f'fix {best_axis[axis]}/{fail_axis} ({100*best_axis[axis]/max(fail_axis,1):.1f}%) | '
              f'oracle fix {oracle_axis[axis]}/{fail_axis} ({100*oracle_axis[axis]/max(fail_axis,1):.1f}%)')
    print('=' * 60)
    return {'best_alpha': best_a, 'best_total': best_total, 'oracle_total': oracle_total,
            'n_golden': n_golden, 'n_fail_total': n_fail_total, 'n_total': n_total}


def print_final_coco_summary(model_label, *, partition,
                             probe_per_alpha, probe_per_alpha_axis,
                             random_per_alpha, random_per_alpha_axis,
                             probe_oracle, random_oracle,
                             probe_oracle_axis=None, random_oracle_axis=None,
                             best_probe_a=None, best_random_a=None,
                             axes=('horizontal', 'vertical')):
    """Print the compact paper-table style summary: None/Random/Random*/Probe/Probe*."""
    n_golden = len(partition['golden'])
    n_fail_total = sum(len(partition[c]) for c in FAIL_CATS)
    n_total = n_golden + n_fail_total

    if best_probe_a is None:
        best_probe_a = max(probe_per_alpha, key=lambda a: sum(probe_per_alpha[a].values()))
    if best_random_a is None:
        best_random_a = max(random_per_alpha, key=lambda a: sum(random_per_alpha[a].values()))

    probe_fixed = sum(probe_per_alpha[best_probe_a].values())
    random_fixed = sum(random_per_alpha[best_random_a].values())
    probe_star_fixed = sum(probe_oracle.values())
    random_star_fixed = sum(random_oracle.values())

    def acc(fixed):
        return (n_golden + fixed) / max(n_total, 1)

    def fix_rate(fixed):
        return fixed / max(n_fail_total, 1)

    def row(label, fixed=None, alpha=None):
        if fixed is None:
            return f'{label:<8} acc={n_golden / max(n_total, 1):.4f}'
        alpha_text = f', alpha={alpha}' if alpha is not None else ''
        return f'{label:<8} acc={acc(fixed):.4f}, fixes={100 * fix_rate(fixed):.1f}%{alpha_text}'

    print('\n' + '=' * 70)
    print(f'FINAL COCO AMPLIFICATION SUMMARY - {model_label}')
    print('=' * 70)
    print(row('None:'))
    print(row('Random:', random_fixed, best_random_a))
    print(row('Random*:', random_star_fixed))
    print(row('Probe:', probe_fixed, best_probe_a))
    print(row('Probe*:', probe_star_fixed))

    print('\nBreakdown by axis:')
    for axis in axes:
        fail_axis = sum(1 for c in FAIL_CATS for item in partition[c] if item['axis'] == axis)
        golden_axis = sum(1 for item in partition['golden'] if item['axis'] == axis)
        total_axis = golden_axis + fail_axis
        random_axis = random_per_alpha_axis[best_random_a][axis]
        probe_axis = probe_per_alpha_axis[best_probe_a][axis]
        random_star_axis = None if random_oracle_axis is None else random_oracle_axis[axis]
        probe_star_axis = None if probe_oracle_axis is None else probe_oracle_axis[axis]

        pieces = [
            f'None={golden_axis / max(total_axis, 1):.4f}',
            f'Random={(golden_axis + random_axis) / max(total_axis, 1):.4f}',
        ]
        if random_star_axis is not None:
            pieces.append(f'Random*={(golden_axis + random_star_axis) / max(total_axis, 1):.4f}')
        pieces.append(f'Probe={(golden_axis + probe_axis) / max(total_axis, 1):.4f}')
        if probe_star_axis is not None:
            pieces.append(f'Probe*={(golden_axis + probe_star_axis) / max(total_axis, 1):.4f}')
        print(f'  {axis:<10} ' + ', '.join(pieces))
    print('=' * 70)

    return {
        'none_acc': n_golden / max(n_total, 1),
        'random_acc': acc(random_fixed),
        'random_star_acc': acc(random_star_fixed),
        'probe_acc': acc(probe_fixed),
        'probe_star_acc': acc(probe_star_fixed),
        'random_fix_rate': fix_rate(random_fixed),
        'random_star_fix_rate': fix_rate(random_star_fixed),
        'probe_fix_rate': fix_rate(probe_fixed),
        'probe_star_fix_rate': fix_rate(probe_star_fixed),
        'best_random_alpha': best_random_a,
        'best_probe_alpha': best_probe_a,
    }


# ---------------------------------------------------------------------------
# Amp artifact pickle + summary.txt + full report.txt  (qwen COCO cells 22 / 23)
# ---------------------------------------------------------------------------
def write_coco_amp_report(artifact_dir, model_label, *, results, clean, partition,
                          probe_per_alpha, probe_per_alpha_axis,
                          random_per_alpha, random_per_alpha_axis,
                          probe_oracle, random_oracle, best_probe_a, best_random_a,
                          alpha_range, axes=('horizontal', 'vertical'), probe_train_stats=None,
                          probe_oracle_axis=None, random_oracle_axis=None,
                          output_prefix='coco_2obj_spatial_anchor',
                          prompt_description=None):
    """Pickle the amp artifact dict and write the summary.txt + full report.txt under ``artifact_dir``.

    ``model_label`` (e.g. "Qwen2-VL-2B") goes into the report header text. File names are
    ``output_prefix`` controls the three file names. If it is empty, files are
    written as ``amp_results.pkl``, ``amp_summary.txt``, and ``full_report.txt``.
    Prints the full report at the end.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    alpha_range = list(alpha_range)

    n_per_cat = {c: len(partition[c]) for c in FAIL_CATS}
    n_golden = len(partition['golden'])
    n_fail_total = sum(n_per_cat.values())
    n_total = n_golden + n_fail_total

    artifact = {'partition_counts': n_per_cat, 'n_golden': n_golden,
                'ALPHA_RANGE': alpha_range, 'probe_per_alpha': probe_per_alpha,
                'probe_per_alpha_axis': probe_per_alpha_axis,
                'random_per_alpha': random_per_alpha,
                'random_per_alpha_axis': random_per_alpha_axis,
                'probe_oracle': probe_oracle,
                'random_oracle': random_oracle,
                'best_probe_alpha': best_probe_a, 'best_random_alpha': best_random_a,
                'probe_train_stats': probe_train_stats}
    def report_name(suffix):
        return f'{output_prefix}_{suffix}' if output_prefix else suffix

    results_path = os.path.join(artifact_dir, report_name('amp_results.pkl'))
    summary_path = os.path.join(artifact_dir, report_name('amp_summary.txt'))
    report_path = os.path.join(artifact_dir, report_name('full_report.txt'))

    with open(results_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f'Saved {results_path}')

    total_fail = sum(n_per_cat.values())
    with open(summary_path, 'w') as f:
        f.write(f'{model_label} / COCO 2obj spatial anchor prompt amplification sweep summary\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Baseline strict-acc:    {n_golden}/{n_total} ({100*n_golden/max(n_total,1):.1f}%)\n')
        f.write(f'Fail items by category: {n_per_cat}\n')
        f.write(f'Best probe alpha:       {best_probe_a}\n')
        f.write(f'Best random alpha:      {best_random_a}\n\n')
        for name, per_alpha in [('PROBE', probe_per_alpha), ('RANDOM', random_per_alpha)]:
            f.write(f'\n{name} direction\n')
            f.write('-' * 60 + '\n')
            f.write(f'{"alpha":>5} | {"fail_lr_only":>15} | {"fail_rl_only":>15} | {"fail_both":>15} | {"total fixed":>15}\n')
            for a in alpha_range:
                tot = sum(per_alpha[a].values())
                f.write(f'{a:>5} | {per_alpha[a]["fail_lr_only"]:>3}/{n_per_cat["fail_lr_only"]:<3} | '
                        f'{per_alpha[a]["fail_rl_only"]:>3}/{n_per_cat["fail_rl_only"]:<3} | '
                        f'{per_alpha[a]["fail_both"]:>3}/{n_per_cat["fail_both"]:<3} | '
                        f'{tot:>3}/{total_fail:<3}\n')
    print(f'Saved {summary_path}')

    n_raw = len(results['lr'])
    n_clean = len(clean['lr'])
    both_ok = sum(1 for a, b in zip(clean['lr'], clean['rl']) if a['ok'] and b['ok'])
    probe_best = sum(probe_per_alpha[best_probe_a].values())
    random_best = sum(random_per_alpha[best_random_a].values())
    probe_oracle_total = sum(probe_oracle.values())
    random_oracle_total = sum(random_oracle.values())

    def axis_report_lines():
        lines = []
        for axis in axes:
            clean_axis = [(a, b) for a, b in zip(clean['lr'], clean['rl']) if a['axis'] == axis]
            n_axis = len(clean_axis)
            ok_axis = sum(1 for a, b in clean_axis if a['ok'] and b['ok'])
            fail_axis = sum(1 for c in FAIL_CATS for item in partition[c] if item['axis'] == axis)
            golden_axis = sum(1 for item in partition['golden'] if item['axis'] == axis)
            probe_best_axis = probe_per_alpha_axis[best_probe_a][axis]
            random_best_axis = random_per_alpha_axis[best_random_a][axis]
            lines.append(f'{axis}: baseline {ok_axis}/{n_axis} ({100*ok_axis/max(n_axis,1):.1f}%), '
                         f'probe best {golden_axis + probe_best_axis}/{golden_axis + fail_axis} '
                         f'({100*(golden_axis + probe_best_axis)/max(golden_axis + fail_axis,1):.1f}%), '
                         f'random best {golden_axis + random_best_axis}/{golden_axis + fail_axis} '
                         f'({100*(golden_axis + random_best_axis)/max(golden_axis + fail_axis,1):.1f}%)')
        return '\n'.join(lines)

    if prompt_description is None:
        prompt_description = (
            'Horizontal: Using the {obj} as the reference point, which side is the {subj} on? '
            'Answer with one word: left or right.\n'
            'Vertical:   Using the {obj} as the reference point, where is the {subj} located? '
            'Answer with one word: above or below.'
        )

    final_summary = print_final_coco_summary(
        model_label,
        partition=partition,
        probe_per_alpha=probe_per_alpha,
        probe_per_alpha_axis=probe_per_alpha_axis,
        random_per_alpha=random_per_alpha,
        random_per_alpha_axis=random_per_alpha_axis,
        probe_oracle=probe_oracle,
        random_oracle=random_oracle,
        probe_oracle_axis=probe_oracle_axis,
        random_oracle_axis=random_oracle_axis,
        best_probe_a=best_probe_a,
        best_random_a=best_random_a,
        axes=axes,
    )

    report = (
        f'{model_label} / COCO 2obj spatial anchor prompt report\n'
        + '=' * 60 + '\n\n'
        + 'Prompts:\n'
        + prompt_description.rstrip() + '\n\n'
        + 'Compact final summary:\n'
        + f'None:    acc={final_summary["none_acc"]:.4f}\n'
        + f'Random:  acc={final_summary["random_acc"]:.4f}, fixes={100 * final_summary["random_fix_rate"]:.1f}%, alpha={best_random_a}\n'
        + f'Random*: acc={final_summary["random_star_acc"]:.4f}, fixes={100 * final_summary["random_star_fix_rate"]:.1f}%\n'
        + f'Probe:   acc={final_summary["probe_acc"]:.4f}, fixes={100 * final_summary["probe_fix_rate"]:.1f}%, alpha={best_probe_a}\n'
        + f'Probe*:  acc={final_summary["probe_star_acc"]:.4f}, fixes={100 * final_summary["probe_star_fix_rate"]:.1f}%\n\n'
        + f'Baseline strict acc:     {both_ok}/{n_clean} ({100*both_ok/max(n_clean,1):.1f}%)\n'
        + axis_report_lines() + '\n\n'
        + f'Probe best alpha:        {best_probe_a}\n'
        + f'Probe best-alpha acc:    {n_golden + probe_best}/{n_total} ({100*(n_golden + probe_best)/max(n_total,1):.1f}%)\n'
        + f'Probe best-alpha fix:    {probe_best}/{n_fail_total} ({100*probe_best/max(n_fail_total,1):.1f}%)\n'
        + f'Probe oracle acc:        {n_golden + probe_oracle_total}/{n_total} ({100*(n_golden + probe_oracle_total)/max(n_total,1):.1f}%)\n'
        + f'Probe oracle fix:        {probe_oracle_total}/{n_fail_total} ({100*probe_oracle_total/max(n_fail_total,1):.1f}%)\n\n'
        + f'Random best alpha:       {best_random_a}\n'
        + f'Random best-alpha acc:   {n_golden + random_best}/{n_total} ({100*(n_golden + random_best)/max(n_total,1):.1f}%)\n'
        + f'Random best-alpha fix:   {random_best}/{n_fail_total} ({100*random_best/max(n_fail_total,1):.1f}%)\n'
        + f'Random oracle acc:       {n_golden + random_oracle_total}/{n_total} ({100*(n_golden + random_oracle_total)/max(n_total,1):.1f}%)\n'
        + f'Random oracle fix:       {random_oracle_total}/{n_fail_total} ({100*random_oracle_total/max(n_fail_total,1):.1f}%)\n\n'
        + f'Dropped invalid items:   {n_raw - n_clean}\n'
        + f'Partition: correct={n_golden}, fail_lr={len(partition["fail_lr_only"])}, '
          f'fail_rl={len(partition["fail_rl_only"])}, fail_both={len(partition["fail_both"])}\n'
    )

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f'>>> Report saved to {report_path}')
    return report
