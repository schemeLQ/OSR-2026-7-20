import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.cub200 import (PROTOCOL_NAME, PROTOCOL_NOTE, cub_root, load_cub_metadata,
                             load_class_attributes, validate_cub_root)


def checksum(meta):
    return 'classes={};images={}'.format(len(meta['class_names']), len(meta['images']))


def max_sim_to_known(attrs, known):
    sims = attrs @ attrs[np.asarray(known)].T
    return sims.max(axis=1)


def class_names(meta, classes):
    return [meta['class_names'][int(c)] for c in classes]


def overlap_matrix(splits, key):
    n = len(splits)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            mat[i, j] = len(set(splits[i][key]) & set(splits[j][key]))
    return mat


def write_matrix(path, mat):
    with open(path, 'w', encoding='utf-8') as f:
        for row in mat:
            f.write(','.join(str(int(x)) for x in row) + '\n')


def random_baseline(attrs, known, selected_unknown, split_seed, trials=1000):
    rng = np.random.RandomState(split_seed + 12345)
    all_classes = np.asarray(sorted(set(range(200)) - set(known)))
    sim = max_sim_to_known(attrs, known)
    vals = []
    for _ in range(trials):
        u = rng.choice(all_classes, size=len(selected_unknown), replace=False)
        vals.append(sim[u].mean())
    vals = np.asarray(vals)
    selected = float(sim[np.asarray(selected_unknown)].mean())
    percentile = float((vals <= selected).mean() * 100.0)
    return selected, float(vals.mean()), float(vals.std()), percentile


def generate_one(split_idx, split_seed, attrs, meta, num_known, num_unknown, easy_pool_ratio,
                 allow_random_unknown_fallback=False):
    all_classes = np.asarray(sorted(range(200)))
    retry = 0
    while True:
        rng = np.random.RandomState(split_seed + retry * 1000)
        known = sorted(rng.choice(all_classes, size=num_known, replace=False).astype(int).tolist())
        remaining = np.asarray(sorted(set(all_classes.tolist()) - set(known)))
        if attrs is None:
            if not allow_random_unknown_fallback:
                raise RuntimeError('CUB attributes unavailable and fallback is disabled.')
            easy_pool = remaining
            sim = np.full(200, np.nan)
            attr_source = 'WARNING: CUB attributes were unavailable. Unknown classes were sampled randomly without semantic-distance control.'
        else:
            sim = max_sim_to_known(attrs, known)
            rem_scores = sim[remaining]
            pool_size = max(num_unknown, int(np.ceil(len(remaining) * easy_pool_ratio)))
            order = np.argsort(rem_scores)
            easy_pool = remaining[order[:pool_size]]
            attr_source = 'attributes/image_attribute_labels.txt'
        unknown = sorted(rng.choice(easy_pool, size=num_unknown, replace=False).astype(int).tolist())
        if len(set(known) & set(unknown)) == 0:
            break
        retry += 1
    selected, rb_mean, rb_std, rb_pct = (np.nan, np.nan, np.nan, np.nan)
    if attrs is not None:
        selected, rb_mean, rb_std, rb_pct = random_baseline(attrs, known, unknown, split_seed)
    return {
        'protocol': PROTOCOL_NAME,
        'protocol_note': PROTOCOL_NOTE,
        'official_benchmark': False,
        'split_idx': int(split_idx),
        'split_seed': int(split_seed),
        'num_known': int(num_known),
        'num_unknown': int(num_unknown),
        'easy_pool_ratio': float(easy_pool_ratio),
        'known_classes': known,
        'unknown_classes': unknown,
        'known_class_names': class_names(meta, known),
        'unknown_class_names': class_names(meta, unknown),
        'unknown_max_similarity_to_known': [None if np.isnan(sim[c]) else float(sim[c]) for c in unknown],
        'mean_unknown_similarity_to_known': None if attrs is None else selected,
        'random_unknown_expected_mean_similarity': None if attrs is None else rb_mean,
        'random_unknown_similarity_std': None if attrs is None else rb_std,
        'selected_unknown_percentile_among_random_sets': None if attrs is None else rb_pct,
        'attribute_source': attr_source,
        'dataset_checksum': checksum(meta),
    }


def main():
    p = argparse.ArgumentParser(description='Generate custom CUB-10/10 Easy-OSR splits.')
    p.add_argument('--data_root', default='./data')
    p.add_argument('--num_splits', type=int, default=5)
    p.add_argument('--num_known', type=int, default=10)
    p.add_argument('--num_unknown', type=int, default=10)
    p.add_argument('--easy_pool_ratio', type=float, default=0.30)
    p.add_argument('--out_dir', default=None)
    p.add_argument('--regenerate_splits', action='store_true')
    p.add_argument('--allow_random_unknown_fallback', action='store_true')
    args = p.parse_args()

    root = cub_root(args.data_root)
    try:
        validate_cub_root(root, require_attributes=not args.allow_random_unknown_fallback)
    except Exception as e:
        if not args.allow_random_unknown_fallback:
            raise
        print('WARNING: CUB attributes/data validation failed:', e)
    meta = load_cub_metadata(root)
    attrs = None
    try:
        attrs, attr_source = load_class_attributes(root)
        print('Loaded class attributes from:', attr_source)
    except Exception as e:
        if not args.allow_random_unknown_fallback:
            raise RuntimeError('CUB attributes were unavailable. Re-run with --allow_random_unknown_fallback to permit random unknown sampling. Error: {}'.format(e))
        print('WARNING: CUB attributes were unavailable. Unknown classes were sampled randomly without semantic-distance control.')

    out_dir = args.out_dir or os.path.join(args.data_root, 'open_set_splits', 'cub_10_10_easy')
    os.makedirs(out_dir, exist_ok=True)
    splits = []
    for i in range(args.num_splits):
        path = os.path.join(out_dir, 'split_{}.json'.format(i))
        if os.path.exists(path) and not args.regenerate_splits:
            with open(path, 'r', encoding='utf-8') as f:
                split = json.load(f)
            print('Loaded existing split:', path)
        else:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    old = json.load(f)
                print('Regenerating split {} old known={} old unknown={}'.format(i, old.get('known_classes'), old.get('unknown_classes')))
            split = generate_one(i, i, attrs, meta, args.num_known, args.num_unknown, args.easy_pool_ratio,
                                 args.allow_random_unknown_fallback)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(split, f, indent=2)
            print('Wrote:', path)
        splits.append(split)

    kmat = overlap_matrix(splits, 'known_classes')
    umat = overlap_matrix(splits, 'unknown_classes')
    for i in range(args.num_splits):
        for j in range(i + 1, args.num_splits):
            if kmat[i, j] == args.num_known:
                raise RuntimeError('Known classes are identical for splits {} and {}'.format(i, j))
            if umat[i, j] == args.num_unknown:
                raise RuntimeError('Unknown classes are identical for splits {} and {}'.format(i, j))
    write_matrix(os.path.join(out_dir, 'split_overlap_known.csv'), kmat)
    write_matrix(os.path.join(out_dir, 'split_overlap_unknown.csv'), umat)
    summary = {
        'protocol': PROTOCOL_NAME,
        'protocol_note': PROTOCOL_NOTE,
        'official_benchmark': False,
        'num_splits': args.num_splits,
        'split_seeds': list(range(args.num_splits)),
        'known_overlap_matrix': kmat.tolist(),
        'unknown_overlap_matrix': umat.tolist(),
        'known_unique_total': len(set(c for s in splits for c in s['known_classes'])),
        'unknown_unique_total': len(set(c for s in splits for c in s['unknown_classes'])),
        'mean_unknown_similarity_to_known': [s.get('mean_unknown_similarity_to_known') for s in splits],
        'splits': splits,
    }
    with open(os.path.join(out_dir, 'split_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('\n{}'.format(PROTOCOL_NOTE))
    print('Known overlap matrix:\n', kmat)
    print('Unknown overlap matrix:\n', umat)
    for s in splits:
        vals = [v for v in s.get('unknown_max_similarity_to_known', []) if v is not None]
        print('=== CUB-10/10 Easy-OSR Split {} ==='.format(s['split_idx']))
        print('Protocol: custom, not official SSB')
        print('Training seed: unchanged by this script')
        print('Split seed:', s['split_seed'])
        print('Known classes:', s['known_classes'])
        print('Unknown classes:', s['unknown_classes'])
        if vals:
            print('Mean/Min/Max unknown max-similarity: {:.4f}/{:.4f}/{:.4f}'.format(np.mean(vals), np.min(vals), np.max(vals)))
            print('Random unknown expected mean similarity: {:.4f} +/- {:.4f}; selected percentile {:.1f}'.format(
                s['random_unknown_expected_mean_similarity'], s['random_unknown_similarity_std'],
                s['selected_unknown_percentile_among_random_sets']))
            if s['mean_unknown_similarity_to_known'] >= s['random_unknown_expected_mean_similarity']:
                print('WARNING: selected easy unknown similarity is not lower than random baseline.')


if __name__ == '__main__':
    main()
