"""Train three target models and eight fixed full-feature references."""
import shutil
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from experiment_core import train_predict
from experiment_utils import (parser, paths, load_data, read_json, empty_directory,
                              versions, digest, STRATEGIES, ROOT)


def train(args):
    data, models, results = paths(args)
    meta, frames, labels, _, mask = load_data(data)
    classes = int(meta.get('classes', 2))
    graph_path = results / 'causal_graph.json'
    graph = read_json(graph_path)
    features = graph['feature_sets']
    reuse_baseline = args.import_historical or args.reuse_historical_baseline
    if reuse_baseline and (not meta['historical'] or graph['mode'] != 'historical'):
        raise ValueError('Historical model reuse requires historical data and graph')
    if features['all_features'] != meta['features']:
        raise ValueError('Full-feature strategy must use all original inputs in order')
    for kept in features.values():
        if any(f not in meta['features'] for f in kept) or len(set(kept)) != len(kept):
            raise ValueError('Invalid feature subset')
    empty_directory(models)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    metadata = dict(dataset=args.dataset, threads=args.threads, device='cpu',
                    data_sha256=digest(data / 'data.npz'), graph_sha256=digest(graph_path),
                    versions=versions(), historical=meta['historical'])
    options = SimpleNamespace(output=models, epochs=args.epochs, batch_size=args.batch_size,
                              hidden=[512, 256, 128], device='cpu', metadata=metadata)
    historical_models = ROOT / 'models/ACSEmployment'
    # Targets use the same seed; references always use the complete feature set.
    for strategy in STRATEGIES:
        name = f'target_{strategy}'
        if args.import_historical or (args.reuse_historical_baseline and strategy == 'all_features'):
            shutil.copy2(historical_models / f'{name}.pt', models / f'{name}.pt')
        else:
            kept = features[strategy]
            train_predict(frames['member'][kept], labels['member'], [], classes,
                          [f for f in meta['numeric'] if f in kept], options, name, meta['seed'])
    pool = pd.concat([frames['ref0'], frames['ref1']], ignore_index=True)
    pool_y = np.concatenate([labels['ref0'], labels['ref1']])
    for index in range(8):
        name = f'ref{index}'
        if reuse_baseline:
            shutil.copy2(historical_models / f'{name}.pt', models / f'{name}.pt')
        else:
            indices = np.flatnonzero(mask[index])
            train_predict(pool.iloc[indices], pool_y[indices], [], classes,
                          meta['numeric'], options, name, meta['seed'] + index + 1)
    print(f'Saved 11 models (with encoders and training parameters): {models}')


def main():
    p = parser(__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--reuse-historical-baseline', action='store_true',
                       help='Reuse full-feature target/references; retrain the two subset targets')
    modes.add_argument('--import-historical', action='store_true', help='Reuse the 11 historical checkpoints')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        p.error('epochs, batch-size and threads must be positive')
    train(args)


if __name__ == '__main__':
    main()
