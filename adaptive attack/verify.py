"""Verify all completed adaptive artifacts against original data and models."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import torch
from experiment_core import Encoder
from experiment_utils import DATASETS, STRATEGIES, digest, load_data, read_json
from run import validate, METRICS


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root, source = args.output, args.source
    checked = 0
    for dataset in args.dataset:
        data = source / 'data' / dataset
        meta, frames, labels, _, membership = load_data(data)
        graph_path = source / 'results' / dataset / 'causal_graph.json'
        graph = read_json(graph_path)
        manifest = read_json(root / 'results' / dataset / 'manifest.json')
        assert manifest['status'] == 'completed'
        assert manifest['data_sha256'] == digest(data / 'data.npz') == graph['data_sha256']
        assert manifest['graph_sha256'] == digest(graph_path)
        assert manifest['baseline_csv_sha256'] == digest(source / 'results' / dataset / 'comparison.csv')
        table = pd.read_csv(root / 'results' / dataset / 'comparison.csv').set_index('experiment')
        baseline = pd.read_csv(source / 'results' / dataset / 'comparison.csv').set_index('experiment')
        assert list(table.index) == STRATEGIES
        pool = pd.concat([frames['ref0'], frames['ref1']], ignore_index=True)
        for strategy in STRATEGIES:
            kept = graph['feature_sets'][strategy]
            row = table.loc[strategy]
            assert json.loads(row['retained_features']) == json.loads(row['reference_features']) == kept
            assert row['reference_count'] == 8
            assert row['rmia_a'] == baseline.loc[strategy, 'rmia_a']
            assert row['gamma'] == baseline.loc[strategy, 'gamma']
            for metric in METRICS:
                assert 0 <= row[metric] <= 1 and 0 <= row[f'calibrated_{metric}'] <= 1
                np.testing.assert_allclose(row[f'delta_{metric}'], row[metric] - baseline.loc[strategy, metric], atol=1e-12)
            if strategy == 'all_features':
                np.testing.assert_allclose(row[METRICS].to_numpy(dtype=float),
                                           baseline.loc[strategy, METRICS].to_numpy(dtype=float), atol=1e-12, rtol=0)
            target = source / 'models' / dataset / f'target_{strategy}.pt'
            assert manifest[strategy]['target_sha256'] == digest(target)
            for index in range(8):
                name = f'ref{index}'
                path = root / 'models' / dataset / strategy / f'{name}.pt'
                assert manifest[strategy]['references'][name] == digest(path)
                original = torch.load(source / 'models' / dataset / f'{name}.pt', weights_only=True)
                expected = {k: original['training'][k] for k in ['epochs', 'batch_size', 'threads', 'device', 'seed']}
                assert expected['epochs'] == 100
                validate(path, kept, manifest['data_sha256'], manifest['graph_sha256'], int(meta.get('classes', 2)), expected)
                encoder = Encoder([f for f in meta['numeric'] if f in kept]).fit(pool.loc[membership[index], kept])
                checkpoint = torch.load(path, weights_only=True)
                assert checkpoint['encoder'] == dict(columns=encoder.columns, numeric=sorted(encoder.numeric), spec=encoder.spec)
                assert checkpoint['input_dim'] == encoder.transform(pool.iloc[:1][kept]).shape[1]
                checked += 1
        print(f'{dataset}: all checks passed', flush=True)
    print(f'Verified {checked} references and {3 * len(args.dataset)} result rows.')


if __name__ == '__main__':
    main()
