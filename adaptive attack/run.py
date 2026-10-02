"""Train feature-matched references and evaluate the existing target models."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from experiment_core import train_predict, predict, score, auc, tpr_at_fpr, best_attack_accuracy
from experiment_utils import DATASETS, STRATEGIES, NAMES, load_data, read_json, digest, versions
from RMIA import choose_a

METRICS = ['rmia_auc', 'rmia_best_accuracy', 'tpr_at_1pct_fpr', 'tpr_at_0_1pct_fpr']


def validate(path, features, classes, expected=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['training']
    if (checkpoint['encoder']['columns'] != features
            or checkpoint['hidden'] != [512, 256, 128]
            or checkpoint['classes'] != classes):
        raise ValueError(f'Incompatible checkpoint: {path}')
    if expected:
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f'Checkpoint {path}: unexpected {key}')
    return config


def attack_metrics(target, references, a, gamma):
    tz = np.concatenate([target['ref0'], target['ref1']])
    rz = np.concatenate([references['reference_ref0'], references['reference_ref1']], axis=1)
    member, nonmember = [score(target[s], references[f'reference_{s}'], tz, rz, a, gamma)
                         for s in ['member', 'nonmember']]
    return dict(rmia_auc=auc(member, nonmember),
                rmia_best_accuracy=best_attack_accuracy(member, nonmember),
                tpr_at_1pct_fpr=tpr_at_fpr(member, nonmember, .01),
                tpr_at_0_1pct_fpr=tpr_at_fpr(member, nonmember, .001))


def run_dataset(args, dataset):
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError('Output must differ from the baseline root')
    data = source / 'data' / dataset
    original_models = source / 'models' / dataset
    graph_path = source / 'results' / dataset / 'causal_graph.json'
    meta, frames, labels, _, mask = load_data(data)
    if not mask[0, :10000].all() or mask[0, 10000:].any() or not np.array_equal(mask[1], ~mask[0]):
        raise ValueError('Auxiliary calibration requires ref0/ref1 complementary ordered splits')
    features = read_json(graph_path)['feature_sets']
    data_hash, graph_hash = digest(data / 'data.npz'), digest(graph_path)
    if features['all_features'] != meta['features']:
        raise ValueError('Data/graph mismatch')
    classes = int(meta.get('classes', 2))
    baseline = pd.read_csv(source / 'results' / dataset / 'comparison.csv').set_index('experiment')
    pool = pd.concat([frames['ref0'], frames['ref1']], ignore_index=True)
    pool_y = np.concatenate([labels['ref0'], labels['ref1']])
    results = output / 'results' / dataset
    results.mkdir(parents=True, exist_ok=True)
    rows, manifest, trained_sets = [], {}, {}
    for strategy in STRATEGIES:
        kept = features[strategy]
        if len(set(kept)) != len(kept) or any(f not in meta['features'] for f in kept):
            raise ValueError('Invalid feature set')
        target_name = f'target_{strategy}'
        target_path = original_models / f'{target_name}.pt'
        validate(target_path, kept, classes)
        directory = output / 'models' / dataset / strategy
        directory.mkdir(parents=True, exist_ok=True)
        manifest[strategy] = {'target_sha256': digest(target_path), 'references': {}}
        for index in range(8):
            name = f'ref{index}'
            path = directory / f'{name}.pt'
            original = original_models / f'{name}.pt'
            config = validate(original, features['all_features'], classes)
            expected = {k: config[k] for k in ['epochs', 'batch_size', 'seed', 'threads', 'device']}
            if config['seed'] != meta['seed'] + index + 1:
                raise ValueError('Unexpected reference seed')
            if not path.exists():
                if args.evaluate_only:
                    raise FileNotFoundError(path)
                if strategy == 'all_features':
                    shutil.copy2(original, path)
                elif tuple(kept) in trained_sets:
                    shutil.copy2(trained_sets[tuple(kept)] / f'{name}.pt', path)
                else:
                    torch.set_num_threads(config['threads'])
                    metadata = dict(config, versions=versions(), adaptive=True,
                                    feature_strategy=strategy, source_reference_sha256=digest(original))
                    options = SimpleNamespace(output=directory, epochs=config['epochs'],
                        batch_size=config['batch_size'], hidden=[512, 256, 128], device='cpu', metadata=metadata)
                    indices = np.flatnonzero(mask[index])
                    print(f'{dataset}/{strategy}/{name}', flush=True)
                    train_predict(pool.iloc[indices][kept], pool_y[indices], [], classes,
                        [f for f in meta['numeric'] if f in kept], options, name, config['seed'])
            validate(path, kept, classes, expected)
            manifest[strategy]['references'][name] = digest(path)
        trained_sets[tuple(kept)] = directory
        references = {}
        for split in NAMES:
            references[f'reference_{split}'] = np.stack([
                predict(directory, f'ref{i}', frames[split], labels[split])[0] for i in range(8)])
        target, accuracy = {}, {}
        for split in NAMES:
            target[split], accuracy[split] = predict(original_models, target_name, frames[split], labels[split])
        base = baseline.loc[strategy]
        a, gamma = float(base['rmia_a']), float(base['gamma'])
        metrics = attack_metrics(target, references, a, gamma)
        calibrated_a = choose_a(references, gamma)
        calibrated = attack_metrics(target, references, calibrated_a, gamma)
        row = dict(dataset=dataset, experiment=strategy, feature_count=len(kept),
            retained_features=json.dumps(kept), reference_feature_count=len(kept),
            reference_features=json.dumps(kept),
            target_test_accuracy=accuracy['nonmember'], **metrics, rmia_a=a, gamma=gamma,
            calibrated_a=calibrated_a)
        for metric in METRICS:
            row[f'baseline_{metric}'] = float(base[metric])
            row[f'delta_{metric}'] = metrics[metric] - float(base[metric])
            row[f'calibrated_{metric}'] = calibrated[metric]
        rows.append(row)
        pd.DataFrame(rows).to_csv(results / 'comparison.csv', index=False)
        print(f'{dataset}/{strategy}: AUC={metrics["rmia_auc"]:.6f}, calibrated AUC={calibrated["rmia_auc"]:.6f}', flush=True)
    manifest.update(data_sha256=data_hash, graph_sha256=graph_hash,
                    baseline_csv_sha256=digest(source / 'results' / dataset / 'comparison.csv'),
                    source=os.path.relpath(source, ROOT), versions=versions(), status='completed')
    (results / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    rows = []
    for dataset in args.dataset:
        rows.extend(run_dataset(args, dataset))
    pd.DataFrame(rows).to_csv(args.output / 'results' / 'comparison.csv', index=False)


if __name__ == '__main__':
    main()
