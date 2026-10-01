"""Train random all-feature-pool targets and evaluate with fixed full-feature RMIA references."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from experiment_core import train_predict, predict, score, auc, best_attack_accuracy, tpr_at_fpr
from experiment_utils import DATASETS, load_data, read_json, write_json, digest, versions

HERE = Path(__file__).resolve().parent
SPLITS = ['member', 'nonmember', 'ref0', 'ref1']


def select_features(features, blanket, seed):
    if len(set(features)) != len(features) or not set(blanket).issubset(features):
        raise ValueError('Invalid feature sets')
    remaining = [f for f in features if f not in blanket]
    chosen = set(np.random.default_rng(seed).choice(features, len(blanket), replace=False))
    return remaining, [f for f in features if f in chosen]


def evaluate(directory, name, frames, labels, references, a, gamma):
    target, accuracy = {}, {}
    for split in SPLITS:
        target[split], accuracy[split] = predict(directory, name, frames[split], labels[split])
    tz = np.concatenate([target['ref0'], target['ref1']])
    rz = np.concatenate([references['ref0'], references['ref1']], axis=1)
    member, nonmember = [score(target[s], references[s], tz, rz, a, gamma)
                         for s in ['member', 'nonmember']]
    metrics = dict(target_test_accuracy=accuracy['nonmember'], rmia_auc=auc(member, nonmember),
                   rmia_best_accuracy=best_attack_accuracy(member, nonmember),
                   tpr_at_1pct_fpr=tpr_at_fpr(member, nonmember, .01),
                   tpr_at_0_1pct_fpr=tpr_at_fpr(member, nonmember, .001))
    return metrics, member, nonmember


def run_dataset(args, dataset):
    started = time.time()
    source = args.source_root
    result_dir = args.output / 'results' / dataset
    result_dir.mkdir(parents=True, exist_ok=True)
    graph_path = source / 'results' / dataset / 'causal_graph.json'
    graph = read_json(graph_path)
    features = graph['feature_sets']['all_features']
    blanket = graph['feature_sets']['markov_blanket']
    remaining, chosen = select_features(features, blanket, args.selection_seed)
    baseline_path = source / 'results' / dataset / 'comparison.csv'
    baseline = pd.read_csv(baseline_path)
    baseline['result_origin'] = 'existing_non_adaptive'
    baseline['status'] = 'completed'
    baseline['attack_mode'] = 'non_adaptive'
    baseline['selection_seed'] = np.nan
    baseline['markov_blanket_count'] = len(blanket)
    baseline['remaining_feature_count'] = len(remaining)
    baseline['reason'] = ''
    mb_row = baseline.set_index('experiment').loc['markov_blanket']
    row = dict(dataset=dataset, experiment='random_selection', feature_count=0,
               retained_features='[]', result_origin='new_experiment', status='pending',
               attack_mode='non_adaptive', selection_seed=args.selection_seed,
               markov_blanket_count=len(blanket), remaining_feature_count=len(remaining),
               rmia_a=float(mb_row.rmia_a), gamma=float(mb_row.gamma), reason='')
    manifest = dict(dataset=dataset, selection_seed=args.selection_seed,
                    all_features=features, markov_blanket=blanket, remaining_features=remaining,
                    selected_features=chosen, selection_pool=features,
                    markov_blanket_overlap=sorted(set(chosen) & set(blanket)), attack_mode='non_adaptive',
                    source_root=os.path.relpath(source, HERE.parent), graph_sha256=digest(graph_path),
                    baseline_csv_sha256=digest(baseline_path), versions=versions())
    shutil.copy2(graph_path, result_dir / 'causal_graph.json')
    print(f'{dataset}: selected {chosen}', flush=True)
    meta, frames, labels, _, _ = load_data(source / 'data' / dataset)
    data_hash = digest(source / 'data' / dataset / 'data.npz')
    if graph['data_sha256'] != data_hash or features != meta['features']:
        raise ValueError(f'{dataset}: data/graph mismatch')
    source_models = source / 'models' / dataset
    mb = torch.load(source_models / 'target_markov_blanket.pt', weights_only=True, map_location='cpu')
    config = mb['training']
    for key, value in [('data_sha256', data_hash), ('graph_sha256', digest(graph_path))]:
        if config[key] != value:
            raise ValueError(f'{dataset}: target {key} mismatch')
    if mb['encoder']['columns'] != blanket or mb['hidden'] != [512, 256, 128]:
        raise ValueError('Unexpected MB target configuration')
    if config['seed'] != meta['seed']:
        raise ValueError('Data and target checkpoint seeds disagree')
    model_dir = args.output / 'models' / dataset
    model_dir.mkdir(parents=True, exist_ok=True)
    target_path = model_dir / 'target_random_selection.pt'
    if target_path.exists() and not args.evaluate_only:
        raise FileExistsError(f'Use --evaluate-only or a fresh --output: {target_path}')
    reference_hashes = {}
    for i in range(8):
        path = source_models / f'ref{i}.pt'
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        if (checkpoint['encoder']['columns'] != features or checkpoint['hidden'] != [512, 256, 128]
                or checkpoint['classes'] != int(meta.get('classes', 2))
                or checkpoint['training']['data_sha256'] != data_hash
                or checkpoint['training']['graph_sha256'] != digest(graph_path)):
            raise ValueError(f'Invalid full-feature reference: {path}')
        reference_hashes[path.name] = digest(path)
        if args.evaluate_only:
            if digest(model_dir / path.name) != reference_hashes[path.name]:
                raise ValueError('Reference snapshot changed')
        else:
            shutil.copy2(path, model_dir / path.name)
    threads = int(config.get('threads', 2))
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    options = SimpleNamespace(output=model_dir, epochs=config['epochs'],
                              batch_size=config['batch_size'], hidden=[512, 256, 128],
                              device='cpu', metadata=dict(config, versions=versions(),
                              selection_seed=args.selection_seed, attack_mode='non_adaptive'))
    if not args.evaluate_only:
        train_predict(frames['member'][chosen], labels['member'], [],
                      int(meta.get('classes', 2)), [f for f in meta['numeric'] if f in chosen],
                      options, 'target_random_selection', config['seed'])
    checkpoint = torch.load(target_path, weights_only=True, map_location='cpu')
    if (checkpoint['encoder']['columns'] != chosen or checkpoint['training'] != options.metadata
            or checkpoint['hidden'] != [512, 256, 128]):
        raise ValueError('Random target does not match requested experiment')
    references = {}
    for split in SPLITS:
        references[split] = np.stack([predict(model_dir, f'ref{i}', frames[split], labels[split])[0]
                                      for i in range(8)])
    metrics, member, nonmember = evaluate(model_dir, 'target_random_selection', frames,
                                           labels, references, row['rmia_a'], row['gamma'])
    row.update(metrics, feature_count=len(chosen), retained_features=json.dumps(chosen), status='completed')
    np.savez_compressed(result_dir / 'attack_scores.npz', member=member, nonmember=nonmember)
    manifest.update(data_sha256=data_hash, reference_sha256=reference_hashes,
                    target_sha256=digest(target_path), training=checkpoint['training'],
                    rmia_a=row['rmia_a'], gamma=row['gamma'])
    print(f'{dataset}: accuracy={metrics["target_test_accuracy"]:.4f}, AUC={metrics["rmia_auc"]:.6f}', flush=True)
    manifest.update(status=row['status'], reason=row['reason'], elapsed_seconds=time.time() - started)
    write_json(result_dir / 'manifest.json', manifest)
    return baseline.to_dict('records') + [row]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-root', type=Path, default=HERE.parent)
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--dataset', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--selection-seed', type=int, default=42)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.output = args.output.resolve()
    rows = []
    for dataset in args.dataset:
        rows.extend(run_dataset(args, dataset))
        pd.DataFrame(rows).to_csv(args.output / 'results' / 'comparison.csv', index=False)
    print('Saved:', os.path.relpath(args.output / 'results' / 'comparison.csv', HERE.parent), flush=True)


if __name__ == '__main__':
    main()
