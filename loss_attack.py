"""Non-adaptive membership inference using negative cross-entropy loss.

The fixed score is log p(true_label | x), i.e. negative per-example loss.
Larger scores indicate membership. No references, tuning or retraining are used.
Run all datasets by default, or select one with --dataset.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_core import auc, predict, tpr_at_fpr
from experiment_utils import DATASETS, ROOT, STRATEGIES, digest, load_data, read_json


def evaluate(root, dataset):
    data = root / 'data' / dataset
    models = root / 'models' / dataset
    results = root / 'results' / dataset
    meta, frames, labels, _, _ = load_data(data)
    graph_path = results / 'causal_graph.json'
    graph = read_json(graph_path)
    features = graph['feature_sets']
    data_hash = digest(data / 'data.npz')
    graph_hash = digest(graph_path)
    if graph['data_sha256'] != data_hash:
        raise ValueError(f'{dataset}: graph and data do not match')
    if features['all_features'] != meta['features']:
        raise ValueError(f'{dataset}: full-feature columns do not match data')

    rows = []
    for strategy in STRATEGIES:
        name = f'target_{strategy}'
        checkpoint = torch.load(models / f'{name}.pt', map_location='cpu', weights_only=True)
        training = checkpoint['training']
        if training['data_sha256'] != data_hash or training['graph_sha256'] != graph_hash:
            raise ValueError(f'{dataset}/{name}: checkpoint does not match data/graph')
        if (checkpoint['encoder']['columns'] != features[strategy]
                or checkpoint['hidden'] != [512, 256, 128]
                or checkpoint['classes'] != int(meta.get('classes', 2))):
            raise ValueError(f'{dataset}/{name}: unexpected features or architecture')

        # predict restores the checkpoint's fitted encoder and returns log p(y|x).
        # Keep log probabilities directly to avoid probability clipping/underflow.
        member, train_accuracy = predict(models, name, frames['member'], labels['member'])
        nonmember, test_accuracy = predict(models, name, frames['nonmember'], labels['nonmember'])
        for scores in [member, nonmember]:
            if not np.isfinite(scores).all() or (scores > 1e-10).any():
                raise ValueError(f'{dataset}/{name}: invalid log probabilities')

        rows.append(dict(
            dataset=dataset, experiment=strategy, attack='loss', adaptive=False,
            score='negative_cross_entropy', feature_count=len(features[strategy]),
            retained_features=json.dumps(features[strategy]),
            member_count=len(member), nonmember_count=len(nonmember),
            attack_auc=auc(member, nonmember),
            tpr_at_1pct_fpr=tpr_at_fpr(member, nonmember, 0.01),
            tpr_at_0_1pct_fpr=tpr_at_fpr(member, nonmember, 0.001),
            member_mean_loss=float(-member.mean()),
            nonmember_mean_loss=float(-nonmember.mean()),
            target_train_accuracy=train_accuracy, target_test_accuracy=test_accuracy,
            seed=training['seed'], epochs=training['epochs'],
            data_sha256=data_hash, graph_sha256=graph_hash,
            model_sha256=digest(models / f'{name}.pt')))

    table = pd.DataFrame(rows)
    output = results / 'loss_attack.csv'
    table.to_csv(output, index=False)
    print(table[['dataset', 'experiment', 'attack_auc']].to_string(index=False), flush=True)
    print(f'Saved: {output}', flush=True)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all'] + DATASETS, default='all')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    tables = []
    for dataset in datasets:
        tables.append(evaluate(args.root, dataset))
    if args.dataset == 'all':
        output = args.root / 'results' / 'loss_attack.csv'
        pd.concat(tables, ignore_index=True).to_csv(output, index=False)
        print(f'Saved combined results: {output}', flush=True)


if __name__ == '__main__':
    main()
