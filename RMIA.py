"""Evaluate three strategies; keep only the final CSV, recomputing predictions in memory."""
import json
import numpy as np
import pandas as pd
import torch
from experiment_core import predict, score, auc, tpr_at_fpr, best_attack_accuracy
from experiment_utils import parser, paths, load_data, read_json, digest, NAMES, STRATEGIES, ROOT


def choose_a(pack, gamma):
    # Auxiliary ref0 is the tuning target; ref1 is OUT for ref0 and calibration.
    n = min(len(pack['reference_calibration'][0]), len(pack['reference_ref0'][0]))
    tx = np.concatenate([pack['reference_ref0'][0, :n], pack['reference_calibration'][0, :n]])
    rx = np.concatenate([pack['reference_ref0'][1, :n], pack['reference_calibration'][1, :n]])[None, :]
    tz = np.concatenate([pack['reference_ref0'][0], pack['reference_ref1'][0]])
    rz = np.concatenate([pack['reference_ref0'][1], pack['reference_ref1'][1]])[None, :]
    best, selected = -1, None
    for candidate in np.linspace(0, 1, 11):
        values = score(tx, rx, tz, rz, float(candidate), gamma)
        value = auc(values[:n], values[n:])
        if value > best:
            best, selected = value, float(candidate)
    return selected


def evaluate(args):
    data, models, results = paths(args)
    meta, frames, labels, _, _ = load_data(data)
    graph_path = results / 'causal_graph.json'
    features = read_json(graph_path)['feature_sets']
    data_hash, graph_hash = digest(data / 'data.npz'), digest(graph_path)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    expected = None
    columns = ['target_train_accuracy', 'target_test_accuracy', 'rmia_auc',
               'rmia_best_accuracy', 'tpr_at_1pct_fpr', 'tpr_at_0_1pct_fpr']
    if args.verify_historical:
        if not meta['historical']:
            raise ValueError('--verify-historical requires historical data')
        baseline = pd.read_csv(ROOT / 'results/ACSEmployment/comparison.csv')
        expected = baseline.set_index('experiment').loc[STRATEGIES, columns].to_numpy()
    model_names = [f'ref{i}' for i in range(8)] + [f'target_{s}' for s in STRATEGIES]
    configs = {}
    for name in model_names:
        checkpoint = torch.load(models / f'{name}.pt', weights_only=True, map_location='cpu')
        config = checkpoint['training']
        if config['data_sha256'] != data_hash or config['graph_sha256'] != graph_hash:
            raise ValueError(f'Data/graph mismatch for {name}')
        kept = meta['features'] if name.startswith('ref') else features[name[len('target_'):]]
        if (checkpoint['encoder']['columns'] != kept or checkpoint['hidden'] != [512, 256, 128]
                or checkpoint['classes'] != int(meta.get('classes', 2))):
            raise ValueError(f'Unexpected features or architecture: {name}')
        configs[name] = config
    references = {}
    for split in NAMES:
        references[f'reference_{split}'] = np.stack([
            predict(models, f'ref{i}', frames[split], labels[split])[0] for i in range(8)])
    a = args.a
    if a is None:
        a = 0.1 if meta['historical'] else choose_a(references, args.gamma)
    rz = np.concatenate([references['reference_ref0'], references['reference_ref1']], axis=1)
    rows = []
    for strategy in STRATEGIES:
        name = f'target_{strategy}'
        target, accuracy = {}, {}
        for split in NAMES:
            target[split], accuracy[split] = predict(models, name, frames[split], labels[split])
        tz = np.concatenate([target['ref0'], target['ref1']])
        m, n = [score(target[split], references[f'reference_{split}'], tz, rz, a, args.gamma)
                for split in ['member', 'nonmember']]
        rows.append(dict(dataset=args.dataset, experiment=strategy,
            feature_count=len(features[strategy]), retained_features=json.dumps(features[strategy]),
            target_train_accuracy=accuracy['member'], target_test_accuracy=accuracy['nonmember'],
            rmia_auc=auc(m, n), rmia_best_accuracy=best_attack_accuracy(m, n),
            tpr_at_1pct_fpr=tpr_at_fpr(m, n, .01), tpr_at_0_1pct_fpr=tpr_at_fpr(m, n, .001),
            rmia_a=a, gamma=args.gamma, reference_count=8, seed=configs[name]['seed'],
            epochs=configs[name]['epochs'], distribution=meta.get('distribution', 'same_state_year')))
    table = pd.DataFrame(rows)
    if expected is not None:
        actual = table.set_index('experiment').loc[STRATEGIES, columns].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        print('Historical verification passed; maximum error:', np.max(np.abs(actual - expected)))
    table.to_csv(results / 'comparison.csv', index=False)
    print(table.to_string(index=False))


def main():
    p = parser(__doc__)
    p.add_argument('--a', type=float)
    p.add_argument('--gamma', type=float, default=2.0)
    p.add_argument('--verify-historical', action='store_true')
    args = p.parse_args()
    if (args.a is not None and not 0 <= args.a <= 1) or not np.isfinite(args.gamma) or args.gamma < 1:
        p.error('Require 0 <= a <= 1 and finite gamma >= 1')
    evaluate(args)


if __name__ == '__main__':
    main()
