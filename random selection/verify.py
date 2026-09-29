"""Check all saved selections, model provenance and RMIA metrics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_core import auc, best_attack_accuracy, tpr_at_fpr
from experiment_utils import DATASETS, digest, read_json
from run import select_features, METRICS


def main():
    root = Path(__file__).resolve().parent
    table = pd.read_csv(root / 'results/comparison.csv')
    assert len(table) == 4 * len(DATASETS)
    assert not table.duplicated(['dataset', 'experiment']).any()
    assert set(table['status']) == {'completed'}
    assert set(table['attack_mode']) == {'non_adaptive'}
    assert np.isfinite(table[METRICS].to_numpy()).all()
    assert ((table[METRICS] >= 0) & (table[METRICS] <= 1)).all().all()
    for dataset in DATASETS:
        manifest = read_json(root / 'results' / dataset / 'manifest.json')
        rows = table[table.dataset == dataset].set_index('experiment')
        row = rows.loc['random_selection']
        selected = json.loads(row.retained_features)
        _, expected = select_features(manifest['all_features'], manifest['markov_blanket'], int(row.selection_seed))
        assert selected == expected == manifest['selected_features']
        assert len(selected) == len(set(selected)) == len(manifest['markov_blanket'])
        assert set(selected).issubset(manifest['all_features'])
        assert manifest['selection_pool'] == manifest['all_features']
        assert manifest['baseline_mb_reproduced']
        assert row.rmia_a == rows.loc['markov_blanket'].rmia_a
        assert row.gamma == rows.loc['markov_blanket'].gamma
        models = root / 'models' / dataset
        target = models / 'target_random_selection.pt'
        assert digest(target) == manifest['target_sha256']
        checkpoint = torch.load(target, weights_only=True, map_location='cpu')
        assert checkpoint['encoder']['columns'] == selected
        assert checkpoint['hidden'] == [512, 256, 128]
        for name, checksum in manifest['reference_sha256'].items():
            assert digest(models / name) == checksum
            ref = torch.load(models / name, weights_only=True, map_location='cpu')
            assert ref['encoder']['columns'] == manifest['all_features']
        with np.load(root / 'results' / dataset / 'attack_scores.npz') as pack:
            member, nonmember = pack['member'], pack['nonmember']
        assert member.shape == nonmember.shape == (10000,)
        values = [auc(member, nonmember), best_attack_accuracy(member, nonmember),
                  tpr_at_fpr(member, nonmember, .01), tpr_at_fpr(member, nonmember, .001)]
        np.testing.assert_allclose(values, row[METRICS[2:]].to_numpy(dtype=float), rtol=0, atol=1e-12)
        print(dataset, 'PASS')
    print('Verified 7 random selections, 56 unchanged full-feature references and 28 CSV rows.')


if __name__ == '__main__':
    main()
