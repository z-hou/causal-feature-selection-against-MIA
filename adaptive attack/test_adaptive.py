"""Checks for feature-matched checkpoint validation and RMIA evaluation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('adaptive_run', Path(__file__).with_name('run.py'))
adaptive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive)


class AdaptiveTests(unittest.TestCase):
    def test_reject_wrong_features_and_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ref.pt'
            torch.save(dict(encoder={'columns': ['x', 'y']}, hidden=[512, 256, 128],
                classes=2, training=dict(data_sha256='data', graph_sha256='graph', seed=43)), path)
            adaptive.validate(path, ['x', 'y'], 'data', 'graph', 2, {'seed': 43})
            for features, data_hash, expected in [(['x'], 'data', None),
                    (['x', 'y'], 'other', None), (['x', 'y'], 'data', {'seed': 44})]:
                with self.assertRaises(ValueError):
                    adaptive.validate(path, features, data_hash, 'graph', 2, expected)

    def test_attack_uses_matched_references(self):
        target = {s: np.log(np.array([.7, .6, .8, .5])) for s in adaptive.NAMES}
        target['nonmember'] = np.log(np.array([.4, .3, .2, .1]))
        references = {f'reference_{s}': np.log(np.full((8, 4), .5)) for s in adaptive.NAMES}
        metrics = adaptive.attack_metrics(target, references, 1., 1.)
        # One member ties all nonmembers at zero: (3 + 0.5) / 4.
        self.assertEqual(metrics['rmia_auc'], .875)
        self.assertEqual(metrics['tpr_at_1pct_fpr'], .75)
        # Matching a different reference family must change the attack result.
        references['reference_nonmember'] = np.log(np.full((8, 4), .01))
        changed = adaptive.attack_metrics(target, references, 1., 1.)
        self.assertLess(changed['rmia_auc'], metrics['rmia_auc'])


if __name__ == '__main__':
    unittest.main()
