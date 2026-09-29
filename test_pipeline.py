"""Regression tests for feature selection, RMIA ties and balanced reference splitting."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from Learn_causal_graph import feature_sets
from data_prepare import split_households, reference_mask
from experiment_core import score, auc, tpr_at_fpr, Encoder, train_predict, predict


class PipelineTests(unittest.TestCase):
    def test_multiclass_training_and_checkpoint_prediction(self):
        frame = pd.DataFrame({'value': ['1', '2', '3', '4'], 'category': ['a', 'b', 'a', 'b']})
        labels = np.array([0, 5, 11, 5], dtype=np.int64)
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), epochs=1, batch_size=4, device='cpu')
            expected = train_predict(frame, labels, [(frame, labels)], 12, ['value'], args, 'target', 42)[0]
            actual, accuracy = predict(Path(folder), 'target', frame, labels)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            self.assertTrue(0 <= accuracy <= 1)
            saved = torch.load(Path(folder) / 'target.pt', weights_only=True, map_location='cpu')
            self.assertEqual(saved['classes'], 12)
            self.assertEqual(saved['hidden'], [512, 256, 128])

    def test_reference_pairs_match_rmia_calibration_assumptions(self):
        mask = reference_mask(42)
        self.assertEqual(mask.shape, (8, 20000))
        self.assertTrue(mask[0, :10000].all() and not mask[0, 10000:].any())
        np.testing.assert_array_equal(mask.sum(0), np.full(20000, 4))
        np.testing.assert_array_equal(mask.sum(1), np.full(8, 10000))
        for pair in range(4):
            np.testing.assert_array_equal(mask[2 * pair], ~mask[2 * pair + 1])

    def test_blanket_includes_spouses_and_excludes_label_and_risk(self):
        edges = [('A', '__label__'), ('__label__', 'B'), ('C', 'B'),
                 ('__label__', 'privacy_risk_score'), ('D', 'privacy_risk_score')]
        sets = feature_sets(['A', 'B', 'C', 'D', '__label__', 'privacy_risk_score'],
                            edges, ['D', 'C', 'B', 'A', 'E'])
        self.assertEqual(sets['label_parents'], ['A'])
        self.assertEqual(sets['markov_blanket'], ['D', 'C', 'B', 'A'])

    def test_empirical_ties_respect_false_positive_budget(self):
        negatives = np.array([.1] * 990 + [.8] * 10)
        positives = np.array([.2, .8, .9])
        self.assertEqual(tpr_at_fpr(positives, negatives, .01), 1.)
        self.assertEqual(tpr_at_fpr(positives, negatives, .001), 1 / 3)
        self.assertEqual(auc(np.ones(3), np.ones(4)), .5)

    def test_rmia_matches_direct_probability_calculation(self):
        tx = np.array([.2, .5, .9])
        rx = np.array([[.3, .5, .6], [.1, .4, .8]])
        tz = np.array([.2, .7, .9, .4])
        rz = np.array([[.4, .4, .8, .2], [.2, .6, .7, .3]])
        x_ratio = tx / ((1 + .1) * rx.mean(0) + 1 - .1) * 2
        z_ratio = tz / rz.mean(0)
        expected = (x_ratio[:, None] / z_ratio[None, :] > 2).mean(1)
        np.testing.assert_array_equal(score(np.log(tx), np.log(rx), np.log(tz), np.log(rz), .1, 2), expected)

    def test_households_are_not_reused_when_truncated(self):
        frame = pd.DataFrame({'SERIALNO': np.repeat(np.arange(20), 3)})
        parts = split_households(frame, [7, 7, 7], 42)
        self.assertTrue(all(len(p) == 7 for p in parts))
        for i in range(3):
            for j in range(i):
                self.assertFalse(set(parts[i].SERIALNO) & set(parts[j].SERIALNO))

    def test_empty_feature_set_is_constant_model_input(self):
        frame = pd.DataFrame(index=range(3))
        np.testing.assert_array_equal(Encoder([]).fit(frame).transform(frame), np.ones((3, 1)))


if __name__ == '__main__':
    unittest.main()
