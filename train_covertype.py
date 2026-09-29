"""Train three seven-class Covertype targets and eight full-feature references.

Reuse the existing deterministic MLP training implementation and defaults.
Prepare original UCI data with data_prepare.py and learn the training-only DAG
with Learn_causal_graph.py before running this script; evaluate with RMIA.py.
"""
from experiment_utils import COVERTYPE, parser, paths, load_data
from train import train


def train_covertype(args):
    data, _, _ = paths(args)
    meta, _, labels, _, _ = load_data(data)
    if (meta['dataset'] != COVERTYPE or meta['target'] != 'Cover_Type'
            or meta['classes'] != 7 or len(meta['features']) != 54):
        raise ValueError('Expected the original 54-feature, seven-class Covertype task')
    if set(labels['member']) != set(range(7)):
        raise ValueError('Every cover type must appear in the target training set')
    train(args)


def main():
    p = parser(__doc__)
    p.set_defaults(dataset=COVERTYPE, import_historical=False, reuse_historical_baseline=False)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    if args.dataset != COVERTYPE:
        p.error('This training script only supports --dataset covertype')
    if min(args.epochs, args.batch_size, args.threads) < 1:
        p.error('epochs, batch-size and threads must be positive')
    train_covertype(args)


if __name__ == '__main__':
    main()
