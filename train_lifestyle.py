"""Train the three Lifestyle Choice targets and eight full-feature references.

Uses the same training implementation, MLP and defaults as the ACS experiments.
Prepare data with data_prepare.py, learn the graph with Learn_causal_graph.py,
and evaluate with RMIA.py; this entry point performs training only.
"""
from experiment_utils import LIFESTYLE, parser, paths, load_data
from train import train


def train_lifestyle(args):
    data, _, _ = paths(args)
    meta, _, labels, _, _ = load_data(data)
    if meta['dataset'] != LIFESTYLE or meta['target'] != 'Lifestyle Choice':
        raise ValueError('Expected the half-a-million-lifestyle Lifestyle Choice task')
    if set(labels['member']) != set(range(meta['classes'])):
        raise ValueError('Every lifestyle class must appear in the target training set')
    train(args)


def main():
    p = parser(__doc__)
    p.set_defaults(dataset=LIFESTYLE, import_historical=False, reuse_historical_baseline=False)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    if args.dataset != LIFESTYLE:
        p.error('Use train.py for ACS datasets')
    if min(args.epochs, args.batch_size, args.threads) < 1:
        p.error('epochs, batch-size and threads must be positive')
    train_lifestyle(args)


if __name__ == '__main__':
    main()
