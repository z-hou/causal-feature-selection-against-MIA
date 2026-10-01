"""Learn a training-only feature/label DAG and extract label parents and blanket."""
import os
import shutil
os.environ.setdefault('MPLCONFIGDIR', '/tmp/causal-mpl-cache')
import pandas as pd
import networkx as nx
from experiment_utils import (parser, paths, load_data, read_json, write_json,
                              HISTORICAL_GRAPH, digest)


def feature_sets(nodes, edges, features):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if '__label__' not in graph or not nx.is_directed_acyclic_graph(graph):
        raise ValueError('Expected a DAG containing __label__')
    parents = set(graph.predecessors('__label__'))
    children = set(graph.successors('__label__'))
    blanket = parents | children
    for child in children:
        blanket.update(graph.predecessors(child))
    blanket.discard('__label__')
    return dict(all_features=list(features),
                label_parents=[f for f in features if f in parents],
                markov_blanket=[f for f in features if f in blanket])


def plot_graph(nodes, edges, selections, output, target_name=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    long_names = any(len(node) > 24 for node in nodes)
    if long_names:
        fig, (ax, legend) = plt.subplots(1, 2, figsize=(24, 13), gridspec_kw={'width_ratios': [1.3, 1]})
        names = {node: f'F{i + 1:02d}' for i, node in enumerate(selections['all_features'])}
        names['__label__'] = 'Y'
        for i, node in enumerate(nodes):
            text = 'Target label' if node == '__label__' else node
            legend.text(0, .98 - i * min(.033, .96 / len(nodes)),
                        f'{names[node]}  {text}', fontsize=9 if len(nodes) > 35 else 10, va='top')
        legend.axis('off')
        positions = nx.spring_layout(graph, seed=42, k=1.0, iterations=300)
        if len(nodes) > 35:
            blanket = selections['markov_blanket']
            other = [node for node in nodes if node != '__label__' and node not in blanket]
            shells = [group for group in [['__label__'], blanket, other] if group]
            positions = nx.shell_layout(graph, nlist=shells)
        size = 900
    else:
        if target_name is None:
            import folktables
            target_name = getattr(folktables, output.parent.name).target
        fig, ax = plt.subplots(figsize=(14, 10))
        names = {node: node for node in nodes}
        names['__label__'] = target_name
        positions = nx.spring_layout(graph, seed=42)
        size = 1800
    colors = []
    for node in nodes:
        color = '#b9dcf2'
        if node in selections['markov_blanket']:
            color = '#c9bbeb'
        if node in selections['label_parents']:
            color = '#f7b267'
        if node == '__label__':
            color = '#ed9696'
        colors.append(color)
    nx.draw_networkx(graph, positions, labels=names, ax=ax, font_size=8,
                     node_size=size, node_color=colors, arrowsize=16)
    ax.set_title('Candidate DAG: orange = parents; purple = other blanket features; red = target')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def learn(args):
    data, _, results = paths(args)
    meta, frames, labels, _, _ = load_data(data)
    out = results
    if args.historical and not meta['historical']:
        raise ValueError('Historical graph requires the historical data splits')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'causal_graph.json').exists():
        raise FileExistsError(out / 'causal_graph.json')
    if args.historical:
        if read_json(HISTORICAL_GRAPH)['mode'] != 'historical':
            raise ValueError('The supplied graph is no longer a historical graph; omit --historical to relearn it')
        shutil.copy2(HISTORICAL_GRAPH, out / 'causal_graph.json')
        shutil.copy2(HISTORICAL_GRAPH.with_suffix('.png'), out / 'causal_graph.png')
        print(read_json(HISTORICAL_GRAPH)['feature_sets'])
        return
    else:
        from pgmpy.estimators import HillClimbSearch, BicScore
        # Read raw saved columns, never the neural network's encoded input.
        if meta.get('split_unit') == 'row':
            raw = frames['member'].copy()
            for column in meta['numeric']:
                raw[column] = pd.to_numeric(raw[column], errors='raise')
        else:
            raw = frames['member'].apply(pd.to_numeric)
        raw['__label__'] = labels['member']
        discrete, mapping = pd.DataFrame(index=raw.index), {}
        for column in raw:
            if column in meta['numeric'] and raw[column].nunique() > 1:
                codes, bins = pd.qcut(raw[column], args.bins, labels=False, retbins=True, duplicates='drop')
                if len(bins) > 1:
                    discrete[column] = codes.astype(int)
                    mapping[column] = dict(edges=bins.tolist())
                    continue
            codes, states = pd.factorize(raw[column], sort=True)
            discrete[column] = codes
            mapping[column] = dict(states=states.tolist())
        graph = HillClimbSearch(discrete).estimate(scoring_method=BicScore(discrete),
                    max_indegree=args.max_indegree, max_iter=args.max_iter, show_progress=False)
        nodes, edges = list(graph.nodes()), sorted(graph.edges())
        provenance = dict(mode='fresh', method='discrete BIC hill climb', bins=args.bins,
                          max_indegree=args.max_indegree, max_iter=args.max_iter,
                          data_sha256=digest(data / 'data.npz'), discretization=mapping,
                          pythonhashseed=os.environ.get('PYTHONHASHSEED'),
                          input_representation=meta.get('input_representation', 'original saved feature columns'),
                          training_rows=len(raw), model_preprocessing_applied=False,
                          split_provenance='saved input splits; graph learned anew',
                          note='Observational candidate DAG; edge directions are not established causal mechanisms.')
    selections = feature_sets(nodes, edges, meta['features'])
    write_json(out / 'causal_graph.json', dict(nodes=nodes, edges=edges, feature_sets=selections, **provenance))
    plot_graph(nodes, edges, selections, out / 'causal_graph.png', meta.get('target'))
    print(selections)


def main():
    p = parser(__doc__)
    p.add_argument('--historical', action='store_true')
    p.add_argument('--bins', type=int, default=5)
    p.add_argument('--max-indegree', type=int, default=3)
    p.add_argument('--max-iter', type=int, default=10000)
    args = p.parse_args()
    if args.bins < 2 or args.max_indegree < 1 or args.max_iter < 1:
        p.error('Require bins >= 2, max-indegree >= 1, max-iter >= 1')
    learn(args)


if __name__ == '__main__':
    main()
