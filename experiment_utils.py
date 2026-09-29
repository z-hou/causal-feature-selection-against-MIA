"""Paths, metadata and split checks shared by the four experiment stages."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd

LIFESTYLE = 'half-a-million-lifestyle'
COVERTYPE = 'covertype'
DATASETS = ['ACSEmployment', 'ACSPublicCoverage', 'ACSMobility', 'ACSIncome', 'ACSTravelTime', LIFESTYLE, COVERTYPE]
NAMES = ['member', 'nonmember', 'ref0', 'ref1', 'calibration']
STRATEGIES = ['all_features', 'label_parents', 'markov_blanket']
ROOT = Path(__file__).resolve().parent
HISTORICAL = ROOT / 'data/ACSEmployment'
HISTORICAL_GRAPH = ROOT / 'results/ACSEmployment/causal_graph.json'


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument('--dataset', choices=DATASETS, default='ACSEmployment')
    result.add_argument('--root', type=Path, default=ROOT, help='Root containing data, models and results')
    return result


def paths(args):
    return tuple(args.root / name / args.dataset for name in ['data', 'models', 'results'])


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def versions():
    return {name: importlib.metadata.version(name)
            for name in ['numpy', 'pandas', 'torch', 'pgmpy', 'folktables']}


def empty_directory(path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)


def load_data(directory):
    with np.load(directory / 'data.npz', allow_pickle=False) as saved:
        meta = json.loads(str(saved['metadata']))
        arrays = {key: saved[key].copy() for key in saved.files if key != 'metadata'}
    frames, labels, ids = {}, {}, {}
    classes = int(meta.get('classes', 2))
    row_split = meta.get('split_unit') == 'row'
    seen_ids, seen_households, seen_inputs = set(), set(), set()
    for name in NAMES:
        frame = pd.DataFrame(arrays[f'x_{name}'], columns=meta['features'])
        ids[name] = arrays[f'ids_{name}']
        labels[name] = arrays[f'y_{name}']
        if (list(frame) != meta['features'] or len(labels[name]) != len(frame)
                or not set(labels[name]).issubset(range(classes))):
            raise ValueError(f'Invalid features or labels: {name}')
        expected = 5000 if name == 'calibration' else 10000
        keys = set(map(tuple, frame.to_numpy()))
        households = set(ids[name]) if row_split else {':'.join(v.split(':')[:3]) for v in ids[name]}
        if (len(frame) != expected or len(set(ids[name])) != expected
                or seen_ids.intersection(ids[name]) or seen_households.intersection(households)
                or len(keys) != expected or seen_inputs.intersection(keys)):
            raise ValueError(f'Incorrect size or overlapping records/households/inputs: {name}')
        if not row_split and any(v.split(':')[:2] != [str(meta['year']), meta['state']] for v in ids[name]):
            raise ValueError(f'Expected the same state/year for every split: {name}')
        seen_ids.update(ids[name]); seen_households.update(households); seen_inputs.update(keys)
        frames[name] = frame
    mask = arrays['reference_membership']
    if mask.dtype != bool or mask.shape != (8, 20000):
        raise ValueError('Expected an 8 x 20000 boolean reference mask')
    if not (mask.sum(0) == 4).all() or not (mask.sum(1) == 10000).all():
        raise ValueError('References must have balanced 10000/10000 train/test splits')
    return meta, frames, labels, ids, mask


def save_data(directory, meta, frames, labels, ids, mask):
    pack = {'metadata': np.array(json.dumps(meta)), 'reference_membership': mask}
    for name in NAMES:
        pack[f'x_{name}'] = frames[name].to_numpy(dtype=str)
        pack[f'y_{name}'] = np.asarray(labels[name], dtype=np.int64)
        pack[f'ids_{name}'] = np.asarray(ids[name], dtype=str)
    np.savez_compressed(directory / 'data.npz', **pack)
