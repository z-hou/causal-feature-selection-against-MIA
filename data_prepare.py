"""Prepare same-source ACS or lifestyle splits, or import historical ACS splits."""
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_utils import (parser, paths, NAMES, HISTORICAL, empty_directory,
                              load_data, save_data, LIFESTYLE, COVERTYPE, digest)


def prepare_covertype(args, out):
    """Read the unscaled UCI file; keep all 54 original input columns."""
    import urllib.request
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
    path = args.covtype_file
    if path is None:
        cache = Path(tempfile.gettempdir()) / 'causal-covertype-cache'
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / 'covtype.data.gz'
        if not path.exists():
            partial = path.with_suffix('.partial')
            with urllib.request.urlopen(url, timeout=120) as source, partial.open('wb') as target:
                shutil.copyfileobj(source, target)
            partial.replace(path)
    numeric = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
               'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
               'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
               'Horizontal_Distance_To_Fire_Points']
    wilderness = [f'Wilderness_Area_{i}' for i in range(1, 5)]
    soil = [f'Soil_Type_{i}' for i in range(1, 41)]
    features = numeric + wilderness + soil
    raw = pd.read_csv(path, header=None)
    if raw.shape != (581012, 55) or raw.isna().any().any():
        raise ValueError('Expected the complete original UCI Covertype file: 581012 rows, 55 columns')
    raw.columns = features + ['Cover_Type']
    if not all(pd.api.types.is_integer_dtype(raw[c]) for c in raw):
        raise ValueError('Original Covertype values must be integers, not scaled values')
    if (set(raw.Cover_Type) != set(range(1, 8))
            or not raw[wilderness + soil].isin([0, 1]).all().all()
            or not (raw[wilderness].sum(axis=1) == 1).all()
            or not (raw[soil].sum(axis=1) == 1).all()):
        raise ValueError('Invalid original Covertype labels or indicator columns')
    shuffled = raw.sample(frac=1, random_state=args.seed).drop_duplicates(subset=features)
    frames, labels, ids = {}, {}, {}
    cursor = 0
    for name in NAMES:
        count = 5000 if name == 'calibration' else 10000
        part = shuffled.iloc[cursor:cursor + count]
        frames[name] = part[features].astype(str)
        labels[name] = part.Cover_Type.to_numpy(dtype=np.int64) - 1
        ids[name] = np.array([f'covertype:{i}' for i in part.index], dtype=str)
        cursor += count
    meta = dict(dataset=COVERTYPE, target='Cover_Type', classes=7,
                label_states=list(range(1, 8)), label_mapping='original label minus one',
                features=features, numeric=numeric, seed=args.seed, historical=False,
                split_unit='row', distribution='same_source', synthetic=False,
                source=url, source_sha256=digest(path), source_rows=len(raw),
                unique_feature_rows=len(shuffled), input_representation='original unscaled 54 UCI columns',
                distribution_note='Disjoint unique-input random records from one source; spatial independence is not guaranteed.')
    save_data(out, meta, frames, labels, ids, reference_mask(args.seed))


def reference_mask(seed):
    mask = np.zeros((8, 20000), dtype=bool)
    for pair in range(4):
        order = np.arange(20000) if pair == 0 else np.random.default_rng(seed + 10000 + pair).permutation(20000)
        mask[2 * pair, order[:10000]] = True
        mask[2 * pair + 1, order[10000:]] = True
    return mask


def lifestyle_csv(csv_path):
    """Use an explicit CSV, otherwise download the public Kaggle archive to a temporary cache."""
    if csv_path is not None:
        return csv_path
    import urllib.request
    import zipfile
    cache = Path(tempfile.gettempdir()) / 'half-a-million-lifestyle-cache'
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / 'user_data.csv'
    if path.exists():
        return path
    url = 'https://www.kaggle.com/api/v1/datasets/download/anthonytherrien/half-a-million-lifestyle'
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    archive = cache / 'dataset.zip'
    with urllib.request.urlopen(request, timeout=120) as response, archive.open('wb') as target:
        shutil.copyfileobj(response, target)
    with zipfile.ZipFile(archive) as saved:
        files = [name for name in saved.namelist() if Path(name).name == 'user_data.csv']
        if len(files) != 1:
            raise ValueError('Expected exactly one user_data.csv in the Kaggle archive')
        with saved.open(files[0]) as source, path.open('wb') as target:
            shutil.copyfileobj(source, target)
    return path


def prepare_lifestyle(args, out):
    path = lifestyle_csv(args.csv)
    raw = pd.read_csv(path)
    target = 'Lifestyle Choice'
    excluded = ['First Name', 'Last Name', 'City', 'State', 'Country']
    if any(column not in raw for column in excluded + [target]):
        raise ValueError('Expected Lifestyle Choice and the excluded name/location columns')
    features = [column for column in raw if column not in excluded + [target]]
    if raw[features + [target]].isna().any().any():
        raise ValueError('Unexpected missing values in lifestyle data; inspect source before processing')
    numeric = raw[features].select_dtypes(include='number').columns.tolist()
    states = sorted(raw[target].unique().tolist())
    mapping = {value: index for index, value in enumerate(states)}
    shuffled = raw.sample(frac=1, random_state=args.seed).drop_duplicates(subset=features)
    if len(shuffled) < 45000:
        raise ValueError('Need at least 45000 unique feature records')
    frames, labels, ids = {}, {}, {}
    cursor = 0
    for name in NAMES:
        count = 5000 if name == 'calibration' else 10000
        part = shuffled.iloc[cursor:cursor + count]
        frames[name] = part[features].astype(str)
        labels[name] = part[target].map(mapping).to_numpy(dtype=np.int64)
        ids[name] = np.array([f'lifestyle:{index}' for index in part.index], dtype=str)
        cursor += count
    meta = dict(dataset=LIFESTYLE, target=target, classes=len(states), label_states=states,
                features=features, numeric=numeric, excluded_features=excluded,
                seed=args.seed, historical=False, split_unit='row', distribution='same_source',
                synthetic=True, source='https://www.kaggle.com/datasets/anthonytherrien/half-a-million-lifestyle',
                source_sha256=digest(path), source_rows=len(raw), unique_feature_rows=len(shuffled),
                distribution_note='Random disjoint unique-feature records from one synthetic source; no household identifiers.')
    save_data(out, meta, frames, labels, ids, reference_mask(args.seed))


def split_households(frame, sizes, seed):
    rng = np.random.default_rng(seed)
    grouped = frame.groupby('SERIALNO', sort=False).indices
    keys = list(grouped)
    rng.shuffle(keys)
    cursor, result = 0, []
    for size in sizes:
        blocks, count = [], 0
        while count < size and cursor < len(keys):
            indices = grouped[keys[cursor]]
            blocks.append(indices)
            count += len(indices)
            cursor += 1
        if count < size:
            raise ValueError(f'Insufficient distinct-household data for {size} rows')
        indices = np.concatenate(blocks)
        rng.shuffle(indices)
        result.append(frame.iloc[indices[:size]].copy())
    return result


def prepare(args):
    out, _, _ = paths(args)
    if args.historical and args.dataset != 'ACSEmployment':
        raise ValueError('Historical reproduction is available only for ACSEmployment')
    empty_directory(out)
    if args.dataset == COVERTYPE:
        prepare_covertype(args, out)
    elif args.dataset == LIFESTYLE:
        prepare_lifestyle(args, out)
    elif args.historical:
        shutil.copy2(HISTORICAL / 'data.npz', out / 'data.npz')
    else:
        import folktables
        source = folktables.ACSDataSource(survey_year=str(args.year), horizon='1-Year',
                                        survey='person', root_dir=str(args.raw_dir))
        raw = source.get_data(states=[args.state], download=True)
        task = getattr(folktables, args.dataset)
        # Apply the task filter BEFORE assigning households; PublicCoverage removes rows.
        raw = task._preprocess(raw.copy())
        raw = raw.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        x, _, _ = task.df_to_numpy(raw)
        raw = raw.loc[~pd.DataFrame(x).duplicated().to_numpy()].copy()
        splits = split_households(raw, [10000, 10000, 10000, 10000, 5000], args.seed)
        frames, labels, ids = {}, {}, {}
        for name, frame in zip(NAMES, splits):
            x, y, _ = task.df_to_numpy(frame)
            saved = pd.DataFrame(x, columns=task.features)
            saved['__label__'] = y.astype(np.int64)
            saved['__record_id__'] = (str(args.year) + ':' + args.state + ':'
                + frame.SERIALNO.astype(str) + ':' + frame.SPORDER.astype(str)).to_numpy()
            ids[name] = saved.pop('__record_id__').to_numpy(dtype=str)
            labels[name] = saved.pop('__label__').to_numpy(dtype=np.int64)
            frames[name] = saved.astype(str)
        mask = reference_mask(args.seed)
        meta = dict(dataset=args.dataset, features=list(task.features),
                    numeric=[f for f in ['AGEP', 'PINCP', 'WKHP', 'JWMNP'] if f in task.features],
                    year=args.year, state=args.state, seed=args.seed, historical=False)
        meta.update(distribution='same_state_year',
                    distribution_note='Household-disjoint, unique-input samples from one state/year; not unconditional IID.')
        save_data(out, meta, frames, labels, ids, mask)
    load_data(out)
    print(f'Data validated: {out}')


def main():
    p = parser(__doc__)
    p.add_argument('--historical', action='store_true')
    p.add_argument('--year', type=int, default=2018)
    p.add_argument('--state', default='CA')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--raw-dir', type=Path, default=Path(tempfile.gettempdir()) / 'causal-folktables-cache')
    p.add_argument('--csv', type=Path, help='Local lifestyle user_data.csv; otherwise download to temporary cache')
    p.add_argument('--covtype-file', type=Path, help='Original UCI covtype.data or covtype.data.gz')
    args = p.parse_args()
    if args.csv is not None and args.dataset != LIFESTYLE:
        p.error('--csv is only supported with --dataset half-a-million-lifestyle')
    if args.covtype_file is not None and args.dataset != COVERTYPE:
        p.error('--covtype-file requires --dataset covertype')
    prepare(args)


if __name__ == '__main__':
    main()
