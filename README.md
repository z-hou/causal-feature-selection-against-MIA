# Causal Feature Selection and Membership Inference

This project compares all input features, the target's parents, and its Markov
blanket using MLP prediction models and membership inference attacks. Additional
experiments evaluate random feature selection and adaptive RMIA references.

## 1. Environment

Run all commands from the project root in Bash. Use Python 3.11 and the pinned
package versions used by the saved checkpoints:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
```

The experiments use CPU execution with two PyTorch threads. Matching package
versions, hardware, and numerical libraries is important for close numerical
agreement; retraining on another platform is not guaranteed to produce identical
weights. The random-selection evaluation also checks the recorded package versions.

## 2. Saved experiment configuration

The seven datasets are `ACSEmployment`, `ACSPublicCoverage`, `ACSMobility`,
`ACSIncome`, `ACSTravelTime`, `half-a-million-lifestyle`, and `covertype`.

- Target training/test sets: 10,000 records each.
- Reference pool: 20,000 records; each of eight references uses 10,000 for training
  and 10,000 for testing. Each record belongs to four reference training sets.
- Additional calibration set: 5,000 records.
- MLP hidden layers: 512, 256, 128; 100 epochs; batch size 256.
- Data and target seed: 42; reference seeds: 43 through 50.
- Baseline RMIA: the same eight full-feature references for all three targets.
- Adaptive RMIA: references use the feature subset of the corresponding target.
- Random selection: sample from all original features without replacement, using
  seed 42 and the same feature count as the Markov blanket; references remain
  full-feature models.
- Reported attack metrics: AUC, TPR at 1% FPR, and TPR at 0.1% FPR.

ACS datasets use California 2018 records. ACSEmployment uses a saved historical
split and graph. The other saved graphs use discrete BIC hill climbing with five
quantile bins, maximum indegree 3, maximum iterations 10,000, and
`PYTHONHASHSEED=0`. Graph discovery uses target training records only.

## 3. Verify the supplied artifacts

These commands check the existing artifacts without training or rewriting results:

```bash
python 'random selection/verify.py'
python 'adaptive attack/verify.py'
```

Expected summaries:

- Random selection: seven datasets, 56 reference files, and 28 result rows.
- Adaptive attack: 168 reference files and 21 result rows.

The random-selection verifier recomputes attack metrics from saved attack scores.
The adaptive verifier checks model/data/graph consistency and saved result
relationships. To recompute predictions and attacks, continue with Section 4.

Optional code tests require pytest, which is not included in `requirements.txt`:

```bash
python -m pip install pytest
python -m pytest -q -p no:cacheprovider test_pipeline.py 'adaptive attack/test_adaptive.py'
```

## 4. Recompute results from saved models

No downloads or model training are needed. Work on copies so the supplied results
remain available for comparison. The following setup requires a new
`reproduction/evaluation` directory and enough disk space for the model copies:

```bash
python - <<'PY'
from pathlib import Path
import shutil

work = Path('reproduction/evaluation')
work.mkdir(parents=True, exist_ok=False)
for name in ['data', 'models', 'results']:
    shutil.copytree(name, work / 'baseline' / name)
for experiment in ['random selection', 'adaptive attack']:
    for name in ['models', 'results']:
        shutil.copytree(Path(experiment) / name, work / experiment / name)
PY
```

### 4.1 Baseline RMIA and loss attack

```bash
set -e
datasets=(ACSEmployment ACSPublicCoverage ACSMobility ACSIncome ACSTravelTime half-a-million-lifestyle covertype)
for dataset in "${datasets[@]}"; do
    python RMIA.py --dataset "$dataset" --root reproduction/evaluation/baseline
done
python loss_attack.py --root reproduction/evaluation/baseline
```

RMIA uses `gamma=2`. With no `--a` argument, it uses `a=0.1` for historical
ACSEmployment and selects `a` using auxiliary calibration for the other datasets.
The loss attack uses negative per-example cross-entropy and needs no references.

### 4.2 Random selection and adaptive RMIA

Use the supplied baseline as the source and the copied experiment models as the
outputs. `--evaluate-only` prevents training missing models:

```bash
python 'random selection/run.py' --source-root . \
    --output 'reproduction/evaluation/random selection' --evaluate-only
python 'adaptive attack/run.py' --source . \
    --output 'reproduction/evaluation/adaptive attack' --evaluate-only
python 'adaptive attack/verify.py' --source . \
    --output 'reproduction/evaluation/adaptive attack'
```

The random-selection verifier has no custom-output option; its Section 3 command
checks the supplied artifacts. The evaluation command above checks model
provenance and reproduces the baseline Markov-blanket metrics before computing the
random-selection results.

### 4.3 Compare recomputed CSV results

Run after all Section 4 evaluation commands complete:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

work = Path('reproduction/evaluation')
pairs = [(Path('results'), work / 'baseline/results'),
         (Path('random selection/results'), work / 'random selection/results'),
         (Path('adaptive attack/results'), work / 'adaptive attack/results')]
checked = 0
for original, reproduced in pairs:
    for path in sorted(original.rglob('*.csv')):
        other = reproduced / path.relative_to(original)
        expected = pd.read_csv(path)
        actual = pd.read_csv(other)
        # Compare results independently of file-serialization provenance.
        hash_columns = [c for c in expected.columns if c.endswith('_sha256')]
        expected = expected.drop(columns=hash_columns)
        actual = actual.drop(columns=hash_columns)
        keys = ['dataset', 'experiment']
        expected = expected.sort_values(keys).reset_index(drop=True)
        actual = actual.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False,
                                      rtol=0, atol=1e-12)
        checked += 1
print(f'Compared {checked} CSV files successfully.')
PY
```

This checks numeric agreement to an absolute tolerance of `1e-12`, along with
non-hash column contents and identifiers. File-hash columns are excluded: the
saved ACSEmployment loss-attack CSV retains hashes from before path anonymization,
whereas a new evaluation records the current artifact hashes. This does not
change the stored metrics. The evaluation scripts check current data/graph/model
consistency independently. A metric mismatch should be investigated rather than
silently replacing the supplied baseline.

## 5. Retrain using the supplied splits and feature sets

Use this route to repeat model training while keeping data splits and selected
features fixed. It requires a new `reproduction/retrain` directory:

```bash
python - <<'PY'
from pathlib import Path
import shutil

work = Path('reproduction/retrain')
work.mkdir(parents=True, exist_ok=False)
shutil.copytree('data', work / 'data')
for graph in Path('results').glob('*/causal_graph.json'):
    destination = work / graph
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(graph, destination)
PY

set -e
datasets=(ACSEmployment ACSPublicCoverage ACSMobility ACSIncome ACSTravelTime half-a-million-lifestyle covertype)
for dataset in "${datasets[@]}"; do
    python train.py --dataset "$dataset" --root reproduction/retrain \
        --epochs 100 --batch-size 256 --threads 2
    python RMIA.py --dataset "$dataset" --root reproduction/retrain
done
python loss_attack.py --root reproduction/retrain
python 'random selection/run.py' --source-root reproduction/retrain \
    --output reproduction/retrain/random_selection --selection-seed 42
python 'adaptive attack/run.py' --source reproduction/retrain \
    --output reproduction/retrain/adaptive_attack
python 'adaptive attack/verify.py' --source reproduction/retrain \
    --output reproduction/retrain/adaptive_attack
```

`train.py` supports all seven datasets. `train_lifestyle.py` and
`train_covertype.py` are optional dataset-specific entry points for the same
training implementation. Training requires empty model output directories.
These commands retrain all baseline models, including ACSEmployment; they do not
import historical weights. Compare retrained metrics separately from the
saved-model evaluation in Section 4.

## 6. Optional: prepare data and learn a graph again

For a fresh experiment, choose an unused root and a dataset. For example:

```bash
python data_prepare.py --dataset ACSPublicCoverage --root reproduction/fresh \
    --year 2018 --state CA --seed 42
python Learn_causal_graph.py --dataset ACSPublicCoverage --root reproduction/fresh \
    --bins 5 --max-indegree 3 --max-iter 10000
python train.py --dataset ACSPublicCoverage --root reproduction/fresh \
    --epochs 100 --batch-size 256 --threads 2
python RMIA.py --dataset ACSPublicCoverage --root reproduction/fresh
python loss_attack.py --dataset ACSPublicCoverage --root reproduction/fresh
```

Data preparation downloads source data when needed. For local source files, use
`--dataset half-a-million-lifestyle --csv path/to/user_data.csv` or
`--dataset covertype --covtype-file path/to/covtype.data.gz` with
`data_prepare.py`. Lifestyle is a synthetic dataset; Covertype uses the original
54 input columns and seven classes.

To reuse the supplied historical ACSEmployment data, graph, and checkpoints in a
new directory, use the explicit import workflow:

```bash
python data_prepare.py --dataset ACSEmployment --root reproduction/historical --historical
python Learn_causal_graph.py --dataset ACSEmployment --root reproduction/historical --historical
python train.py --dataset ACSEmployment --root reproduction/historical --import-historical
python RMIA.py --dataset ACSEmployment --root reproduction/historical --verify-historical
```

The historical workflow copies supplied artifacts; it does not reconstruct their
original discovery process. Running ACSEmployment without these import options
creates a fresh experiment and need not reproduce the saved historical results.
Likewise, fresh downloads or a newly learned graph may differ from saved artifacts;
use Sections 4 and 5 when the saved splits and graph are required.

## 7. Output files

| Output | Contents |
|---|---|
| `results/<dataset>/comparison.csv` | Baseline RMIA, three feature strategies |
| `results/<dataset>/loss_attack.csv` | Loss attack, three feature strategies |
| `results/loss_attack.csv` | Combined loss-attack results for all datasets |
| `random selection/results/comparison.csv` | Seven random-selection rows plus 21 baseline rows |
| `adaptive attack/results/comparison.csv` | 21 adaptive RMIA rows, including calibrated variants |
| `results/<dataset>/causal_graph.json` | Graph, feature sets, and data provenance |
| Experiment `results/<dataset>/manifest.json` | Model/data hashes and experiment metadata |

Custom `--root` or `--output` arguments relocate these outputs as shown above.
Keep data, graphs, checkpoints, and manifests together: the scripts enforce hash
consistency. Source paths recorded in experiment manifests are relative to the
project root. Logs are optional and are not required by the reproduction scripts.
