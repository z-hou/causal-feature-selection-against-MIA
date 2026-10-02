# Reproducing the Experiments
Main experiment: we conduct three feature selection strategies under RMIA attack,

Addition experiment 1: Adaptive attack  
Addition experiment 2: Three feature selection strategies under Loss-based attack  
Addition experiment 3: Causal feature selection(Markov blanket) Vs Random selection  

The prepared datasets are saved in directory "data", Main experiment contains three stages:

1. Learn the causal graph and extract the target's parents and Markov blanket.
2. Train the target models and reference models.
3. Run RMIA, compute the metrics, and save the results as CSV files.

This workflow uses the existing data splits. No data downloading, generation, or
splitting is required. Steps 1–3 reproduce the main experiment. Steps 4–6 run
adaptive RMIA, the loss attack, and random feature selection using those outputs.

## Preparation

Use Python 3.11 and run commands from the project directory containing the Python
scripts. Install the dependencies:

```text
python -m pip install -r requirements.txt
```

Before starting Python, set `PYTHONHASHSEED=0` in your IDE's run configuration or
the environment settings used by your terminal. This fixes Python hash ordering
for graph discovery. Setting it inside an already running Python session is not
sufficient. The `-B` option in the commands below prevents bytecode cache creation.

Using your file manager:

1. Create a new directory named `reproduction/main/` inside the project.
2. Copy the supplied `data/` folder into it as `reproduction/main/data/`.
3. Leave `reproduction/main/models/` and `reproduction/main/results/` absent.
   The scripts will create them. Do not copy existing models, graphs, or CSVs.

The input layout should be:

```text
reproduction/main/data/
    ACSEmployment/data.npz
    ACSPublicCoverage/data.npz
    ACSMobility/data.npz
    ACSIncome/data.npz
    ACSTravelTime/data.npz
    half-a-million-lifestyle/data.npz
    covertype/data.npz
```

Copying these archives preserves all existing training, test, calibration, and
reference-membership splits. Keep the terminal in the project directory for all
commands below. Wait for each command to finish successfully before continuing.

## 1. Learn the causal graphs

Run graph discovery on the target training data. Each graph contains the original
input features and the target label. The code uses discrete BIC hill climbing,
five quantile bins, maximum indegree 3, and up to 10,000 search iterations.

```text
python -B Learn_causal_graph.py --dataset ACSEmployment --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000
python -B Learn_causal_graph.py --dataset ACSPublicCoverage --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000
python -B Learn_causal_graph.py --dataset ACSMobility --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000
python -B Learn_causal_graph.py --dataset ACSIncome --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000
python -B Learn_causal_graph.py --dataset ACSTravelTime --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000
python -B Learn_causal_graph.py --dataset half-a-million-lifestyle --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000
python -B Learn_causal_graph.py --dataset covertype --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000
```

For each dataset, outputs are saved under `reproduction/main/results/<dataset>/`:

- `causal_graph.json`: graph edges, selected feature sets, and metadata.
- `causal_graph.png`: visualization of the learned graph.

These commands learn new graphs from the supplied data, including ACSEmployment.
Do not add `--historical`; this workflow does not import previous graphs.

## 2. Train target and reference models

Each command trains:

- Three target models using all features, the target's parents, and its Markov blanket.
- Eight reference models using all input features. The same references are used
  for all three target strategies.

The MLP has hidden layers `(512, 256, 128)`. Training uses 100 epochs, batch size
256, two CPU threads.

```text
python -B train.py --dataset ACSEmployment --root reproduction/main --epochs 100 --batch-size 256 --threads 2
python -B train.py --dataset ACSPublicCoverage --root reproduction/main --epochs 100 --batch-size 256 --threads 2
python -B train.py --dataset ACSMobility --root reproduction/main --epochs 100 --batch-size 256 --threads 2
python -B train.py --dataset ACSIncome --root reproduction/main --epochs 100 --batch-size 256 --threads 2
python -B train.py --dataset ACSTravelTime --root reproduction/main --epochs 100 --batch-size 256 --threads 2
python -B train.py --dataset half-a-million-lifestyle --root reproduction/main --epochs 100 --batch-size 256 --threads 2
python -B train.py --dataset covertype --root reproduction/main --epochs 100 --batch-size 256 --threads 2
```

Each dataset produces 11 checkpoints in `reproduction/main/models/<dataset>/`:

```text
target_all_features.pt
target_label_parents.pt
target_markov_blanket.pt
8 reference models: ref0.pt ... ref7.pt
```

The target has 10,000 training and 10,000 test records. Each reference
uses 10,000 training and 10,000 test records from its separate reference pool.
All models are newly trained; no existing weights are imported.

## 3. Run RMIA and save CSV results

Run RMIA on the newly trained models. Keep `gamma=2` and omit `--a` so the code
selects `a` using auxiliary calibration for the newly learned graphs.

```text
python -B RMIA.py --dataset ACSEmployment --root reproduction/main --gamma 2
python -B RMIA.py --dataset ACSPublicCoverage --root reproduction/main --gamma 2
python -B RMIA.py --dataset ACSMobility --root reproduction/main --gamma 2
python -B RMIA.py --dataset ACSIncome --root reproduction/main --gamma 2
python -B RMIA.py --dataset ACSTravelTime --root reproduction/main --gamma 2
python -B RMIA.py --dataset half-a-million-lifestyle --root reproduction/main --gamma 2
python -B RMIA.py --dataset covertype --root reproduction/main --gamma 2
```

Each command prints the results and writes:

```text
reproduction/main/results/<dataset>/comparison.csv
```

There are seven CSV files and 21 result rows in total. Each file contains three
rows: `all_features`, `label_parents`, and `markov_blanket`.

| Column | Meaning |
|---|---|
| `dataset` | Dataset name |
| `experiment` | Feature-selection strategy |
| `feature_count` | Number of retained original features |
| `retained_features` | Names of retained features |
| `target_test_accuracy` | Target prediction accuracy on its test set |
| `rmia_auc` | Membership-inference ROC-AUC |
| `rmia_best_accuracy` | Maximum empirical attack accuracy over score thresholds |
| `tpr_at_1pct_fpr` | Maximum empirical TPR with FPR no greater than 1% |
| `tpr_at_0_1pct_fpr` | Maximum empirical TPR with FPR no greater than 0.1% |
| `rmia_a` | Selected RMIA correction parameter |
| `gamma` | RMIA likelihood-ratio threshold |


## 4. Adaptive RMIA

Important!!! Complete Steps 1–3 for all seven datasets first (Mandatory). Continue running commands from
the project directory. Use the newly generated `reproduction/main/` artifacts as
the baseline and choose a new output directory:

```text
python -B "adaptive attack/run.py" --source reproduction/main --output reproduction/adaptive_attack
```

The command runs all seven datasets. It keeps the main experiment's target models
and trains reference models using the same feature set as each target. Full-feature
references are copied from the main experiment; identical feature sets can reuse
references. Each target is attacked using eight feature-matched references.

The attack uses the baseline `rmia_a` and `gamma` from Step 3. It also evaluates a
separately calibrated `a` and records the calibrated metrics.

Outputs:

- `reproduction/adaptive_attack/models/<dataset>/<strategy>/ref0.pt` through `ref7.pt`.
- `reproduction/adaptive_attack/results/<dataset>/comparison.csv`: three strategy rows.
- `reproduction/adaptive_attack/results/comparison.csv`: 21 rows across seven datasets.

## 5. Loss attack

Use the target models trained in Step 2. This attack requires no additional model
training or reference models:

```text
python -B loss_attack.py --root reproduction/main
```

The command runs all seven datasets and all three target strategies. It uses each
sample's negative cross-entropy loss as the membership score and computes AUC,
TPR at 1% FPR, and TPR at 0.1% FPR. It does not use average training loss as a fixed
membership-decision threshold.

Outputs:

- `reproduction/main/results/<dataset>/loss_attack.csv`: three strategy rows.
- `reproduction/main/results/loss_attack.csv`: 21 rows across seven datasets.

The main RMIA `comparison.csv` files are not overwritten.

## 6. Random feature selection

Complete Steps 1–3 first and use a new output directory:

```text
python -B "random selection/run.py" --source-root reproduction/main --output reproduction/random_selection --selection-seed 42
```

The command runs all seven datasets. For each dataset, it samples features without
replacement from all original inputs, retaining the same number of features as
the learned Markov blanket. Overlap with Markov-blanket
features is allowed.

It trains one new target model per dataset and copies the eight full-feature
reference models from Step 2. It then runs non-adaptive RMIA using the baseline
`rmia_a` and `gamma` from Step 3.

Outputs:

- `reproduction/random_selection/models/<dataset>/target_random_selection.pt`.
- `reproduction/random_selection/models/<dataset>/ref0.pt` through `ref7.pt`.
- `reproduction/random_selection/results/comparison.csv`: 28 rows, consisting of
  seven new random-selection rows and 21 baseline rows copied from Step 3.

