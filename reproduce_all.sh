# Run from the project directory: bash -e reproduce_all.sh
# The -e option stops execution if a command fails; -B prevents Python bytecode caches.
# Preparation (see README.md):
# - Install requirements.txt in the active Python 3.11 environment.
# - Copy the supplied data/ folder to reproduction/main/data/.
# - Start without reproduction/main/models/ or reproduction/main/results/,
#   and without reproduction/adaptive_attack/ or reproduction/random_selection/.
# No data generation or downloading is performed.

# Fix Python hash ordering before starting any Python process.
export PYTHONHASHSEED=0

# 1. Learn causal graphs and extract label parents and Markov blankets.
# Outputs: reproduction/main/results/<dataset>/causal_graph.json and .png.
python -B Learn_causal_graph.py --dataset ACSEmployment --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000
python -B Learn_causal_graph.py --dataset ACSPublicCoverage --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000
python -B Learn_causal_graph.py --dataset ACSMobility --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000
python -B Learn_causal_graph.py --dataset ACSIncome --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000
python -B Learn_causal_graph.py --dataset ACSTravelTime --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000
python -B Learn_causal_graph.py --dataset half-a-million-lifestyle --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000
python -B Learn_causal_graph.py --dataset covertype --root reproduction/main --bins 5 --max-indegree 3 --max-iter 10000

# 2. Train three target models and eight full-feature references per dataset.
# Outputs: reproduction/main/models/<dataset>/*.pt.
python -B train.py --dataset ACSEmployment --root reproduction/main --epochs 100 --batch-size 256 --threads 2
python -B train.py --dataset ACSPublicCoverage --root reproduction/main --epochs 100 --batch-size 256 --threads 2
python -B train.py --dataset ACSMobility --root reproduction/main --epochs 100 --batch-size 256 --threads 2
python -B train.py --dataset ACSIncome --root reproduction/main --epochs 100 --batch-size 256 --threads 2
python -B train.py --dataset ACSTravelTime --root reproduction/main --epochs 100 --batch-size 256 --threads 2
python -B train.py --dataset half-a-million-lifestyle --root reproduction/main --epochs 100 --batch-size 256 --threads 2
python -B train.py --dataset covertype --root reproduction/main --epochs 100 --batch-size 256 --threads 2

# 3. Evaluate main RMIA; select a using auxiliary calibration and keep gamma=2.
# Outputs: reproduction/main/results/<dataset>/comparison.csv (three rows each).
python -B RMIA.py --dataset ACSEmployment --root reproduction/main --gamma 2
python -B RMIA.py --dataset ACSPublicCoverage --root reproduction/main --gamma 2
python -B RMIA.py --dataset ACSMobility --root reproduction/main --gamma 2
python -B RMIA.py --dataset ACSIncome --root reproduction/main --gamma 2
python -B RMIA.py --dataset ACSTravelTime --root reproduction/main --gamma 2
python -B RMIA.py --dataset half-a-million-lifestyle --root reproduction/main --gamma 2
python -B RMIA.py --dataset covertype --root reproduction/main --gamma 2

# 4. Run adaptive RMIA for all seven datasets after Steps 1-3 finish.
# Reuse main targets; train feature-matched references and evaluate both a settings.
# Combined output: reproduction/adaptive_attack/results/comparison.csv.
python -B "adaptive attack/run.py" --source reproduction/main --output reproduction/adaptive_attack

# 5. Evaluate loss attack using existing targets; no additional training.
# Combined output: reproduction/main/results/loss_attack.csv.
python -B loss_attack.py --root reproduction/main

# 6. Train random-selection targets with the same feature count as the Markov blanket.
# Reuse full-feature references and baseline RMIA parameters.
# Combined output: reproduction/random_selection/results/comparison.csv.
python -B "random selection/run.py" --source-root reproduction/main --output reproduction/random_selection --selection-seed 42
