# Reproducibility

From repository root, create a virtual environment with Python 3.12 and install `requirements.txt`, then run commands below. CPU execution is supported; GPU is not required. Default seed is 7; evaluations use a separate held-out seed range.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
python scripts/collect_demonstrations.py --episodes 40 --seed 7
python scripts/validate_demonstrations.py
python scripts/train_behavior_cloning.py --epochs 100 --seed 7
python scripts/train_ppo.py --timesteps 8192 --n-steps 256 --batch-size 64 --seed 7
python scripts/evaluate_policies.py --episodes 20 --seed-start 1000
python scripts/plot_results.py
python scripts/evaluate_policies.py --episodes 20 --seed-start 1000 --observation-noise-std 0.01 --output experiments/results/robustness_noise_0_01.json
python scripts/run_demo.py --policy expert
pytest -q
```

Use a smaller PPO budget such as 2,048 timesteps for a quick smoke run. The precise CLI options and outputs are printed by each command. Generated datasets/checkpoints/results are intentionally excluded from Git by default; retain them locally or configure an artifact store for larger runs. Tests generate small temporary datasets and model files.
