# Evaluation protocol

All policies are evaluated on the same list of held-out reset seeds, separate from the default demo seed range. Each episode is capped at 50 steps. Reported measures are success rate, wall-clock episode completion time, mean episode steps, grasp-lift rate, estimated object-drop rate, mean dense return, invalid-action episodes, and per-episode failure category.

The evaluation JSON stores the seed and metrics for every episode plus aggregate statistics. `scripts/plot_results.py` reads that JSON directly to create the comparison image. Collision rate is not reported because a collision counter was not implemented or validated for this Fetch wrapper. The evaluation sample size and PPO timestep count are declared in generated results; they should not be interpreted as a statistically powered benchmark or proof of convergence.

The expert baseline has strong prior structure and is expected to outperform short CPU PPO training. Learned-policy results are reported without post-hoc selection of favorable seeds.
