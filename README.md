
# PufferMO

**PufferMO** is a performant framework for multi-objective reinforcement learning (MORL), built as an extension of [PufferLib 3.0](https://github.com/PufferAI/PufferLib/tree/3.0). It provides three multi-objective environment variants, a vectorized reward interface, and support for weight-conditioned Multi-Objective PPO (MOPPO).

> This work accompanies the paper **"Controllability in preference-conditioned multi-objective reinforcement learning"**, accepted at NeuS 2026.

---

## Contents

- [What is PufferMO?](#what-is-puffermo)
- [Setup and usage](#setup-and-usage)
- [MOPPO and linear preference](#moppo-and-linear-preference)
- [Reproducing the experiments reported in the paper](#reproducing-the-experiments-reported-in-the-paper)
- [Citation](#citation)

---

## What is PufferMO?

PufferMO extends PufferLib 3.0 with the following additions:

**Environments** — Three multi-objective variants of PufferLib's built-in environments, each exposing a decomposed vector reward instead of a scalar:

| Environment | Reward components |
|---|---|
| `puffer_tetris_mo` | drop, rotate, combo |
| `puffer_snake_mo` | food, corpse, death |
| `puffer_moba_mo` | death, distance, tower |

The original scalar-reward environments (`puffer_tetris`, `puffer_snake`, `puffer_moba`) remain available as baselines.

**Algorithm** — Multi-Objective PPO (MOPPO), which extends standard PPO with:
- A vectorized critic that operates on the reward vector
- Optional preference weight conditioning: the policy network receives a preference weight vector $\mathbf{w}$ as input alongside the observation

---

## Setup and usage

**Install**

```bash
uv venv
uv pip install -e .
source .venv/bin/activate

# Compile C environments after installation or any changes to C code
python setup.py build_ext --inplace --force
```

**Evaluate (random policy)**

```bash
puffer eval puffer_tetris_mo
puffer eval puffer_tetris_mo --train.device cpu --vec.overwork True  # CPU fallback
```

**Train**

```bash
# MOPPO without weight conditioning
puffer train puffer_tetris_mo

# MOPPO with weight conditioning, logging to wandb
puffer train puffer_tetris_mo --policy.weight-conditioning True --wandb

# See all options (default values found in <env_name>.ini and default.ini)
puffer train puffer_tetris_mo --help
```

**Evaluate a trained model**

```bash
puffer eval puffer_tetris_mo --load-model-path latest
puffer eval puffer_tetris_mo --wandb --load-id <wandb-run-id>
puffer eval puffer_tetris_mo --env.max-ticks 1000
```

---

## MOPPO and linear preference

### Reward decomposition

The multi-objective environments expose a reward vector rather than a scalar. For example, Tetris goes from:

$$r = r_\text{drop} + r_\text{rotate} + r_\text{combo}$$

to:

$$\mathbf{r} = \left[r_\text{drop},\ r_\text{rotate},\ r_\text{combo}\right]$$

### Linear preference

A preference is a weight vector $\mathbf{w}$ on the simplex ($w_i \geq 0$, $\sum_i w_i = 1$) that linearly combines reward components into a scalar utility:

$$u = \mathbf{w} \cdot \mathbf{r} = w_\text{drop} \cdot r_\text{drop} + w_\text{rotate} \cdot r_\text{rotate} + w_\text{combo} \cdot r_\text{combo}$$

![The linear preference simplex for Tetris](readme/simplex_tetris.png)

### Algorithm

MOPPO samples $\mathbf{w}$ from a Dirichlet distribution at each rollout, concatenates it with the policy's observation input (when weight conditioning is enabled), and uses the vectorized critic to compute per-component advantages before scalarizing via $\mathbf{w}$.

![Extending PPO to MOPPO](readme/ppo_to_moppo.png)

---

## Reproducing the experiments reported in the paper

Experiments compare three algorithms across all three environments:

| Algorithm | Description |
|---|---|
| PPO | Standard PPO on scalar-reward environment |
| MOPPO (no cond.) | MOPPO with vector rewards, no weight conditioning |
| MOPPO | MOPPO with weight conditioning (Dirichlet-sampled $\mathbf{w}$) |

### Static adaptation

Preference weights are fixed for the duration of each episode. Evaluation used 30 uniformly spaced preference vectors per model, with the two best-performing checkpoints per environment and algorithm. See [`run_static_adaptation_experiment.py`](experiments/paper/run_static_adaptation_experiment.py) for the full evaluation script.

### Dynamic adaptation

Preference weights change during an episode according to a user-specified schedule. The model is expected to adapt its behavior on the fly.

![Dynamic adaptation](readme/dynamic_adaptation.png)

To run dynamic adaptation evaluation (and record a video):

```bash
./record.sh tetris.mp4 puffer eval puffer_tetris_mo \
  --policy.weight-conditioning True \
  --wandb \
  --load-id <wandb-run-id> \
  --env.max-ticks 1000 \
  --eval-weights "{0: [0.5, 0.5, 0], 500: [0, 0, 1]}"
```

The `--eval-weights` argument takes a dictionary mapping tick offsets to weight vectors, allowing arbitrary preference schedules within a single episode.

### Recreating the figures

The data used to generate all figures is included under `experiments/paper/data/`. To recreate the figures, install the plotting dependencies and run the plotting scripts from within `experiments/paper/` (the figures will be saved to `experiments/paper/figures/`):

```bash
cd experiments/paper
uv sync
```

The exact commands are documented as comments at the bottom of [`plot_curves.py`](experiments/paper/plot_curves.py) (training curves and dynamic adaptation reward plot) and [`plot_metrics.py`](experiments/paper/plot_metrics.py) (static adaptation Pareto fronts, metric bars, and correlation plots).

The static adaptation data (`data/static_adaptation/`) was produced by [`run_static_adaptation_experiment.py`](experiments/paper/run_static_adaptation_experiment.py), which evaluates trained models across a grid of preference weights. Training curve data (`data/training/`) was collected from WandB, and the dynamic adaptation data (`data/dynamic_adaptation/`) was generated manually by using `puffer eval [...]` commands.

---

## Citation

If you use PufferMO in your research, please cite:

```bibtex
@inproceedings{delasherasmolins:2026,
  title     = {Controllability in preference-conditioned multi-objective reinforcement learning},
  author    = {P. de las Heras Molins and B. Yalcinkaya and L. Peters and D. Fridovich-Keil and G. Bakirtzis},
  year      = {2026},
  booktitle = {NeuS},
  note      = {In press}
}
```


