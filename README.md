
> **⚠️ Fork notice**: This is a fork of [PufferLib](https://github.com/PufferAI/PufferLib/tree/3.0)
> with algorithmic and environment extensions for multi-objective reinforcement learning (MORL).
> This work is part of the paper "Controllability in preference-conditioned multi-objective reinforcement learning", currently under review at NeuS 2026.

## Key differences from PufferLib

### Algorithmic extensions

- **Weight Conditioning**: Policy networks conditioned on preference weights via Dirichlet sampling
- **Multi-Objective PPO (MOPPO)**: Extended PPO to handle vector rewards with preference weight conditioning

### Environment extensions

- **Multi-objective environment variants**: `snake_mo`, `moba_mo`, `tetris_mo` with vectorized rewards
- **Reward decomposition**: Separate reward components (e.g., food/corpse/death in Snake, death/xp/distance/tower in MOBA)

### Experiments

Experiments on dynamic adaptation were performed using the default evaluation command in PufferLib on `tetris_mo` while recording the virtual environment screen using script `./record.sh`, e.g.:

```bash
./record.sh tetris.mp4 puffer eval puffer_tetris_mo --wandb --load-id <wandb-load-id>  # for MOPPO
```

Experiments on static adaptation were run on three environments
comparing non-conditioned baselines (PPO, MOPPO without conditioning)
against the weight-conditioned MOPPO.
Evaluation used 30 randomly sampled preference vectors per each of the two best models per environment and algorithm.
See [run_adaptation_experiment.py](run_adaptation_experiment.py) for details.

---

![figure](https://pufferai.github.io/source/resource/header.png)

[![PyPI version](https://badge.fury.io/py/pufferlib.svg)](https://badge.fury.io/py/pufferlib)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pufferlib)
![Github Actions](https://github.com/PufferAI/PufferLib/actions/workflows/install.yml/badge.svg)
[![](https://dcbadge.vercel.app/api/server/spT4huaGYV?style=plastic)](https://discord.gg/spT4huaGYV)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez5341)](https://twitter.com/jsuarez5341)

PufferLib is the reinforcement learning library I wish existed during my PhD. It started as a compatibility layer to make working with complex environments a breeze. Now, it's a high-performance toolkit for research and industry with optimized parallel simulation, environments that run and train at 1M+ steps/second, and tons of quality of life improvements for practitioners. All our tools are free and open source. We also offer priority service for companies, startups, and labs!

![Trailer](https://github.com/PufferAI/puffer.ai/blob/main/docs/assets/puffer_2.gif?raw=true)

All of our documentation is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support -- post here before opening issues. We're always looking for new contributors, too!

## Star to puff up the project!

<a href="https://star-history.com/#pufferai/pufferlib&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
 </picture>
</a>
