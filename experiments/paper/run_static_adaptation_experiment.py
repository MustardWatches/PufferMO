import os
import pickle
import numpy as np
import logging
import subprocess
import glob
from functools import lru_cache
from pymoo.util.ref_dirs import get_reference_directions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)

NUM_EVAL_WEIGHTS = 30
OUTPUT_DIR = "./data/static_adaptation"

REWARD_DIM_PER_ENV = {
    "puffer_moba_mo": 3,
    "puffer_snake_mo": 3,
    "puffer_tetris_mo": 3,
}

ADDITIONAL_ENV_ARGS = {
    "puffer_moba": {"max_ticks": 5000, "reward_death": -0.1},
    "puffer_moba_mo": {"max_ticks": 5000},
    "puffer_snake": {"max_ticks": 3000, "max_ticks_offset_mod": 1},
    "puffer_snake_mo": {"max_ticks": 3000, "max_ticks_offset_mod": 1},
    "puffer_tetris": {"max_ticks": 3000},
    "puffer_tetris_mo": {"max_ticks": 3000},
}

# The top models achieving higher performance and discounted return per environment and algorithm
MODELS = {
    "puffer_moba_mo": {
        "moppo": ["lltfqvn0", "wmf1spi3"],
        "moppo_no_cond": ["v2pvphn8", "b270xqzl"],
    },
    "puffer_snake_mo": {
        "moppo": ["ltvs6kjh", "2zmxsbeo"],
        "moppo_no_cond": ["mfc412qw", "wtjv38vo"],
    },
    "puffer_tetris_mo": {
        "moppo": ["d1cmozp0", "cchozc4a"],
        "moppo_no_cond": ["h5v2pfi4", "cyn4hl4q"],
    },
    "puffer_moba": {
        "ppo": ["tttx7bqy", "ma2on7l1"],
    },
    "puffer_snake": {
        "ppo": ["4hvxjp0c", "eacxwib0"],
    },
    "puffer_tetris": {
        "ppo": ["ljjvx7mw", "ht4k8745"],
    },
}

@lru_cache
def equally_spaced_weights(dim: int, n: int, seed: int = 42) -> list[np.ndarray]:
    """Taken from MORL-Baselines: https://github.com/LucasAlegre/morl-baselines/blob/main/morl_baselines/common/weights.py
    
    Generate weight vectors that are equally spaced in the weight simplex.

    It uses the Riesz s-Energy method from pymoo: https://pymoo.org/misc/reference_directions.html

    Args:
        dim: size of the weight vector
        n: number of weight vectors to generate
        seed: random seed
    """
    return list(get_reference_directions("energy", dim, n, seed=seed))

def parse_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)

    data_by_key = {key: [] for key in data['infos'][0].keys()}
    for info in data['infos']:
        for key, value in info.items():
            data_by_key[key].append(value)

    for key, value in data_by_key.items():
        data_by_key[key] = np.array(value) # type: ignore

    return data_by_key

def get_weight_conditioning(model_type):
    return model_type.startswith("mo") and "_no_cond" not in model_type

def get_additional_args(env_name, model_type):
    additional_args = []
    
    # Environment-specific arguments
    for k, v in ADDITIONAL_ENV_ARGS[env_name].items():
        additional_args += [f"--env.{k.replace('_', '-')}", str(v)]

    # Weight conditioning
    weight_conditioning = get_weight_conditioning(model_type)
    if model_type.startswith("mo"):
        additional_args += [
            "--policy.weight-conditioning", str(weight_conditioning),
        ]

    return additional_args

def adaptation_experiment(env_name):
    results = {}
    for model_type, load_ids in sorted(MODELS[env_name].items()):
        additional_args = get_additional_args(env_name, model_type)

        results[model_type] = []
        for load_id in load_ids:
            log = f"env:{env_name} algo:{model_type} id:{load_id}"
            logging.info(log)

            weight_conditioning = get_weight_conditioning(model_type)
            eval_weights = None if not weight_conditioning else equally_spaced_weights(
                dim=REWARD_DIM_PER_ENV[env_name],
                n=NUM_EVAL_WEIGHTS,
                seed=42,
            )

            for run_idx in range(NUM_EVAL_WEIGHTS):
                additional_run_args = []
                if weight_conditioning:
                    weight_str = '"[' + ",".join(
                        [f"{w:.6f}" for w in eval_weights[run_idx]] # type: ignore
                    ) + ']"'
                    additional_run_args += ["--eval-weights", weight_str]

                # Call puffer eval via subprocess to avoid memory leaks
                cmd = [
                    "puffer",
                    "eval",
                    env_name,
                    "--wandb",
                    "--load-id", load_id,
                    "--train.seed", str(run_idx),
                    "--skip-render",
                    "--save-eval-data",
                    "--env.freeze-on-done", "True",
                ] + additional_args + additional_run_args

                # Clean up old eval files, if any
                for f in glob.glob(f"eval_{env_name}_*.pkl"):
                    os.remove(f)

                output_file = os.path.join(
                    OUTPUT_DIR,
                    f"{env_name}_{model_type}_{load_id}_{run_idx}.pkl",
                )

                if os.path.exists(output_file):
                    logging.info(f"  Run {run_idx + 1}/{NUM_EVAL_WEIGHTS} - Skipping (already exists)")
                    continue

                logging.info(f"  Run {run_idx + 1}/{NUM_EVAL_WEIGHTS} - {' '.join(cmd)}")

                try:
                    # Log file for errors
                    subprocess_errors_file = "subprocess_errors.log"
                    with open(subprocess_errors_file, "w") as err:
                        subprocess.run(
                            cmd,
                            stderr=err,
                            stdout=subprocess.DEVNULL,
                            check=True,
                        )
                except Exception as e:
                    logging.error(f"Subprocess failed (check {subprocess_errors_file}): {e}")
                    continue

                # Load and store data
                result_files = glob.glob(f"eval_{env_name}_*.pkl")
                if not result_files:
                    logging.error(f"No eval data file found for run {run_idx}.")
                    continue

                os.makedirs(OUTPUT_DIR, exist_ok=True)
                os.rename(result_files[0], output_file)


for env_name in MODELS.keys():
    adaptation_experiment(env_name)
