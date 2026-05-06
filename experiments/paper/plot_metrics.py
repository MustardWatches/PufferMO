from pymoo.indicators.hv import HV  # type: ignore
from scipy import stats  # type: ignore
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import os

# CONFIGURATION
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
    }
)

ENVS = ["moba", "snake", "tetris"]
ALGORITHMS = ["PPO", "MOPPO", "MOPPO (no conditioning)"]
COLORS = {"PPO": "#1f77b4", "MOPPO": "#d62728", "MOPPO (no conditioning)": "#2ca02c"}
OBJECTIVE_LABELS = {
    "moba": ["death", "xp", "tower"],
    "snake": ["food", "corpse", "death"],
    "tetris": ["combo", "drop", "rotate"],
}


# DATA LOADING
def load_env_data(path, env):
    """Load NPZ files for an environment, return dict keyed by algorithm."""
    path = os.path.join(path, env)
    data = {}
    for algo in ALGORITHMS:
        filepath = os.path.join(path, f"{algo}.npz")
        if os.path.exists(filepath):
            data[algo] = dict(np.load(filepath, allow_pickle=True))
    return data


def load_all_data(path):
    """Load data for all environments."""
    return {env: load_env_data(path, env) for env in ENVS}


# METRIC COMPUTATION
def compute_nondominated_mask(points):
    """Return boolean mask of Pareto-optimal points (maximization)."""
    n = len(points)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if mask[i]:
            dominated = np.all(points >= points[i], axis=1) & np.any(
                points > points[i], axis=1
            )
            if dominated.any():
                mask[i] = False
    return mask


def normalize_returns(returns, min_vals, ranges):
    """Min-max normalize returns to [0, 1] using provided range."""
    return (returns - min_vals) / ranges


def compute_returns_range(all_returns, eps=1e-12):
    """Compute min-max range for each dimension."""
    min_vals = all_returns.min(axis=0)
    max_vals = all_returns.max(axis=0)
    ranges = np.maximum(max_vals - min_vals, eps)
    return min_vals, ranges


def compute_hypervolume(normalized_front):
    """Compute hypervolume (maximization) with normalized front in [0,1]."""
    if len(normalized_front) == 0:
        return 0.0

    # Convert maximization -> minimization
    front = 1.0 - normalized_front

    # Reference strictly worse than worst (worst ≈ 1 after transform)
    ref_point = np.ones(front.shape[1]) * 1.1

    return HV(ref_point=ref_point)(front)


def compute_sparsity(front):
    """Compute sparsity of normalized Pareto front."""
    if len(front) <= 1:
        return 0.0
    sorted_front = np.sort(front, axis=0)
    diffs = np.diff(sorted_front, axis=0)
    return np.sum(diffs**2) / (len(front) - 1)


def compute_utility(weights, returns):
    """Compute expected utility (max over returns for each weight vector)."""
    return (returns @ weights.T).max(axis=0)


def compute_cosine_similarity(weights, returns):
    """Compute cosine similarity between weights and best matching returns."""
    # Compute cosine similarity for each weight-return pair
    w_norm = np.linalg.norm(weights, axis=1)
    r_norm = np.linalg.norm(returns, axis=1)
    dot_product = np.sum(weights * returns, axis=1)

    # Avoid division by zero
    denom = w_norm * r_norm
    denom = np.maximum(denom, 1e-8)

    return dot_product / denom


def compute_spearman(weights, returns, dim):
    """Compute Spearman correlation for a dimension."""
    w, r = weights[:, dim], returns[:, dim]
    if np.std(w) > 1e-8 and np.std(r) > 1e-8:
        result = stats.spearmanr(w, r)
        return result.statistic, result.pvalue
    return np.nan, np.nan


def compute_metrics(all_data):
    """Compute all metrics for all environments and algorithms."""
    metrics = {env: {} for env in ENVS}

    for env in ENVS:
        env_data = all_data[env]

        # Compute returns range across all algorithms for normalization
        all_returns = np.vstack(
            [d["discounted_returns"][:, 0] for d in env_data.values()]
        )
        min_vals, ranges = compute_returns_range(all_returns)

        # Use MOPPO weights as surrogate weight distribution
        moppo_weights = env_data["MOPPO"]["weights"]

        for algo in ALGORITHMS:
            d = env_data[algo]
            returns = d["discounted_returns"][:, 0]
            nd_mask = compute_nondominated_mask(returns)

            norm_returns = normalize_returns(returns, min_vals, ranges)
            norm_nd_returns = norm_returns[nd_mask]

            # Utility and cosine similarity use all returns, not just Pareto front
            utilities = compute_utility(moppo_weights, returns)
            # Cosine similarity is computed with normalized returns to be consistent with other metrics
            cosine_sims = compute_cosine_similarity(moppo_weights, norm_returns)

            m = {
                "hypervolume": compute_hypervolume(norm_nd_returns),
                "sparsity": compute_sparsity(norm_nd_returns),
                "expected_utility": {
                    "mean": utilities.mean(),
                    "std": utilities.std(),
                },
                "cosine_similarity": {
                    "mean": cosine_sims.mean(),
                    "std": cosine_sims.std(),
                },
            }

            # Spearman only for MOPPO
            if algo == "MOPPO":
                weights = d["weights"]
                m["spearman"] = [
                    compute_spearman(weights, returns, dim)
                    for dim in range(returns.shape[1])
                ]

            metrics[env][algo] = m

    return metrics


# PLOTTING
def plot_pareto_fronts(all_data, output_dir):
    """Plot 2D projections of Pareto fronts for each environment."""
    dims_pairs = [(0, 1), (0, 2), (1, 2)]

    for env in ENVS:
        env_data = all_data[env]
        obj_labels = OBJECTIVE_LABELS[env]

        fig, axes = plt.subplots(1, 3, figsize=(10, 2.5))

        for algo in ALGORITHMS:
            returns = env_data[algo]["discounted_returns"][:, 0]
            nd_mask = compute_nondominated_mask(returns)
            color = COLORS[algo]

            for col, (d1, d2) in enumerate(dims_pairs):
                ax = axes[col]
                ax.scatter(
                    returns[nd_mask, d1],
                    returns[nd_mask, d2],
                    color=color,
                    label=algo,
                    s=30,
                    edgecolors="black",
                    linewidth=0.5,
                    zorder=3,
                )
                ax.scatter(
                    returns[~nd_mask, d1],
                    returns[~nd_mask, d2],
                    color=color,
                    s=15,
                    alpha=0.3,
                    zorder=2,
                )
                ax.set_xlabel(obj_labels[d1])
                ax.set_ylabel(obj_labels[d2])
                ax.grid(alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=3,
            frameon=False,
        )
        plt.tight_layout()
        outpath = os.path.join(output_dir, f"pareto_{env}.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {outpath}")


def plot_metric_bars(metrics, metric_name, output_dir):
    """Plot bars for a metric across all envs (3 subplots, independent y-axes)."""
    fig, axes = plt.subplots(1, 3, figsize=(5, 1.5))

    x = np.arange(len(ALGORITHMS))
    width = 0.85

    for i, env in enumerate(ENVS):
        ax = axes[i]
        env_metrics = metrics[env]

        values, errors, colors_list = [], [], []
        for algo in ALGORITHMS:
            m = env_metrics[algo].get(metric_name)
            if isinstance(m, dict):
                values.append(m["mean"])
                errors.append(m["std"])
            elif m is not None:
                values.append(m)
                errors.append(0)
            else:
                values.append(0)
                errors.append(0)
            colors_list.append(COLORS[algo])

        ax.bar(
            x,
            values,
            width,
            yerr=errors if any(errors) else None,
            capsize=2,
            color=colors_list,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 0.8},
        )
        ax.set_xlabel(env.capitalize(), labelpad=8)
        ax.set_xticks([])
        ax.set_xlim(-0.5, len(ALGORITHMS) - 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    axes[0].set_ylabel(metric_name.replace("_", " "))

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[a], edgecolor="black")
        for a in ALGORITHMS
    ]
    fig.legend(
        handles,
        ALGORITHMS,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=False,
    )
    plt.tight_layout()
    outpath = os.path.join(output_dir, f"bars_{metric_name}.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def plot_correlation_scatter(all_data, output_dir):
    """Plot weight vs return: MOPPO as scatter with mean line and error bars, others as horizontal lines."""
    for env in ENVS:
        env_data = all_data[env]
        obj_labels = OBJECTIVE_LABELS[env]

        fig, axes = plt.subplots(1, 3, figsize=(10, 2.5))

        for algo in ALGORITHMS:
            d = env_data[algo]
            returns = d["discounted_returns"][:, 0]
            color = COLORS[algo]

            for dim in range(3):
                ax = axes[dim]

                if algo == "MOPPO":
                    weights = d["weights"]
                    unique_w = np.unique(weights[:, dim])
                    means = [
                        returns[weights[:, dim] == w, dim].mean() for w in unique_w
                    ]
                    stds = [returns[weights[:, dim] == w, dim].std() for w in unique_w]
                    ax.errorbar(
                        unique_w,
                        means,
                        yerr=stds,
                        color=color,
                        lw=1,
                        label=algo,
                        capsize=2,
                        capthick=1,
                        elinewidth=1,
                        zorder=3,
                    )
                else:
                    mean, std = returns[:, dim].mean(), returns[:, dim].std()
                    x_line = np.array([0, 1])
                    y_line = np.array([mean, mean])

                    # Plot line and shaded band for std
                    ax.plot(
                        x_line, y_line, "-", color=color, label=algo, lw=1, alpha=0.9
                    )
                    ax.fill_between(
                        x_line, mean - std, mean + std, color=color, alpha=0.2
                    )

        for dim in range(3):
            axes[dim].set_xlabel(rf"$w_{{\text{{{obj_labels[dim]}}}}}$")
            axes[dim].set_ylabel(obj_labels[dim])
            axes[dim].grid(alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=3,
            frameon=False,
        )
        plt.tight_layout()
        outpath = os.path.join(output_dir, f"correlation_scatter_{env}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {outpath}")


def plot_correlation_bars(metrics, output_dir):
    """Plot Spearman correlation bars for MOPPO only (3 envs, shared y-axis)."""
    _, axes = plt.subplots(1, 3, figsize=(5, 2), sharey=True)

    x = np.arange(3)
    width = 0.85
    color = COLORS["MOPPO"]

    for i, env in enumerate(ENVS):
        ax = axes[i]
        obj_labels = OBJECTIVE_LABELS[env]
        spearman = metrics[env]["MOPPO"]["spearman"]
        values = [s[0] for s in spearman]
        pvalues = [s[1] for s in spearman]

        ax.bar(x, values, width, color=color, edgecolor="black", linewidth=0.5)

        for j, (bar_val, pval) in enumerate(zip(values, pvalues)):
            if np.isnan(pval):
                continue
            txt = r"$p<.001$" if pval < 0.001 else rf"$p={pval:.2f}$"
            y_pos = bar_val + 0.03 if bar_val >= 0 else bar_val - 0.03
            va = "bottom" if bar_val >= 0 else "top"
            ax.text(x[j], y_pos, txt, ha="center", va=va, fontsize=6, alpha=0.7)

        ax.set_xlabel(env.capitalize(), labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(obj_labels)
        ax.set_xlim(-0.5, 2.5)
        ax.tick_params(
            axis="x",
            top=True,
            bottom=False,
            labeltop=True,
            labelbottom=False,
            length=0,
            pad=8,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    axes[0].set_ylabel(r"spearman $\rho$")
    plt.tight_layout()
    outpath = os.path.join(output_dir, "bars_correlation.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# OUTPUT METRICS TEXT FILE
def save_metrics_file(metrics, output_dir):
    """Save minimal metrics text file."""
    lines = ["METRICS SUMMARY", "=" * 60, ""]

    for env in ENVS:
        lines.append(f"[{env.upper()}]")
        lines.append(
            f"{'Algorithm':<25} {'HV':>12} {'Sparsity':>12} {'Utility':>20} {'Cosine Sim':>20}"
        )
        lines.append("-" * 90)

        for algo in ALGORITHMS:
            m = metrics[env][algo]
            hv = f"{m['hypervolume']:.4f}"
            sp = f"{m['sparsity']:.4f}"
            ut = f"{m['expected_utility']['mean']:.3f} ± {m['expected_utility']['std']:.3f}"
            cs = f"{m['cosine_similarity']['mean']:.3f} ± {m['cosine_similarity']['std']:.3f}"
            lines.append(f"{algo:<25} {hv:>12} {sp:>12} {ut:>20} {cs:>20}")

        lines.append("")
        lines.append(
            f"{'Algorithm':<25} "
            + " ".join([f"{o:>18}" for o in OBJECTIVE_LABELS[env]])
        )
        for algo in ALGORITHMS:
            if "spearman" not in metrics[env][algo]:
                continue
            corrs = metrics[env][algo]["spearman"]
            corr_strs = []
            for stat, pval in corrs:
                if np.isnan(stat):
                    corr_strs.append("N/A")
                else:
                    p = "p<.001" if pval < 0.001 else f"p={pval:.3f}"
                    corr_strs.append(f"{stat:.3f} ({p})")
            lines.append(f"{algo:<25} " + " ".join([f"{c:>18}" for c in corr_strs]))

        lines.append("")

    outpath = os.path.join(output_dir, "metrics.txt")
    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {outpath}")


def main():
    data_dir = "./data/static_adaptation"
    output_dir = "./figures/static_adaptation"

    print("Loading data...")
    all_data = load_all_data(data_dir)

    print("Computing metrics...")
    metrics = compute_metrics(all_data)

    print("Generating plots...")
    plot_pareto_fronts(all_data, output_dir)
    plot_metric_bars(metrics, "hypervolume", output_dir)
    plot_metric_bars(metrics, "sparsity", output_dir)
    plot_metric_bars(metrics, "expected_utility", output_dir)
    plot_metric_bars(metrics, "cosine_similarity", output_dir)
    plot_correlation_scatter(all_data, output_dir)
    plot_correlation_bars(metrics, output_dir)
    save_metrics_file(metrics, output_dir)

    print("Done!")


if __name__ == "__main__":
    main()


# Command used to generate the plots in the paper:
"""
uv run python plot_metrics.py
"""
