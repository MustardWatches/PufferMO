import os

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mticker  # type: ignore
import numpy as np
from pymoo.indicators.hv import HV  # type: ignore
from scipy import stats  # type: ignore

# CONFIGURATION
FONT = "times"  # "lucidabr" or "times"
IMAGE_FORMAT = "pdf"
ENVS = ["moba", "snake", "tetris"]
ALGORITHMS = ["MOPPO", "MOPPO (no conditioning)", "PPO"]
COLORS = {"PPO": "#65a98f", "MOPPO (no conditioning)": "#0757b9", "MOPPO": "#ff8c00"}
OBJECTIVE_LABELS = {
    "moba": ["death", "xp", "tower"],
    "snake": ["food", "corpse", "death"],
    "tetris": ["combo", "drop", "rotate"],
}
ALPHA_BARS = 0.7

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": rf"\usepackage[T1]{{fontenc}}\usepackage{{amsmath}}\usepackage{{{FONT}}}",
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
def _set_dynamic_tick_format(ax, axis="both"):
    """Use the shared compact numeric formatting for tick labels."""
    formatter = mticker.FormatStrFormatter("%.3g")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)


def _set_plot_axes(
    ax, x_min, x_max, y_min, y_max, x_pad=0.0, y_pad=0.0, bottom_outward=16
):
    """Apply the shared axis formatting used by the paper plots."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.spines["bottom"].set_position(("outward", bottom_outward))
    ax.spines["left"].set_position(("outward", 16))
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xticks([x_min, x_max])
    _set_dynamic_tick_format(ax)
    ax.tick_params(
        axis="x", direction="in", length=4, labelbottom=True, width=1.2, pad=12
    )
    ax.spines["bottom"].set_bounds(x_min, x_max)

    ax.set_yticks([y_min, y_max])
    ax.tick_params(
        axis="y", direction="in", length=4, labelleft=True, width=1.2, pad=12
    )
    ax.spines["left"].set_bounds(y_min, y_max)


def _add_rug_ticks(ax, series):
    """Add colored inward ticks for each Pareto series."""
    import matplotlib.transforms as mtransforms

    trans_xd = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    trans_yd = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    # Offset the ticks off the axis line so they don't crowd the spine/data edge
    rug_gap_y = 0.015
    rug_gap_x = 0.03
    tick_len = 0.04

    for x_values, y_values, color in series:
        for x_value in np.unique(x_values):
            ax.plot(
                [x_value, x_value],
                [-rug_gap_x, -rug_gap_x + tick_len],
                transform=trans_xd,
                color=color,
                clip_on=False,
                lw=0.8,
                alpha=0.6,
            )
        for y_value in np.unique(y_values):
            ax.plot(
                [-rug_gap_y, -rug_gap_y + tick_len],
                [y_value, y_value],
                transform=trans_yd,
                color=color,
                clip_on=False,
                lw=0.8,
                alpha=0.6,
            )


def _align_axis_labels(fig, axes, gap=14, x_gap=None, y_gap=None):
    """Anchor visible axis labels a fixed display-space gap from each axes edge."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_gap = gap if x_gap is None else x_gap
    y_gap = gap if y_gap is None else y_gap

    for ax in np.atleast_1d(axes):
        axes_bbox = ax.get_window_extent(renderer)
        bottom_edge = (
            ax.spines["bottom"].get_window_extent(renderer)
            if ax.spines["bottom"].get_visible()
            else axes_bbox
        )
        left_edge = (
            ax.spines["left"].get_window_extent(renderer)
            if ax.spines["left"].get_visible()
            else axes_bbox
        )

        x_label_y = bottom_edge.y0 - x_gap
        x_axes_coord = ax.transAxes.inverted().transform((axes_bbox.x0, x_label_y))[1]
        ax.xaxis.set_label_coords(0.5, x_axes_coord)
        ax.xaxis.label.set_verticalalignment("top")

        y_label_x = left_edge.x0 - y_gap
        y_axes_coord = ax.transAxes.inverted().transform((y_label_x, axes_bbox.y0))[0]
        ax.yaxis.set_label_coords(y_axes_coord, 0.5)
        ax.yaxis.label.set_horizontalalignment("center")


def plot_pareto_fronts(all_data, output_dir, show_legend=False):
    """Plot 2D projections of Pareto fronts for each environment."""
    dims_pairs = [(0, 1), (0, 2), (1, 2)]

    for env_idx, env in enumerate(ENVS):
        env_data = all_data[env]
        obj_labels = OBJECTIVE_LABELS[env]

        fig, axes = plt.subplots(1, 3, figsize=(10, 2.5))
        series = [[] for _ in range(3)]

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
                    s=40,
                    edgecolors="black",
                    linewidth=0.5,
                    zorder=3,
                )
                ax.scatter(
                    returns[~nd_mask, d1],
                    returns[~nd_mask, d2],
                    color=color,
                    s=30,
                    edgecolors="none",
                    linewidth=0,
                    alpha=0.3,
                    zorder=2,
                )

                ax.set_xlabel(obj_labels[d1], fontsize=12)
                ax.set_ylabel(obj_labels[d2], fontsize=12)
                series[col].append((returns[:, d1], returns[:, d2], color))
        for col, (d1, d2) in enumerate(dims_pairs):
            x_values = np.concatenate(
                [env_data[algo]["discounted_returns"][:, 0, d1] for algo in ALGORITHMS]
            )
            y_values = np.concatenate(
                [env_data[algo]["discounted_returns"][:, 0, d2] for algo in ALGORITHMS]
            )
            x_min, x_max = x_values.min(), x_values.max()
            y_min, y_max = y_values.min(), y_values.max()
            _set_plot_axes(
                axes[col],
                x_min,
                x_max,
                y_min,
                y_max,
                # Extra padding vs. other plots leaves room for the rug ticks near the axis line
                x_pad=(x_max - x_min) * 0.16,
                y_pad=(y_max - y_min) * 0.16,
            )
            _add_rug_ticks(axes[col], series[col])

        has_legend = show_legend or env_idx == 0
        if has_legend:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.1),
                ncol=3,
                frameon=False,
                fontsize=12,
            )
        plt.tight_layout()
        _align_axis_labels(fig, axes)
        suffix = "_with_legend" if has_legend else ""
        outpath = os.path.join(output_dir, f"pareto_{env}{suffix}.{IMAGE_FORMAT}")
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        print(f"Saved: {outpath}")


def plot_metric_bars(metrics, metric_name, output_dir):
    """Plot bars for a metric across all envs (3 subplots, independent y-axes)."""
    fig, axes = plt.subplots(1, 3, figsize=(5, 1.5))

    x = np.arange(len(ALGORITHMS))
    width = 0.85
    metric_label = metric_name.replace("_", " ")

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
            capsize=1,
            color=colors_list,
            edgecolor="none",
            linewidth=0,
            error_kw={"linewidth": 0.6, "capsize": 2, "capthick": 0.6},
            alpha=ALPHA_BARS,
        )
        ax.set_xlabel(env, fontsize=12)
        ax.set_xticks([])
        ax.set_xlim(-0.5, len(ALGORITHMS) - 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ymax = max(v + e for v, e in zip(values, errors))
        if ymax > 0:
            magnitude = 10 ** np.floor(np.log10(ymax))
            ymax = np.ceil(ymax / magnitude * 10) / 10 * magnitude
        yticks = [0, ymax]
        ax.set_yticks(yticks)
        _set_dynamic_tick_format(ax, axis="y")
        ax.yaxis.set_tick_params(direction="in", length=4, pad=12)
        ax.spines["left"].set_bounds(yticks[0], yticks[-1])
        ax.spines["left"].set_position(("outward", 16))
        if i == 0:
            fontsize = 6 if len(metric_label) > 15 else 8
            ax.set_ylabel(metric_label, fontsize=fontsize)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[a], edgecolor="none")
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
    _align_axis_labels(fig, axes)
    outpath = os.path.join(output_dir, f"bars_{metric_name}.{IMAGE_FORMAT}")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def plot_correlation_scatter(all_data, output_dir, show_legend=False):
    """Plot weight vs return: MOPPO as scatter with mean line and error bars, others as horizontal lines."""
    for env_idx, env in enumerate(ENVS):
        env_data = all_data[env]
        obj_labels = OBJECTIVE_LABELS[env]

        fig, axes = plt.subplots(1, 3, figsize=(10, 2.5))

        series = [[] for _ in range(3)]
        y_lo = [np.inf] * 3
        y_hi = [-np.inf] * 3

        for algo in ALGORITHMS[::1]:
            d = env_data[algo]
            returns = d["discounted_returns"][:, 0]
            color = COLORS[algo]

            for dim in range(3):
                ax = axes[dim]

                if algo == "MOPPO":
                    weights = d["weights"]
                    unique_w = np.unique(weights[:, dim])
                    means = np.array(
                        [returns[weights[:, dim] == w, dim].mean() for w in unique_w]
                    )
                    ax.plot(
                        unique_w,
                        means,
                        color=color,
                        lw=1.5,
                        label=algo,
                        zorder=3,
                    )
                    series[dim].append((unique_w, means, color, False))
                    y_lo[dim] = min(y_lo[dim], means.min())
                    y_hi[dim] = max(y_hi[dim], means.max())
                else:
                    mean, std = returns[:, dim].mean(), returns[:, dim].std()
                    x_line = np.array([0, 1])
                    ax.plot(
                        x_line,
                        np.array([mean, mean]),
                        "-",
                        color=color,
                        label=algo,
                        lw=1.5,
                        alpha=0.9,
                    )
                    ax.fill_between(
                        x_line,
                        mean - std,
                        mean + std,
                        color=color,
                        alpha=0.2,
                        edgecolor="none",
                        linewidth=0,
                        antialiased=False,
                    )
                    series[dim].append((x_line, np.array([mean]), color, False))
                    y_lo[dim] = min(y_lo[dim], mean - std)
                    y_hi[dim] = max(y_hi[dim], mean + std)

        for dim in range(3):
            ax = axes[dim]
            ax.set_xlabel(rf"$w_{{\text{{{obj_labels[dim]}}}}}$", fontsize=16)
            ax.set_ylabel(obj_labels[dim], fontsize=12)

            pad = (y_hi[dim] - y_lo[dim]) * 0.1
            _set_plot_axes(ax, 0, 1, y_lo[dim], y_hi[dim], y_pad=pad, bottom_outward=8)

        has_legend = show_legend or env_idx == 0
        if has_legend:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.1),
                ncol=3,
                frameon=False,
                fontsize=12,
            )
        plt.tight_layout()
        _align_axis_labels(fig, axes)
        suffix = "_with_legend" if has_legend else ""
        outpath = os.path.join(
            output_dir, f"correlation_scatter_{env}{suffix}.{IMAGE_FORMAT}"
        )
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        print(f"Saved: {outpath}")


def plot_correlation_bars(metrics, output_dir):
    """Plot Spearman correlation bars for MOPPO only (3 envs, shared y-axis)."""
    fig, axes = plt.subplots(1, 3, figsize=(5, 2), sharey=True)

    x = np.arange(3)
    width = 0.85
    color = COLORS["MOPPO"]

    for i, env in enumerate(ENVS):
        ax = axes[i]
        obj_labels = OBJECTIVE_LABELS[env]
        spearman = metrics[env]["MOPPO"]["spearman"]
        values = [s[0] for s in spearman]
        pvalues = [s[1] for s in spearman]

        ax.bar(
            x,
            values,
            width,
            color=color,
            edgecolor="none",
            linewidth=0,
            alpha=ALPHA_BARS,
        )

        for j, (bar_val, pval) in enumerate(zip(values, pvalues)):
            if np.isnan(pval):
                continue
            txt = r"$p<.001$" if pval < 0.001 else rf"$p={pval:.3g}$"
            y_pos = bar_val + 0.05 if bar_val >= 0 else bar_val - 0.05
            va = "bottom" if bar_val >= 0 else "top"
            ax.text(
                x[j],
                y_pos,
                txt,
                ha="center",
                va=va,
                fontsize=6,
                alpha=0.7,
            )

        ax.set_xlabel(env, fontsize=12)
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
            pad=12,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # Only show y-axis on leftmost plot; fixed -1 to 1 range with margin
        if i == 0:
            yticks = [-1.0, 0.0, 1.0]
            ax.set_yticks(yticks)
            _set_dynamic_tick_format(ax, axis="y")
            ax.yaxis.set_tick_params(direction="in", length=4, pad=12)
            ax.spines["left"].set_bounds(yticks[0], yticks[-1])
            ax.spines["left"].set_position(("outward", 16))
        else:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)

    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_ylabel(r"spearman $\rho$", fontsize=12)
    plt.tight_layout()
    _align_axis_labels(fig, axes, x_gap=8, y_gap=36)
    outpath = os.path.join(output_dir, f"bars_correlation.{IMAGE_FORMAT}")
    plt.savefig(outpath, bbox_inches="tight")
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
    os.makedirs(output_dir, exist_ok=True)

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
    plot_correlation_scatter(all_data, output_dir, show_legend=True)
    plot_correlation_bars(metrics, output_dir)
    save_metrics_file(metrics, output_dir)

    print("Done!")


if __name__ == "__main__":
    main()


# Command used to generate the plots in the paper:
"""
uv run python plot_metrics.py
"""
