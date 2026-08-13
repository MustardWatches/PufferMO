import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

STEP_KEYS = ["_step", "Step"]
FONT = "times"  # "lucidabr" or "times"

COLORS_BY_NAME = {
    "RED": "#FF0F19",
    "GREEN": "#3FB45D",
    "BLUE": "#5A67D8",
    "BLACK": "#000000",
    "ORANGE": "#E6B459",
    "PURPLE": "#805AD5",
    "TEAL": "#319795",
}

DEFAULT_COLOR_CYCLE = [
    COLORS_BY_NAME[color]
    for color in ["BLUE", "ORANGE", "GREEN", "RED", "PURPLE", "TEAL", "BLACK"]
]

SPINE_PADDING = 0.15
SPINE_PADDING_BOTTOM = 0.05

COLORS_BY_LABEL = {
    "PPO": "#65a98f",
    "MOPPO (no conditioning)": "#0757b9",
    "MOPPO": "#ff8c00"
}

OBJECTIVE_LABELS = {
    "moba": ["death", "xp", "tower"],
    "snake": ["food", "corpse", "death"],
    "tetris": ["combo", "drop", "rotate"],
}


def prettify_label(label):
    if "mo_moqppo_no_cond" in label:
        return "MOQPPO (no conditioning)"
    if "mo_moqppo" in label:
        return "MOQPPO"
    if "mo_moppo_no_cond" in label:
        return "MOPPO (no conditioning)"
    if "mo_moppo" in label:
        return "MOPPO"
    if "qppo" in label:
        return "QPPO"
    if "ppo" in label:
        return "PPO"
    return label


def order_from_labels(labels):
    return [labels.index(label) for label in COLORS_BY_LABEL if label in labels]


def setup_plot_style():
    """Apply the formatting styles from format.py"""
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


def get_env_name_from_path(args):
    for name in OBJECTIVE_LABELS:
        if name in args.path.lower():
            return name
    return ""


def smooth_data(series, weight=0.9):
    """Exponential moving average smoothing (from plots.py)."""
    return series.ewm(alpha=1 - weight).mean()


def align_axis_labels(fig, ax, gap=14):
    """Anchor axis labels a fixed display-space gap from the visible spines."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
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

    x_label_y = bottom_edge.y0 - gap
    x_center = (bottom_edge.x0 + bottom_edge.x1) / 2
    x_label_x, x_axes_coord = ax.transAxes.inverted().transform(
        (x_center, x_label_y)
    )
    ax.xaxis.set_label_coords(x_label_x, x_axes_coord)
    ax.xaxis.label.set_verticalalignment("top")

    y_label_x = left_edge.x0 - gap
    y_center = (left_edge.y0 + left_edge.y1) / 2
    y_axes_coord, y_label_y = ax.transAxes.inverted().transform(
        (y_label_x, y_center)
    )
    ax.yaxis.set_label_coords(y_axes_coord, y_label_y)
    ax.yaxis.label.set_horizontalalignment("center")


def main(args):
    # Use specified colors or default palette
    colors = (
        [COLORS_BY_NAME.get(c, c) for c in args.colors]
        if args.colors
        else DEFAULT_COLOR_CYCLE
    )

    # Determine files to process
    if os.path.isdir(args.path):
        files = glob.glob(os.path.join(args.path, "*.csv"))
        files.sort()
        output_path = args.output_path or args.path
    else:
        files = [args.path]
        output_path = args.output_path or os.path.dirname(args.path)

    # Track metric name for title
    metric_name = args.metric_name

    plt.figure(figsize=(3.5, 2.5))

    # Apply style
    try:
        setup_plot_style()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not apply tex style: {e}")

    # Store processed data for plotting
    plot_data = []
    max_step = 0

    for i, file_path in enumerate(files):
        # Parse CSV data
        try:
            df = pd.read_csv(file_path, quotechar='"', skipinitialspace=True)

            # Identify step key
            step_key = next((k for k in STEP_KEYS if k in df.columns), None)
            assert step_key is not None, f"No step key found in {file_path}"

            df = df.sort_values(step_key)
        except Exception as e:  # noqa: BLE001
            print(f"Error reading {file_path}: {e}")
            continue

        # Fill missing values
        df = df.interpolate(method="linear").fillna(0)

        # Identify columns for calculation
        columns = [c for c in df.columns if c not in STEP_KEYS]
        for excluded_col in args.exclude_columns:
            columns = [c for c in columns if excluded_col not in c]

        if not columns:
            continue

        # Fetch metric name from the first file found
        if metric_name == "value":
            metric_name = columns[0].split("/")[-1].split("_")[0]

        # Apply Smoothing
        for col in columns:
            df[col] = smooth_data(df[col], weight=args.smooth)

        # Calculate Aggregate Stats
        if args.aggregate:
            df["mean"] = df[columns].mean(axis=1)
            df["std"] = df[columns].std(axis=1)

            # Label based on filename
            label = os.path.splitext(os.path.basename(file_path))[0]
            label = prettify_label(label)

            # Select color
            color = COLORS_BY_LABEL.get(label, colors[i % len(colors)])

            plot_data.append(
                {
                    "df": df,
                    "label": label,
                    "color": color,
                }
            )
        else:
            for j, col in enumerate(columns):
                df_single = df[[step_key, col]].copy()
                df_single.rename(columns={col: "mean"}, inplace=True)
                df_single["std"] = 0  # No std deviation for single runs

                raw_label = os.path.splitext(os.path.basename(file_path))[0]
                label = f"{col} - {prettify_label(raw_label)}"
                color = colors[j % len(colors)]

                plot_data.append(
                    {
                        "df": df_single,
                        "label": label,
                        "color": color,
                    }
                )

        # Update max step for scaling
        if not df.empty and step_key:
            max_step = max(max_step, df[step_key].max())

    # Determine scale factor for x-axis
    if max_step > 0:
        exponent = int(np.floor(np.log10(max_step)))
        scale = 10**exponent
    else:
        scale = 1
        exponent = 0

    # Plotting
    for i, item in enumerate(plot_data):
        df = item["df"]
        label = item["label"]
        color = item["color"]

        min_val = df["mean"].min()
        max_val = df["mean"].max()

        if args.normalize:
            df["mean"] = (df["mean"] - min_val) / (max_val - min_val + 1e-8)
            df["std"] = df["std"] / (max_val - min_val + 1e-8)

        # Prepare data for plotting
        plot_mean = df["mean"]
        plot_lower = df["mean"] - df["std"]
        plot_upper = df["mean"] + df["std"]

        # For positive-only metrics, clip lower bound at 0
        if "return" not in metric_name.lower():
            plot_mean = plot_mean.clip(lower=0)
            plot_lower = plot_lower.clip(lower=0)

        # Main line
        plt.plot(
            df[step_key] / scale, plot_mean, label=label, color=color, linewidth=1.5
        )

        # Add shading for std deviation bands if enabled
        if args.std_bands:
            assert args.aggregate, "Standard deviation bands require aggregation."
            plt.fill_between(
                df[step_key] / scale,
                plot_lower,
                plot_upper,
                color=color,
                alpha=0.2,
                linewidth=0,
            )
            item["band_min"] = plot_lower.min()
            item["band_max"] = plot_upper.max()

    # Add vertical markers if specified
    for vm in args.vertical_markers:
        plt.axvline(x=vm / scale, color="gray", linestyle="--", linewidth=1)

    process = args.process
    xlabel = (
        f"{process} step ($\\mathsf{{\\times 10^{{{exponent}}}}}$)"
        if exponent != 0
        else f"{process} step"
    )
    plt.xlabel(xlabel)
    plt.ylabel(metric_name.lower())

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    # Bound spines to data extremes and show only limit ticks
    all_x = np.concatenate([d["df"][step_key].values / scale for d in plot_data])
    all_y = np.concatenate([d["df"]["mean"].values for d in plot_data])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    if abs(x_min - round(x_min)) < 0.05:
        x_min = float(round(x_min))
    if abs(x_max - round(x_max)) < 0.05:
        x_max = float(round(x_max))
    ax.spines["bottom"].set_bounds(x_min, x_max)
    ax.spines["left"].set_bounds(y_min, y_max)
    ax.spines["bottom"].set_position(("outward", 16))
    ax.spines["left"].set_position(("outward", 16))

    # Fix the y-spine's fraction of the axes height regardless of std-band
    # overshoot, so spines look similarly sized across envs/metrics with
    # different variance instead of being squashed by wide std bands.
    # Widen the limits further if the bands themselves
    # extend past the padding so they never get clipped.
    y_range = y_max - y_min
    band_min = min((d["band_min"] for d in plot_data if "band_min" in d), default=y_min)
    band_max = max((d["band_max"] for d in plot_data if "band_max" in d), default=y_max)
    ylim_low = min(y_min - SPINE_PADDING_BOTTOM * y_range, band_min)
    ylim_high = max(y_max + SPINE_PADDING * y_range, band_max)
    ax.set_ylim(ylim_low, ylim_high)
    ax.set_xticks([x_min, x_max])
    ax.set_yticks([y_min, y_max])
    formatter = mticker.FormatStrFormatter("%.3g")
    y_formatter = mticker.FuncFormatter(
        lambda value, position: "0" if abs(value) <= 1e-3 else f"{value:.3g}"
    )
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.xaxis.set_tick_params(direction="in", length=4, labelbottom=True, width=1.2, pad=12)
    ax.yaxis.set_tick_params(direction="in", length=4, labelleft=True, width=1.2, pad=12)

    if not args.disable_legend:
        # Define label order
        handles, labels = plt.gca().get_legend_handles_labels()
        order = order_from_labels(labels)
        handles = [handles[idx] for idx in order]
        labels = [labels[idx] for idx in order]
        plt.legend(handles, labels, loc="lower right", frameon=False)
    plt.tight_layout()
    align_axis_labels(plt.gcf(), ax)

    env_name = args.env_name or get_env_name_from_path(args)
    image_format = (args.format or "pdf").lower().lstrip(".")
    file_name = (
        f"plot_{metric_name.replace(' ', '_')}"
        + (f"_{env_name}" if env_name else "")
        + f".{image_format}"
    )

    output_file = os.path.join(output_path, file_name)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(output_file, dpi=500, bbox_inches="tight")
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data from CSV files.")
    parser.add_argument("--path", type=str, help="Path to the CSV file or folder")
    parser.add_argument(
        "--smooth", type=float, default=0.95, help="Smoothing weight (0 to 1)"
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate multiple runs in a file by mean and std deviation",
    )
    parser.add_argument(
        "--std-bands", action="store_true", help="Enable standard deviation bands"
    )
    parser.add_argument(
        "--process", type=str, default="training", help="Process name for x-axis label"
    )
    parser.add_argument(
        "--exclude-columns",
        type=str,
        nargs="*",
        default=[],
        help="Columns to exclude from plotting",
    )
    parser.add_argument(
        "--disable-legend", action="store_true", help="Disable legend in the plot"
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize data to [0, 1] range"
    )
    parser.add_argument(
        "--vertical-markers",
        type=float,
        nargs="*",
        default=[],
        help="Add vertical markers at specified x-values",
    )
    parser.add_argument(
        "--colors", type=str, nargs="*", help="List of colors to use for plotting"
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        default="value",
        help="Name of the metric for the title",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="",
        help="Name of the environment for the output file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="Output directory for the plot",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        help="Output file format for the saved plot (default: pdf)",
    )
    args = parser.parse_args()
    main(args)

# Commands used to generate the plots in the paper:
"""
# Performance and discounted returns during training
uv run python plot_curves.py --path ./data/training/moba/discounted_return   --output-path ./figures/training           --process training --aggregate --std-bands --disable-legend --metric-name "disc return"
uv run python plot_curves.py --path ./data/training/moba/performance         --output-path ./figures/training           --process training --aggregate --std-bands
uv run python plot_curves.py --path ./data/training/snake/discounted_return  --output-path ./figures/training           --process training --aggregate --std-bands --disable-legend --metric-name "disc return"
uv run python plot_curves.py --path ./data/training/snake/performance        --output-path ./figures/training           --process training --aggregate --std-bands --disable-legend
uv run python plot_curves.py --path ./data/training/tetris/discounted_return --output-path ./figures/training           --process training --aggregate --std-bands --disable-legend --metric-name "disc return"
uv run python plot_curves.py --path ./data/training/tetris/performance       --output-path ./figures/training           --process training --aggregate --std-bands --disable-legend
# Reward plot for the dynamic adaptation experiment
uv run python plot_curves.py --path ./data/dynamic_adaptation                --output-path ./figures/dynamic_adaptation --process episode                          --disable-legend --metric-name "reward"      --exclude-columns weights --smooth 0.99 --normalize --vertical-markers 500 --colors BLUE PURPLE ORANGE
"""
