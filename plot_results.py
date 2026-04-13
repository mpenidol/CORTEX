"""
Visualización de resultados del batch de experimentos CORTEX.
Genera gráficas con colores pastel, barras de desviación típica,
ticks grandes para legibilidad en paper.

Usage: python3 plot_results.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CSV_PATH    = os.path.join(os.path.dirname(__file__), "results", "sessions.csv")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Estilo global ─────────────────────────────────────────────────────────────
PASTEL = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#E8BAFF"]
FONT   = "DejaVu Sans"
TICK_SIZE   = 18
LABEL_SIZE  = 20
TITLE_SIZE  = 22
LEGEND_SIZE = 16

plt.rcParams.update({
    "font.family":        FONT,
    "axes.titlesize":     TITLE_SIZE,
    "axes.labelsize":     LABEL_SIZE,
    "xtick.labelsize":    TICK_SIZE,
    "ytick.labelsize":    TICK_SIZE,
    "legend.fontsize":    LEGEND_SIZE,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.4,
    "grid.linestyle":     "--",
})

# ── Métricas a representar ────────────────────────────────────────────────────
METRICS = {
    "total_time_s":             "Total Time (s)",
    "semantic_time_s":          "Semantic Agent Time (s)",
    "retrieval_time_s":         "Retrieval Time (s)",
    "asset_time_s":             "Asset Agent Time (s)",
    "semantic_input_tokens":    "Semantic Input Tokens",
    "semantic_output_tokens":   "Semantic Output Tokens",
    "asset_input_tokens":       "Asset Input Tokens",
    "asset_output_tokens":      "Asset Output Tokens",
    "objects_spawned":          "Objects Spawned",
}

SWEEP_X = {
    "db_sweep": ("db_size",         "Database Size (# assets)"),
    "k_sweep":  ("retrieval_k",     "Retrieval K"),
    "n_sweep":  ("retrieval_top_n", "Top-N (reranker)"),
}

# ─────────────────────────────────────────────────────────────────────────────

def bar_chart(df_sweep, x_col, x_label, metric, y_label, title, out_path, color):
    groups  = sorted(df_sweep[x_col].unique())
    means   = [df_sweep[df_sweep[x_col] == g][metric].mean() for g in groups]
    stds    = [df_sweep[df_sweep[x_col] == g][metric].std()  for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(groups))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=color, edgecolor="grey", linewidth=0.8,
                  error_kw={"elinewidth": 2, "ecolor": "#555555", "capthick": 2},
                  zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{mean:.1f}",
                ha="center", va="bottom",
                fontsize=TICK_SIZE - 2, color="#333333")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def line_chart_all_metrics(df_sweep, x_col, x_label, sweep_name, out_path):
    """One figure with subplots for timing metrics across a sweep."""
    timing_metrics = ["semantic_time_s", "retrieval_time_s", "asset_time_s", "total_time_s"]
    labels         = ["Semantic Agent", "Retrieval", "Asset Agent", "Total"]
    groups = sorted(df_sweep[x_col].unique())
    x = np.arange(len(groups))

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=False)
    fig.suptitle(f"Timing breakdown — {sweep_name}", fontsize=TITLE_SIZE + 2, y=1.02)

    for ax, metric, label, color in zip(axes, timing_metrics, labels, PASTEL):
        means = [df_sweep[df_sweep[x_col] == g][metric].mean() for g in groups]
        stds  = [df_sweep[df_sweep[x_col] == g][metric].std()  for g in groups]

        ax.bar(x, means, yerr=stds, capsize=5,
               color=color, edgecolor="grey", linewidth=0.7,
               error_kw={"elinewidth": 1.5, "ecolor": "#555555"},
               zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in groups], rotation=30, ha="right")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Time (s)")
        ax.set_title(label)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def token_comparison(df_sweep, x_col, x_label, sweep_name, out_path):
    """Grouped bar: input vs output tokens for each agent side by side."""
    groups = sorted(df_sweep[x_col].unique())
    x      = np.arange(len(groups))
    width  = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Token usage — {sweep_name}", fontsize=TITLE_SIZE + 2, y=1.02)

    for ax, agent, colors, title in zip(
        axes,
        [("semantic_input_tokens", "semantic_output_tokens"),
         ("asset_input_tokens",    "asset_output_tokens")],
        [PASTEL[0:2], PASTEL[3:5]],
        ["Semantic Agent", "Asset Agent"],
    ):
        for i, (metric, color, lbl) in enumerate(zip(agent, colors, ["Input", "Output"])):
            means = [df_sweep[df_sweep[x_col] == g][metric].mean() for g in groups]
            stds  = [df_sweep[df_sweep[x_col] == g][metric].std()  for g in groups]
            offset = (i - 0.5) * width * 2
            ax.bar(x + offset, means, width * 1.8, yerr=stds, capsize=4,
                   color=color, edgecolor="grey", linewidth=0.7, label=lbl,
                   error_kw={"elinewidth": 1.5, "ecolor": "#555555"},
                   zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in groups], rotation=30, ha="right")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Tokens")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} sessions from {CSV_PATH}")

    for sweep_key, (x_col, x_label) in SWEEP_X.items():
        df_s = df[df["sweep_type"] == sweep_key].copy()
        if df_s.empty:
            print(f"  No data for sweep: {sweep_key} — skipping")
            continue

        print(f"  Plotting {sweep_key} ({len(df_s)} rows)...")

        # Individual metric bar charts
        for i, (metric, y_label) in enumerate(METRICS.items()):
            out = os.path.join(OUTPUT_DIR, f"{sweep_key}_{metric}.png")
            bar_chart(
                df_s, x_col, x_label, metric, y_label,
                title=f"{y_label}  vs  {x_label}  [{sweep_key}]",
                out_path=out,
                color=PASTEL[i % len(PASTEL)],
            )

        # Timing breakdown (4-panel)
        line_chart_all_metrics(
            df_s, x_col, x_label, sweep_key,
            out_path=os.path.join(OUTPUT_DIR, f"{sweep_key}_timing_breakdown.png"),
        )

        # Token comparison (2-panel)
        token_comparison(
            df_s, x_col, x_label, sweep_key,
            out_path=os.path.join(OUTPUT_DIR, f"{sweep_key}_tokens.png"),
        )

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
