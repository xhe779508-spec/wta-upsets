"""Two-panel chart of upset rates with 95% Wilson intervals."""

import matplotlib

matplotlib.use("Agg")  # headless: never opens a window

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from wta_upsets.matches import SURFACES, format_seasons  # noqa: E402
from wta_upsets.rates import ALL, GAP_LABELS  # noqa: E402

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
BACKGROUND = "#fcfcfb"
OVERALL_COLOR = "#33322f"
# Colour and marker shape both identify the surface.
SURFACE_STYLE = {
    "Hard": {"color": "#2a78d6", "marker": "o"},
    "Clay": {"color": "#eb6834", "marker": "s"},
    "Grass": {"color": "#1baf7a", "marker": "^"},
}
DODGE = 0.22


def _plot_series(ax, rows, offset, color, marker, label=None):
    """Draw one series of points with interval bars, skipping empty groups."""
    rows = rows.set_index("rank_gap").reindex(list(GAP_LABELS))
    rows = rows[rows["matches"].fillna(0) > 0]
    positions = [GAP_LABELS.index(gap) + offset for gap in rows.index]
    ax.errorbar(
        positions,
        rows["upset_rate"],
        yerr=[
            (rows["upset_rate"] - rows["wilson_low"]).clip(lower=0),
            (rows["wilson_high"] - rows["upset_rate"]).clip(lower=0),
        ],
        fmt=marker,
        color=color,
        markersize=7,
        markeredgecolor=BACKGROUND,
        markeredgewidth=1,
        elinewidth=1.6,
        capsize=0,
        label=label,
    )
    return positions, rows


def _style_axis(ax, title):
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=10)
    ax.set_xticks(range(len(GAP_LABELS)))
    ax.set_xticklabels([label.replace("-", "–") for label in GAP_LABELS])
    ax.set_xlim(-0.5, len(GAP_LABELS) - 0.5)
    ax.set_xlabel("Ranking gap (places)", color=MUTED)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_facecolor(BACKGROUND)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_upset_rates(rates, seasons):
    """Build the figure from the tidy rates table and return it."""
    by_gap = rates[rates["rank_gap"] != ALL]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.8), sharey=True, facecolor=BACKGROUND
    )

    _style_axis(left, "All surfaces")
    positions, rows = _plot_series(
        left, by_gap[by_gap["surface"] == ALL], 0.0, OVERALL_COLOR, "o"
    )
    for x, (_, row) in zip(positions, rows.iterrows()):
        left.annotate(
            f"{row['upset_rate']:.1%}\nn = {int(row['matches']):,}",
            (x, row["wilson_low"]),
            xytext=(0, -6),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            color=MUTED,
            linespacing=1.3,
        )
    left.set_ylabel("Matches won by the lower-ranked player", color=MUTED)
    left.annotate(
        "even odds",
        (len(GAP_LABELS) - 0.55, 0.5),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=MUTED,
    )

    _style_axis(right, "By court surface")
    for index, surface in enumerate(SURFACES):
        _plot_series(
            right,
            by_gap[by_gap["surface"] == surface],
            (index - (len(SURFACES) - 1) / 2) * DODGE,
            SURFACE_STYLE[surface]["color"],
            SURFACE_STYLE[surface]["marker"],
            label=surface,
        )
    handles, labels = right.get_legend_handles_labels()
    legend = right.legend(
        [handle[0] for handle in handles],  # marker only, without the bar
        labels,
        loc="upper right", frameon=False, ncols=len(SURFACES), handletextpad=0.2,
        columnspacing=1.2, fontsize=9.5,
    )
    for text in legend.get_texts():
        text.set_color(INK)

    bounds = by_gap[by_gap["matches"] > 0]
    if not bounds.empty:
        low = bounds["wilson_low"].min() - 0.07
        high = min(1.0, max(bounds["wilson_high"].max(), 0.5) + 0.06)
        left.set_ylim(low, high)
        # Room below a 0% estimate is for its label, not for negative ticks.
        left.set_yticks([tick for tick in left.get_yticks() if low <= tick <= high and tick >= 0])
        left.set_ylim(low, high)

    fig.suptitle(
        "WTA upset rates by ranking gap and court surface",
        x=0.012, y=0.985, ha="left", fontsize=14, fontweight="bold", color=INK,
    )
    fig.text(
        0.012, 0.9,
        f"Completed tour-level singles matches, {format_seasons(seasons, dash='–')}. "
        "Bars show 95% Wilson intervals.",
        ha="left", fontsize=10, color=MUTED,
    )
    fig.text(
        0.012, 0.012,
        "Data: Jeff Sackmann / Tennis Abstract, CC BY-NC-SA 4.0",
        ha="left", fontsize=8, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.9), w_pad=2.5)
    return fig


def save_figure(fig, path):
    """Write the figure as a PNG without machine-specific metadata."""
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), metadata={"Software": None})
    plt.close(fig)
