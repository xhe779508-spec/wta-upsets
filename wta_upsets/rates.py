"""Upset counts, rates and Wilson intervals by ranking gap and surface."""

import math

import pandas as pd

from wta_upsets.matches import SURFACES

# Upper edges of the ranking-gap groups; the last group is open-ended.
GAP_EDGES = (10, 25, 50, 100)
GAP_LABELS = ("1-10", "11-25", "26-50", "51-100", "101+")
ALL = "All"

Z_95 = 1.959963984540054  # 97.5th percentile of the standard normal

COLUMNS = (
    "surface",
    "rank_gap",
    "matches",
    "upsets",
    "upset_rate",
    "wilson_low",
    "wilson_high",
)


def wilson_interval(successes, trials, z=Z_95):
    """Wilson score interval for a binomial proportion.

    Returns ``(nan, nan)`` when there are no trials, so an empty group has
    an undefined rate rather than a misleading one.
    """
    if trials < 0 or not 0 <= successes <= max(trials, 0):
        raise ValueError("Need 0 <= successes <= trials.")
    if trials == 0:
        return (math.nan, math.nan)
    rate = successes / trials
    denominator = 1 + z**2 / trials
    centre = (rate + z**2 / (2 * trials)) / denominator
    half_width = (
        z * math.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2)) / denominator
    )
    # The exact bounds are 0 with no successes and 1 with no failures. Setting
    # them directly avoids floating-point residue such as a 5.6e-17 lower
    # bound, which would put the interval on the wrong side of the estimate.
    low = 0.0 if successes == 0 else max(0.0, centre - half_width)
    high = 1.0 if successes == trials else min(1.0, centre + half_width)
    return (low, high)


def gap_group(rank_gap):
    """Assign each positive ranking gap to its labelled group."""
    edges = [0, *GAP_EDGES, float("inf")]
    return pd.cut(rank_gap, bins=edges, labels=list(GAP_LABELS), right=True)


def upset_rates(matches):
    """Tidy table of upset rates for every surface and ranking-gap group.

    One row per combination of surface (``All``, then each surface) and
    ranking gap (``All``, then each group). Groups with no matches are kept
    with zero counts and blank rate and interval.
    """
    groups = gap_group(matches["rank_gap"])
    everything = pd.Series(True, index=matches.index)
    rows = []
    for surface in (ALL, *SURFACES):
        on_surface = everything if surface == ALL else matches["surface"] == surface
        for gap in (ALL, *GAP_LABELS):
            in_gap = everything if gap == ALL else groups == gap
            selected = matches.loc[on_surface & in_gap, "upset"]
            n_matches = int(selected.size)
            n_upsets = int(selected.sum())
            low, high = wilson_interval(n_upsets, n_matches)
            rows.append(
                {
                    "surface": surface,
                    "rank_gap": gap,
                    "matches": n_matches,
                    "upsets": n_upsets,
                    "upset_rate": n_upsets / n_matches if n_matches else math.nan,
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )
    return pd.DataFrame(rows, columns=list(COLUMNS))
