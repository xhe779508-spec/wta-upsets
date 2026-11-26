"""Load season files, select eligible matches and audit every exclusion."""

from pathlib import Path

import numpy as np
import pandas as pd

FILE_PATTERN = "wta_matches_{season}.csv"

# Tour-level events: Grand Slams, WTA 1000 (mandatory and not), other tour
# events, tour finals, team events and the Olympics. ITF and WTA 125 codes
# such as "50+H" or "W" are lower tiers.
TOUR_LEVELS = ("G", "PM", "P", "I", "F", "D", "O")
SURFACES = ("Hard", "Clay", "Grass")
UNFINISHED_MARKERS = ("RET", "W/O", "DEF", "ABD")

# Checks run in this order. A row is excluded at the first one it fails, so
# the audit columns never overlap.
EXCLUSION_REASONS = (
    "lower_tier",
    "surface",
    "date",
    "ranking",
    "equal_ranks",
    "score",
)

REQUIRED_COLUMNS = (
    "tourney_level",
    "surface",
    "tourney_date",
    "winner_rank",
    "loser_rank",
    "score",
)


def available_seasons(data_dir):
    """Return the seasons that have a match file in ``data_dir``, sorted."""
    prefix, suffix = FILE_PATTERN.split("{season}")
    seasons = []
    for path in Path(data_dir).glob(FILE_PATTERN.format(season="*")):
        label = path.name[len(prefix) : -len(suffix)]
        if label.isdigit():
            seasons.append(int(label))
    return sorted(seasons)


def format_seasons(seasons, dash="-"):
    """Describe seasons accurately: runs become ranges, gaps stay visible.

    ``[2022, 2023, 2024]`` gives ``"2022-2024"`` and ``[2022, 2024, 2025]``
    gives ``"2022, 2024-2025"``.
    """
    runs = []
    for season in sorted({int(season) for season in seasons}):
        if runs and season == runs[-1][1] + 1:
            runs[-1][1] = season
        else:
            runs.append([season, season])
    return ", ".join(
        str(first) if first == last else f"{first}{dash}{last}" for first, last in runs
    )


def load_matches(data_dir, seasons):
    """Read the requested season files into one frame of raw strings.

    Every field is kept as text so that the eligibility checks see exactly
    what the source file contains. The ``season`` column is the year in the
    file name: an event that starts in late December belongs to the season
    of the file that lists it.
    """
    seasons = [int(season) for season in seasons]
    if not seasons:
        raise ValueError("Select at least one season.")
    if len(set(seasons)) != len(seasons):
        raise ValueError("Each season can be selected only once.")

    frames = []
    for season in sorted(seasons):
        path = Path(data_dir) / FILE_PATTERN.format(season=season)
        if not path.is_file():
            raise FileNotFoundError(
                f"No match file for season {season}: expected {path.name} "
                "in the data folder."
            )
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
        if missing:
            raise ValueError(f"{path.name} lacks columns: {', '.join(missing)}")
        frame.insert(0, "season", season)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def parse_rank(values):
    """Parse ranks, returning NaN unless the value is a positive integer."""
    ranks = pd.to_numeric(values.str.strip(), errors="coerce").astype(float)
    valid = np.isfinite(ranks) & (ranks > 0) & (ranks == np.floor(ranks))
    return ranks.where(valid)


def is_finished_score(scores):
    """True for a nonempty score with no retirement, walkover or default."""
    text = scores.str.strip().str.upper()
    unfinished = pd.Series(False, index=scores.index)
    for marker in UNFINISHED_MARKERS:
        unfinished |= text.str.contains(marker, regex=False)
    return (text != "") & ~unfinished


def classify(raw):
    """Label each source row "eligible" or with its first failed check."""
    surface = raw["surface"].str.strip().str.title()
    date = pd.to_datetime(raw["tourney_date"].str.strip(), format="%Y%m%d", errors="coerce")
    winner_rank = parse_rank(raw["winner_rank"])
    loser_rank = parse_rank(raw["loser_rank"])

    failed = {
        "lower_tier": ~raw["tourney_level"].str.strip().isin(TOUR_LEVELS),
        "surface": ~surface.isin(SURFACES),
        "date": date.isna(),
        "ranking": winner_rank.isna() | loser_rank.isna(),
        "equal_ranks": winner_rank == loser_rank,
        "score": ~is_finished_score(raw["score"]),
    }

    status = pd.Series("eligible", index=raw.index)
    for reason in reversed(EXCLUSION_REASONS):
        status = status.mask(failed[reason], reason)
    return status


def select_matches(raw):
    """Return the eligible matches with typed analysis columns.

    ``upset`` is True when the winner had the larger ranking number, i.e.
    the lower-ranked player won. ``rank_gap`` is the absolute difference
    between the two pre-match rankings.
    """
    keep = classify(raw) == "eligible"
    winner_rank = parse_rank(raw.loc[keep, "winner_rank"]).astype(int)
    loser_rank = parse_rank(raw.loc[keep, "loser_rank"]).astype(int)
    matches = pd.DataFrame(
        {
            "season": raw.loc[keep, "season"].astype(int),
            "date": pd.to_datetime(raw.loc[keep, "tourney_date"].str.strip(), format="%Y%m%d"),
            "tourney_level": raw.loc[keep, "tourney_level"].str.strip(),
            "surface": raw.loc[keep, "surface"].str.strip().str.title(),
            "winner_rank": winner_rank,
            "loser_rank": loser_rank,
            "rank_gap": (winner_rank - loser_rank).abs(),
            "upset": winner_rank > loser_rank,
        }
    )
    return matches.reset_index(drop=True)


def audit_exclusions(raw):
    """Count source, excluded and eligible rows for each season.

    For every season, ``source_rows`` equals the exclusion columns plus
    ``eligible``.
    """
    counts = (
        pd.crosstab(raw["season"], classify(raw))
        .reindex(columns=[*EXCLUSION_REASONS, "eligible"], fill_value=0)
        .astype(int)
    )
    counts.columns.name = None
    counts.insert(0, "source_rows", counts.sum(axis=1))
    return counts.reset_index()
