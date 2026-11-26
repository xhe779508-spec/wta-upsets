"""Command line: ``python -m wta_upsets [--seasons ...] [--data DIR] [--output DIR]``."""

import argparse
import sys
from pathlib import Path

from wta_upsets.benchmark import run_benchmark
from wta_upsets.figure import plot_upset_rates, save_figure
from wta_upsets.matches import (
    audit_exclusions,
    available_seasons,
    format_seasons,
    load_matches,
    select_matches,
)
from wta_upsets.rates import upset_rates

RATES_FILE = "rates.csv"
EXCLUSIONS_FILE = "exclusions.csv"
BENCHMARK_FILE = "benchmark.csv"
FIGURE_FILE = "upset-rates.png"

FLOAT_FORMAT = "%.6f"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m wta_upsets",
        description="Upset rates by ranking gap and surface for WTA singles, "
        "with a rank-ratio benchmark fitted on the earlier selected seasons "
        "and scored retrospectively on the latest one.",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        metavar="YEAR",
        help="seasons to analyse (default: every season file in the data folder)",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data"), help="folder with the season files"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports"), help="folder for tables and figure"
    )
    return parser


def run(seasons, data_dir, output_dir):
    """Run the analysis, write the outputs and return a printable summary."""
    seasons = sorted(seasons) if seasons else available_seasons(data_dir)
    if not seasons:
        raise FileNotFoundError("No season files found in the data folder.")

    raw = load_matches(data_dir, seasons)
    matches = select_matches(raw)
    if matches.empty:
        raise ValueError("No eligible matches in the selected seasons.")
    rates = upset_rates(matches)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_exclusions(raw).to_csv(output_dir / EXCLUSIONS_FILE, index=False)
    rates.to_csv(output_dir / RATES_FILE, index=False, float_format=FLOAT_FORMAT)
    save_figure(plot_upset_rates(rates, seasons), output_dir / FIGURE_FILE)

    upsets = int(matches["upset"].sum())
    lines = [
        f"Seasons: {format_seasons(seasons)}",
        f"Source rows: {len(raw):,}; eligible matches: {len(matches):,}",
        f"Upsets: {upsets:,} ({upsets / len(matches):.1%})",
    ]

    benchmark_path = output_dir / BENCHMARK_FILE
    if matches["season"].nunique() < 2:
        # A benchmark left over from an earlier run would describe other data.
        benchmark_path.unlink(missing_ok=True)
        lines.append(
            "Benchmark omitted: it needs at least two seasons with eligible "
            f"matches, so no {BENCHMARK_FILE} was written."
        )
    else:
        benchmark = run_benchmark(matches)
        benchmark.to_csv(benchmark_path, index=False, float_format=FLOAT_FORMAT)
        first = benchmark.iloc[0]
        lines.append(
            f"Retrospective benchmark: fitted on {first['train_matches']:,} matches "
            f"({first['train_seasons']}), scored on {first['test_matches']:,} "
            f"matches ({first['test_season']})"
        )
        for _, row in benchmark.iterrows():
            lines.append(
                f"  {row['model']}: Brier {row['brier_score']:.6f}, "
                f"log loss {row['log_loss']:.6f}"
            )
    lines.append(f"Outputs written to {output_dir}")
    return "\n".join(lines)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        print(run(args.seasons, args.data, args.output))
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
