"""Rank players by held-out outlet forecasts and compare them with revised SVA."""

import argparse
from collections import defaultdict
import csv
from math import isfinite
from pathlib import Path
from statistics import fmean

from safety_valve import ROOT, write_csv


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def rank(rows, field, rank_field):
    ordered = sorted(rows, key=lambda row: (-row[field], -row["in_range_onset_routes"], row["nfl_id"]))
    for place, row in enumerate(ordered, 1):
        row[rank_field] = place
    return ordered


def player_leaderboard(outlets, previous, min_routes):
    previous_by_id = {int(row["nfl_id"]): row for row in previous}
    by_player = defaultdict(list)
    seen = set()
    for row in outlets:
        nfl_id = int(row["nfl_id"])
        key = (row["game_id"], row["play_id"], nfl_id)
        if key in seen:
            raise ValueError(f"Duplicate outlet forecast for {key}.")
        seen.add(key)
        catch, epa, sva = (float(row[field]) for field in
                           ("catch_probability", "hurdle_epa_on_comparable_targets", "onset_sva"))
        if not (0 <= catch <= 1 and isfinite(epa) and 0 <= sva <= 100):
            raise ValueError(f"Invalid outlet forecast for {key}.")
        by_player[nfl_id].append((row, catch, epa, sva))

    rows = []
    for nfl_id, routes in by_player.items():
        if len(routes) < min_routes:
            continue
        first = routes[0][0]
        prior = previous_by_id.get(nfl_id)
        rows.append(dict(
            nfl_id=nfl_id, player=first["player"], position=first["position"],
            in_range_onset_routes=len(routes),
            named_targets=sum(bool(row["observed_catch"]) for row, _, _, _ in routes),
            mean_catch_pct=100 * fmean(catch for _, catch, _, _ in routes),
            mean_expected_epa=fmean(epa for _, _, epa, _ in routes),
            mean_onset_sva=fmean(sva for _, _, _, sva in routes),
            previous_sva_rank=int(prior["rank"]) if prior else None,
            previous_sva=float(prior["sva"]) if prior else None,
            previous_in_range_routes=int(prior["in_range_routes"]) if prior else None,
        ))

    rank(rows, "mean_catch_pct", "catch_probability_rank")
    rank(rows, "mean_onset_sva", "same_routes_sva_rank")
    ordered = rank(rows, "mean_expected_epa", "expected_epa_rank")
    for row in ordered:
        row["epa_places_gained_vs_same_routes_sva"] = row["same_routes_sva_rank"] - row["expected_epa_rank"]
        row["catch_places_gained_vs_same_routes_sva"] = row["same_routes_sva_rank"] - row["catch_probability_rank"]
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outlets", type=Path, default=ROOT / "sva_joint_outlets.csv")
    parser.add_argument("--previous", type=Path, default=ROOT / "safety_valve_separation_weighted_leaderboard.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "sva_joint_leaderboard.csv")
    parser.add_argument("--min-routes", type=int, default=30)
    args = parser.parse_args()
    if args.min_routes < 1:
        parser.error("--min-routes must be at least 1")
    if args.output.resolve() in {args.outlets.resolve(), args.previous.resolve()}:
        parser.error("The output must be different from both inputs.")

    rows = player_leaderboard(read_csv(args.outlets), read_csv(args.previous), args.min_routes)
    if not rows:
        parser.error("No players meet the minimum in-range route count.")
    fields = ("expected_epa_rank", "catch_probability_rank", "nfl_id", "player", "position",
              "in_range_onset_routes", "named_targets", "mean_catch_pct", "mean_expected_epa",
              "previous_sva_rank", "previous_sva", "previous_in_range_routes", "same_routes_sva_rank",
              "mean_onset_sva", "epa_places_gained_vs_same_routes_sva", "catch_places_gained_vs_same_routes_sva")
    write_csv(args.output, fields, rows)
    overlap = sum(row["previous_sva_rank"] is not None for row in rows)
    print(f"Ranked {len(rows)} players with at least {args.min_routes} onset routes. "
          f"{overlap} were also ranked in the previous leaderboard.")


if __name__ == "__main__":
    main()
