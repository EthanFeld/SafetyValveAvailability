"""Combine catch and EPA forecasts into a position- and situation-adjusted SVA."""

import argparse
from collections import defaultdict
from math import isfinite, tanh
from pathlib import Path
from statistics import fmean

import numpy as np

from analyze_catch_rate import category_controls, pre_snap_context
from rank_joint_sva import player_leaderboard, rank, read_csv
from safety_valve import ROOT, write_csv
from sva_epa_models import game_folds
from sva_joint_models import fit_ridge


BENCHMARK_RIDGE = 20
DEFAULT_SCORE_SPREAD = 10.0
PRESSURE_FIELDS = ("onset_seconds", "onset_rusher_distance_yd", "onset_rusher_closing_yd_s",
                   "onset_credited_rushers", "onset_qb_speed_yd_s")


def key(row):
    return int(row["game_id"]), int(row["play_id"])


def situation_design(outlets, situations, pressure):
    keys = [key(row) for row in outlets]
    missing = set(keys) - (situations.keys() & pressure.keys())
    if missing:
        raise ValueError(f"Missing first-pressure or game situation data for {len(missing)} plays.")
    to_go, to_goal, margin, action, down, quarter, coverage, _ = zip(*(situations[k] for k in keys))
    position = category_controls([row["position"] for row in outlets])
    depth = np.array([float(row["projected_depth_yd"]) for row in outlets]) / 10
    sticks = np.array([float(row["projected_margin_to_sticks_yd"]) for row in outlets]) / 10
    shortfall = np.maximum(-sticks, 0)
    reached_sticks = sticks >= 0
    late_down = np.isin(down, ("3", "4"))
    field = np.asarray(to_goal) / 50
    onset = np.array([[float(pressure[k][field]) for field in PRESSURE_FIELDS] for k in keys])
    context = np.column_stack((
        position, depth, depth ** 2, depth[:, None] * position,
        np.log1p(to_go), reached_sticks, shortfall, shortfall * late_down,
        field, field ** 2, np.asarray(margin) / 14, action, category_controls(down),
        category_controls(quarter), category_controls(coverage), onset,
    ))
    return context, depth, shortfall, late_down, reached_sticks


def benchmark_forecasts(outlets, joint_plays, situations, pressure):
    games = np.array([int(row["game_id"]) for row in outlets])
    play_games = np.array([int(row["game_id"]) for row in joint_plays])
    # Forecasts in each original fold were made by the same model without those games.
    # Fit the context benchmark among those games and hold out each evaluated game again.
    game_to_fold = dict(zip(play_games, game_folds(play_games)))
    if set(games) - game_to_fold.keys():
        raise ValueError("Outlet forecasts include games absent from the original model folds.")
    outer_folds = np.array([game_to_fold[game] for game in games])
    catch = np.array([float(row["catch_probability"]) for row in outlets])
    epa = np.array([float(row["hurdle_epa_on_comparable_targets"]) for row in outlets])
    context, depth, shortfall, late_down, reached_sticks = situation_design(outlets, situations, pressure)
    # A catch near the sticks is worth more than one well short of them. Condition
    # on catch probability and its situational value to avoid counting it twice.
    epa_context = np.column_stack((context, catch, catch ** 2, catch * depth,
                                   catch * shortfall, catch * late_down, catch * reached_sticks))
    expected_catch = np.empty_like(catch)
    expected_epa = np.empty_like(epa)
    for fold in np.unique(outer_folds):
        rows = np.flatnonzero(outer_folds == fold)
        reference_folds = game_folds(games[rows], n_folds=4, seed=2021 + int(fold))
        for held_out in np.unique(reference_folds):
            test = rows[reference_folds == held_out]
            train = rows[reference_folds != held_out]
            expected_catch[test] = fit_ridge(context[train], catch[train], BENCHMARK_RIDGE).predict(context[test])
            expected_epa[test] = fit_ridge(epa_context[train], epa[train], BENCHMARK_RIDGE).predict(epa_context[test])
    return catch - np.clip(expected_catch, 0, 1), epa - expected_epa


def score_players(outlets, prior, min_routes, catch_lift, epa_lift, catch_weight,
                  score_spread=DEFAULT_SCORE_SPREAD):
    players = player_leaderboard(outlets, prior, min_routes)
    scales = np.array([np.std(catch_lift), np.std(epa_lift)])
    if not (np.isfinite(scales).all() and (scales > 0).all()):
        raise ValueError("Catch and EPA residuals must have nonzero finite variation.")
    standardized = catch_weight * catch_lift / scales[0] + (1 - catch_weight) * epa_lift / scales[1]
    composite = np.clip(50 + 10 * standardized, 0, 100)
    by_player = defaultdict(list)
    for i, row in enumerate(outlets):
        by_player[int(row["nfl_id"])].append(i)
    for row in players:
        indices = by_player[row["nfl_id"]]
        base = fmean(composite[indices])
        # Stretch the player average around 50 without clipping the tails.
        widened = 50 + 50 * tanh(score_spread * ((base - 50) / 50))
        row.update(context_sva=widened, base_context_sva=base,
                   catch_above_context_pp=100 * fmean(catch_lift[indices]),
                   epa_above_context_given_catch=fmean(epa_lift[indices]))
    ordered = rank(players, "base_context_sva", "context_sva_rank")
    for row in ordered:
        place = row["context_sva_rank"]
        row.update(places_gained_vs_same_routes_sva=row["same_routes_sva_rank"] - place,
                   places_gained_vs_raw_catch=row["catch_probability_rank"] - place,
                   places_gained_vs_raw_epa=row["expected_epa_rank"] - place)
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "nfl-big-data-bowl-regional-event-data-main.zip")
    parser.add_argument("--outlets", type=Path, default=ROOT / "sva_joint_outlets.csv")
    parser.add_argument("--joint-plays", type=Path, default=ROOT / "sva_joint_plays.csv")
    parser.add_argument("--pressure-plays", type=Path, default=ROOT / "sva_epa_plays.csv")
    parser.add_argument("--previous", type=Path, default=ROOT / "safety_valve_separation_weighted_leaderboard.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "sva_context_leaderboard.csv")
    parser.add_argument("--min-routes", type=int, default=30)
    parser.add_argument("--catch-weight", type=float, default=0.5, help="Share of the standardized score from catch lift.")
    parser.add_argument("--score-spread", type=float, default=DEFAULT_SCORE_SPREAD,
                        help="How much to stretch scores near 50 after averaging routes (default 10).")
    args = parser.parse_args()
    if args.min_routes < 1 or not 0 <= args.catch_weight <= 1:
        parser.error("Use a positive minimum route count and a catch weight between zero and one.")
    if not isfinite(args.score_spread) or args.score_spread <= 0:
        parser.error("--score-spread must be positive and finite.")
    inputs = (args.archive, args.outlets, args.joint_plays, args.pressure_plays, args.previous)
    if args.output.resolve() in {path.resolve() for path in inputs}:
        parser.error("The output must be different from the inputs.")

    outlets = list(read_csv(args.outlets))
    joint_plays = list(read_csv(args.joint_plays))
    pressure = {key(row): row for row in read_csv(args.pressure_plays)}
    situations = dict(pre_snap_context(args.archive, {key(row) for row in outlets}))
    lift = benchmark_forecasts(outlets, joint_plays, situations, pressure)
    rows = score_players(outlets, read_csv(args.previous), args.min_routes, *lift,
                         args.catch_weight, args.score_spread)
    if not rows:
        parser.error("No players meet the minimum in-range route count.")
    fields = ("context_sva_rank", "nfl_id", "player", "position", "in_range_onset_routes", "named_targets",
              "context_sva", "base_context_sva", "catch_above_context_pp", "epa_above_context_given_catch", "mean_catch_pct",
              "mean_expected_epa", "catch_probability_rank", "expected_epa_rank", "same_routes_sva_rank",
              "mean_onset_sva", "previous_sva_rank", "previous_sva", "previous_in_range_routes",
              "places_gained_vs_same_routes_sva", "places_gained_vs_raw_catch", "places_gained_vs_raw_epa")
    write_csv(args.output, fields, ({field: row[field] for field in fields} for row in rows))
    print(f"Ranked {len(rows)} players from {len(outlets):,} outlet forecasts with catch weight {args.catch_weight:.0%}.")


if __name__ == "__main__":
    main()
