"""Build temporal SVA and test incremental early-window value on held-out games."""

import argparse
from collections import defaultdict
from math import tanh
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from analyze_catch_rate import archive_rows, category_controls
from rank_context_sva import BENCHMARK_RIDGE, DEFAULT_SCORE_SPREAD, benchmark_forecasts, situation_design
from rank_joint_sva import read_csv
from safety_valve import ROOT, write_csv
from sva_epa_models import game_folds
from sva_joint_features import _context
from sva_joint_models import fit_logistic, fit_ridge
from sva_temporal_data import (DEFAULT_LANDMARK_FRAMES, DEFAULT_OPEN_THRESHOLD, FRAME_SECONDS,
                               collect_temporal_data)
from sva_temporal_validation import evaluate_temporal


MIN_ROUTES = 30
MIN_SPLIT_ROUTES = 10


def play_key(row):
    return int(row["game_id"]), int(row["play_id"])


def route_key(row):
    return (*play_key(row), int(row["nfl_id"]))


def numbers(rows, *fields):
    return np.array([[float(row[field]) for field in fields] for row in rows])


def geometry(routes, first):
    prefix = "full_first" if first else "onset"
    speed = "full_first_receiver_speed" if first else "onset_speed_yd_s"
    score, distance, separation, lane, hand, radius, movement = numbers(
        routes, f"{prefix}_sva", f"{prefix}_distance_yd", f"{prefix}_separation_yd",
        f"{prefix}_lane_ease", f"{prefix}_hand_ease", "catch_radius_yd", speed,
    ).T
    return np.column_stack((score / 10, distance / 10, (distance / 10) ** 2, np.log1p(separation),
                            lane, hand, radius, movement / 5))


def play_context(plays):
    return _context([dict(row, rush_distance=row["onset_rusher_distance_yd"],
                          rush_closing=row["onset_rusher_closing_yd_s"],
                          rushers=row["onset_credited_rushers"], qb_speed=row["onset_qb_speed_yd_s"])
                     for row in plays])


def first_eligible_forecasts(routes, plays, context, outer):
    """Apply the same onset-trained heads at first eligibility, on other games."""
    indices = np.array([row["play_index"] for row in routes])
    x = geometry(routes, first=True)
    depth, margin, motion = numbers(routes, "full_first_depth_yd", "full_first_margin_yd",
                                    "full_first_downfield_speed").T
    late_down = np.array([plays[i]["situation"][4] in {"3", "4"} for i in indices])
    value_x = np.column_stack((x, depth / 10, margin / 10, margin >= 0,
                               np.maximum(-margin, 0) * late_down / 10, motion / 5,
                               context[indices], category_controls([row["position"] for row in routes])))
    labeled_onset = np.array([row["caught"] is not None and np.isfinite(row["onset_sva"]) for row in routes])
    caught = np.array([row["caught"] == 1 for row in routes])
    observed_epa = np.array([plays[i]["epa"] for i in indices])
    probability = np.empty(len(routes))
    epa = np.empty_like(probability)
    for fold in np.unique(outer):
        test = outer[indices] == fold
        train = ~test & labeled_onset
        probability[test] = fit_logistic(x[train], caught[train], ridge=35).predict(x[test])
        completed = train & caught
        missed = train & ~caught
        on_catch = fit_ridge(value_x[completed], observed_epa[completed], ridge=100).predict(value_x[test])
        on_miss = fit_ridge(value_x[missed], observed_epa[missed], ridge=100).predict(value_x[test])
        epa[test] = probability[test] * on_catch + (1 - probability[test]) * on_miss
    return [dict(row, projected_depth_yd=row["full_first_depth_yd"],
                 projected_margin_to_sticks_yd=row["full_first_margin_yd"], catch_probability=p,
                 hurdle_epa_on_comparable_targets=e) for row, p, e in zip(routes, probability, epa)]


def availability_lift(routes, situations, pressure, outer, window="full"):
    context = situation_design(routes, situations, pressure)[0]
    exposure, entry, observed = numbers(routes, f"{window}_in_range_frames", f"{window}_first_in_range_seconds",
                                        f"{window}_early_weighted_open_share").T
    design = np.column_stack((context, np.log1p(exposure), entry))
    games = np.array([row["game_id"] for row in routes])
    route_folds = outer[[row["play_index"] for row in routes]]
    expected = np.empty(len(routes))
    for fold in np.unique(route_folds):
        rows = np.flatnonzero(route_folds == fold)
        inner = game_folds(games[rows], n_folds=4, seed=2021 + int(fold))
        for held_out in np.unique(inner):
            train, test = rows[inner != held_out], rows[inner == held_out]
            expected[test] = fit_ridge(design[train], observed[train], BENCHMARK_RIDGE).predict(design[test])
    return observed - np.clip(expected, 0, 1)


def standardized(lift):
    return lift / lift.std()


def temporal_scores(catch, epa, availability):
    return np.clip(50 + 10 * (.25 * standardized(catch) + .25 * standardized(epa)
                              + .50 * standardized(availability)), 0, 100)


def player_rows(routes, scores, catch, epa, availability, previous):
    by_player = defaultdict(list)
    for i, row in enumerate(routes):
        by_player[row["nfl_id"]].append(i)
    result = []
    for nfl_id, indices in by_player.items():
        if len(indices) < MIN_ROUTES:
            continue
        subset = [routes[i] for i in indices]
        opened = [row["full_first_open_seconds"] for row in subset if row["full_open_frames"]]
        old = previous.get(nfl_id)
        base = scores[indices].mean()
        result.append(dict(nfl_id=nfl_id, player=subset[0]["player"], position=subset[0]["position"],
                           temporal_sva=50 + 50 * tanh(DEFAULT_SCORE_SPREAD * (base - 50) / 50), base_temporal_sva=base,
                           in_range_routes=len(indices), later_entry_routes=sum(not np.isfinite(row["onset_sva"]) for row in subset),
                           mean_open_share=np.mean([row["full_open_share"] for row in subset]),
                           mean_early_weighted_open_share=np.mean([row["full_early_weighted_open_share"] for row in subset]),
                           opened_route_share=len(opened) / len(indices), median_first_open_seconds=np.median(opened) if opened else None,
                           mean_catch_above_context_pp=100 * catch[indices].mean(), mean_epa_above_context=epa[indices].mean(),
                           mean_open_share_above_context=availability[indices].mean(),
                           previous_rank=int(old["context_sva_rank"]) if old else None,
                           previous_sva=float(old["context_sva"]) if old else None))
    result.sort(key=lambda row: (-row["base_temporal_sva"], -row["in_range_routes"], row["nfl_id"]))
    common_place = 0
    for rank, row in enumerate(result, 1):
        row["temporal_rank"] = rank
        if row["previous_rank"] is not None:
            common_place += 1
            row["temporal_rank_among_previous_players"] = common_place
            row["places_gained_among_previous_players"] = row["previous_rank"] - common_place
        else:
            row["temporal_rank_among_previous_players"] = None
            row["places_gained_among_previous_players"] = None
    return result


def ranks(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def stability(routes, temporal_scores, old_outlets, old_scores, weeks, player_ids):
    game_ids = sorted(weeks)
    game_index = {game: i for i, game in enumerate(game_ids)}
    player_index = {player: i for i, player in enumerate(player_ids)}
    old_key = {route_key(row) for row in old_outlets}

    def totals(rows, scores, matched=False):
        sums = np.zeros((len(game_ids), len(player_ids)))
        counts = np.zeros_like(sums)
        for row, score in zip(rows, scores):
            player = int(row["nfl_id"])
            if player in player_index and (not matched or route_key(row) in old_key):
                i, j = game_index[int(row["game_id"])], player_index[player]
                sums[i, j] += score
                counts[i, j] += 1
        return sums, counts

    models = (totals(old_outlets, old_scores), totals(routes, temporal_scores, matched=True),
              totals(routes, temporal_scores))

    def compare(first):
        old_counts = models[0][1]
        eligible = (old_counts[first].sum(axis=0) >= MIN_SPLIT_ROUTES) & (old_counts[~first].sum(axis=0) >= MIN_SPLIT_ROUTES)
        result = []
        for sums, counts in models:
            left = sums[first].sum(axis=0)[eligible] / counts[first].sum(axis=0)[eligible]
            right = sums[~first].sum(axis=0)[eligible] / counts[~first].sum(axis=0)[eligible]
            overlap = len(set(np.argsort(-left)[:10]) & set(np.argsort(-right)[:10]))
            result.append((np.corrcoef(ranks(left), ranks(right))[0, 1], overlap))
        return int(eligible.sum()), result

    week_array = np.array([weeks[game] for game in game_ids])
    n, chronological = compare(week_array <= 4)
    print(f"Split-half stability ({n} common players, at least {MIN_SPLIT_ROUTES} onset routes per half):")
    names = ("Onset SVA", "Temporal on same routes", "Temporal with later entries")
    for name, (rho, overlap) in zip(names, chronological):
        print(f"  {name}: Spearman {rho:+.3f}, top-10 overlap {overlap}.")
    rng = np.random.default_rng(2021)
    weekly = [np.flatnonzero(week_array == week) for week in np.unique(week_array)]
    draws = []
    for _ in range(500):
        first = np.zeros(len(game_ids), dtype=bool)
        for group in weekly:
            first[rng.permutation(group)[:len(group) // 2]] = True
        _, result = compare(first)
        draws.append([rho for rho, _ in result])
    rho = np.asarray(draws)
    print("Median rank correlations across 500 week-balanced game splits:", np.median(rho, axis=0).round(3))
    print("Temporal beat onset on same/expanded routes in", (rho[:, 1:] > rho[:, :1]).mean(axis=0).round(3))


def early_features(rows, landmark_seconds):
    history = numbers(rows, "early_mean_sva", "early_max_sva", "early_mean_separation_yd", "early_mean_lane_ease")
    opened = np.array([row["early_open_frames"] > 0 for row in rows])
    timing = np.column_stack((numbers(rows, "early_in_range_frames", "early_open_share",
                                      "early_early_weighted_open_share", "early_longest_open_seconds"), opened,
                              [row["early_first_open_seconds"] if flag else landmark_seconds + FRAME_SECONDS
                               for row, flag in zip(rows, opened)]))
    return history, timing


def team_features(routes, plays, landmark_seconds):
    groups = defaultdict(list)
    for row in routes:
        groups[row["play_index"]].append(row)
    onset, history, timing = [], [], []
    for i in range(len(plays)):
        at_onset = [r for r in groups[i] if np.isfinite(r["onset_sva"])]
        early = [r for r in groups[i] if r["early_in_range_frames"]]
        opened = [r for r in early if r["early_open_frames"]]
        onset_scores = sorted((r["onset_sva"] for r in at_onset), reverse=True)
        onset.append((len(at_onset), max(onset_scores, default=0), np.mean(onset_scores[:2]) if onset_scores else 0,
                      max((r["onset_sva"] for r in at_onset if r["onset_projected_margin_to_sticks_yd"] >= 0), default=0)))
        history.append((len(early), max((r["early_mean_sva"] for r in early), default=0),
                        max((r["early_max_sva"] for r in early), default=0),
                        max((r["early_max_sva"] for r in early if r["early_mean_projected_margin_to_sticks_yd"] >= 0), default=0)))
        timing.append((len(opened), max((r["early_open_share"] for r in early), default=0),
                       max((r["early_early_weighted_open_share"] for r in early), default=0),
                       max((r["early_longest_open_seconds"] for r in early), default=0),
                       min((r["early_first_open_seconds"] for r in opened), default=landmark_seconds + FRAME_SECONDS)))
    return tuple(np.asarray(values) for values in (onset, history, timing))


def landmark_tests(routes, plays, context, situations, pressure, landmark_seconds, old_index, outer):
    eligible_plays = np.array([row["landmark_eligible"] for row in plays])
    targets = [row for row in routes if eligible_plays[row["play_index"]] and row["caught"] is not None
               and np.isfinite(row["onset_sva"])]
    target_context_rows = [dict(row, projected_depth_yd=row["onset_projected_depth_yd"],
                                projected_margin_to_sticks_yd=row["onset_projected_margin_to_sticks_yd"]) for row in targets]
    play_idx = np.array([row["play_index"] for row in targets])
    early_pressure = np.array([row["early_pressure_frames"] for row in plays])
    catch_context = np.column_stack((situation_design(target_context_rows, situations, pressure)[0], early_pressure[play_idx]))
    catch_onset = geometry(targets, first=False)
    catch_history, catch_timing = early_features(targets, landmark_seconds)
    team_onset, team_history, team_timing = (matrix[eligible_plays] for matrix in team_features(routes, plays, landmark_seconds))
    team_context = np.column_stack((context, early_pressure))[eligible_plays]
    games = np.array([row["game_id"] for row in plays])
    epa = np.array([row["epa"] for row in plays])[eligible_plays]
    caught = np.array([row["caught"] for row in targets])
    full = evaluate_temporal(caught, games[play_idx], catch_context, catch_onset,
                             np.column_stack((catch_history, catch_timing)), epa, games[eligible_plays],
                             team_context, team_onset, np.column_stack((team_history, team_timing)))
    timing = evaluate_temporal(caught, games[play_idx], catch_context, np.column_stack((catch_onset, catch_history)),
                               catch_timing, epa, games[eligible_plays], team_context,
                               np.column_stack((team_onset, team_history)), team_timing)
    print(f"+{landmark_seconds:.1f}s landmark: {len(epa):,} still-ongoing plays, {len(caught):,} matched onset targets.")
    print(f"Catch AUC: context {full.catch_context.auc:.4f}, onset {full.catch_onset.auc:.4f}, "
          f"history quality {timing.catch_onset.auc:.4f}, full temporal {full.catch_temporal.auc:.4f}; "
          f"gain vs onset CI {full.catch_auc_gain_ci}.")
    print(f"Catch Brier: onset {full.catch_onset.brier:.4f}, full temporal {full.catch_temporal.brier:.4f}; "
          f"reduction CI {full.catch_brier_gain_ci}.")
    print(f"Team EPA RMSE: context {full.epa_context.rmse:.4f}, onset {full.epa_onset.rmse:.4f}, "
          f"history quality {timing.epa_onset.rmse:.4f}, full temporal {full.epa_temporal.rmse:.4f}; "
          f"reduction vs onset CI {full.epa_rmse_gain_ci}.")
    print(f"Timing/frequency beyond history quality: catch AUC gain CI {timing.catch_auc_gain_ci}, "
          f"team EPA RMSE reduction CI {timing.epa_rmse_gain_ci}.")

    # Also test the composite itself, recomputed using only eligible early frames.
    early_routes = [row for row in routes if eligible_plays[row["play_index"]] and row["early_in_range_frames"]]
    catch, value = benchmark_forecasts(early_routes, plays, situations, pressure)
    availability = availability_lift(early_routes, situations, pressure, outer, window="early")
    early_index = dict(zip(map(route_key, early_routes), temporal_scores(catch, value, availability)))
    old_target = np.array([old_index[route_key(row)] for row in targets])[:, None]
    early_target = np.array([early_index[route_key(row)] for row in targets])[:, None]
    best_old = np.full(len(plays), -np.inf)
    best_early = np.full(len(plays), -np.inf)
    for row in early_routes:
        k = route_key(row)
        i = row["play_index"]
        best_early[i] = max(best_early[i], early_index[k])
        if k in old_index:
            best_old[i] = max(best_old[i], old_index[k])
    old_team = np.where(np.isfinite(best_old), best_old, 50)[eligible_plays, None]
    early_team = np.where(np.isfinite(best_early), best_early, 50)[eligible_plays, None]
    team_availability_context = np.column_stack((team_context, team_onset[:, 0], team_history[:, 0]))
    composite = evaluate_temporal(caught, games[play_idx], catch_context, old_target, early_target,
                                  epa, games[eligible_plays], team_availability_context, old_team, early_team,
                                  forecast_games=games)
    print(f"Adding the early composite to the onset index: catch AUC {composite.catch_onset.auc:.4f} to "
          f"{composite.catch_temporal.auc:.4f}, gain CI {composite.catch_auc_gain_ci}; "
          f"team EPA RMSE {composite.epa_onset.rmse:.4f} to {composite.epa_temporal.rmse:.4f}, "
          f"reduction CI {composite.epa_rmse_gain_ci}.")


def run(data, archive, board_output, route_output, landmark_frames=DEFAULT_LANDMARK_FRAMES):
    plays = data.plays
    routes = [row for row in data.routes if row["full_in_range_frames"]]
    pressure = {play_key(row): row for row in plays}
    situations = {play_key(row): row["situation"] for row in plays}
    games = np.array([row["game_id"] for row in plays])
    outer = game_folds(games)
    context = play_context(plays)
    routes = first_eligible_forecasts(routes, plays, context, outer)
    old_outlets = list(read_csv(ROOT / "sva_joint_outlets.csv"))
    old_by_key = {route_key(row): row for row in old_outlets}
    if {route_key(row) for row in routes if np.isfinite(row["onset_sva"])} != old_by_key.keys():
        raise ValueError("The onset route cohort does not match the previous forecasts.")
    for row in routes:
        old = old_by_key.get(route_key(row))
        if old and (abs(row["catch_probability"] - float(old["catch_probability"])) > .00051
                    or abs(row["hurdle_epa_on_comparable_targets"] - float(old["hurdle_epa_on_comparable_targets"])) > .00051):
            raise ValueError(f"Earliest-eligible forecast does not match onset baseline for {route_key(row)}.")
    catch, epa = benchmark_forecasts(routes, plays, situations, pressure)
    availability = availability_lift(routes, situations, pressure, outer)
    # Half the score is the prior quality recipe, half rewards early/frequent openings.
    scores = temporal_scores(catch, epa, availability)
    previous_pressure = {play_key(row): row for row in read_csv(ROOT / "sva_epa_plays.csv")}
    old_catch, old_epa = benchmark_forecasts(old_outlets, plays, situations, previous_pressure)
    old_scores = np.clip(50 + 5 * (standardized(old_catch) + standardized(old_epa)), 0, 100)
    previous = {int(row["nfl_id"]): row for row in read_csv(ROOT / "sva_context_leaderboard.csv")}
    leaderboard = player_rows(routes, scores, catch, epa, availability, previous)
    print(f"{len(plays):,} pressure plays; {len(routes):,} in-range routes ({len(routes)-len(old_outlets):,} added); "
          f"{len(leaderboard)} qualified players.", flush=True)
    with ZipFile(archive) as source:
        weeks = {int(row["gameId"]): int(row["week"]) for row in archive_rows(source, "games.csv")}
    stability(routes, scores, old_outlets, old_scores, weeks, sorted(previous))
    landmark_tests(routes, plays, context, situations, pressure, landmark_frames * FRAME_SECONDS,
                   dict(zip(map(route_key, old_outlets), old_scores)), outer)
    fields = ("temporal_rank", "nfl_id", "player", "position", "temporal_sva", "base_temporal_sva", "in_range_routes",
              "later_entry_routes", "mean_open_share", "mean_early_weighted_open_share", "opened_route_share",
              "median_first_open_seconds", "mean_catch_above_context_pp", "mean_epa_above_context",
              "mean_open_share_above_context", "previous_rank", "previous_sva", "temporal_rank_among_previous_players",
              "places_gained_among_previous_players")
    write_csv(board_output, fields, leaderboard)
    route_fields = ("game_id", "play_id", "nfl_id", "player", "position", "onset_sva", "full_in_range_frames",
                    "full_open_frames", "full_open_share", "full_early_weighted_open_share", "full_first_in_range_seconds",
                    "full_first_open_seconds", "full_wait_for_open_seconds", "full_longest_open_seconds",
                    "catch_probability", "hurdle_epa_on_comparable_targets", "caught")
    write_csv(route_output, (*route_fields, "base_temporal_sva"), (
        dict({field: None if isinstance(row[field], (float, np.floating)) and not np.isfinite(row[field]) else row[field]
              for field in route_fields}, base_temporal_sva=score) for row, score in zip(routes, scores)
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "nfl-big-data-bowl-regional-event-data-main.zip")
    parser.add_argument("--epa", type=Path, default=ROOT / "nflverse_2021_tracking_epa.csv")
    parser.add_argument("--hands", type=Path, default=ROOT / "qb_handedness.csv")
    parser.add_argument("--board-output", type=Path, default=ROOT / "sva_temporal_leaderboard.csv")
    parser.add_argument("--route-output", type=Path, default=ROOT / "sva_temporal_routes.csv")
    args = parser.parse_args()
    inputs = (args.archive, args.epa, args.hands, ROOT / "sva_joint_outlets.csv",
              ROOT / "sva_context_leaderboard.csv", ROOT / "sva_epa_plays.csv")
    outputs = {args.board_output.resolve(), args.route_output.resolve()}
    if len(outputs) != 2 or outputs.intersection(path.resolve() for path in inputs):
        parser.error("Use distinct output files separate from the inputs.")
    data = collect_temporal_data(args.archive, args.epa, args.hands, DEFAULT_OPEN_THRESHOLD, DEFAULT_LANDMARK_FRAMES)
    run(data, args.archive, args.board_output, args.route_output)


if __name__ == "__main__":
    main()
