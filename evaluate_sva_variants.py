"""Test prespecified SVA changes on the same targets, holding out Weeks 7–8."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from analyze_catch_rate import archive_rows, category_controls, pre_snap_context, target_outcomes
from safety_valve import (BASELINE_THROW_MODEL, DEFAULT_MAX_THROW_DISTANCE, ROOT, SEPARATION_WEIGHTED_THROW_MODEL,
                          catch_radius, pressure_frame_scores, read_hands)
from sva_data import open_data
from sva_pressure_context import contexts_for_play
from sva_variant_evaluation import evaluate_variants


BASE = BASELINE_THROW_MODEL
LIGHTER_LANE = replace(BASE, defender_lane_penalty=0.50, teammate_lane_penalty=0.10)
BROADER_SEPARATION = replace(BASE, separation_midpoint=2.35, separation_width=1.10)
VARIANTS = (
    ("Current SVA", BASE, "mean"),
    ("Recent frames (0.35 s half-life)", BASE, "recent"),
    ("Best sustained opening (up to 3 frames)", BASE, "sustained"),
    ("Lighter lane penalties", LIGHTER_LANE, "mean"),
    ("Broader separation curve", BROADER_SEPARATION, "mean"),
    ("Shorter distance scale (12 yd)", replace(BASE, distance_scale=12), "mean"),
    ("Longer distance scale (24 yd)", replace(BASE, distance_scale=24), "mean"),
    ("Lighter throwing-hand penalty", replace(BASE, hand_penalty=0.11), "mean"),
    ("Higher stretch penalty", replace(BASE, stretch_penalty=0.28), "mean"),
    ("Lighter lane penalties + broader separation", SEPARATION_WEIGHTED_THROW_MODEL, "mean"),
    ("Lighter lane penalties + recent frames", LIGHTER_LANE, "recent"),
)


def aggregate(frame_ids, scores, method):
    if method == "recent":
        return np.average(scores, weights=2.0 ** ((frame_ids - frame_ids[-1]) / 3.5))
    if method == "sustained":
        runs = np.split(scores, np.flatnonzero(np.diff(frame_ids) != 1) + 1)
        width = min(3, max(map(len, runs)))
        return max(np.convolve(run, np.ones(width) / width, mode="valid").max()
                   for run in runs if len(run) >= width)
    return scores.mean()


def matched_routes(archive, routes_path):
    outcomes, _ = target_outcomes(archive)
    matches = {}
    with routes_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            game_id, play_id, nfl_id = (int(row[field]) for field in ("game_id", "play_id", "nfl_id"))
            if row["sva"] and (game_id, play_id, nfl_id) in outcomes:
                matches[(game_id, play_id)] = (nfl_id, row, outcomes[(game_id, play_id, nfl_id)][0])
    return matches


def score_targets(archive, hands, matches, max_throw_distance):
    models = tuple(dict.fromkeys(model for _, model, _ in VARIANTS))
    variant_indices = [models.index(model) for _, model, _ in VARIANTS]
    games = {game_id for game_id, _ in matches}
    with open_data(archive) as data:
        for game_id in data.game_ids:
            if game_id not in games:
                continue
            for play_id, frames in data.tracking(game_id):
                key = game_id, play_id
                if key not in matches:
                    continue
                nfl_id, route, caught = matches[key]
                play = data.plays[key]
                radii = np.array([catch_radius(data.players[nfl_id].height_inches)])
                frame_ids, scores = pressure_frame_scores(play, frames, (nfl_id,), radii,
                                                          hands.get(play.qb_id, ""), max_throw_distance, models)
                if not len(frame_ids):
                    raise ValueError(f"No pressure frames for previously scored target {key}.")
                eligible = np.isfinite(scores[:, 0, 0, 0])
                if not eligible.any() or not np.all(np.isfinite(scores[:, :, 0, 0]) == eligible[:, None]):
                    raise ValueError(f"Model variants do not share the same frame eligibility for {key}.")
                eligible_ids = frame_ids[eligible]
                values = [aggregate(eligible_ids, scores[eligible, i, 0, 0], method)
                          for (_, _, method), i in zip(VARIANTS, variant_indices)]
                if len(eligible_ids) != int(route["in_range_frames"]) or abs(values[0] - float(route["sva"])) > 0.00051:
                    raise ValueError(f"Current SVA does not reproduce for {key}; check the range cutoff and route CSV.")
                for _, context in contexts_for_play(play, frames, (nfl_id,), max_throw_distance, set(frame_ids)):
                    yield key, route["position"], caught, context, values


def design_matrix(rows, situations):
    release = [row[3] for row in rows]
    yards_to_go, yards_to_goal, margin, play_action, down, _, coverage, _ = zip(
        *(situations[row[0]] for row in rows)
    )
    rush_distance = np.array([item.rusher_distance_yd for item in release])
    target_distance = np.array([item.target_distance_yd for item in release])
    return np.column_stack((
        rush_distance, rush_distance ** 2, [item.rusher_closing_yd_s for item in release],
        [item.credited_rushers for item in release], [item.pressure_seconds for item in release],
        [item.pressure_at_release for item in release], target_distance, target_distance ** 2,
        [item.target_downfield_yd for item in release], [item.qb_speed_yd_s for item in release],
        [item.time_to_throw_s for item in release], yards_to_go, yards_to_goal, margin, play_action,
        category_controls([row[1] for row in rows]), category_controls(down), category_controls(coverage),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "nfl-big-data-bowl-regional-event-data-main.zip")
    parser.add_argument("--routes", type=Path, default=ROOT / "safety_valve_plays.csv")
    parser.add_argument("--hands", type=Path, default=ROOT / "qb_handedness.csv")
    parser.add_argument("--max-throw-distance", type=float, default=DEFAULT_MAX_THROW_DISTANCE)
    args = parser.parse_args()
    if not np.isfinite(args.max_throw_distance) or args.max_throw_distance <= 0:
        parser.error("--max-throw-distance must be a positive, finite number")

    matches = matched_routes(args.archive, args.routes)
    situations = dict(pre_snap_context(args.archive, matches.keys()))
    with ZipFile(args.archive) as archive:
        weeks_by_game = {int(row["gameId"]): int(row["week"]) for row in archive_rows(archive, "games.csv")}
    print(f"Rescoring {len(matches):,} matched targets with fixed frame eligibility.", flush=True)
    rows = [row for row in score_targets(args.archive, read_hands(args.hands), matches, args.max_throw_distance)
            if row[0] in situations]
    games = np.array([row[0][0] for row in rows])
    weeks = np.array([weeks_by_game[game] for game in games])
    caught = np.array([row[2] for row in rows])
    scores = np.array([row[4] for row in rows])
    names = [name for name, _, _ in VARIANTS]
    fit = evaluate_variants(scores, caught, games, design_matrix(rows, situations), names, weeks=weeks)

    print(f"Development (Weeks 1–6): {fit.development_n:,} targets across {len(fit.development_games)} games.")
    for index in np.argsort(-fit.development_adjusted):
        print(f"{names[index]}: raw r={fit.development_raw[index]:+.3f}, "
              f"adjusted r={fit.development_adjusted[index]:+.3f}.")
    print(f"\nHeld-out test (Weeks 7–8): {fit.test_n:,} targets across {len(fit.test_games)} games.")
    for index in dict.fromkeys((fit.baseline, fit.winner)):
        print(f"{names[index]}: raw r={fit.test_raw[index]:+.3f}, "
              f"adjusted r={fit.test_adjusted[index]:+.3f}, AUC={fit.test_auc[index]:.3f}.")
    for label, values, interval in (("Raw", fit.test_raw, fit.test_raw_gain_ci),
                                    ("Adjusted", fit.test_adjusted, fit.test_adjusted_gain_ci)):
        gain = values[fit.winner] - values[fit.baseline]
        print(f"{label} gain {gain:+.3f} (95% paired game-bootstrap interval {interval[0]:+.3f} to {interval[1]:+.3f}).")


if __name__ == "__main__":
    main()
