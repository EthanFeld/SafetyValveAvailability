"""Test whether available outlets predict team EPA on pressured dropbacks."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from analyze_catch_rate import archive_rows, category_controls, play_key, pre_snap_context
from safety_valve import (BASELINE_THROW_MODEL, DEFAULT_MAX_THROW_DISTANCE, ROOT, SEPARATION_WEIGHTED_THROW_MODEL,
                          catch_radius, pressure_frame_scores, read_hands, write_csv)
from sva_data import PRESSURE_FIELDS, open_data
from sva_epa_models import clustered_coefficient, game_folds, incremental_epa, out_of_fold_predictions
from sva_pressure_context import _nearest_rusher


RIDGE = 10.0
MODELS = (BASELINE_THROW_MODEL, SEPARATION_WEIGHTED_THROW_MODEL)


def read_dropbacks(archive_path, epa_path):
    with epa_path.open(encoding="utf-8-sig", newline="") as file:
        epa_rows = list(csv.DictReader(file))
    epa = {(int(row["game_id"]), int(row["play_id"])): row for row in epa_rows}
    if len(epa) != len(epa_rows):
        raise ValueError("Duplicate game/play keys in EPA input.")

    with ZipFile(archive_path) as archive:
        plays = {play_key(row): row for row in archive_rows(archive, "plays.csv")}
        missing = plays.keys() - epa.keys()
        if missing:
            raise ValueError(f"EPA input is missing {len(missing)} tracking plays.")
        qb_ids = {int(row["nflId"]) for row in archive_rows(archive, "players.csv")
                  if row["officialPosition"] == "QB"}
        quarterbacks = defaultdict(set)
        pressures = set()
        for row in archive_rows(archive, "pffScoutingData.csv"):
            key, player = play_key(row), int(row["nflId"])
            if row["pff_role"] == "Pass" and player in qb_ids:
                quarterbacks[key].add(player)
            if any(row[field] == "1" for field in PRESSURE_FIELDS):
                pressures.add(key)

    candidates = {key for key, row in plays.items() if epa[key]["qb_dropback"] == "1"
                  and len(quarterbacks[key]) == 1 and "two-point conversion attempt" not in row["playDescription"].lower()}
    situations = dict(pre_snap_context(archive_path, candidates))
    dropbacks = {}
    for key in sorted(candidates & situations.keys()):
        row = plays[key]
        minute, second = row["gameClock"].split(":")
        dropbacks[key] = dict(game_id=key[0], play_id=key[1], offense=row["possessionTeam"],
                              qb_id=next(iter(quarterbacks[key])), situation=situations[key],
                              formation=row["offenseFormation"], clock=60 * int(minute) + int(second),
                              pressured=key in pressures, epa=float(epa[key]["epa"]),
                              sack=int(epa[key]["sack"] == "1"),
                              scramble=int(epa[key]["qb_scramble"] == "1"), result=row["passResult"])
    return dropbacks


def snap_frame(frames):
    snap = next((frame.id for frame in frames if frame.event == "ball_snap"), None)
    return snap if snap is not None else next(frame.id for frame in frames if frame.event == "autoevent_ballsnap")


def pressure_outlets(archive_path, hands, dropbacks):
    with open_data(archive_path) as data:
        for game_id in data.game_ids:
            for play_id, frames in data.tracking(game_id):
                key = game_id, play_id
                if key not in dropbacks or not dropbacks[key]["pressured"]:
                    continue
                play = data.plays[key]
                radii = np.array([catch_radius(data.players[i].height_inches) for i in play.receiver_ids])
                frame_ids, scores = pressure_frame_scores(play, frames, play.receiver_ids, radii,
                                                          hands.get(play.qb_id, ""), DEFAULT_MAX_THROW_DISTANCE, MODELS)
                if not len(frame_ids):
                    continue

                values = scores[..., 0]
                eligible = np.isfinite(values)
                counts = eligible.sum(axis=0)
                means = np.divide(np.nansum(values, axis=0), counts, out=np.zeros_like(counts, dtype=float),
                                  where=counts > 0)
                original, revised = means
                order = np.argsort(-revised, kind="stable")
                best = revised[order[0]]
                second = revised[order[1]] if len(order) > 1 else 0.0
                best_id = play.receiver_ids[order[0]] if best > 0 else None
                best_player = data.players[best_id] if best_id is not None else None

                onset = next(frame for frame in frames if frame.id == frame_ids[0])
                qb = onset.positions[play.qb_id]
                rushers = [onset.positions[i] for i in play.pressure_ids
                           if i in onset.positions and onset.positions[i].team == play.defense]
                distance, closing = _nearest_rusher(qb, rushers)
                yield key, dict(original_best_sva=original.max(), revised_best_sva=best, second_best_sva=second,
                                best_sva_margin=best - second, onset_best_sva=np.nan_to_num(values[0, 1]).max(),
                                top_receiver_id=best_id, top_receiver=best_player.name if best_player else None,
                                top_receiver_position=best_player.position if best_player else None,
                                in_range_receivers=int((counts[1] > 0).sum()), route_runners=len(play.receiver_ids),
                                pressure_frames=len(frame_ids), onset_seconds=(onset.id - snap_frame(frames)) / 10,
                                onset_rusher_distance_yd=distance, onset_rusher_closing_yd_s=closing,
                                onset_credited_rushers=len(rushers), onset_qb_speed_yd_s=qb.speed)


def situation_design(rows):
    to_go, to_goal, margin, play_action, down, quarter, coverage, defense = zip(*(row["situation"] for row in rows))
    to_go = np.log1p(to_go)
    to_goal = np.asarray(to_goal) / 50
    clock = np.array([row["clock"] for row in rows]) / 900
    return np.column_stack((
        to_go, to_go * np.isin(down, ("3", "4")), to_goal, to_goal ** 2, np.asarray(margin) / 14,
        play_action, clock, category_controls(down), category_controls(quarter), category_controls(coverage),
        category_controls(defense), category_controls([row["offense"] for row in rows]),
        category_controls([row["qb_id"] for row in rows]), category_controls([row["formation"] for row in rows]),
    ))


def pressure_design(rows, situation):
    distance = np.array([row["onset_rusher_distance_yd"] for row in rows]) / 3
    return np.column_stack((situation, distance, distance ** 2,
                            [row["onset_rusher_closing_yd_s"] / 5 for row in rows],
                            [row["onset_credited_rushers"] for row in rows],
                            [row["onset_qb_speed_yd_s"] / 4 for row in rows],
                            [row["onset_seconds"] for row in rows], [row["route_runners"] for row in rows]))


def interval_text(interval):
    return f"{interval[0]:+.4f} to {interval[1]:+.4f}"


def describe_fit(label, fit):
    print(f"{label}: {fit.coefficient_per_10:+.4f} EPA / +10 SVA (95% CI {interval_text(fit.coefficient_ci)}); "
          f"OOF R2 {fit.baseline.r2:+.4f} to {fit.augmented.r2:+.4f} "
          f"(gain {fit.delta_r2:+.4f}, 95% CI {interval_text(fit.delta_r2_ci)}); "
          f"RMSE {fit.baseline.rmse:.4f} to {fit.augmented.rmse:.4f}.")


def receiver_summaries(rows):
    by_receiver = defaultdict(list)
    for row in rows:
        if row["top_receiver_id"] is not None:
            by_receiver[row["top_receiver_id"]].append(row)
    fields = ("revised_best_sva", "best_sva_margin", "epa", "expected_clean_epa", "epa_vs_clean", "epa_vs_pressure_context")
    result = []
    for nfl_id, plays in by_receiver.items():
        result.append(dict(nfl_id=nfl_id, player=plays[0]["top_receiver"], position=plays[0]["top_receiver_position"],
                           top_outlet_plays=len(plays),
                           **{f"mean_{field}": np.mean([row[field] for row in plays]) for field in fields}))
    return sorted(result, key=lambda row: (-row["top_outlet_plays"], row["nfl_id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "nfl-big-data-bowl-regional-event-data-main.zip")
    parser.add_argument("--epa", type=Path, default=ROOT / "nflverse_2021_tracking_epa.csv")
    parser.add_argument("--hands", type=Path, default=ROOT / "qb_handedness.csv")
    parser.add_argument("--play-output", type=Path, default=ROOT / "sva_epa_plays.csv")
    parser.add_argument("--player-output", type=Path, default=ROOT / "sva_epa_receivers.csv")
    args = parser.parse_args()
    outputs = {args.play_output.resolve(), args.player_output.resolve()}
    if len(outputs) < 2 or outputs.intersection(path.resolve() for path in (args.archive, args.epa, args.hands)):
        parser.error("Use separate files for each output and input.")

    dropbacks = read_dropbacks(args.archive, args.epa)
    all_rows = list(dropbacks.values())
    epa = np.array([row["epa"] for row in all_rows])
    all_games = np.array([row["game_id"] for row in all_rows])
    pressure = np.array([row["pressured"] for row in all_rows])
    all_folds = game_folds(all_games)
    situation = situation_design(all_rows)
    print(f"{len(np.unique(all_games))} games, {int(pressure.sum()):,} pressured dropbacks ({epa[pressure].mean():+.3f} EPA) "
          f"and {int((~pressure).sum()):,} clean dropbacks ({epa[~pressure].mean():+.3f} EPA).", flush=True)

    outlets = dict(pressure_outlets(args.archive, read_hands(args.hands), dropbacks))
    usable = np.array([key in outlets for key in dropbacks])
    if not usable.any():
        parser.error("No usable pressure windows joined to EPA.")
    rows = [dict(row, **outlets[key]) for key, row in dropbacks.items() if key in outlets]
    controls = pressure_design(rows, situation[usable])
    games, folds, observed = all_games[usable], all_folds[usable], epa[usable]
    clean_epa = out_of_fold_predictions(epa, situation, all_folds, train_mask=~pressure, predict_mask=usable,
                                        ridge=RIDGE)[usable]
    versus_clean = observed - clean_epa
    original = np.array([row["original_best_sva"] for row in rows])
    revised = np.array([row["revised_best_sva"] for row in rows])
    onset = np.array([row["onset_best_sva"] for row in rows])

    print(f"{len(rows):,} pressure windows across {len(np.unique(games))} games, "
          f"including {sum(not row['in_range_receivers'] for row in rows):,} plays without an in-range outlet; "
          f"{int(pressure.sum() - usable.sum()):,} pressure plays lack a usable window.")
    print(f"Usable pressure plays: observed {observed.mean():+.3f}, predicted clean {clean_epa.mean():+.3f}, "
          f"gap {versus_clean.mean():+.3f} EPA per play. "
          f"Includes {sum(row['sack'] for row in rows):,} sacks and {sum(row['scramble'] for row in rows):,} scrambles.")

    fit = incremental_epa(observed, controls, revised, games, folds=folds, ridge=RIDGE)
    describe_fit("Revised best SVA", fit)
    describe_fit("Original best SVA", incremental_epa(observed, controls, original, games, folds=folds, ridge=RIDGE))
    describe_fit("Revised SVA at pressure onset", incremental_epa(observed, controls, onset, games, folds=folds, ridge=RIDGE))

    for label, outcome in (("EPA vs clean expectation", versus_clean), ("Downside EPA shortfall", np.maximum(-versus_clean, 0))):
        slope, ci = clustered_coefficient(outcome, revised / 10, controls, games)
        print(f"{label}: {slope:+.4f} EPA / +10 SVA (95% CI {interval_text(ci)}).")
    second = np.array([row["second_best_sva"] for row in rows])
    slope, ci = clustered_coefficient(observed, (revised - second) / 10,
                                     np.column_stack((controls, second / 10)), games)
    print(f"Best receiver's margin over the next outlet: {slope:+.4f} EPA / +10 margin (95% CI {interval_text(ci)}).")

    for i, quartile in enumerate(np.array_split(np.argsort(revised, kind="stable"), 4), 1):
        print(f"Revised SVA quartile {i} (n={len(quartile)}, mean SVA={revised[quartile].mean():.1f}): "
              f"observed {observed[quartile].mean():+.3f}, vs clean {versus_clean[quartile].mean():+.3f} EPA.")

    has_outlet = np.array([row["in_range_receivers"] > 0 for row in rows])
    for label, group in (("In-range outlet", has_outlet), ("No in-range outlet", ~has_outlet)):
        if group.any():
            print(f"{label} (n={group.sum():,}): observed {observed[group].mean():+.3f}, "
                  f"vs clean {versus_clean[group].mean():+.3f} EPA.")

    for row, expected, gap, context_prediction in zip(rows, clean_epa, versus_clean, fit.baseline_predictions):
        row.update(expected_clean_epa=expected, epa_vs_clean=gap, context_only_epa=context_prediction,
                   epa_vs_pressure_context=row["epa"] - context_prediction)
        for field in ("situation", "formation", "clock", "pressured"):
            del row[field]
    player_rows = receiver_summaries(rows)
    write_csv(args.play_output, tuple(rows[0]), rows)
    if player_rows:
        write_csv(args.player_output, tuple(player_rows[0]), player_rows)


if __name__ == "__main__":
    main()
