"""Export the scored dropbacks as compact static replay files for GitHub Pages."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from io import TextIOWrapper
from math import tanh
from pathlib import Path, PurePosixPath
from statistics import fmean
from zipfile import ZipFile

import numpy as np

from analyze_catch_rate import archive_rows, play_key, pre_snap_context
from analyze_sva_epa import snap_frame
from rank_context_sva import DEFAULT_SCORE_SPREAD, benchmark_forecasts, score_players
from safety_valve import (DEFAULT_MAX_THROW_DISTANCE, FIELD_WIDTH, ROOT, SEPARATION_WEIGHTED_THROW_MODEL,
                          catch_radius, pressure_frame_scores, read_hands)
from sva_data import MISSING, Frame, Position, open_data
from sva_temporal_data import DEFAULT_OPEN_THRESHOLD, FRAME_SECONDS, _terminal_frame


OFFENSIVE_ROLES = {"Pass", "Pass Block", "Pass Route"}
PRE_SNAP_FRAMES = 4
POST_PLAY_FRAMES = 15
PROVISIONAL_MIN_ROUTES = 15


@dataclass(slots=True)
class Tracking:
    frames: dict = field(default_factory=dict)
    balls: dict = field(default_factory=dict)
    numbers: dict = field(default_factory=dict)


def rows_at(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)


def scored_key(row):
    return int(row["game_id"]), int(row["play_id"])


def archive_tracking(archive, member, play_ids):
    """Read one game and retain the ball and jersey data needed by the player."""
    rows_by_play = {str(play_id): Tracking() for play_id in play_ids}
    with TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="") as source:
        rows = csv.reader(source)
        columns = {name: i for i, name in enumerate(next(rows))}
        play_col, frame_col, player_col, number_col, team_col, play_dir_col, x_col, y_col, speed_col, direction_col, event_col = (
            columns[name] for name in ("playId", "frameId", "nflId", "jerseyNumber", "team", "playDirection", "x", "y", "s", "dir", "event")
        )
        for row in rows:
            track = rows_by_play.get(row[play_col])
            if track is None:
                continue
            frame_id = int(row[frame_col])
            state = track.frames.get(frame_id)
            if state is None:
                state = Frame(frame_id, "", row[play_dir_col], {})
                track.frames[frame_id] = state
            team = row[team_col]
            has_location = row[x_col] not in MISSING and row[y_col] not in MISSING
            if team == "football":
                state.event = row[event_col]
                if has_location:
                    track.balls[frame_id] = (float(row[x_col]), float(row[y_col]))
            elif row[player_col] not in MISSING and has_location:
                nfl_id = int(row[player_col])
                state.positions[nfl_id] = Position(float(row[x_col]), float(row[y_col]),
                                                   0.0 if row[speed_col] in MISSING else float(row[speed_col]),
                                                   None if row[direction_col] in MISSING else float(row[direction_col]),
                                                   team)
                if row[number_col] not in MISSING:
                    track.numbers[nfl_id] = row[number_col].removesuffix(".0")
    return {int(play_id): tracking for play_id, tracking in rows_by_play.items() if tracking.frames}


def opening_lineups(archive, plays, dropbacks):
    slots = defaultdict(list)
    for row in archive_rows(archive, "pffScoutingData.csv"):
        key = play_key(row)
        if key in dropbacks and row["pff_role"] in OFFENSIVE_ROLES:
            slots[key].append((int(row["nflId"]), row["pff_positionLinedUp"]))
    by_game = defaultdict(dict)
    for key in sorted(slots):
        if len(slots[key]) != 11 or len({nfl_id for nfl_id, _ in slots[key]}) != 11:
            continue
        game, play = key
        team = plays[key]["possessionTeam"]
        by_game[game].setdefault(team, (play, slots[key]))
    return by_game


def normal_point(x, y, right):
    if not right:
        x, y = 120 - x, FIELD_WIDTH - y
    return [round(x, 1), round(y, 1)]


def frame_highlights(ids, scores, receiver_ids):
    highlights = {}
    for frame_id, row in zip(ids, scores[:, 0, :, 0]):
        ranked = [(float(value), receiver_id) for receiver_id, value in zip(receiver_ids, row) if np.isfinite(value)]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        best, player = ranked[0] if ranked else (None, None)
        highlights[int(frame_id)] = (player, round(best, 1) if best is not None else None,
                                     [receiver_id for value, receiver_id in ranked if value >= DEFAULT_OPEN_THRESHOLD])
    return highlights


def replay(play_id, metadata, play, track, players, hands, epa, routes):
    frames = sorted(track.frames.values(), key=lambda frame: frame.id)
    radii = np.array([catch_radius(players[nfl_id].height_inches) for nfl_id in play.receiver_ids])
    pressure_ids, scores = pressure_frame_scores(play, frames, play.receiver_ids, radii, hands.get(play.qb_id, ""),
                                                  DEFAULT_MAX_THROW_DISTANCE, (SEPARATION_WEIGHTED_THROW_MODEL,))
    if not len(pressure_ids):
        raise ValueError(f"Previously scored play {metadata['gameId']}/{play_id} has no pressure frames.")
    snap = snap_frame(frames)
    terminal = _terminal_frame(play, frames, snap)
    if terminal is None:
        raise ValueError(f"Replay {metadata['gameId']}/{play_id} has no terminal frame.")
    selected = [frame for frame in frames if snap - PRE_SNAP_FRAMES <= frame.id <= terminal + POST_PLAY_FRAMES]
    right = track.frames[int(pressure_ids[0])].direction == "right"
    active_players = {nfl_id: pos.team for frame in selected for nfl_id, pos in frame.positions.items()}
    player_order = sorted(active_players, key=lambda nfl_id: (active_players[nfl_id] != play.offense, nfl_id))
    highlights = frame_highlights(pressure_ids, scores, play.receiver_ids)
    plot = []
    for frame in selected:
        coords = []
        for nfl_id in player_order:
            pos = frame.positions.get(nfl_id)
            coords.extend(normal_point(pos.x, pos.y, right) if pos else (None, None))
        best, score, opened = highlights.get(frame.id, (None, None, []))
        ball = track.balls.get(frame.id)
        plot.append(dict(t=round((frame.id - snap) * FRAME_SECONDS, 1), xy=coords,
                         ball=normal_point(*ball, right) if ball else None, pressure=frame.id in highlights,
                         best=best, score=score, open=opened,
                         event=frame.event if frame.event not in ("", "None", "NA") else None))

    yards = int(metadata["yardsToGo"])
    raw_los = float(metadata["absoluteYardlineNumber"])
    los = raw_los if right else 120 - raw_los
    yard_side = metadata["yardlineSide"]
    yard_line = f"{yard_side} {metadata['yardlineNumber']}" if yard_side not in MISSING else metadata["yardlineNumber"]
    return dict(id=play_id, offense=play.offense, defense=play.defense, down=int(metadata["down"]), toGo=yards,
                yardLine=yard_line, quarter=int(metadata["quarter"]), clock=metadata["gameClock"], result=play.result,
                epa=epa, description=metadata["playDescription"], qb=play.qb_id, los=los, firstDown=min(110, los + yards),
                snap=0.0, terminal=round((terminal - snap) * FRAME_SECONDS, 1),
                players=[dict(id=nfl_id, team=active_players[nfl_id], number=track.numbers.get(nfl_id, ""))
                         for nfl_id in player_order], frames=plot, routes=routes)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(data, output, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        output.write("\n")


def load_site_sources():
    epa = {scored_key(row): round(float(row["epa"]), 3) for row in rows_at(ROOT / "sva_epa_plays.csv")}
    dropbacks = {scored_key(row) for row in rows_at(ROOT / "nflverse_2021_tracking_epa.csv") if row["qb_dropback"] == "1"}
    onset = {int(row["nfl_id"]): row for row in rows_at(ROOT / "sva_context_leaderboard.csv")}
    temporal = {int(row["nfl_id"]): row for row in rows_at(ROOT / "sva_temporal_leaderboard.csv")}
    routes = defaultdict(list)
    for row in rows_at(ROOT / "sva_temporal_routes.csv"):
        routes[scored_key(row)].append(dict(id=int(row["nfl_id"]), score=float(row["base_temporal_sva"]),
                                         openShare=float(row["full_open_share"]),
                                         firstOpen=float(row["full_first_open_seconds"]) if row["full_first_open_seconds"] else None,
                                         longestOpen=float(row["full_longest_open_seconds"]),
                                         catchProbability=float(row["catch_probability"]),
                                         expectedEpa=float(row["hurdle_epa_on_comparable_targets"])))
    return epa, dropbacks, onset, temporal, routes


def player_index(archive_path, players, onset, temporal, routes):
    """Keep published grades exact and expose the same scores for smaller samples."""
    outlets = list(rows_at(ROOT / "sva_joint_outlets.csv"))
    joint_plays = list(rows_at(ROOT / "sva_joint_plays.csv"))
    pressure = {scored_key(row): row for row in rows_at(ROOT / "sva_epa_plays.csv")}
    situations = dict(pre_snap_context(archive_path, {scored_key(row) for row in outlets}))
    catch_lift, epa_lift = benchmark_forecasts(outlets, joint_plays, situations, pressure)
    onset_counts = Counter(int(row["nfl_id"]) for row in outlets)
    onset_grades = {row["nfl_id"]: row["context_sva"] for row in score_players(
        outlets, [], PROVISIONAL_MIN_ROUTES, catch_lift, epa_lift, 0.5,
    )}
    temporal_by_player = defaultdict(list)
    for play_routes in routes.values():
        for route in play_routes:
            temporal_by_player[route["id"]].append(route["score"])

    index = {}
    for nfl_id, player in players.items():
        start = onset.get(nfl_id)
        end = temporal.get(nfl_id)
        provisional_start = onset_grades.get(nfl_id) if not start else None
        temporal_scores = temporal_by_player.get(nfl_id, [])
        provisional_end = None
        if not end and len(temporal_scores) >= PROVISIONAL_MIN_ROUTES:
            base = fmean(temporal_scores)
            provisional_end = round(50 + 50 * tanh(DEFAULT_SCORE_SPREAD * (base - 50) / 50), 3)
        index[nfl_id] = dict(name=player.name, position=player.position,
                             onset=float(start["context_sva"]) if start else None,
                             onsetRank=int(start["context_sva_rank"]) if start else None,
                             onsetProvisional=round(provisional_start, 3) if provisional_start is not None else None,
                             onsetRoutes=onset_counts[nfl_id],
                             temporal=float(end["temporal_sva"]) if end else None,
                             temporalRank=int(end["temporal_rank"]) if end else None,
                             temporalProvisional=provisional_end,
                             temporalRoutes=len(temporal_scores))
    return index


def refresh_index(archive_path, output):
    _, _, onset, temporal, routes = load_site_sources()
    path = output / "index.json"
    with path.open(encoding="utf-8") as existing:
        index = json.load(existing)
    with open_data(archive_path) as data:
        index["players"] = player_index(archive_path, data.players, onset, temporal, routes)
    save_json(path, index)
    print(f"Refreshed scores for {len(index['players']):,} players in {path}.")


def build(archive_path, output, selected_games):
    epa, dropbacks, onset, temporal, routes = load_site_sources()
    hands = read_hands(ROOT / "qb_handedness.csv")
    epa_plays = defaultdict(set)
    for game, play in epa:
        epa_plays[game].add(play)
    with ZipFile(archive_path) as archive, open_data(archive_path) as data:
        game_rows = {int(row["gameId"]): row for row in archive_rows(archive, "games.csv")}
        missing = set(selected_games) - game_rows.keys()
        if missing:
            raise ValueError(f"Unknown game IDs: {', '.join(map(str, sorted(missing)))}")
        plays = {play_key(row): row for row in archive_rows(archive, "plays.csv")}
        lineups = opening_lineups(archive, plays, dropbacks)
        members = {int(path.stem.removeprefix("tracking_")): name for name in archive.namelist()
                   if (path := PurePosixPath(name)).name.startswith("tracking_") and path.suffix == ".csv"}
        index = []
        total_plays = 0
        for game_id in sorted(selected_games or game_rows):
            row = game_rows[game_id]
            game = dict(id=str(game_id), week=int(row["week"]), home=row["homeTeamAbbr"], away=row["visitorTeamAbbr"],
                        date=row["gameDate"])
            expected = epa_plays[game_id]
            opening = lineups[game_id]
            if set(opening) != {game["home"], game["away"]}:
                raise ValueError(f"Missing opening dropback lineup for {game_id}.")
            selected = expected | {play for play, _ in opening.values()}
            tracking = archive_tracking(archive, members[game_id], selected)
            if not expected.issubset(tracking):
                raise ValueError(f"Missing tracking frames for scored plays in {game_id}.")
            game_lineups = {}
            for team, (play_id, members_at_snap) in opening.items():
                numbers = tracking[play_id].numbers if play_id in tracking else {}
                game_lineups[team] = dict(playId=play_id, players=[dict(id=nfl_id, slot=slot, number=numbers.get(nfl_id, ""))
                                                                for nfl_id, slot in members_at_snap])
            game_replays = [replay(play_id, plays[game_id, play_id], data.plays[game_id, play_id], tracking[play_id],
                                   data.players, hands, epa[game_id, play_id], routes[game_id, play_id])
                            for play_id in sorted(expected)]
            game.update(playCount=len(game_replays), offenseCounts=dict(Counter(play["offense"] for play in game_replays)))
            save_json(output / "games" / f"{game_id}.json", dict(game, lineups=game_lineups, plays=game_replays))
            index.append(game)
            total_plays += len(game_replays)
            if len(index) == 1 or len(index) % 20 == 0:
                print(f"Exported {len(index)} games ({total_plays:,} replays).", flush=True)

        site_players = player_index(archive_path, data.players, onset, temporal, routes)
        save_json(output / "index.json", dict(games=index, players=site_players, playCount=total_plays))
    print(f"Wrote static data for {len(index)} games and {total_plays:,} replay plays to {output}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "nfl-big-data-bowl-regional-event-data-main.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "data")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--game", type=int, action="append", help="Export selected games for testing.")
    mode.add_argument("--index-only", action="store_true", help="Refresh player grades without regenerating replays.")
    args = parser.parse_args()
    if args.index_only:
        refresh_index(args.archive, args.output)
    else:
        build(args.archive, args.output, args.game or [])


if __name__ == "__main__":
    main()
