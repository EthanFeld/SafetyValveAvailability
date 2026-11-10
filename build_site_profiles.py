"""Export the leaderboard score breakdowns and pressure openings for the Film Room."""

import argparse
from collections import defaultdict
from math import tanh
from pathlib import Path
from statistics import fmean, median

from analyze_catch_rate import pre_snap_context
from build_site_data import save_json, scored_key
from build_site_leaderboard import MIN_ROUTES, load_json
from rank_context_sva import DEFAULT_SCORE_SPREAD, benchmark_forecasts, score_players
from rank_joint_sva import read_csv
from safety_valve import ROOT


SITE_DATA = ROOT / "docs" / "data"


def route_key(row):
    return str(row["game_id"]), int(row["play_id"]), int(row["nfl_id"])


def onset_components(archive_path):
    outlets = list(read_csv(ROOT / "sva_joint_outlets.csv"))
    plays = list(read_csv(ROOT / "sva_joint_plays.csv"))
    pressure = {scored_key(row): row for row in read_csv(ROOT / "sva_epa_plays.csv")}
    situations = dict(pre_snap_context(archive_path, {scored_key(row) for row in outlets}))
    catch, epa = benchmark_forecasts(outlets, plays, situations, pressure)
    return {row["nfl_id"]: row for row in score_players(outlets, [], MIN_ROUTES, catch, epa, 0.5)}


def opened_replays(games, temporal_routes):
    by_player = defaultdict(list)
    expected = {key for key, row in temporal_routes.items() if int(row["full_open_frames"]) > 0}
    seen = set()
    for game in games:
        game_id = str(game["id"])
        shard = load_json(SITE_DATA / "games" / f"{game_id}.json")
        for play in shard["plays"]:
            pressure = [frame for frame in play["frames"] if frame["pressure"]]
            if not pressure:
                continue
            openings = defaultdict(list)
            for frame in pressure:
                for player_id in frame["open"]:
                    key = game_id, play["id"], player_id
                    if key in temporal_routes:
                        openings[player_id].append(frame["t"])
            for player_id, frames in openings.items():
                key = game_id, play["id"], player_id
                route = temporal_routes[key]
                open_count = len(frames)
                first_open = round(frames[0] - pressure[0]["t"], 1)
                if (key in seen or open_count != int(route["full_open_frames"])
                        or first_open != float(route["full_first_open_seconds"])):
                    raise ValueError(f"Replay openings do not match temporal metrics for {key}")
                seen.add(key)
                by_player[player_id].append(dict(
                    gameId=game_id, playId=play["id"], team=play["offense"], opponent=play["defense"],
                    week=game["week"], quarter=play["quarter"], clock=play["clock"], down=play["down"], toGo=play["toGo"],
                    description=play["description"], epa=play["epa"], firstOpen=first_open,
                    openShare=float(route["full_open_share"]), openFrames=open_count,
                    longestOpen=float(route["full_longest_open_seconds"]),
                ))
    if seen != expected:
        raise ValueError("Replay openings and scored openings do not cover the same routes")
    return by_player


def temporal_components(player, routes, published):
    if player["temporal"] is None:
        return dict(temporalCatchLiftPp=None, temporalEpaLift=None, temporalAvailabilityLiftPp=None,
                    openShare=None, weightedOpenShare=None, openedRouteShare=None, medianFirstOpen=None, longestOpen=None)
    if len(routes) != player["temporalRoutes"]:
        raise ValueError(f"Incorrect temporal route count for {player['id']}")
    if published:
        grade = float(published["temporal_sva"])
        catch_lift = float(published["mean_catch_above_context_pp"])
        epa_lift = float(published["mean_epa_above_context"])
        availability_lift = float(published["mean_open_share_above_context"])
    else:
        base = fmean(float(row["base_temporal_sva"]) for row in routes)
        grade = 50 + 50 * tanh(DEFAULT_SCORE_SPREAD * (base - 50) / 50)
        catch_lift = fmean(float(row["catch_above_context_pp"]) for row in routes)
        epa_lift = fmean(float(row["epa_above_context_given_catch"]) for row in routes)
        availability_lift = fmean(float(row["open_share_above_context"]) for row in routes)
    if abs(grade - player["temporal"]) > .00051:
        raise ValueError(f"Incorrect temporal grade for {player['id']}")
    opened = [float(row["full_first_open_seconds"]) for row in routes if row["full_first_open_seconds"]]
    return dict(
        temporalCatchLiftPp=round(catch_lift, 4), temporalEpaLift=round(epa_lift, 5),
        temporalAvailabilityLiftPp=round(100 * availability_lift, 4),
        openShare=round(fmean(float(row["full_open_share"]) for row in routes), 5),
        weightedOpenShare=round(fmean(float(row["full_early_weighted_open_share"]) for row in routes), 5),
        openedRouteShare=round(len(opened) / len(routes), 5), medianFirstOpen=round(median(opened), 2) if opened else None,
        longestOpen=max(float(row["full_longest_open_seconds"]) for row in routes),
    )


def build(archive_path):
    leaderboard = load_json(SITE_DATA / "leaderboard.json")["players"]
    ids = {player["id"] for player in leaderboard}
    games = load_json(SITE_DATA / "index.json")["games"]
    onset = onset_components(archive_path)
    temporal_routes = {}
    by_player = defaultdict(list)
    for row in read_csv(ROOT / "sva_temporal_routes.csv"):
        key = route_key(row)
        player_id = key[2]
        if player_id not in ids:
            continue
        if key in temporal_routes:
            raise ValueError(f"Duplicate temporal route {key}")
        temporal_routes[key] = row
        by_player[player_id].append(row)
    published = {int(row["nfl_id"]): row for row in read_csv(ROOT / "sva_temporal_leaderboard.csv")}
    replays = opened_replays(games, temporal_routes)
    profiles = {}
    for player in leaderboard:
        player_id = player["id"]
        start = onset.get(player_id)
        if player["onset"] is not None and (start is None or start["in_range_onset_routes"] != player["onsetRoutes"]
                                             or abs(start["context_sva"] - player["onset"]) > .00051):
            raise ValueError(f"Incorrect onset grade or route count for {player_id}")
        profiles[player_id] = dict(
            player,
            onsetCatchLiftPp=round(start["catch_above_context_pp"], 4) if start else None,
            onsetEpaLift=round(start["epa_above_context_given_catch"], 5) if start else None,
            **temporal_components(player, by_player[player_id], published.get(player_id)),
            plays=replays[player_id],
        )
    return {"players": profiles}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "nfl-big-data-bowl-regional-event-data-main.zip")
    parser.add_argument("--output", type=Path, default=SITE_DATA / "player-profiles.json")
    args = parser.parse_args()
    data = build(args.archive)
    save_json(args.output, data)
    count = sum(len(row["plays"]) for row in data["players"].values())
    print(f"Wrote {len(data['players'])} profiles and {count:,} open-pressure replay links to {args.output}.")


if __name__ == "__main__":
    main()
