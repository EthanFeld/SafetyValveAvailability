"""Score the outlets available to a pressured quarterback from NFL tracking."""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sva_data import open_data


ROOT = Path(__file__).resolve().parent
FIELD_WIDTH = 160 / 3
DEFAULT_MAX_THROW_DISTANCE = 20.0
END_EVENTS = {"pass_forward", "qb_sack", "qb_strip_sack", "run"}
PASS_RESULTS = {"C", "I", "IN"}
ANGLES = np.arange(8) * np.pi / 4
AIM_DIRECTIONS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))
OFFSET_RINGS = np.array([0.25, 0.50, 0.75, 1.00])
AIM_OFFSETS = np.vstack(([0, 0], (OFFSET_RINGS[:, None, None] * AIM_DIRECTIONS).reshape(-1, 2)))
OFFSET_DISTANCE = np.concatenate(([0], np.repeat(OFFSET_RINGS, len(ANGLES))))
FRAME_COLUMNS = ("sva", "distance_yd", "separation_yd", "lane_ease", "hand_ease")
PLAY_COLUMNS = (
    "game_id", "play_id", "nfl_id", "player", "position", "qb_id", "qb_hand",
    "pressure_frames", "in_range_frames", "catch_radius_yd", *FRAME_COLUMNS,
)
LEADERBOARD_COLUMNS = (
    "rank", "nfl_id", "player", "position", "sva", "in_range_routes", "pressured_routes",
    "in_range_frames", "pressure_frames", "known_hand_routes", "catch_radius_yd", *FRAME_COLUMNS[1:],
)


@dataclass(frozen=True, slots=True)
class ThrowModel:
    separation_midpoint: float = 1.25
    separation_width: float = 0.65
    defender_lane_penalty: float = 0.90
    teammate_lane_penalty: float = 0.20
    distance_scale: float = 16.0
    hand_penalty: float = 0.22
    stretch_penalty: float = 0.12


BASELINE_THROW_MODEL = ThrowModel()
SEPARATION_WEIGHTED_THROW_MODEL = ThrowModel(separation_midpoint=2.35, separation_width=1.10,
                                             defender_lane_penalty=0.50, teammate_lane_penalty=0.10)
THROW_MODELS = {"baseline": BASELINE_THROW_MODEL, "separation-weighted": SEPARATION_WEIGHTED_THROW_MODEL}


def catch_radius(height_inches):
    """Estimated horizontal placement tolerance, in yards."""
    height = height_inches if height_inches is not None else 72
    return float(np.clip(0.75 + 0.025 * (height - 72), 0.55, 1.05))


def velocity(players):
    # In tracking, 0 degrees points toward increasing y and 90 toward increasing x.
    angles = np.radians(np.asarray([player.direction for player in players], dtype=float))
    speed = np.asarray([player.speed for player in players], dtype=float)
    return np.nan_to_num(np.column_stack((np.sin(angles), np.cos(angles))) * speed[:, None])


def xy(players):
    return np.array([(player.x, player.y) for player in players], dtype=float).reshape(-1, 2)


def under_pressure(qb, rushers):
    if not rushers:
        return False
    delta = xy(rushers) - xy([qb])
    distance = np.linalg.norm(delta, axis=1)
    relative_velocity = velocity(rushers) - velocity([qb])
    closing = -np.sum(delta * relative_velocity, axis=1) / np.maximum(distance, 0.01)
    # A nearby rusher can affect the throw even when tracking shows little closing speed.
    return bool(np.any((distance <= 2.5) | ((distance <= 4) & (distance - 0.30 * closing <= 2.5))))


def throwing_hand_ease(throw, distance, direction, hand, penalty=0.22):
    # Increasing y is to the passer's left when the offense is moving toward +x.
    rightward = -throw[..., 1] / np.maximum(distance, 0.01)
    if direction == "left":
        rightward = -rightward
    if hand == "R":
        return 1 - penalty * np.maximum(-rightward, 0)
    if hand == "L":
        return 1 - penalty * np.maximum(rightward, 0)
    return 1 - (penalty / 2) * np.abs(rightward)


def lane_risk(start, throw, flight_time, players_xy, players_velocity, width, excluded=None):
    """Largest corridor intrusion from one group, for each receiver and aim point."""
    if not len(players_xy):
        return np.zeros(throw.shape[:2])

    # Estimate when the ball crosses each player's current position along the lane.
    # Shapes are receiver x aim point x obstacle, with a last axis for x/y vectors.
    ray = throw[:, :, None, :]
    length_squared = np.maximum(np.sum(ray ** 2, axis=-1), 0.01)
    relative_xy = players_xy - start
    initial_fraction = np.sum(relative_xy * ray, axis=-1) / length_squared
    crossing_time = flight_time[:, None, None] * np.clip(initial_fraction, 0, 1)
    obstacle = relative_xy + crossing_time[..., None] * players_velocity
    fraction = np.sum(obstacle * ray, axis=-1) / length_squared
    perpendicular = np.linalg.norm(obstacle - fraction[..., None] * ray, axis=-1)
    length = np.sqrt(length_squared)
    between = (fraction * length >= 0.65) & ((1 - fraction) * length >= 0.75)
    if excluded is not None:
        between &= ~excluded[:, None, :]
    return np.max(np.where(between, np.exp(-0.5 * (perpendicular / width) ** 2), 0), axis=-1)


def score_frame(qb, receivers, radii, defenders, teammates, receiver_mask, direction, hand,
                max_throw_distance=DEFAULT_MAX_THROW_DISTANCE):
    """Return the best throw for each in-range receiver, with NaNs for receivers outside range."""
    return score_frame_models(qb, receivers, radii, defenders, teammates, receiver_mask, direction, hand,
                              max_throw_distance, (BASELINE_THROW_MODEL,))[0]


def score_frame_models(qb, receivers, radii, defenders, teammates, receiver_mask, direction, hand,
                       max_throw_distance, models):
    """Score several weightings on the same geometry, returning model x receiver x component."""
    start = xy([qb])[0]
    receivers_xy = xy(receivers)
    flight_time = np.minimum(0.15 + np.linalg.norm(receivers_xy - start, axis=1) / 28, 0.8)
    # Dampen constant-velocity forecasts over the short time to an outlet catch.
    center = receivers_xy + 0.65 * flight_time[:, None] * velocity(receivers)
    # Eligibility depends on the route location, independent of reach or coverage.
    in_range = np.linalg.norm(center - start, axis=1) <= max_throw_distance
    result = np.full((len(models), len(receivers), len(FRAME_COLUMNS)), np.nan)
    if not np.any(in_range):
        return result

    center = center[in_range]
    radii = radii[in_range]
    flight_time = flight_time[in_range]
    defenders_xy = xy(defenders)
    defender_velocity = 0.65 * velocity(defenders)
    aim = center[:, None, :] + AIM_OFFSETS
    throw = aim - start
    distance = np.linalg.norm(throw, axis=-1)
    inbounds = ((0 <= aim[..., 0]) & (aim[..., 0] <= 120)
                & (0 <= aim[..., 1]) & (aim[..., 1] <= FIELD_WIDTH))
    stretch = OFFSET_DISTANCE / radii[:, None]
    catchable = inbounds & (stretch <= 1) & (distance <= max_throw_distance)

    defenders_at_catch = defenders_xy + flight_time[:, None, None] * defender_velocity
    separation = np.min(np.linalg.norm(aim[:, :, None, :] - defenders_at_catch[:, None, :, :],
                                       axis=-1), axis=-1)
    defender_risk = lane_risk(start, throw, flight_time, defenders_xy, defender_velocity, 0.9)
    # A route runner is an obstacle only for the other receivers' throws.
    teammate_risk = lane_risk(start, throw, flight_time, xy(teammates),
                              0.65 * velocity(teammates), 0.6, excluded=receiver_mask[in_range])
    for i, model in enumerate(models):
        margin = separation - model.separation_midpoint - 0.8 * flight_time[:, None]
        open_window = 1 / (1 + np.exp(-margin / model.separation_width))
        lane = ((1 - model.defender_lane_penalty * defender_risk)
                * (1 - model.teammate_lane_penalty * teammate_risk))
        hand_ease = throwing_hand_ease(throw, distance, direction, hand, model.hand_penalty)
        distance_ease = 1 / (1 + (distance / model.distance_scale) ** 2)
        placement_ease = 1 - model.stretch_penalty * stretch ** 2
        scores = np.where(catchable, 100 * open_window * lane * hand_ease * distance_ease * placement_ease, 0)
        best = (np.arange(len(center)), np.argmax(scores, axis=1))
        result[i, in_range] = np.column_stack([
            values[best] for values in (scores, distance, separation, lane, hand_ease)
        ])
    return result


def pressure_frames(play, frames):
    snap = next((frame.id for frame in frames if frame.event == "ball_snap"), None)
    if snap is None:
        snap = next((frame.id for frame in frames if frame.event == "autoevent_ballsnap"), None)
    if snap is None:
        return
    stop = next((frame.id for frame in frames if frame.id > snap and frame.event in END_EVENTS), None)
    if stop is None and play.result in PASS_RESULTS:
        stop = next((frame.id for frame in frames
                     if frame.id > snap and frame.event == "autoevent_passforward"), None)
    if stop is None:
        return

    for frame in frames:
        if not snap < frame.id < stop or frame.direction not in {"left", "right"}:
            continue
        positions = frame.positions
        qb = positions.get(play.qb_id)
        if qb is None or qb.team != play.offense:
            continue
        offense = [player for player in positions.values() if player.team == play.offense]
        if len(offense) < 11 or not all(i in positions and positions[i].team == play.offense
                                       for i in play.receiver_ids):
            continue
        defenders = [player for player in positions.values() if player.team == play.defense]
        if len(defenders) < 11:
            continue
        rushers = [positions[i] for i in play.pressure_ids
                   if i in positions and positions[i].team == play.defense]
        if under_pressure(qb, rushers):
            yield frame, qb, defenders


def pressure_frame_scores(play, frames, receiver_ids, radii, hand, max_throw_distance, models):
    frame_ids = []
    scores = []
    for frame, qb, defenders in pressure_frames(play, frames):
        receivers = [frame.positions[i] for i in receiver_ids]
        teammate_ids = [i for i, player in frame.positions.items()
                        if player.team == play.offense and i != play.qb_id]
        teammates = [frame.positions[i] for i in teammate_ids]
        receiver_mask = np.asarray(receiver_ids)[:, None] == np.asarray(teammate_ids)
        frame_ids.append(frame.id)
        scores.append(score_frame_models(qb, receivers, radii, defenders, teammates, receiver_mask,
                                         frame.direction, hand, max_throw_distance, models))
    return np.array(frame_ids, dtype=int), np.asarray(scores)


def score_play(game_id, play_id, play, frames, players, hand, max_throw_distance=DEFAULT_MAX_THROW_DISTANCE,
               throw_model=BASELINE_THROW_MODEL):
    radii = np.array([catch_radius(players[i].height_inches) for i in play.receiver_ids])
    frame_ids, scores = pressure_frame_scores(play, frames, play.receiver_ids, radii, hand, max_throw_distance,
                                              (throw_model,))
    if not len(frame_ids):
        return

    for i, nfl_id in enumerate(play.receiver_ids):
        route_frames = scores[:, 0, i, :]
        in_range = route_frames[~np.isnan(route_frames[:, 0])]
        means = in_range.mean(axis=0) if len(in_range) else [None] * len(FRAME_COLUMNS)
        yield dict(game_id=game_id, play_id=play_id, nfl_id=nfl_id, player=players[nfl_id].name,
                   position=players[nfl_id].position, qb_id=play.qb_id, qb_hand=hand,
                   pressure_frames=len(scores), in_range_frames=len(in_range), catch_radius_yd=radii[i],
                   **dict(zip(FRAME_COLUMNS, means)))


def leaderboard(route_rows, min_routes):
    by_player = defaultdict(list)
    for row in route_rows:
        by_player[row["nfl_id"]].append(row)

    rows = []
    for nfl_id, routes in by_player.items():
        # Each route with at least one in-range pressure frame counts once.
        scored = [route for route in routes if route["in_range_frames"]]
        if len(scored) < min_routes:
            continue
        rows.append(dict(
            nfl_id=nfl_id, player=routes[0]["player"], position=routes[0]["position"],
            in_range_routes=len(scored), pressured_routes=len(routes),
            in_range_frames=sum(r["in_range_frames"] for r in scored),
            pressure_frames=sum(r["pressure_frames"] for r in routes),
            known_hand_routes=sum(r["qb_hand"] in {"L", "R"} for r in scored),
            catch_radius_yd=routes[0]["catch_radius_yd"],
            **{column: np.mean([r[column] for r in scored]) for column in FRAME_COLUMNS},
        ))
    rows.sort(key=lambda row: (-row["sva"], -row["in_range_routes"], row["nfl_id"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def read_hands(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {int(row["nflId"]): row["throwingHand"].strip().upper()
                for row in csv.DictReader(file) if row["throwingHand"].strip().upper() in {"L", "R"}}


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: f"{v:.3f}" if isinstance(v, (float, np.floating)) else v
                          for k, v in row.items()} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "nfl-big-data-bowl-regional-event-data-main.zip")
    parser.add_argument("--hands", type=Path, default=ROOT / "qb_handedness.csv")
    parser.add_argument("--game", type=int, action="append", help="Only score this game ID (repeatable).")
    parser.add_argument("--min-routes", type=int, default=30, help="Minimum in-range routes to rank (default 30).")
    parser.add_argument("--max-throw-distance", type=float, default=DEFAULT_MAX_THROW_DISTANCE,
                        help="Maximum QB-to-catch distance in yards (default 20).")
    parser.add_argument("--throw-model", choices=THROW_MODELS, default="baseline",
                        help="Use the original or the separation-weighted throw scoring model.")
    parser.add_argument("--output", type=Path, help="Leaderboard CSV (defaults to a name for the chosen model).")
    parser.add_argument("--play-output", type=Path, help="Optional receiver-by-play score breakdown.")
    args = parser.parse_args()
    if args.min_routes < 1:
        parser.error("--min-routes must be at least 1")
    if not np.isfinite(args.max_throw_distance) or args.max_throw_distance <= 0:
        parser.error("--max-throw-distance must be a positive, finite number")
    if args.output is None:
        filename = ("safety_valve_leaderboard.csv" if args.throw_model == "baseline"
                    else "safety_valve_separation_weighted_leaderboard.csv")
        args.output = ROOT / filename
    output_paths = [path.resolve() for path in (args.output, args.play_output) if path is not None]
    if len(set(output_paths)) != len(output_paths):
        parser.error("--output and --play-output must be different files")
    if set(output_paths).intersection({args.archive.resolve(), args.hands.resolve()}):
        parser.error("output files must be different from the archive and handedness inputs")

    hands = read_hands(args.hands)
    throw_model = THROW_MODELS[args.throw_model]
    route_rows = []
    tracked_plays = 0
    with open_data(args.archive) as data:
        game_ids = sorted(set(args.game or data.game_ids))
        unknown = set(game_ids).difference(data.game_ids)
        if unknown:
            parser.error(f"unknown game IDs: {', '.join(map(str, sorted(unknown)))}")
        selected_games = set(game_ids)
        eligible_plays = sum(game_id in selected_games for game_id, _ in data.plays)
        for game_id in game_ids:
            for play_id, frames in data.tracking(game_id):
                play = data.plays[(game_id, play_id)]
                rows = list(score_play(game_id, play_id, play, frames, data.players,
                                       hands.get(play.qb_id, ""), args.max_throw_distance, throw_model))
                tracked_plays += bool(rows)
                route_rows.extend(rows)

    rows = leaderboard(route_rows, args.min_routes)
    write_csv(args.output, LEADERBOARD_COLUMNS, rows)
    if args.play_output:
        write_csv(args.play_output, PLAY_COLUMNS, route_rows)
    scored_routes = [row for row in route_rows if row["in_range_frames"]]
    known_hand = sum(row["qb_hand"] in {"L", "R"} for row in scored_routes)
    print(f"{len(game_ids)} games, usable pressure frames on {tracked_plays:,}/{eligible_plays:,} plays, "
          f"{len(scored_routes):,}/{len(route_rows):,} in-range routes ({known_hand:,} with known QB hand), "
          f"{len(rows)} ranked players.")


if __name__ == "__main__":
    main()
