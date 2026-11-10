"""Build outlet and play features at the first qualifying pressure frame."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from analyze_catch_rate import category_controls, target_outcomes
from analyze_sva_epa import read_dropbacks, snap_frame
from safety_valve import (DEFAULT_MAX_THROW_DISTANCE, SEPARATION_WEIGHTED_THROW_MODEL, catch_radius,
                          pressure_frames, read_hands, score_frame_models, velocity, xy)
from sva_data import open_data
from sva_pressure_context import _nearest_rusher


@dataclass(slots=True)
class JointData:
    plays: list
    routes: list
    play_context: np.ndarray
    play_geometry: np.ndarray
    play_sticks: np.ndarray
    route_geometry: np.ndarray
    route_sticks: np.ndarray
    route_play_rows: np.ndarray
    target_route_rows: np.ndarray
    target_caught: np.ndarray


def _outlet_rows(frame, qb, defenders, play, players, hand):
    receivers = [frame.positions[nfl_id] for nfl_id in play.receiver_ids]
    radii = np.array([catch_radius(players[nfl_id].height_inches) for nfl_id in play.receiver_ids])
    teammate_ids = [i for i, item in frame.positions.items() if item.team == play.offense and i != play.qb_id]
    teammates = [frame.positions[i] for i in teammate_ids]
    mask = np.asarray(play.receiver_ids)[:, None] == np.asarray(teammate_ids)
    scores = score_frame_models(qb, receivers, radii, defenders, teammates, mask, frame.direction, hand,
                                DEFAULT_MAX_THROW_DISTANCE, (SEPARATION_WEIGHTED_THROW_MODEL,))[0]
    start = xy([qb])[0]
    xy_receiver = xy(receivers)
    receiver_velocity = velocity(receivers)
    flight = np.minimum(0.15 + np.linalg.norm(xy_receiver - start, axis=1) / 28, 0.8)
    projected = xy_receiver + 0.65 * flight[:, None] * receiver_velocity
    return scores, radii, projected, receiver_velocity


def _collect(archive, epa, hands):
    plays = []
    routes = []
    dropbacks = read_dropbacks(archive, epa)
    outcomes, _ = target_outcomes(archive)
    with open_data(archive) as data:
        for game_id in data.game_ids:
            for play_id, frames in data.tracking(game_id):
                key = game_id, play_id
                if key not in dropbacks or not dropbacks[key]["pressured"]:
                    continue
                play = data.plays[key]
                onset = next(pressure_frames(play, frames), None)
                if onset is None:
                    continue
                frame, qb, defenders = onset
                state = dropbacks[key]
                to_go, to_goal, _, _, down, _, _, _ = state["situation"]
                right = frame.direction == "right"
                scrimmage_x = 110 - to_goal if right else 10 + to_goal
                sign = 1 if right else -1
                rushers = [frame.positions[i] for i in play.pressure_ids if i in frame.positions
                           and frame.positions[i].team == play.defense]
                distance, closing = _nearest_rusher(qb, rushers)
                scores, radii, projected, motion = _outlet_rows(frame, qb, defenders, play, data.players,
                                                                  hands.get(play.qb_id, ""))
                play_index = len(plays)
                plays.append(dict(state, onset_frame=frame.id, rush_distance=distance, rush_closing=closing,
                                  rushers=len(rushers), qb_speed=qb.speed, onset_seconds=(frame.id - snap_frame(frames)) / 10,
                                  route_runners=len(play.receiver_ids)))
                for i, nfl_id in enumerate(play.receiver_ids):
                    if not np.isfinite(scores[i, 0]):
                        continue
                    player = data.players[nfl_id]
                    score, distance_yd, separation, lane, hand_ease = scores[i]
                    depth = (projected[i, 0] - scrimmage_x) * sign
                    margin = depth - to_go
                    # Positive separation is still useful after the original sigmoid saturates.
                    geometry = (score / 10, distance_yd / 10, (distance_yd / 10) ** 2, np.log1p(separation),
                                lane, hand_ease, radii[i], frame.positions[nfl_id].speed / 5)
                    sticks = (depth / 10, margin / 10, float(margin >= 0),
                              max(-margin, 0) * (down in {"3", "4"}) / 10, motion[i, 0] * sign / 5)
                    outcome = outcomes.get((*key, nfl_id))
                    routes.append(dict(play_index=play_index, game_id=game_id, play_id=play_id, nfl_id=nfl_id,
                                       player=player.name, position=player.position, revised_sva=score,
                                       projected_depth=depth, projected_margin=margin, geometry=geometry,
                                       sticks=sticks, caught=outcome[0] if outcome else None))
    return plays, routes


def _context(plays):
    to_go, to_goal, margin, action, down, quarter, coverage, _ = zip(*(row["situation"] for row in plays))
    to_go = np.log1p(to_go)
    field = np.asarray(to_goal) / 50
    return np.column_stack((to_go, to_go * np.isin(down, ("3", "4")), field, field ** 2,
                            np.asarray(margin) / 14, action, category_controls(down), category_controls(quarter),
                            category_controls(coverage), [row["clock"] / 900 for row in plays],
                            [row["rush_distance"] for row in plays], [row["rush_closing"] for row in plays],
                            [row["rushers"] for row in plays], [row["qb_speed"] for row in plays],
                            [row["onset_seconds"] for row in plays], [row["route_runners"] for row in plays]))


def _summaries(plays, routes):
    by_play = defaultdict(list)
    for row in routes:
        by_play[row["play_index"]].append(row)
    geometry = np.zeros((len(plays), 6))
    sticks = np.zeros((len(plays), 4))
    for i, outlets in by_play.items():
        top = sorted(outlets, key=lambda row: -row["revised_sva"])
        geometry[i] = (top[0]["revised_sva"] / 10, len(top),
                       np.mean([row["revised_sva"] / 10 for row in top[:2]]),
                       max(row["geometry"][3] for row in top), min(row["geometry"][1] for row in top),
                       max(row["geometry"][4] for row in top))
        conversions = [row for row in top if row["projected_margin"] >= 0]
        sticks[i] = (len(conversions), max((row["revised_sva"] / 10 for row in conversions), default=0),
                     max(row["projected_margin"] / 10 for row in top),
                     max(row["revised_sva"] / 100 * max(row["projected_depth"], 0) / 10 for row in top))
    return geometry, sticks


def collect_joint_data(archive, epa, hands_path):
    plays, routes = _collect(archive, epa, read_hands(hands_path))
    if not plays or not routes:
        raise ValueError("No usable pressure windows and in-range outlets.")
    route_play_rows = np.array([row["play_index"] for row in routes])
    context = _context(plays)
    play_geometry, play_sticks = _summaries(plays, routes)
    geometry = np.array([row["geometry"] for row in routes])
    sticks = np.column_stack((geometry, [row["sticks"] for row in routes], context[route_play_rows],
                             category_controls([row["position"] for row in routes])))
    target_rows = np.array([i for i, row in enumerate(routes) if row["caught"] is not None], dtype=int)
    target_caught = np.array([routes[i]["caught"] for i in target_rows], dtype=int)
    return JointData(plays, routes, context, play_geometry, play_sticks, geometry, sticks,
                     route_play_rows, target_rows, target_caught)
