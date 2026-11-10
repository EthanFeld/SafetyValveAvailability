"""Collect outlet opportunities during pressure and at an early landmark."""

from dataclasses import dataclass

import numpy as np

from analyze_catch_rate import target_outcomes
from analyze_sva_epa import read_dropbacks, snap_frame
from safety_valve import (DEFAULT_MAX_THROW_DISTANCE, END_EVENTS, FRAME_COLUMNS, PASS_RESULTS,
                          SEPARATION_WEIGHTED_THROW_MODEL, catch_radius, pressure_frame_scores,
                          read_hands, velocity, xy)
from sva_data import open_data
from sva_pressure_context import _nearest_rusher


FRAME_SECONDS = 0.1
OPEN_HALF_LIFE_SECONDS = 0.4
DEFAULT_OPEN_THRESHOLD = 25.0
DEFAULT_LANDMARK_FRAMES = 3
GEOMETRY_FIELDS = (*FRAME_COLUMNS, "projected_depth_yd", "projected_margin_to_sticks_yd", "speed_yd_s",
                   "downfield_speed_yd_s")
FIRST_FIELDS = (*FRAME_COLUMNS, "depth_yd", "margin_yd", "receiver_speed", "downfield_speed")


@dataclass(slots=True)
class TemporalData:
    plays: list
    routes: list


def _terminal_frame(play, frames, snap):
    terminal = next((frame.id for frame in frames if frame.id > snap and frame.event in END_EVENTS), None)
    if terminal is None and play.result in PASS_RESULTS:
        terminal = next((frame.id for frame in frames
                         if frame.id > snap and frame.event == "autoevent_passforward"), None)
    return terminal


def _route_motion(pressure_frames, receiver_ids, qb_id, to_go, to_goal):
    """Use the same projected catch point as the existing onset model."""
    values = []
    for frame in pressure_frames:
        receivers = [frame.positions[receiver_id] for receiver_id in receiver_ids]
        start = xy([frame.positions[qb_id]])[0]
        receiver_xy = xy(receivers)
        motion = velocity(receivers)
        flight = np.minimum(0.15 + np.linalg.norm(receiver_xy - start, axis=1) / 28, 0.8)
        projected_x = receiver_xy[:, 0] + 0.65 * flight * motion[:, 0]
        right = frame.direction == "right"
        scrimmage_x = 110 - to_goal if right else 10 + to_goal
        direction = 1 if right else -1
        depth = direction * (projected_x - scrimmage_x)
        values.append(np.column_stack((depth, depth - to_go, [p.speed for p in receivers], motion[:, 0] * direction)))
    return np.asarray(values)


def summarize_window(frame_ids, geometry, onset_frame, open_threshold=DEFAULT_OPEN_THRESHOLD):
    """Measure opening frequency and timing conditional on the route being in range."""
    in_range = np.isfinite(geometry[:, 0])
    eligible = frame_ids[in_range]
    eligible_geometry = geometry[in_range]
    missing = float("nan")
    first = eligible_geometry[0] if len(eligible) else [missing] * len(GEOMETRY_FIELDS)
    means = np.mean(eligible_geometry, axis=0) if len(eligible) else [missing] * len(GEOMETRY_FIELDS)
    summary = {f"mean_{name}": value for name, value in zip(GEOMETRY_FIELDS, means)}
    summary.update({f"first_{name}": value for name, value in zip(FIRST_FIELDS, first)})
    if not len(eligible):
        summary.update(in_range_frames=0, open_frames=0, open_share=missing, early_weighted_open_share=missing,
                       first_in_range_seconds=missing, first_open_seconds=missing, wait_for_open_seconds=missing,
                       longest_open_seconds=0.0, max_sva=missing)
        return summary

    opened = eligible_geometry[:, 0] >= open_threshold
    open_ids = eligible[opened]
    weights = np.exp2(-(eligible - eligible[0]) * FRAME_SECONDS / OPEN_HALF_LIFE_SECONDS)
    # Missing pressure or out-of-range frames break a streak, even when the route
    # is open on both sides of the gap.
    streak = max(map(len, np.split(open_ids, np.flatnonzero(np.diff(open_ids) != 1) + 1)))
    summary.update(in_range_frames=len(eligible), open_frames=len(open_ids), open_share=float(np.mean(opened)),
                   early_weighted_open_share=float(np.average(opened, weights=weights)),
                   first_in_range_seconds=float((eligible[0] - onset_frame) * FRAME_SECONDS),
                   first_open_seconds=float((open_ids[0] - onset_frame) * FRAME_SECONDS) if len(open_ids) else missing,
                   wait_for_open_seconds=float((open_ids[0] - eligible[0]) * FRAME_SECONDS) if len(open_ids) else missing,
                   longest_open_seconds=streak * FRAME_SECONDS, max_sva=float(np.max(eligible_geometry[:, 0])))
    return summary


def collect_temporal_data(archive, epa, hands_path, open_threshold=DEFAULT_OPEN_THRESHOLD,
                          landmark_frames=DEFAULT_LANDMARK_FRAMES):
    """Return the prior EPA play cohort and every charted receiver route within it."""
    if not np.isfinite(open_threshold) or not 0 <= open_threshold <= 100:
        raise ValueError("Open threshold must be between 0 and 100.")
    if not isinstance(landmark_frames, int) or landmark_frames < 0:
        raise ValueError("Landmark frames must be a nonnegative integer.")
    plays = []
    routes = []
    dropbacks = read_dropbacks(archive, epa)
    outcomes, _ = target_outcomes(archive)
    hands = read_hands(hands_path)
    with open_data(archive) as data:
        for game_id in data.game_ids:
            for play_id, frames in data.tracking(game_id):
                key = game_id, play_id
                state = dropbacks.get(key)
                if state is None or not state["pressured"]:
                    continue
                play = data.plays[key]
                receiver_ids = play.receiver_ids
                radii = np.array([catch_radius(data.players[nfl_id].height_inches) for nfl_id in receiver_ids])
                frame_ids, scores = pressure_frame_scores(play, frames, receiver_ids, radii,
                                                          hands.get(play.qb_id, ""), DEFAULT_MAX_THROW_DISTANCE,
                                                          (SEPARATION_WEIGHTED_THROW_MODEL,))
                if not len(frame_ids):
                    continue

                frames_by_id = {frame.id: frame for frame in frames}
                pressure_frames = [frames_by_id[frame_id] for frame_id in frame_ids]
                onset = pressure_frames[0]
                qb = onset.positions[play.qb_id]
                rushers = [onset.positions[i] for i in play.pressure_ids if i in onset.positions
                           and onset.positions[i].team == play.defense]
                rusher_distance, rusher_closing = _nearest_rusher(qb, rushers)
                snap = snap_frame(frames)
                terminal = _terminal_frame(play, frames, snap)
                landmark = onset.id + landmark_frames
                early = frame_ids <= landmark
                to_go, to_goal = state["situation"][:2]
                motion = _route_motion(pressure_frames, receiver_ids, play.qb_id, to_go, to_goal)
                geometry = np.concatenate((scores[:, 0], motion), axis=2)
                # Motion is recorded for all players, but context summaries are
                # defined on the same eligible frames as the outlet score.
                geometry[~np.isfinite(geometry[:, :, 0])] = np.nan
                play_index = len(plays)
                plays.append(dict(state, onset_frame=onset.id, terminal_frame=terminal, pressure_frames=len(frame_ids),
                                  landmark_frame=landmark,
                                  landmark_eligible=terminal > landmark and landmark in frames_by_id,
                                  early_pressure_frames=int(early.sum()), onset_rusher_distance_yd=rusher_distance,
                                  onset_rusher_closing_yd_s=rusher_closing, onset_credited_rushers=len(rushers),
                                  onset_qb_speed_yd_s=qb.speed, onset_seconds=(onset.id - snap) * FRAME_SECONDS,
                                  route_runners=len(receiver_ids)))
                for i, nfl_id in enumerate(receiver_ids):
                    player = data.players[nfl_id]
                    outcome = outcomes.get((*key, nfl_id))
                    row = dict(play_index=play_index, game_id=game_id, play_id=play_id, nfl_id=nfl_id,
                               player=player.name, position=player.position, caught=outcome[0] if outcome else None,
                               catch_radius_yd=radii[i])
                    row.update({f"onset_{name}": value for name, value in zip(GEOMETRY_FIELDS, geometry[0, i])})
                    for label, ids, values in (("full", frame_ids, geometry[:, i]),
                                               ("early", frame_ids[early], geometry[early, i])):
                        stats = summarize_window(ids, values, onset.id, open_threshold)
                        row.update({f"{label}_{name}": value for name, value in stats.items()})
                    routes.append(row)
    return TemporalData(plays, routes)
