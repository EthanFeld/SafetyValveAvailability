"""Pre-release tracking context for matched targets in the SVA analysis."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from safety_valve import DEFAULT_MAX_THROW_DISTANCE, END_EVENTS, PASS_RESULTS, pressure_frames, velocity, xy
from sva_data import open_data


SECONDS_PER_FRAME = 0.1


@dataclass(frozen=True, slots=True)
class ReleaseContext:
    """Pre-release geometry, with positive closing speed toward the quarterback."""

    rusher_distance_yd: float
    rusher_closing_yd_s: float
    credited_rushers: int
    pressure_seconds: float
    target_distance_yd: float
    target_downfield_yd: float
    projected_target_distance_yd: float
    time_to_throw_s: float
    qb_speed_yd_s: float
    pressure_at_release: bool
    target_in_range_at_release: bool
    release_frame_id: int


def _snap_and_release(play, frames):
    snap = next((frame.id for frame in frames if frame.event == "ball_snap"), None)
    if snap is None:
        snap = next((frame.id for frame in frames if frame.event == "autoevent_ballsnap"), None)
    if snap is None:
        return None

    terminal = next((frame for frame in frames if frame.id > snap and frame.event in END_EVENTS), None)
    if terminal is None and play.result in PASS_RESULTS:
        terminal = next((frame for frame in frames
                         if frame.id > snap and frame.event == "autoevent_passforward"), None)
    if terminal is None or terminal.event not in {"pass_forward", "autoevent_passforward"}:
        return None
    return snap, terminal.id


def _nearest_rusher(qb, rushers):
    offsets = xy(rushers) - xy([qb])
    distances = np.linalg.norm(offsets, axis=1)
    nearest = np.argmin(distances)
    relative_velocity = velocity([rushers[nearest]])[0] - velocity([qb])[0]
    distance = float(distances[nearest])
    closing = -float(offsets[nearest] @ relative_velocity) / max(distance, 0.01)
    return distance, closing


def _target_distances(qb, target):
    offset = xy([target])[0] - xy([qb])[0]
    distance = float(np.linalg.norm(offset))
    # Use the same short-flight projection as the SVA range gate.
    flight_time = min(0.15 + distance / 28, 0.8)
    projected_distance = float(np.linalg.norm(offset + 0.65 * flight_time * velocity([target])[0]))
    return distance, projected_distance


def contexts_for_play(play, frames, target_ids, max_throw_distance, pressure_ids=None):
    events = _snap_and_release(play, frames)
    if events is None:
        return
    snap, release = events
    before = next((frame for frame in frames if frame.id == release - 1 and frame.id > snap), None)
    if before is None or before.direction not in {"left", "right"}:
        return

    positions = before.positions
    qb = positions.get(play.qb_id)
    rushers = [positions[i] for i in play.pressure_ids if i in positions and positions[i].team == play.defense]
    if qb is None or qb.team != play.offense or not rushers:
        return

    if pressure_ids is None:
        pressure_ids = {frame.id for frame, _, _ in pressure_frames(play, frames)}
    rusher_distance, rusher_closing = _nearest_rusher(qb, rushers)
    for target_id in target_ids:
        target = positions.get(target_id)
        if target is None or target.team != play.offense:
            continue
        target_distance, projected_distance = _target_distances(qb, target)
        yield target_id, ReleaseContext(
            rusher_distance_yd=rusher_distance,
            rusher_closing_yd_s=rusher_closing,
            credited_rushers=len(rushers),
            pressure_seconds=len(pressure_ids) * SECONDS_PER_FRAME,
            target_distance_yd=target_distance,
            target_downfield_yd=(target.x - qb.x) * (1 if before.direction == "right" else -1),
            projected_target_distance_yd=projected_distance,
            time_to_throw_s=(release - snap) * SECONDS_PER_FRAME,
            qb_speed_yd_s=qb.speed,
            pressure_at_release=before.id in pressure_ids,
            target_in_range_at_release=projected_distance <= max_throw_distance,
            release_frame_id=release,
        )


def release_contexts(archive_path, target_keys, max_throw_distance=DEFAULT_MAX_THROW_DISTANCE):
    """Map (game ID, play ID, receiver ID) to context, omitting missing release windows."""
    targets = defaultdict(lambda: defaultdict(set))
    for game_id, play_id, target_id in target_keys:
        targets[game_id][play_id].add(target_id)

    contexts = {}
    with open_data(archive_path) as data:
        for game_id in data.game_ids:
            if game_id not in targets:
                continue
            for play_id, frames in data.tracking(game_id):
                if play_id not in targets[game_id]:
                    continue
                play = data.plays[(game_id, play_id)]
                for target_id, context in contexts_for_play(play, frames, targets[game_id][play_id],
                                                            max_throw_distance):
                    contexts[(game_id, play_id, target_id)] = context
    return contexts
