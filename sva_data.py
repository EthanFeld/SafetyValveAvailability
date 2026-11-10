"""Read the Big Data Bowl regional data archive for Safety Valve Availability."""

import csv
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import PurePosixPath
from zipfile import ZipFile


RECEIVER_POSITIONS = {"WR", "TE", "RB", "FB"}
PRESSURE_FIELDS = ("pff_hit", "pff_hurry", "pff_sack")
MISSING = {"", "NA"}


@dataclass(frozen=True, slots=True)
class Player:
    name: str
    position: str
    height_inches: int | None


@dataclass(frozen=True, slots=True)
class Play:
    offense: str
    defense: str
    result: str
    qb_id: int
    receiver_ids: tuple[int, ...]
    pressure_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Position:
    x: float
    y: float
    speed: float
    direction: float | None
    team: str


@dataclass(slots=True)
class Frame:
    id: int
    event: str
    direction: str
    positions: dict[int, Position]


def _height_inches(height: str) -> int | None:
    feet, _, inches = height.partition("-")
    if feet.isdecimal() and inches.isdecimal():
        return int(feet) * 12 + int(inches)
    return None


def _optional_float(value: str) -> float | None:
    return None if value in MISSING else float(value)


class Dataset:
    """Pressure-play metadata and one game's tracking at a time."""

    def __init__(self, archive: ZipFile):
        self._archive = archive
        metadata = {}
        self._tracking_members = {}
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.name in {"players.csv", "plays.csv", "pffScoutingData.csv"}:
                metadata[path.name] = name
            elif path.name.startswith("tracking_") and path.suffix == ".csv":
                self._tracking_members[int(path.stem.removeprefix("tracking_"))] = name
        self.game_ids = sorted(self._tracking_members)
        self.players = self._load_players(metadata["players.csv"])
        self.plays = self._load_plays(metadata["plays.csv"], metadata["pffScoutingData.csv"])
        self._pressure_play_ids = defaultdict(set)
        for game_id, play_id in self.plays:
            self._pressure_play_ids[game_id].add(str(play_id))

    def _open_member(self, name):
        return TextIOWrapper(self._archive.open(name), encoding="utf-8-sig", newline="")

    def _load_players(self, name):
        with self._open_member(name) as source:
            return {
                int(row["nflId"]): Player(
                    row["displayName"], row["officialPosition"], _height_inches(row["height"])
                )
                for row in csv.DictReader(source)
            }

    def _load_plays(self, plays_name, scouting_name):
        quarterbacks = defaultdict(set)
        receivers = defaultdict(set)
        pressure_players = defaultdict(set)
        with self._open_member(scouting_name) as source:
            for row in csv.DictReader(source):
                key = (int(row["gameId"]), int(row["playId"]))
                player_id = int(row["nflId"])
                role = row["pff_role"]
                player = self.players.get(player_id)
                if role == "Pass" and player and player.position == "QB":
                    quarterbacks[key].add(player_id)
                if role == "Pass Route" and player and player.position in RECEIVER_POSITIONS:
                    receivers[key].add(player_id)
                if any(row[field] == "1" for field in PRESSURE_FIELDS):
                    pressure_players[key].add(player_id)

        plays = {}
        with self._open_member(plays_name) as source:
            for row in csv.DictReader(source):
                key = (int(row["gameId"]), int(row["playId"]))
                if key not in pressure_players or len(quarterbacks[key]) != 1 or not receivers[key]:
                    continue
                plays[key] = Play(
                    row["possessionTeam"],
                    row["defensiveTeam"],
                    row["passResult"],
                    next(iter(quarterbacks[key])),
                    tuple(sorted(receivers[key])),
                    tuple(sorted(pressure_players[key])),
                )
        return plays

    def tracking(self, game_id):
        """Yield (play ID, ordered frames) for qualifying pressure plays in a game."""
        pressure_play_ids = self._pressure_play_ids[game_id]
        if not pressure_play_ids:
            return

        frames_by_play = defaultdict(dict)
        with self._open_member(self._tracking_members[game_id]) as source:
            rows = csv.reader(source)
            header = next(rows)
            columns = (
                "playId", "nflId", "frameId", "team", "playDirection",
                "x", "y", "s", "dir", "event",
            )
            play, player, frame, team, play_dir, x, y, speed, direction, event = (
                header.index(name) for name in columns
            )
            for row in rows:
                if row[play] not in pressure_play_ids:
                    continue
                frames = frames_by_play[row[play]]
                frame_id = int(row[frame])
                state = frames.get(frame_id)
                if state is None:
                    state = Frame(frame_id, "", row[play_dir], {})
                    frames[frame_id] = state
                if row[team] == "football":
                    state.event = row[event]
                elif row[player] not in MISSING and row[x] not in MISSING and row[y] not in MISSING:
                    state.positions[int(row[player])] = Position(
                        float(row[x]), float(row[y]), _optional_float(row[speed]) or 0.0,
                        _optional_float(row[direction]), row[team],
                    )

        for play_id in sorted(frames_by_play, key=int):
            frames = frames_by_play[play_id]
            yield int(play_id), tuple(frames[frame_id] for frame_id in sorted(frames))


@contextmanager
def open_data(archive_path):
    """Open the archive for the lifetime of a with block."""
    with ZipFile(archive_path) as archive:
        yield Dataset(archive)
