"""Compare Safety Valve Availability with catch rate on matched pressured targets."""

import argparse
from collections import defaultdict
import csv
from io import TextIOWrapper
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile

import numpy as np

from safety_valve import DEFAULT_MAX_THROW_DISTANCE, ROOT
from sva_pressure_context import release_contexts


PASS_TARGET = re.compile(
    r"\bpass\s+(?:incomplete\s+)?(?:(short|deep)\s+)?(?:left\s+|middle\s+|right\s+)?"
    r"(?:to|intended for)\s+", re.IGNORECASE,
)
EXCLUDED_PLAYS = re.compile(r"\bno play\b|\btwo[- ]point conversion attempt\b", re.IGNORECASE)
NAME_PART = re.compile(r"[\w.'’\-]+")
NAME_SUFFIX = re.compile(r"\s+(?:Jr\.?|Sr\.?|II|III|IV|V)$", re.IGNORECASE)
OFFENSIVE_ROLES = {"Pass", "Pass Route", "Pass Block"}
# The Saints used Harris in 2021 before the roster name changed to Harty.
# https://www.neworleanssaints.com/news/deonte-harris-harty-wide-receiver-new-orleans-saints-season-recap-2021
HISTORICAL_ALIASES = {48415: {"dharris"}}


def normalized(text):
    return "".join(char for char in unicodedata.normalize("NFKD", text).casefold() if char.isalnum())


def name_aliases(name, nfl_id):
    first, _, surname = name.partition(" ")
    initials = normalized(first)
    surname = normalized(NAME_SUFFIX.sub("", surname))
    aliases = {initials[:length] + surname for length in range(1, min(3, len(initials)) + 1)}
    return aliases | HISTORICAL_ALIASES.get(nfl_id, set())


def match_target(description, aliases_by_id):
    clauses = list(PASS_TARGET.finditer(description))
    if not clauses:
        return None, None
    clause = clauses[-1]
    # Allow multiword surnames like St. Brown without treating the whole description as a name.
    parts = NAME_PART.findall(description[clause.end():])[:4]
    candidates = {normalized("".join(parts[:i])) for i in range(1, len(parts) + 1)}
    matches = [nfl_id for nfl_id, aliases in aliases_by_id.items() if candidates & aliases]
    depth = clause.group(1)
    return (matches[0] if len(matches) == 1 else None), depth.lower() if depth else None


def archive_rows(archive, filename):
    member = next(name for name in archive.namelist() if name.rsplit("/", 1)[-1] == filename)
    with TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def play_key(row):
    return int(row["gameId"]), int(row["playId"])


def target_outcomes(archive_path):
    with ZipFile(archive_path) as archive:
        plays = {
            play_key(row): row for row in archive_rows(archive, "plays.csv")
            if row["passResult"] in {"C", "I", "IN"}
            and not EXCLUDED_PLAYS.search(row["playDescription"])
        }
        aliases = {
            int(row["nflId"]): name_aliases(row["displayName"], int(row["nflId"]))
            for row in archive_rows(archive, "players.csv")
        }
        offense = defaultdict(dict)
        for row in archive_rows(archive, "pffScoutingData.csv"):
            key = play_key(row)
            nfl_id = int(row["nflId"])
            if key in plays and row["pff_role"] in OFFENSIVE_ROLES and nfl_id in aliases:
                offense[key][nfl_id] = aliases[nfl_id]

    outcomes = {}
    for key, play in plays.items():
        nfl_id, depth = match_target(play["playDescription"], offense[key])
        if nfl_id is not None:
            outcomes[(*key, nfl_id)] = (int(play["passResult"] == "C"), depth)
    return outcomes, plays.keys()


def pearson(x, y):
    if len(x) < 2 or not np.ptp(x) or not np.ptp(y):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def game_bootstrap(x, y, games):
    groups = [np.flatnonzero(games == game) for game in np.unique(games)]
    rng = np.random.default_rng(2021)
    estimates = []
    for sample in rng.integers(len(groups), size=(2000, len(groups))):
        rows = np.concatenate([groups[i] for i in sample])
        estimates.append(pearson(x[rows], y[rows]))
    return np.nanquantile(estimates, [0.025, 0.975])


def category_controls(values):
    values = np.asarray(values)
    return values[:, None] == np.unique(values)[None, 1:]


def adjusted_pearson(x, y, controls):
    if len(x) < 2:
        return float("nan")
    design = np.column_stack((np.ones(len(x)), controls))
    paired = np.column_stack((x, y))
    residuals = paired - design @ np.linalg.lstsq(design, paired, rcond=None)[0]
    return pearson(*residuals.T)


def pre_snap_context(archive_path, play_keys):
    with ZipFile(archive_path) as archive:
        home_teams = {int(row["gameId"]): row["homeTeamAbbr"]
                      for row in archive_rows(archive, "games.csv")}
        for row in archive_rows(archive, "plays.csv"):
            game, play = play_key(row)
            if (game, play) not in play_keys or game not in home_teams:
                continue
            offense = row["possessionTeam"]
            try:
                yardline = int(row["yardlineNumber"])
                home_margin = int(row["preSnapHomeScore"]) - int(row["preSnapVisitorScore"])
                yards_to_go = int(row["yardsToGo"])
                play_action = int(row["pff_playAction"])
            except ValueError:
                continue
            yield (game, play), (
                yards_to_go, 100 - yardline if row["yardlineSide"] == offense else yardline,
                home_margin if offense == home_teams[game] else -home_margin, play_action,
                row["down"], row["quarter"], row["pff_passCoverageType"], row["defensiveTeam"],
            )


def catch_rate_fit(sva, caught, games, controls):
    """Linear probability slope per 10 SVA points, with a game-clustered CR1 interval."""
    design = np.column_stack((np.ones(len(sva)), controls))
    pair = np.column_stack((sva / 10, caught))
    nuisance_fit, _, rank, _ = np.linalg.lstsq(design, pair, rcond=None)
    score, outcome = (pair - design @ nuisance_fit).T
    _, game_index = np.unique(games, return_inverse=True)
    n_games = len(np.unique(game_index))
    score_ss = score @ score
    dof = len(sva) - rank - 1
    if n_games < 2 or dof < 1 or score_ss < 1e-12:
        return (float("nan"),) * 3

    slope = (score @ outcome) / score_ss
    error = outcome - slope * score
    game_scores = np.bincount(game_index, weights=score * error)
    correction = n_games / (n_games - 1) * (len(sva) - 1) / dof
    half_width = 1.96 * np.sqrt(correction * (game_scores @ game_scores)) / score_ss
    return tuple(100 * value for value in (slope, slope - half_width, slope + half_width))


def print_fit(label, sva, caught, games, controls):
    estimate, low, high = catch_rate_fit(sva, caught, games, controls)
    print(f"{label}: {estimate:+.2f} pp (95% game-clustered CI {low:+.2f} to {high:+.2f}).")


def analyze_release_adjustment(archive_path, matched_targets, max_throw_distance):
    contexts = release_contexts(archive_path, matched_targets, max_throw_distance)
    situations = dict(pre_snap_context(archive_path, {key[:2] for key in contexts}))
    keys = [key for key in matched_targets if key in contexts and key[:2] in situations]
    if not keys:
        print("No matched targets have pre-release tracking and game context.")
        return

    sva = np.array([float(matched_targets[key][0]["sva"]) for key in keys])
    caught = np.array([matched_targets[key][1] for key in keys])
    position = [matched_targets[key][0]["position"] for key in keys]
    qb = [matched_targets[key][0]["qb_id"] for key in keys]
    games, _, receiver = np.asarray(keys).T
    release = [contexts[key] for key in keys]
    distance = np.array([item.rusher_distance_yd for item in release])
    pressure = np.column_stack((
        distance, distance ** 2,
        [item.rusher_closing_yd_s for item in release],
        [item.credited_rushers for item in release],
        [item.pressure_seconds for item in release],
        [item.pressure_at_release for item in release],
    ))
    to_go, to_goal, margin, play_action, down, quarter, coverage, defense = zip(
        *(situations[key[:2]] for key in keys)
    )
    throw_distance = np.array([item.target_distance_yd for item in release])
    football = np.column_stack((
        pressure, throw_distance, throw_distance ** 2,
        [item.target_downfield_yd for item in release],
        [item.time_to_throw_s for item in release], [item.qb_speed_yd_s for item in release],
        to_go, to_goal, margin, play_action, category_controls(position), category_controls(down),
        category_controls(quarter), category_controls(coverage), category_controls(defense),
    ))
    player_effects = np.column_stack((football, category_controls(qb), category_controls(receiver)))

    print(f"\nAdjusted catch analysis ({len(keys):,}/{len(matched_targets):,} targets with release context "
          f"across {len(np.unique(games))} games), per 10 SVA points:")
    print_fit("Unadjusted, same targets", sva, caught, games, np.empty((len(keys), 0)))
    print_fit("Pressure severity", sva, caught, games, pressure)
    print_fit("Pressure, throw and game context", sva, caught, games, football)
    print_fit("Also accounting for QB and receiver", sva, caught, games, player_effects)

    at_release = np.array([item.pressure_at_release and item.target_in_range_at_release for item in release])
    if np.any(at_release):
        print_fit(f"Pressure, throw and game context, still pressured and in range at release (n={at_release.sum():,})",
                  sva[at_release], caught[at_release], games[at_release], football[at_release])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "nfl-big-data-bowl-regional-event-data-main.zip")
    parser.add_argument("--routes", type=Path, default=ROOT / "safety_valve_plays.csv")
    parser.add_argument("--adjusted", action="store_true", help="Load tracking for pressure and context controls.")
    parser.add_argument("--max-throw-distance", type=float, default=DEFAULT_MAX_THROW_DISTANCE,
                        help="Range cutoff for the release check, matching the route scores (default 20).")
    args = parser.parse_args()
    if not np.isfinite(args.max_throw_distance) or args.max_throw_distance <= 0:
        parser.error("--max-throw-distance must be a positive, finite number")

    with args.routes.open(encoding="utf-8-sig", newline="") as file:
        routes = list(csv.DictReader(file))
    pressure_plays = {(int(row["game_id"]), int(row["play_id"])) for row in routes}
    outcomes, pass_plays = target_outcomes(args.archive)
    pressure_targets = sum((game_id, play_id) in pressure_plays for game_id, play_id, _ in outcomes)
    print(f"Identified {pressure_targets:,}/{len(pressure_plays & pass_plays):,} intended targets "
          "on eligible pressure pass plays.")
    print(f"Identified {len(outcomes):,}/{len(pass_plays):,} intended targets across all charted pass plays.")

    routes_by_player = defaultdict(list)
    positions_by_player = {}
    targets = []
    matched_targets = {}
    for row in routes:
        if not row["sva"]:
            continue
        game_id, play_id, player = (int(row[key]) for key in ("game_id", "play_id", "nfl_id"))
        score = float(row["sva"])
        routes_by_player[player].append(score)
        positions_by_player[player] = row["position"]
        key = game_id, play_id, player
        if key in outcomes:
            caught, depth = outcomes[key]
            targets.append((game_id, score, float(row["distance_yd"]), caught, depth == "short", row["position"]))
            if args.adjusted:
                matched_targets[key] = (row, caught)
    if not targets:
        parser.error("no in-range routes matched to an intended target")

    games, sva, distance, caught, short, positions = map(np.asarray, zip(*targets))
    print(f"{int(caught.sum()):,}/{len(caught):,} in-range targets caught ({caught.mean():.1%}).")
    if len(np.unique(games)) > 1 and np.ptp(sva) and np.ptp(caught):
        lo, hi = game_bootstrap(sva, caught, games)
        print(f"Target-level r={pearson(sva, caught):.3f} (95% game-bootstrap interval {lo:.3f} to {hi:.3f}).")
    else:
        print(f"Target-level r={pearson(sva, caught):.3f}.")

    bins = np.digitize(sva, np.quantile(sva, [0.25, 0.5, 0.75]))
    for i in range(4):
        group = bins == i
        if np.any(group):
            print(f"SVA quartile {i + 1} ({sva[group].min():.2f}-{sva[group].max():.2f}): "
                  f"{int(caught[group].sum())}/{int(group.sum())} caught ({caught[group].mean():.1%}).")

    controls = np.column_stack((distance, distance ** 2, category_controls(positions)))
    print(f"SVA r adjusted for modeled distance and position={adjusted_pearson(sva, caught, controls):.3f}.")
    print(f"Short-labeled passes only: r={pearson(sva[short], caught[short]):.3f} (n={short.sum():,}).")

    all_catches_by_player = defaultdict(list)
    for (_, _, player), (caught, _) in outcomes.items():
        all_catches_by_player[player].append(caught)
    qualified = [player for player, scores in routes_by_player.items()
                 if len(scores) >= 30 and len(all_catches_by_player[player]) >= 20]
    player_sva = np.array([np.mean(routes_by_player[player]) for player in qualified])
    player_catch_pct = np.array([np.mean(all_catches_by_player[player]) for player in qualified])
    player_positions = np.array([positions_by_player[player] for player in qualified])
    player_r = pearson(player_sva, player_catch_pct)
    within_r = adjusted_pearson(player_sva, player_catch_pct, category_controls(player_positions))
    print(f"Player-level SVA vs overall catch rate (30+ in-range routes and 20+ targets, n={len(qualified)}): "
          f"r={player_r:.3f} pooled, r={within_r:.3f} after adjusting for position.")
    if args.adjusted:
        analyze_release_adjustment(args.archive, matched_targets, args.max_throw_distance)


if __name__ == "__main__":
    main()
