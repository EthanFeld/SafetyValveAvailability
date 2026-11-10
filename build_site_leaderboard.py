"""Export the comparison leaderboard from the existing Film Room data."""

import argparse
import csv
import json
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parent
MIN_ROUTES = 15
TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Football Team",
}


def load_json(path):
    with path.open(encoding="utf-8-sig") as source:
        return json.load(source)


def observed_teams(games, data, lineups):
    teams = defaultdict(set)
    for game in games:
        replay = load_json(data / "games" / f"{game['id']}.json")
        for play in replay["plays"]:
            for player in play["players"]:
                if player["team"] == play["offense"]:
                    teams[player["id"]].add(player["team"])
        for team, starters in lineups.get(game["id"], {}).items():
            for player in starters:
                if player["id"] is not None:
                    teams[player["id"]].add(team)
    return teams


def onset_forecasts(path):
    forecasts = defaultdict(list)
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            player_id = int(row["nfl_id"])
            route = row["game_id"], row["play_id"], player_id
            if route in seen:
                raise ValueError(f"Duplicate onset route {route}")
            seen.add(route)
            catch = float(row["catch_probability"])
            epa = float(row["hurdle_epa_on_comparable_targets"])
            if not 0 <= catch <= 1 or not isfinite(epa):
                raise ValueError(f"Invalid onset forecast for {route}")
            forecasts[player_id].append((catch, epa))
    return forecasts


def display_score(player, model):
    published = player[model]
    return published if published is not None else player[f"{model}Provisional"]


def build():
    data = ROOT / "docs" / "data"
    index = load_json(data / "index.json")
    games = index["games"]
    lineups = load_json(ROOT / "docs" / "lineups.json")["lineups"]
    team_codes = {game[side] for game in games for side in ("home", "away")}
    teams = [{"code": code, "name": TEAM_NAMES[code]} for code in sorted(team_codes, key=TEAM_NAMES.__getitem__)]
    memberships = observed_teams(games, data, lineups)
    forecasts = onset_forecasts(ROOT / "sva_joint_outlets.csv")
    players = []

    for nfl_id, player in index["players"].items():
        player_id = int(nfl_id)
        onset = display_score(player, "onset")
        temporal = display_score(player, "temporal")
        if onset is None and temporal is None:
            continue
        routes = forecasts[player_id]
        if len(routes) != player["onsetRoutes"]:
            raise ValueError(f"Onset route count does not match the replay index for {player_id}")
        player_teams = memberships[player_id]
        if not player_teams or not player_teams <= team_codes:
            raise ValueError(f"Missing or unknown team for {player_id}")
        catch = epa = None
        if len(routes) >= MIN_ROUTES:
            catch = round(fmean(p for p, _ in routes), 5)
            epa = round(fmean(value for _, value in routes), 5)
        players.append({
            "id": player_id,
            "name": player["name"],
            "position": player["position"],
            "teams": sorted(player_teams),
            "onset": onset,
            "temporal": temporal,
            "catchProbability": catch,
            "expectedEpa": epa,
            "onsetRoutes": player["onsetRoutes"],
            "temporalRoutes": player["temporalRoutes"],
        })
    players.sort(key=lambda row: (row["name"].casefold(), row["id"]))
    return {"teams": teams, "players": players}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "data" / "leaderboard.json")
    output = parser.parse_args().output
    leaderboard = build()
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(leaderboard, destination, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        destination.write("\n")
    print(f"Wrote {len(leaderboard['players'])} players across {len(leaderboard['teams'])} teams to {output}")


if __name__ == "__main__":
    main()
