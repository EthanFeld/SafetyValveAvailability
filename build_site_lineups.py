"""Export game-day offensive starters for the tracking archive's film room."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from zipfile import ZipFile

from analyze_catch_rate import archive_rows, name_aliases, normalized


ROOT = Path(__file__).resolve().parent
SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"
ESPN = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
ESPN_TEAMS = dict(ATL=1, BUF=2, CHI=3, CIN=4, CLE=5, DAL=6, DEN=7, DET=8, GB=9, TEN=10,
                  IND=11, KC=12, LV=13, LAR=14, LA=14, MIA=15, MIN=16, NE=17, NO=18, NYG=19,
                  NYJ=20, PHI=21, ARI=22, PIT=23, LAC=24, SF=25, SEA=26, TB=27, WAS=28,
                  WSH=28, CAR=29, JAX=30, JAC=30, BAL=33, HOU=34)
POSITION_ORDER = ("QB", "RB", "FB", "WR", "TE", "LT", "LG", "C", "RG", "RT", "OT", "OG", "T", "G", "OL")
LINEUP_ENTRY = re.compile(r"\b([A-Z][A-Z/]{0,4})\s+(\d{1,2})\s+(\S+)")


def download(url):
    request = Request(url, headers={"User-Agent": "SVA-Film-Room/1.0"})
    with urlopen(request, timeout=25) as response:
        return response.read().decode("utf-8-sig")


def get_roster(job):
    game_id, team, event_id = job
    url = f"{ESPN}/events/{event_id}/competitions/{event_id}/competitors/{ESPN_TEAMS[team]}/roster"
    try:
        payload = json.loads(download(url))
        entries = [row for row in payload["entries"] if row.get("period", 0) == 0]
        return game_id, team, entries
    except (OSError, ValueError, KeyError) as error:
        print(f"Roster unavailable for {game_id} {team}: {error}")
        return game_id, team, []


def position_id(entry):
    match = re.search(r"/positions/(\d+)", entry.get("position", {}).get("$ref", ""))
    return match.group(1) if match else None


def position_lookup(position):
    payload = json.loads(download(f"{ESPN}/positions/{position}"))
    return position, payload["abbreviation"]


def without_suffix(name):
    return normalized(re.sub(r"\s+(?:jr\.?|sr\.?|ii|iii|iv|v)$", "", name, flags=re.IGNORECASE))


def tracking_player(entry, profile, tracking, by_name):
    tracked_id = profile.get("nfl_id", "")
    if tracked_id in tracking:
        return int(tracked_id), tracking[tracked_id]["displayName"]
    display_name = profile.get("display_name") or profile.get("football_name") or entry["displayName"]
    candidates = by_name[without_suffix(display_name)]
    if len(candidates) == 1:
        return int(candidates[0]["nflId"]), candidates[0]["displayName"]
    return None, display_name


def jersey_number(jersey):
    return str(int(jersey)) if jersey and jersey.isdecimal() else jersey or ""


def lineup_player(entry, position, profiles, tracking, by_name):
    tracked_id, name = tracking_player(entry, profiles.get(str(entry["playerId"]), {}), tracking, by_name)
    return dict(id=tracked_id, name=name, position=position, jersey=jersey_number(entry.get("jersey")))


def gamebook_lineups(schedule_row):
    # Only the factual starting units are extracted. The PDF is never copied to the site.
    from pypdf import PdfReader

    url = f"https://www.nflgsis.com/2021/reg/{int(schedule_row['week']):02}/{schedule_row['gsis']}/Gamebook.pdf"
    with urlopen(Request(url, headers={"User-Agent": "SVA-Film-Room/1.0"}), timeout=25) as response:
        pdf = PdfReader(BytesIO(response.read()))
    page = pdf.pages[0].extract_text(extraction_mode="layout")
    body = page.split("Lineups", 1)[1].split("Substitutions", 1)[0]
    rows = [matches for line in body.splitlines() if len(matches := LINEUP_ENTRY.findall(line)) == 4]
    return [row[0] for row in rows], [row[2] for row in rows]


def get_gamebook(job):
    game_id, schedule_row = job
    try:
        return game_id, gamebook_lineups(schedule_row)
    except (ImportError, OSError, ValueError, KeyError, IndexError) as error:
        print(f"Gamebook lineup unavailable for {game_id}: {error}")
        return game_id, ([], [])


def complete_tracking_ids(archive_path, lineups, tracking):
    # A gamebook-only starter may have no matching ESPN roster entry. The
    # tracked team and game-day jersey give a second, game-specific crosswalk.
    with ZipFile(archive_path) as archive:
        for game_id, teams in lineups.items():
            pending = {(team, row["jersey"]): row for team, roster in teams.items() for row in roster
                       if row["id"] is None and row["jersey"]}
            if not pending:
                continue
            for row in archive_rows(archive, f"tracking_{game_id}.csv"):
                key = (row["team"], jersey_number(row["jerseyNumber"]))
                player = tracking.get(row["nflId"])
                if player and key in pending:
                    starter = pending.pop(key)
                    starter.update(id=int(player["nflId"]), name=player["displayName"])
                    if not pending:
                        break

    # A starter can leave before any charted passing play. Accept a short
    # gamebook name only when it identifies exactly one archive player.
    aliases = defaultdict(list)
    for nfl_id, player in tracking.items():
        for alias in name_aliases(player["displayName"], int(nfl_id)):
            aliases[alias].append(player)
    for teams in lineups.values():
        for roster in teams.values():
            for row in roster:
                if row["id"] is None:
                    candidates = aliases[without_suffix(row["name"])]
                    if len(candidates) == 1:
                        row.update(id=int(candidates[0]["nflId"]), name=candidates[0]["displayName"])


def build_lineups(archive_path, output_path):
    with ZipFile(archive_path) as archive:
        games = list(archive_rows(archive, "games.csv"))
        tracking = {row["nflId"]: row for row in archive_rows(archive, "players.csv")}
    by_name = defaultdict(list)
    for row in tracking.values():
        by_name[without_suffix(row["displayName"])].append(row)
    schedule = {row["old_game_id"]: row for row in csv.DictReader(download(SCHEDULE_URL).splitlines())
                if row["season"] == "2021" and row["espn"].isdecimal()}
    profiles = {row["espn_id"]: row for row in csv.DictReader(download(PLAYERS_URL).splitlines()) if row["espn_id"]}
    archive_games = {row["gameId"]: row for row in games}
    jobs = [(game["gameId"], game[team], schedule[game["gameId"]]["espn"]) for game in games
            if game["gameId"] in schedule for team in ("homeTeamAbbr", "visitorTeamAbbr")]
    with ThreadPoolExecutor(max_workers=12) as pool:
        rosters = list(pool.map(get_roster, jobs))
        used_positions = {position_id(entry) for _, _, entries in rosters for entry in entries if entry.get("starter")} - {None}
        positions = dict(pool.map(position_lookup, sorted(used_positions)))

    lineups = defaultdict(dict)
    starter_counts = Counter()
    incomplete = []
    for game_id, team, entries in rosters:
        starters = []
        seen = set()
        for entry in entries:
            espn_id = str(entry["playerId"])
            position = positions.get(position_id(entry))
            if not entry.get("starter") or position not in POSITION_ORDER or espn_id in seen:
                continue
            seen.add(espn_id)
            starters.append(lineup_player(entry, position, profiles, tracking, by_name))
        starter_counts[len(starters)] += 1
        if len(starters) == 11:
            starters.sort(key=lambda row: (POSITION_ORDER.index(row["position"]), row["name"]))
            lineups[game_id][team] = starters
        else:
            incomplete.append((game_id, team, entries))

    gamebook_units = defaultdict(list)
    with ThreadPoolExecutor(max_workers=8) as pool:
        gamebooks = dict(pool.map(get_gamebook, [(game_id, schedule[game_id])
                                                for game_id in sorted({job[0] for job in incomplete})]))
    for game_id, team, entries in incomplete:
        away, home = gamebooks[game_id]
        unit = home if team == archive_games[game_id]["homeTeamAbbr"] else away
        if len(unit) != 11 or len({jersey for _, jersey, _ in unit}) != 11:
            print(f"Incomplete gamebook starters for {game_id} {team}: {len(unit)} offensive starters.")
            continue
        by_jersey = {jersey_number(entry.get("jersey")): entry for entry in entries}
        starters = []
        for position, jersey, name in unit:
            jersey = jersey_number(jersey)
            entry = by_jersey.get(jersey)
            if entry:
                starters.append(lineup_player(entry, position, profiles, tracking, by_name))
            else:
                starters.append(dict(id=None, name=name, position=position, jersey=jersey))
        lineups[game_id][team] = starters
        gamebook_units[game_id].append(team)

    if not lineups:
        raise ValueError("No complete offensive starter units found.")
    complete_tracking_ids(archive_path, lineups, tracking)
    result = dict(source=dict(label="ESPN and NFL game-day starters",
                              url="https://www.espn.com/nfl/scoreboard/_/week/1/year/2021/seasontype/2"),
                  lineups=dict(lineups), gamebookUnits=dict(gamebook_units))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    unmatched = Counter(row["name"] for teams in lineups.values() for roster in teams.values()
                        for row in roster if row["id"] is None)
    print(f"Exported {sum(map(len, lineups.values()))}/{len(games)*2} team-game starter units.")
    print(f"ESPN offensive starter counts: {dict(sorted(starter_counts.items()))}.")
    print(f"Units completed from NFL gamebooks: {sum(map(len, gamebook_units.values()))}.")
    print(f"Starters without a tracking ID: {sum(unmatched.values())} ({', '.join(sorted(unmatched)) or 'none'}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "nfl-big-data-bowl-regional-event-data-main.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "lineups.json")
    args = parser.parse_args()
    if args.output.resolve() == args.archive.resolve():
        parser.error("The output must be different from the tracking archive.")
    build_lineups(args.archive, args.output)
