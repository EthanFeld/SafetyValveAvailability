# Safety Valve Availability

[**Read the story**](https://ethanfeld.github.io/SafetyValveAvailability/) · [**Open the Film Room**](https://ethanfeld.github.io/SafetyValveAvailability/film-room.html) · [**Player leaderboard**](https://ethanfeld.github.io/SafetyValveAvailability/film-room.html#leaderboard)

Safety Valve Availability (SVA) measures the outlets a quarterback has under pressure, including routes that never receive a target. The Film Room pairs receiver grades with frame-by-frame tracking from **2,756 pressure plays across 122 games** in Weeks 1–8 of the 2021 NFL season.

## Explore the site

- **Story** — See why an available outlet matters under pressure, how SVA measures it, and what the tracking sample supports.
- **Film Room** — Pick an offense, game, and play. Replay the tracking, scrub the timeline, or step through individual frames. The gold ring identifies the highest-scoring starter on the displayed lineup. The dashed aqua ring identifies the best in-range throw on the current pressure frame.
- **Leaderboard** — Search 183 players across all 32 teams by name. Combine the search with team and position filters, then sort by onset SVA, full-window SVA, catch probability, or expected EPA. A player tracked with multiple teams can appear under each. Displayed scores are pooled across the sample.
- **Side-by-side comparison** — Check the box beside each of two receivers and choose **Compare players** to see their grades, forecasts, route samples, and opening breakdowns on the same rows. Selections stay in place when you search or change filters, so you can pair receivers from different teams or positions.
- **Player profiles** — Select a player’s name for a breakdown of the forecasts, context benchmarks, and opening frequency behind their scores. Choose a play from the profile to jump to the player’s first open pressure frame, where their in-range throw score reached at least 25 out of 100.
- **Sample sizes** — The analysis leaderboards require at least 30 eligible routes. The site also shows provisional grades based on 15–29 eligible routes, marked with an asterisk. A dash indicates that a metric is unavailable.

## What the scores mean

| Metric | Interpretation |
| --- | --- |
| **Onset SVA** | A 0–100 receiver grade using the first qualifying pressure frame. It equally weights standardized catch and EPA forecasts above position and situation benchmarks. The EPA benchmark also conditions on catch probability to avoid counting catchability twice. |
| **Full-window SVA** | A 0–100 grade that also accounts for opening frequency and timing. Half comes from context-adjusted opening frequency, weighted toward earlier frames with a 0.4-second half-life after first entering range. The catch and EPA forecasts at first eligibility supply the other half. |
| **Catch probability** | Mean modeled chance of a catch if targeted, across the player's eligible pressure-onset routes. |
| **Expected EPA** | Mean modeled offensive expected points added if targeted, across the same onset routes. The hurdle forecast weights the EPA estimates for a catch and a miss by their predicted probabilities. |

The pressure window ends at the first pass, sack, or run marker. Within that window, a PFF-credited rusher qualifies as pressure if they are within 2.5 yards of the quarterback, or within 4 yards and projected to close to 2.5 yards in 0.3 seconds. A receiver is eligible when their projected position is within 20 yards of the quarterback. An opening requires a revised frame-level throw score of at least 25.

The throw score considers projected separation, throw distance, a height-based catch radius, traffic in the passing lane, and quarterback handedness. This frame-level score drives the replay highlight. The player grades aggregate routes against their benchmarks, with **50** as the reference point. [Frame scoring](safety_valve.py) · [Onset benchmarks](rank_context_sva.py) · [Full-window method](evaluate_temporal_sva.py)

## Validation and interpretation

Full-window scoring covered **6,535 in-range routes**, compared with **6,270 at onset**, and raised the number of qualified players from 66 to 70. Among common players, split-half rank correlation was **0.150 at onset** and **0.016 with the full-window score**.

In held-out-game tests on plays still ongoing 0.3 seconds after pressure began, adding early timing to the onset index raised catch AUC from **0.553 to 0.563** across **834 matched targets**. The gain was smaller and uncertain when detailed throw geometry was already included. Team EPA RMSE moved from **1.7685 to 1.7699** across **2,230 plays**, with lower values indicating better forecasts. These results favor onset SVA for player ranking and full-window timing for film review. [Temporal validation](evaluate_temporal_sva.py) · [Route timing data](sva_temporal_routes.csv)

The sample covers eight weeks of one season. Catch and EPA forecasts are trained on observed targets, so estimates for untargeted outlets rely on how well those examples generalize. The grades describe modeled opportunities in this sample. A causal estimate of changing the target would require separate validation.

## Run locally

The prebuilt site needs only a static HTTP server. With Python installed, run these commands from the repository root and open the story or go straight to the Film Room. [Local story](http://127.0.0.1:8000/) · [Local Film Room](http://127.0.0.1:8000/film-room.html)

```sh
git clone https://github.com/EthanFeld/SafetyValveAvailability.git
cd SafetyValveAvailability
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

### Rebuild the data and site

Use Python 3.10+ and install the analysis dependencies with `python -m pip install -r requirements.txt`. Place `nfl-big-data-bowl-regional-event-data-main.zip` in the repository root. The 179 MB input is distributed separately because it exceeds GitHub's 100 MiB limit for regular Git files. [Project release download](https://github.com/EthanFeld/SafetyValveAvailability/releases/download/data-2021/nfl-big-data-bowl-regional-event-data-main.zip) · [Upstream ZIP](https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data/archive/refs/heads/main.zip) · [GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

The repository includes the joined EPA extract and sourced quarterback-handedness table. Run the analysis and export steps in order.

```sh
python safety_valve.py --play-output safety_valve_plays.csv
python safety_valve.py --throw-model separation-weighted
python analyze_sva_epa.py
python train_sva_joint.py
python rank_joint_sva.py
python rank_context_sva.py
python evaluate_temporal_sva.py
python build_site_data.py
python build_site_leaderboard.py
python build_site_profiles.py
```

The committed `docs/lineups.json` is enough for a local rebuild. To refresh it from the original sources, install `pypdf` and run `python build_site_lineups.py`. This step uses the network to fetch ESPN lineups and NFL gamebooks.

### Repository layout

| Path | Contents |
| --- | --- |
| [`docs/`](docs/) | Story page, Film Room, leaderboard, player profiles, starting lineups, and per-game replay files. |
| [`safety_valve.py`](safety_valve.py), [`sva_data.py`](sva_data.py) | Tracking ingestion, pressure detection, throw geometry, and frame-level scoring. |
| [`train_sva_joint.py`](train_sva_joint.py), [`rank_context_sva.py`](rank_context_sva.py), [`evaluate_temporal_sva.py`](evaluate_temporal_sva.py) | Catch and EPA models, receiver grading, and held-out evaluation. |
| [`build_site_data.py`](build_site_data.py), [`build_site_lineups.py`](build_site_lineups.py), [`build_site_leaderboard.py`](build_site_leaderboard.py), [`build_site_profiles.py`](build_site_profiles.py) | Static site data exporters. |
| [`sva_context_leaderboard.csv`](sva_context_leaderboard.csv), [`sva_temporal_leaderboard.csv`](sva_temporal_leaderboard.csv), [`sva_joint_outlets.csv`](sva_joint_outlets.csv) | Onset grades, full-window grades, and route-level forecasts. |

### Publish on GitHub Pages

The live repository publishes from `main` → `/docs`. To enable Pages on a fork, push the `docs/` folder with `.nojekyll` and its generated JSON files. In **Settings → Pages**, choose **Deploy from a branch**, select `main` and `/docs`, and save. The site uses relative URLs so it works under a repository path. [GitHub Pages publishing guide](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

## Data and attribution

- **Tracking and scouting** — NFL Next Gen Stats player and ball tracking and Pro Football Focus scouting labels. [Big Data Bowl regional archive and documentation](https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data)
- **Play outcomes and EPA** — The 2021 nflverse play-by-play release, distributed under CC BY 4.0. The local extract matches `old_game_id` and `play_id` to the tracking archive and checks the possession team. [nflverse release](https://github.com/nflverse/nflverse-data/releases/tag/pbp) · [EPA field documentation](https://nflfastr.com/reference/fast_scraper.html) · [nflverse licensing](https://github.com/nflverse)
- **Starting lineups and handedness** — ESPN and NFL gamebooks supply game-day starters. The handedness file records its sources per quarterback. [ESPN scoreboard](https://www.espn.com/nfl/scoreboard/_/week/1/year/2021/seasontype/2) · [Quarterback source table](qb_handedness.csv)
