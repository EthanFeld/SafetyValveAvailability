import { frameAt, pickPlayer, renderReplay } from "./replay.js";

const TEAMS = {
  ARI: "Arizona Cardinals", ATL: "Atlanta Falcons", BAL: "Baltimore Ravens", BUF: "Buffalo Bills",
  CAR: "Carolina Panthers", CHI: "Chicago Bears", CIN: "Cincinnati Bengals", CLE: "Cleveland Browns",
  DAL: "Dallas Cowboys", DEN: "Denver Broncos", DET: "Detroit Lions", GB: "Green Bay Packers",
  HOU: "Houston Texans", IND: "Indianapolis Colts", JAX: "Jacksonville Jaguars", KC: "Kansas City Chiefs",
  LA: "Los Angeles Rams", LAR: "Los Angeles Rams", LAC: "Los Angeles Chargers", LV: "Las Vegas Raiders",
  MIA: "Miami Dolphins", MIN: "Minnesota Vikings", NE: "New England Patriots", NO: "New Orleans Saints",
  NYG: "New York Giants", NYJ: "New York Jets", PHI: "Philadelphia Eagles", PIT: "Pittsburgh Steelers",
  SEA: "Seattle Seahawks", SF: "San Francisco 49ers", TB: "Tampa Bay Buccaneers", TEN: "Tennessee Titans",
  WAS: "Washington Football Team",
};
const RESULTS = { C: "Complete", I: "Incomplete", IN: "Interception", S: "Sack", R: "Scramble" };
const MODELS = { onset: "Onset SVA", temporal: "Full-window SVA" };
const ids = ["team-select", "game-select", "play-select", "model-select", "field-canvas", "play-toggle",
  "step-back", "step-forward", "timeline", "speed-select", "time-label", "frame-label", "event-label", "frame-outlet",
  "pressure-label", "play-title", "play-description", "play-epa", "play-down", "play-distance", "play-week",
  "lineup-title", "lineup-note", "lineup-rows", "valve-name", "valve-score", "valve-position", "status",
  "offense-label", "defense-label", "selected-player"];
const ui = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
const initial = new URLSearchParams(location.search);
const initialHash = location.hash;
const state = {
  index: null, starters: null, team: "", game: null, play: null, model: "onset",
  lineup: [], valve: null, selected: null, time: 0, animation: null, lastTick: null, loadTicket: 0,
};
const gameCache = new Map();

function text(id, value) {
  const node = ui[id];
  if (node.textContent !== value) node.textContent = value;
}

function status(message, kind = "ready") {
  text("status", message);
  ui.status.dataset.state = kind;
}

function teamName(code) {
  return TEAMS[code] || code;
}

function player(id) {
  return state.index?.players[String(id)];
}

function playerName(id) {
  return player(id)?.name || state.lineup.find((p) => p.id === id)?.name || `Player ${id}`;
}

function scoreInfo(id) {
  const metadata = player(id);
  const published = metadata?.[state.model];
  const provisional = metadata?.[`${state.model}Provisional`];
  return {
    score: Number.isFinite(published) ? published : Number.isFinite(provisional) ? provisional : null,
    provisional: !Number.isFinite(published) && Number.isFinite(provisional),
    routes: metadata?.[`${state.model}Routes`],
  };
}

function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function option(value, label) {
  return new Option(label, String(value));
}

function placeholder(select, label) {
  select.replaceChildren(option("", label));
  select.disabled = true;
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path} (HTTP ${response.status})`);
  return response.json();
}

function updateLocation() {
  const url = new URL(location.href);
  for (const [key, value] of Object.entries({ team: state.team, game: state.game?.id,
    play: state.play?.id, model: state.model })) {
    if (value != null && value !== "") url.searchParams.set(key, value);
    else url.searchParams.delete(key);
  }
  history.replaceState(null, "", url);
}

function controlsAvailable(available) {
  for (const id of ["play-toggle", "step-back", "step-forward", "timeline", "speed-select"]) {
    ui[id].disabled = !available;
  }
}

function phase(frame, play) {
  if (frame.pressure) return "Pressure active";
  if (frame.t < play.snap) return "Before snap";
  if (frame.t >= play.terminal) return "After decision";
  return "No pressure at this frame";
}

function clock(t) {
  return `${t < 0 ? "−" : ""}${Math.abs(t).toFixed(1)} s`;
}

function eventName(value) {
  const pretty = { ball_snap: "Ball snapped", pass_forward: "Pass released", autoevent_passforward: "Pass released",
    qb_sack: "Quarterback sacked", qb_strip_sack: "Strip sack", run: "Scramble", pass_arrived: "Pass arrives" };
  return pretty[value] || (value ? value.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase()) : "—");
}

function paint() {
  const { play, time, valve, selected } = state;
  renderReplay(ui["field-canvas"], play, time, { valve: valve?.id, selected, nameOf: playerName });
  if (!play?.frames.length) return;
  const index = frameAt(play.frames, time);
  const frame = play.frames[index];
  ui.timeline.value = String(index);
  ui.timeline.setAttribute("aria-valuetext", `${clock(frame.t)} from snap, frame ${index + 1} of ${play.frames.length}`);
  text("time-label", clock(frame.t));
  text("frame-label", `${index + 1} / ${play.frames.length}`);
  text("event-label", eventName(frame.event));
  const outlet = frame.pressure && frame.best != null;
  text("frame-outlet", outlet ? `${playerName(frame.best)} · ${formatNumber(frame.score, 1)} frame SVA`
    : frame.pressure ? "No in-range outlet" : "—");
  text("pressure-label", phase(frame, play));
  ui["pressure-label"].dataset.active = String(frame.pressure);
}

function stopPlayback() {
  if (state.animation !== null) cancelAnimationFrame(state.animation);
  state.animation = null;
  state.lastTick = null;
  text("play-toggle", "Play");
  ui["play-toggle"].setAttribute("aria-label", "Play replay");
}

function tick(now) {
  const end = state.play.frames.at(-1).t;
  const elapsed = state.lastTick === null ? 0 : Math.min(0.1, (now - state.lastTick) / 1000);
  state.lastTick = now;
  state.time = Math.min(end, state.time + elapsed * Number(ui["speed-select"].value));
  paint();
  if (state.time >= end) stopPlayback();
  else state.animation = requestAnimationFrame(tick);
}

function togglePlayback() {
  if (!state.play?.frames.length) return;
  if (state.animation !== null) {
    stopPlayback();
    return;
  }
  if (state.time >= state.play.frames.at(-1).t) state.time = state.play.frames[0].t;
  text("play-toggle", "Pause");
  ui["play-toggle"].setAttribute("aria-label", "Pause replay");
  state.animation = requestAnimationFrame(tick);
}

function seek(index) {
  if (!state.play?.frames.length) return;
  stopPlayback();
  const frames = state.play.frames;
  state.time = frames[Math.max(0, Math.min(frames.length - 1, index))].t;
  paint();
}

function step(delta) {
  if (state.play?.frames.length) seek(frameAt(state.play.frames, state.time) + delta);
}

function updateSelected() {
  for (const button of ui["lineup-rows"].querySelectorAll("button[data-player]")) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.player) === state.selected));
  }
  if (state.selected === null) {
    text("selected-player", "Select a player on the field or in the lineup to inspect their route.");
    return;
  }
  const route = state.play?.routes.find((row) => row.id === state.selected);
  const isOnField = state.play?.players.some((row) => row.id === state.selected);
  const { score, provisional, routes } = scoreInfo(state.selected);
  const facts = [playerName(state.selected)];
  if (score !== null) facts.push(`${MODELS[state.model]} ${formatNumber(score, 1)}`);
  if (Number.isFinite(routes) && score !== null) facts.push(`${routes} eligible routes${provisional ? " · small sample" : ""}`);
  if (route && Number.isFinite(route.openShare)) facts.push(`Open in ${Math.round(100 * route.openShare)}% of eligible frames`);
  if (route && Number.isFinite(route.firstOpen)) facts.push(`First open ${formatNumber(route.firstOpen, 1)} s after pressure began`);
  if (!isOnField && state.play) facts.push("Off the field on this play");
  text("selected-player", facts.join(" · "));
}

function choosePlayer(id) {
  state.selected = id === state.selected ? null : id;
  updateSelected();
  paint();
}

function currentLineup() {
  const official = state.starters?.lineups?.[state.game?.id]?.[state.team];
  const tracked = state.game?.lineups?.[state.team];
  if (Array.isArray(official) && official.length) {
    return { official: true, rows: official.map((row) => ({ id: row.id, name: row.name,
      position: row.position, number: row.jersey })) };
  }
  return { official: false, trackedPlayId: tracked?.playId,
    rows: (tracked?.players || []).map((row) => ({ id: row.id, name: playerName(row.id),
      position: row.slot || player(row.id)?.position, number: row.number })) };
}

function lineupRow(row) {
  const tr = document.createElement("tr");
  const isValve = row.id != null && state.valve?.id === row.id;
  if (isValve) tr.classList.add("valve-row");
  const position = tr.insertCell();
  position.textContent = row.position || "—";
  const name = tr.insertCell();
  const label = row.number ? `#${row.number} ${row.name}` : row.name;
  if (isValve) {
    const marker = document.createElement("span");
    marker.className = "valve-ring";
    marker.setAttribute("aria-label", "Designated safety valve");
    name.append(marker);
  }
  if (Number.isFinite(row.id)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "lineup-player";
    button.dataset.player = String(row.id);
    button.textContent = label;
    button.setAttribute("aria-pressed", String(row.id === state.selected));
    button.addEventListener("click", () => choosePlayer(row.id));
    name.append(button);
  } else {
    name.append(document.createTextNode(label));
  }
  const score = tr.insertCell();
  score.textContent = formatNumber(row.score, 1);
  if (row.score === null) score.setAttribute("aria-label", "No qualifying score");
  else if (row.provisional) {
    const marker = document.createElement("span");
    marker.className = "provisional-score";
    marker.textContent = "*";
    score.append(marker);
    score.setAttribute("aria-label", `${formatNumber(row.score, 1)}, small sample of ${row.routes} eligible routes`);
  }
  return tr;
}

function renderLineup() {
  const { official, trackedPlayId, rows } = currentLineup();
  state.lineup = rows.map((row) => ({ ...row, ...scoreInfo(row.id) }));
  state.valve = state.lineup.filter((row) => row.score !== null)
    .sort((a, b) => b.score - a.score || a.id - b.id)[0] || null;
  text("lineup-title", state.game ? `${state.team} ${official ? "offensive starters" : "opening lineup"}` : "Game lineup");
  text("lineup-note", state.game ? (official ? `Game-day offensive starters · Week ${state.game.week}`
    : `First tracked offensive dropback${trackedPlayId ? ` · Play ${trackedPlayId}` : ""}`)
    : "Select a game to see the offensive lineup and available receiver scores.");
  text("valve-name", state.valve?.name || (state.game ? "No ranked starter" : "Awaiting lineup"));
  text("valve-score", formatNumber(state.valve?.score, 1));
  text("valve-position", state.valve ? [state.valve.position || "Receiver", MODELS[state.model],
    Number.isFinite(state.valve.routes) ? `${state.valve.routes} routes${state.valve.provisional ? " (small sample)" : ""}` : null]
    .filter(Boolean).join(" · ") : "");
  const table = ui["lineup-rows"];
  table.replaceChildren(...state.lineup.map(lineupRow));
  if (!state.lineup.length) {
    const cell = table.insertRow().insertCell();
    cell.colSpan = 3;
    cell.className = "empty-lineup";
    cell.textContent = state.game ? "No lineup is available for this team and game." : "Select a game to load its lineup.";
  }
  updateSelected();
  paint();
}

function selectPlay(id) {
  stopPlayback();
  state.play = state.game?.plays.find((row) => row.offense === state.team && String(row.id) === String(id)) || null;
  const play = state.play;
  const ready = Boolean(play?.frames.length);
  controlsAvailable(ready);
  if (!ready) {
    text("play-title", "Pressure replay");
    text("play-week", state.game ? `Week ${state.game.week} · Film replay` : "Film replay");
    text("play-description", state.game ? "This offense has no scored pressure replays in the selected game." : "Choose a play to see the tracking and available outlets.");
    for (const id of ["play-down", "play-distance", "play-epa", "event-label", "frame-outlet"]) text(id, "—");
    text("time-label", "0.0 s");
    text("frame-label", "Frame —");
    text("pressure-label", "Awaiting tracking");
    ui["pressure-label"].dataset.active = "false";
    text("offense-label", "Offense");
    text("defense-label", "Defense");
    ui.timeline.max = "0";
    ui.timeline.value = "0";
  } else {
    text("play-title", `${play.offense} offense vs. ${play.defense}`);
    text("play-week", `Week ${state.game.week} · Q${play.quarter} ${play.clock} · Play ${play.id}`);
    text("play-description", play.description);
    text("play-down", String(play.down ?? "—"));
    text("play-distance", Number.isFinite(play.toGo) ? `${play.toGo} yd` : "—");
    text("play-epa", Number.isFinite(play.epa) ? `${play.epa > 0 ? "+" : ""}${play.epa.toFixed(2)}` : "—");
    text("offense-label", `${play.offense} offense`);
    text("defense-label", `${play.defense} defense`);
    ui.timeline.max = String(play.frames.length - 1);
    state.time = play.frames[0].t;
  }
  updateSelected();
  paint();
  updateLocation();
}

function playLabel(play) {
  const outcome = RESULTS[play.result] || play.result || "Dropback";
  return `Q${play.quarter} ${play.clock} · ${play.down}&${play.toGo} · ${play.yardLine} · ${outcome}`;
}

async function selectGame(id, preferredPlay) {
  const ticket = ++state.loadTicket;
  stopPlayback();
  state.game = null;
  selectPlay(null);
  renderLineup();
  placeholder(ui["play-select"], "Loading pressure plays…");
  status("Loading game tracking…", "loading");
  try {
    let game = gameCache.get(id);
    if (!game) {
      game = await loadJson(`./data/games/${id}.json`);
      if (gameCache.size >= 6) gameCache.delete(gameCache.keys().next().value);
      gameCache.set(id, game);
    }
    if (ticket !== state.loadTicket) return;
    state.game = game;
    const candidates = game.plays.filter((play) => play.offense === state.team);
    if (candidates.length) {
      const selector = ui["play-select"];
      selector.replaceChildren(...candidates.map((play) => option(play.id, playLabel(play))));
      selector.disabled = false;
      const selected = candidates.find((play) => String(play.id) === preferredPlay) || candidates[0];
      selector.value = String(selected.id);
      selectPlay(selected.id);
    } else {
      placeholder(ui["play-select"], "No pressure replays for this offense");
      selectPlay(null);
    }
    renderLineup();
    status(`${candidates.length} pressure ${candidates.length === 1 ? "play" : "plays"} for ${teamName(state.team)} in this game.`);
  } catch (error) {
    if (ticket !== state.loadTicket) return;
    placeholder(ui["play-select"], "Game data unavailable");
    status(`Game tracking could not be loaded. Choose another game or reload. ${error.message}`, "error");
  }
}

function selectTeam(team, preferredGame, preferredPlay) {
  state.team = team;
  state.selected = null;
  const games = state.index.games.filter((game) => game.home === team || game.away === team);
  const selector = ui["game-select"];
  selector.replaceChildren(...games.map((game) => {
    const home = game.home === team;
    const other = home ? game.away : game.home;
    return option(game.id, `Week ${game.week} · ${home ? "vs." : "at"} ${other}`);
  }));
  selector.disabled = false;
  const game = games.find((item) => item.id === preferredGame) || games[0];
  selector.value = game.id;
  return selectGame(game.id, preferredPlay);
}

function bindControls() {
  ui["team-select"].addEventListener("change", (e) => selectTeam(e.target.value, state.game?.id));
  ui["game-select"].addEventListener("change", (e) => selectGame(e.target.value));
  ui["play-select"].addEventListener("change", (e) => selectPlay(e.target.value));
  ui["model-select"].addEventListener("change", (e) => {
    state.model = e.target.value;
    renderLineup();
    updateLocation();
  });
  ui["play-toggle"].addEventListener("click", togglePlayback);
  ui["step-back"].addEventListener("click", () => step(-1));
  ui["step-forward"].addEventListener("click", () => step(1));
  ui.timeline.addEventListener("input", (e) => seek(Number(e.target.value)));
  const canvas = ui["field-canvas"];
  canvas.tabIndex = 0;
  canvas.addEventListener("click", (event) => choosePlayer(pickPlayer(canvas, state.play, state.time, event)));
  canvas.addEventListener("keydown", (event) => {
    if (![" ", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === " ") togglePlayback();
    else step(event.key === "ArrowLeft" ? -1 : 1);
  });
  new ResizeObserver(paint).observe(canvas);
}

async function init() {
  controlsAvailable(false);
  ui["team-select"].disabled = true;
  placeholder(ui["game-select"], "Select a team");
  placeholder(ui["play-select"], "Select a game");
  paint();
  try {
    state.index = await loadJson("./data/index.json");
    const codes = [...new Set(state.index.games.flatMap((game) => [game.home, game.away]))]
      .sort((a, b) => teamName(a).localeCompare(teamName(b)));
    ui["team-select"].replaceChildren(...codes.map((code) => option(code, teamName(code))));
    ui["team-select"].disabled = false;
    const team = codes.includes(initial.get("team")) ? initial.get("team") : codes.includes("TB") ? "TB" : codes[0];
    state.model = Object.hasOwn(MODELS, initial.get("model")) ? initial.get("model") : "onset";
    ui["model-select"].value = state.model;
    ui["team-select"].value = team;
    bindControls();
    const starters = loadJson("./lineups.json").then((data) => { state.starters = data; renderLineup(); }).catch(() => {
      // Static exports can use the charted first-dropback lineup on their own.
    });
    await Promise.all([starters, selectTeam(team, initial.get("game"), initial.get("play"))]);
    if (initialHash === "#leaderboard" && location.hash === initialHash) {
      document.getElementById("leaderboard")?.scrollIntoView();
    }
  } catch (error) {
    status(`The play library could not be loaded. ${error.message}`, "error");
  }
}

init();
