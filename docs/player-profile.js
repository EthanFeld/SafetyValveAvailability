const dialog = document.getElementById("player-profile");
const title = document.getElementById("player-profile-title");
const teams = document.getElementById("player-profile-teams");
const scores = document.getElementById("player-profile-scores");
const status = document.getElementById("player-profile-status");
const breakdown = document.getElementById("player-profile-breakdown");
const plays = document.getElementById("player-profile-plays");
const playsTitle = document.getElementById("player-profile-plays-title");

const signedPp = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 1, maximumFractionDigits: 1, signDisplay: "exceptZero",
});
const signedEpa = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 3, maximumFractionDigits: 3, signDisplay: "exceptZero",
});

let profilesRequest;
let currentPlayer;
let launcher;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function number(value, formatter) {
  return Number.isFinite(value) ? formatter(value) : "—";
}

const grade = (value) => number(value, (v) => v.toFixed(1));
const percent = (value) => number(value, (v) => `${(100 * v).toFixed(1)}%`);
const points = (value) => number(value, (v) => `${signedPp.format(v)} pp`);
const epa = (value) => number(value, (v) => signedEpa.format(v));
const time = (value) => number(value, (v) => `${v.toFixed(1)} s`);

function loadProfiles() {
  if (!profilesRequest) {
    profilesRequest = fetch("./data/player-profiles.json")
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((data) => {
        if (!data.players || typeof data.players !== "object") throw new Error("Player profile data missing");
        return data.players;
      })
      .catch((error) => {
        profilesRequest = undefined;
        throw error;
      });
  }
  return profilesRequest;
}

function summaryCard(label, value, sample) {
  const card = element("div", undefined, "profile-score");
  const detail = element("dd", value);
  card.append(element("dt", label), detail);
  if (value === "—") {
    detail.append(element("small", "No qualifying sample"));
  } else {
    const provisional = sample < 30 ? " · provisional" : "";
    detail.append(element("small", `${sample} eligible routes${provisional}`));
  }
  return card;
}

function renderScores(player) {
  scores.replaceChildren(
    summaryCard("Onset SVA", grade(player.onset), player.onsetRoutes),
    summaryCard("Full-window SVA", grade(player.temporal), player.temporalRoutes),
    summaryCard("Catch probability", percent(player.catchProbability), player.onsetRoutes),
    summaryCard("Expected EPA per target", epa(player.expectedEpa), player.onsetRoutes),
  );
}

function breakdownCard(title, rows, note, wide = false) {
  const card = element("section", undefined, `profile-detail${wide ? " profile-detail-wide" : ""}`);
  const table = element("dl", undefined, "profile-detail-grid");
  for (const [label, value] of rows) {
    const row = element("div");
    row.append(element("dt", label), element("dd", value));
    table.append(row);
  }
  card.append(element("h3", title), table, element("p", note, "score-note"));
  return card;
}

function renderBreakdown(player) {
  breakdown.replaceChildren(
    breakdownCard("Onset score components", [
      ["Catch probability lift", points(player.onsetCatchLiftPp)],
      ["EPA lift at the same catch probability", epa(player.onsetEpaLift)],
    ], "Both lifts are compared with similar in-range opportunities at pressure onset and weighted equally after standardization."),
    breakdownCard("Full-window score components", [
      ["Catch probability lift", points(player.temporalCatchLiftPp)],
      ["EPA lift at the same catch probability", epa(player.temporalEpaLift)],
      ["Early-weighted opening lift", points(player.temporalAvailabilityLiftPp)],
    ], "Opening frequency gets half the weight and favors earlier openings. Catch and EPA each get a quarter and are estimated at first eligibility."),
    breakdownCard("Opening frequency and timing", [
      ["Mean open share", percent(player.openShare)],
      ["Early-weighted open share", percent(player.weightedOpenShare)],
      ["Routes with an opening", percent(player.openedRouteShare)],
      ["Median first opening", time(player.medianFirstOpen)],
      ["Longest opening streak", time(player.longestOpen)],
    ], "Open share uses eligible frames within 20 yards. First openings are timed from pressure onset.", true),
  );
}

function replayCard(play, player) {
  const item = element("li", undefined, "profile-play");
  const heading = `Week ${play.week} · ${play.team} vs ${play.opponent} · Q${play.quarter} ${play.clock}`;
  const summary = `Down ${play.down} · ${play.toGo} to go · First open ${time(play.firstOpen)} · Open ${percent(play.openShare)} · Longest opening ${time(play.longestOpen)} · Play EPA ${epa(play.epa)}`;
  const link = element("a", "Replay", "profile-replay-link");
  const query = new URLSearchParams({
    team: play.team, game: play.gameId, play: play.playId, player: player.id, open: "1",
  });
  link.href = `./?${query}#replay`;
  link.setAttribute("aria-label", `Replay ${player.name} opening in ${heading}, play ${play.playId}`);
  const body = element("div", undefined, "profile-play-body");
  body.append(element("p", heading, "profile-play-heading"), element("p", play.description, "play-description"),
    element("p", summary, "profile-play-facts"));
  item.append(body, link);
  return item;
}

export async function openPlayerProfile(player, button) {
  currentPlayer = player.id;
  launcher = button;
  title.textContent = player.name;
  teams.textContent = [player.position, ...player.teams].join(" · ");
  renderScores(player);
  breakdown.replaceChildren();
  plays.replaceChildren();
  playsTitle.textContent = "Open under pressure";
  status.textContent = "Loading score breakdown and replays…";
  delete status.dataset.state;
  if (!dialog.open) dialog.showModal();
  dialog.scrollTop = 0;

  try {
    const profiles = await loadProfiles();
    if (currentPlayer !== player.id || !dialog.open) return;
    const profile = profiles[String(player.id)];
    if (!profile) throw new Error("Player profile missing");

    renderBreakdown(profile);
    const count = profile.plays.length;
    playsTitle.textContent = `Open under pressure · ${count} ${count === 1 ? "play" : "plays"}`;
    if (count) {
      plays.replaceChildren(...profile.plays.map((play) => replayCard(play, player)));
      status.textContent = "";
    } else {
      status.textContent = "No recorded openings for this player in the pressure sample.";
    }
  } catch {
    if (currentPlayer !== player.id || !dialog.open) return;
    status.textContent = "The breakdown and replays could not load. Close and reopen the profile to try again.";
    status.dataset.state = "error";
  }
}

dialog.addEventListener("close", () => {
  if (dialog.open) return;
  currentPlayer = undefined;
  launcher?.focus();
  launcher = undefined;
});
