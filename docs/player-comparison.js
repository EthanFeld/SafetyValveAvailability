import { epa, grade, loadProfiles, percent, points, time } from "./player-profile.js";

const dialog = document.getElementById("player-comparison");
const content = dialog.querySelector(".comparison-content");
const status = document.getElementById("comparison-status");
const head = document.getElementById("comparison-head");
const body = document.getElementById("comparison-rows");

const routeCount = new Intl.NumberFormat("en-US");

const headline = [
  ["Onset SVA", "onset", grade, "onsetRoutes"],
  ["Full-window SVA", "temporal", grade, "temporalRoutes"],
  ["Catch probability", "catchProbability", percent, "onsetRoutes"],
  ["Expected EPA per target", "expectedEpa", epa, "onsetRoutes"],
];

const breakdown = [
  ["Onset score components", [
    ["Catch probability lift", "onsetCatchLiftPp", points],
    ["EPA lift at the same catch probability", "onsetEpaLift", epa],
  ]],
  ["Full-window score components", [
    ["Catch probability lift", "temporalCatchLiftPp", points],
    ["EPA lift at the same catch probability", "temporalEpaLift", epa],
    ["Early-weighted opening lift", "temporalAvailabilityLiftPp", points],
  ]],
  ["Opening frequency and timing", [
    ["Mean open share", "openShare", percent],
    ["Early-weighted open share", "weightedOpenShare", percent],
    ["Routes with an opening", "openedRouteShare", percent],
    ["Median first opening", "medianFirstOpen", time],
    ["Longest opening streak", "longestOpen", time],
  ]],
];

let currentComparison;
let launcher;

function element(tag, content = "") {
  const node = document.createElement(tag);
  node.textContent = content;
  return node;
}

function header(text, scope) {
  const cell = element("th", text);
  cell.scope = scope;
  return cell;
}

function renderHead(players) {
  const row = element("tr");
  row.append(header("Metric", "col"));
  for (const player of players) {
    const cell = header(player.name, "col");
    cell.append(element("small", [player.position, ...player.teams].join(" · ")));
    row.append(cell);
  }
  head.replaceChildren(row);
}

function appendSection(label, metrics, players) {
  const fragment = document.createDocumentFragment();
  const section = element("tr");
  const sectionHeader = header(label, "row");
  section.className = "comparison-section";
  sectionHeader.colSpan = 3;
  section.append(sectionHeader);
  fragment.append(section);

  for (const [metric, key, format, routes] of metrics) {
    const row = element("tr");
    row.append(header(metric, "row"));
    for (const player of players) {
      const value = player?.[key];
      const cell = element("td", format(value));
      if (routes) {
        const sample = player[routes];
        const hasValue = Number.isFinite(value);
        const note = hasValue ? (sample < 30 ? " · provisional" : "") : " · no qualifying score";
        cell.append(element("small", `${routeCount.format(sample)} eligible routes${note}`));
      }
      row.append(cell);
    }
    fragment.append(row);
  }

  body.append(fragment);
}

export async function openComparison([left, right], button) {
  const players = [left, right];
  currentComparison = players;
  launcher = button;
  renderHead(players);
  body.replaceChildren();
  appendSection("Leaderboard scores", headline, players);
  status.textContent = "Loading score breakdowns…";
  delete status.dataset.state;
  if (!dialog.open) dialog.showModal();
  content.scrollTop = 0;
  content.scrollLeft = 0;

  try {
    const profiles = await loadProfiles();
    if (currentComparison !== players || !dialog.open) return;
    const details = players.map((player) => profiles[String(player.id)]);
    if (!details.some(Boolean)) throw new Error("Comparison profiles missing");
    for (const [label, metrics] of breakdown) appendSection(label, metrics, details);
    const missing = details.findIndex((detail) => !detail);
    if (missing === -1) {
      status.textContent = "";
    } else {
      status.textContent = `The score breakdown for ${players[missing].name} is unavailable.`;
      status.dataset.state = "error";
    }
  } catch {
    if (currentComparison !== players || !dialog.open) return;
    status.textContent = "The score breakdowns could not load. Close and reopen the comparison to try again.";
    status.dataset.state = "error";
  }
}

dialog.addEventListener("close", () => {
  if (dialog.open) return;
  currentComparison = undefined;
  launcher?.focus();
  launcher = undefined;
});
