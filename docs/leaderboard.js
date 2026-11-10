import { openPlayerProfile } from "./player-profile.js";
import { openComparison } from "./player-comparison.js";

const epaFormat = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 3,
  maximumFractionDigits: 3,
  signDisplay: "exceptZero",
});
const metrics = {
  onset: { name: "onset SVA", routes: "onsetRoutes", format: (value) => value.toFixed(1) },
  temporal: { name: "full-window SVA", routes: "temporalRoutes", format: (value) => value.toFixed(1) },
  catchProbability: { name: "catch probability", routes: "onsetRoutes", format: (value) => `${(value * 100).toFixed(1)}%` },
  expectedEpa: {
    name: "expected EPA",
    routes: "onsetRoutes",
    format: (value) => epaFormat.format(value),
  },
};

const teamFilter = document.getElementById("leaderboard-team");
const positionFilter = document.getElementById("leaderboard-position");
const searchFilter = document.getElementById("leaderboard-search");
const status = document.getElementById("leaderboard-status");
const body = document.getElementById("leaderboard-rows");
const sortButtons = document.querySelectorAll("#leaderboard [data-sort]");
const selection = document.getElementById("leaderboard-selection");
const compareButton = document.getElementById("leaderboard-compare");
const clearButton = document.getElementById("leaderboard-clear");
const comparisonStatus = document.getElementById("leaderboard-compare-status");
const numberFormat = new Intl.NumberFormat("en-US");
const initialHash = location.hash;
const selectedPlayers = new Map();

let players = [];
let sortKey = "onset";
let ascending = false;

function searchKey(name) {
  return name.normalize("NFD").toLowerCase().replace(/[\p{M}\p{P}\s]/gu, "");
}

function updateComparisonSelection() {
  const count = selectedPlayers.size;
  compareButton.disabled = count !== 2;
  clearButton.disabled = count === 0;
  comparisonStatus.textContent = [
    "Check the boxes beside two players to compare them.",
    "1 of 2 players selected. Choose one more player.",
    "2 of 2 players selected. Use Compare players to see their breakdowns.",
  ][count];

  for (const checkbox of body.querySelectorAll("[data-compare-player]")) {
    checkbox.checked = selectedPlayers.has(checkbox.dataset.comparePlayer);
    checkbox.disabled = count === 2 && !checkbox.checked;
  }

  const chips = [...selectedPlayers].map(([id, player], index) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "leaderboard-selection-chip";
    chip.textContent = player.name;
    chip.setAttribute("aria-label", `Remove ${player.name} from comparison`);
    const icon = document.createElement("span");
    icon.textContent = "×";
    icon.setAttribute("aria-hidden", "true");
    chip.append(icon);
    chip.addEventListener("click", () => {
      selectedPlayers.delete(id);
      updateComparisonSelection();
      (selection.children[index] ?? selection.lastElementChild ?? searchFilter).focus();
    });
    return chip;
  });
  selection.replaceChildren(...chips);
}

function comparisonOption(player) {
  const option = document.createElement("label");
  option.className = "leaderboard-compare-option";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.dataset.comparePlayer = String(player.id);
  checkbox.checked = selectedPlayers.has(checkbox.dataset.comparePlayer);
  checkbox.disabled = selectedPlayers.size === 2 && !checkbox.checked;
  checkbox.setAttribute("aria-label", `Compare ${player.name}`);
  checkbox.addEventListener("change", () => {
    const id = checkbox.dataset.comparePlayer;
    if (checkbox.checked && selectedPlayers.size < 2) selectedPlayers.set(id, player);
    if (!checkbox.checked) selectedPlayers.delete(id);
    updateComparisonSelection();
  });
  option.append(checkbox);
  return option;
}

function messageRow(message) {
  const row = document.createElement("tr");
  const cell = row.insertCell();
  cell.colSpan = 8;
  cell.className = "empty-lineup";
  cell.textContent = message;
  body.replaceChildren(row);
}

function metricCell(row, player, key) {
  const cell = row.insertCell();
  const value = player[key];
  if (!Number.isFinite(value)) {
    cell.textContent = "—";
    cell.setAttribute("aria-label", "No qualifying score");
    return;
  }

  const metric = metrics[key];
  const sample = player[metric.routes];
  const formatted = metric.format(value);
  cell.textContent = formatted;
  cell.title = `${numberFormat.format(sample)} eligible routes`;
  if (sample < 30) {
    const marker = document.createElement("span");
    marker.className = "provisional-score";
    marker.textContent = "*";
    marker.setAttribute("aria-hidden", "true");
    cell.append(marker);
    cell.setAttribute("aria-label", `${formatted}, provisional with ${sample} eligible routes`);
  }
}

function comparePlayers(a, b) {
  const left = a[sortKey];
  const right = b[sortKey];
  const hasLeft = Number.isFinite(left);
  const hasRight = Number.isFinite(right);
  if (hasLeft !== hasRight) return hasLeft ? -1 : 1;
  if (hasLeft && left !== right) return ascending ? left - right : right - left;
  return a.name.localeCompare(b.name) || a.id - b.id;
}

function updateSortHeaders() {
  for (const button of sortButtons) {
    const header = button.closest("th");
    const active = button.dataset.sort === sortKey;
    if (active) header.setAttribute("aria-sort", ascending ? "ascending" : "descending");
    else header.removeAttribute("aria-sort");
    const direction = active && !ascending ? "ascending" : "descending";
    button.setAttribute("aria-label", `Sort by ${metrics[button.dataset.sort].name} ${direction}`);
  }
}

function render() {
  const team = teamFilter.value;
  const position = positionFilter.value;
  const query = searchKey(searchFilter.value);
  const focusedPlayer = document.activeElement?.dataset.comparePlayer;
  const filtered = players.filter((player) => (!team || player.teams.includes(team))
    && (!position || player.position === position)
    && player.searchName.includes(query)).sort(comparePlayers);

  updateSortHeaders();
  status.textContent = `${numberFormat.format(filtered.length)} of ${numberFormat.format(players.length)} players · Sorted by ${metrics[sortKey].name}, ${ascending ? "lowest" : "highest"} first`;
  if (!filtered.length) {
    messageRow("No players match your search and filters.");
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const [index, player] of filtered.entries()) {
    const row = document.createElement("tr");
    row.insertCell().textContent = index + 1;
    const name = document.createElement("th");
    name.scope = "row";
    name.setAttribute("aria-label", player.name);
    const nameContents = document.createElement("div");
    nameContents.className = "leaderboard-player-cell";
    const profile = document.createElement("button");
    profile.type = "button";
    profile.className = "leaderboard-player";
    profile.setAttribute("aria-haspopup", "dialog");
    profile.setAttribute("aria-controls", "player-profile");
    profile.textContent = player.name;
    profile.addEventListener("click", () => openPlayerProfile(player, profile));
    nameContents.append(comparisonOption(player), profile);
    name.append(nameContents);
    row.append(name);
    row.insertCell().textContent = player.teams.join(", ") || "—";
    row.insertCell().textContent = player.position;
    for (const key of Object.keys(metrics)) metricCell(row, player, key);
    fragment.append(row);
  }
  body.replaceChildren(fragment);
  if (focusedPlayer) body.querySelector(`[data-compare-player="${focusedPlayer}"]`)?.focus();
}

async function initLeaderboard() {
  try {
    const response = await fetch("./data/leaderboard.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    players = data.players.map((player) => ({ ...player, searchName: searchKey(player.name) }));

    for (const { code, name } of data.teams) teamFilter.add(new Option(name, code));
    for (const position of [...new Set(players.map((player) => player.position))].sort()) {
      positionFilter.add(new Option(position, position));
    }
    teamFilter.disabled = false;
    positionFilter.disabled = false;
    searchFilter.disabled = false;
    teamFilter.addEventListener("change", render);
    positionFilter.addEventListener("change", render);
    searchFilter.addEventListener("input", render);
    compareButton.addEventListener("click", () => openComparison([...selectedPlayers.values()], compareButton));
    clearButton.addEventListener("click", () => {
      selectedPlayers.clear();
      updateComparisonSelection();
      searchFilter.focus();
    });
    for (const button of sortButtons) {
      button.disabled = false;
      button.addEventListener("click", () => {
        ascending = button.dataset.sort === sortKey && !ascending;
        sortKey = button.dataset.sort;
        render();
      });
    }
    updateComparisonSelection();
    render();
    if (initialHash === "#leaderboard" && location.hash === initialHash) {
      document.getElementById("leaderboard").scrollIntoView();
    }
  } catch {
    status.textContent = "The leaderboard data could not be loaded.";
    status.dataset.state = "error";
    messageRow("Reload the page to try again.");
  }
}

initLeaderboard();
