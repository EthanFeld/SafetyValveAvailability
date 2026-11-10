function forwardLegacyFilmLink() {
  const { search, hash } = location;
  const params = new URLSearchParams(search);
  const filmSections = ["#replay", "#lineup", "#leaderboard"];
  const replayParams = ["team", "game", "play", "model", "player", "open"];

  if (filmSections.includes(hash) || replayParams.some((key) => params.has(key))) {
    location.replace(`./film-room.html${search}${hash}`);
  }
}

forwardLegacyFilmLink();
window.addEventListener("hashchange", forwardLegacyFilmLink);
