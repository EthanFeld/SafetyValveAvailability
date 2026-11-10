const WIDTH = 120;
const HEIGHT = 160 / 3;
const GOLD = "#f9ca67";
const AQUA = "#81f5df";

export function frameAt(frames, time) {
  let left = 0;
  let right = frames.length - 1;
  while (left < right) {
    const mid = Math.ceil((left + right) / 2);
    if (frames[mid].t <= time + 1e-8) left = mid;
    else right = mid - 1;
  }
  return left;
}

function pointAt(values, index) {
  const x = values?.[index];
  const y = values?.[index + 1];
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}

function interpolate(a, b, ratio) {
  if (!a || !b) return a;
  return [a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio];
}

function sample(play, time) {
  const index = frameAt(play.frames, time);
  const frame = play.frames[index];
  const next = play.frames[index + 1] || frame;
  const span = next.t - frame.t;
  const ratio = span > 0 ? Math.max(0, Math.min(1, (time - frame.t) / span)) : 0;
  return {
    frame,
    index,
    players: play.players.map((p, i) => ({
      ...p,
      point: interpolate(pointAt(frame.xy, i * 2), pointAt(next.xy, i * 2), ratio),
    })),
    ball: interpolate(pointAt(frame.ball, 0), pointAt(next.ball, 0), ratio),
  };
}

function line(ctx, x1, y1, x2, y2) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function circle(ctx, point, radius, fill, stroke, width = 0.2, dash = []) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(point[0], point[1], radius, 0, Math.PI * 2);
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.stroke();
  }
  ctx.restore();
}

function drawTurf(ctx, play) {
  ctx.fillStyle = "#113a31";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  for (let x = 10; x < 110; x += 5) {
    ctx.fillStyle = x % 10 === 0 ? "#1a4b3d" : "#174437";
    ctx.fillRect(x, 0, 5, HEIGHT);
  }
  ctx.fillStyle = "#142e2d";
  ctx.fillRect(0, 0, 10, HEIGHT);
  ctx.fillRect(110, 0, 10, HEIGHT);

  ctx.lineWidth = 0.15;
  ctx.strokeStyle = "#a8c9ad62";
  for (let x = 10; x <= 110; x += 5) line(ctx, x, 0, x, HEIGHT);
  ctx.strokeStyle = "#b1d0b368";
  ctx.lineWidth = 0.1;
  for (let x = 11; x < 110; x += 1) {
    if (x % 5 === 0) continue;
    for (const y of [1.0, 23.58, 29.75, HEIGHT - 1.0]) line(ctx, x, y - 0.34, x, y + 0.34);
  }
  ctx.strokeStyle = "#bfd7c0a8";
  ctx.lineWidth = 0.22;
  ctx.strokeRect(0.16, 0.16, WIDTH - 0.32, HEIGHT - 0.32);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "600 1.85px system-ui, sans-serif";
  ctx.fillStyle = "#d5e7d4a0";
  for (let x = 20; x <= 100; x += 10) {
    const yard = Math.min(x - 10, 110 - x);
    ctx.fillText(yard, x, 7.0);
    ctx.fillText(yard, x, HEIGHT - 7.0);
  }
  if (!play) return;
  ctx.fillStyle = "#d1e9d294";
  ctx.font = "700 3.25px system-ui, sans-serif";
  for (const [x, label, rotation] of [[5, play.defense, -Math.PI / 2], [115, play.offense, Math.PI / 2]]) {
    ctx.save();
    ctx.translate(x, HEIGHT / 2);
    ctx.rotate(rotation);
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }
  ctx.lineWidth = 0.32;
  ctx.strokeStyle = "#8eb8fbde";
  if (Number.isFinite(play.los)) line(ctx, play.los, 0, play.los, HEIGHT);
  ctx.strokeStyle = "#f5cf7199";
  ctx.setLineDash([0.7, 0.6]);
  if (Number.isFinite(play.firstDown)) line(ctx, play.firstDown, 0, play.firstDown, HEIGHT);
  ctx.setLineDash([]);
}

function drawTrail(ctx, play, index, id) {
  const i = play.players.findIndex((player) => player.id === id);
  if (i === -1) return;
  ctx.save();
  ctx.lineWidth = 0.3;
  ctx.lineCap = "round";
  ctx.strokeStyle = "#f9ca679c";
  let previous = null;
  for (let j = Math.max(0, index - 12); j <= index; j += 1) {
    const point = pointAt(play.frames[j].xy, i * 2);
    if (point && previous) line(ctx, ...previous, ...point);
    previous = point;
  }
  ctx.restore();
}

function labelPlayer(ctx, point, label, color, below = false) {
  ctx.font = "700 1.64px system-ui, sans-serif";
  const w = ctx.measureText(label).width + 1.7;
  const x = Math.max(w / 2 + 0.4, Math.min(WIDTH - w / 2 - 0.4, point[0]));
  const y = Math.max(1.55, Math.min(HEIGHT - 1.55, point[1] + (below || point[1] < 6 ? 4.0 : -4.0)));
  ctx.fillStyle = "#0a1d1aed";
  ctx.fillRect(x - w / 2, y - 1.25, w, 2.5);
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x, y + 0.04);
}

export function renderReplay(canvas, play, time, { valve, selected, nameOf } = {}) {
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(rect.width * ratio);
  const height = Math.round(rect.height * ratio);
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(width / WIDTH, 0, 0, height / HEIGHT, 0, 0);
  drawTurf(ctx, play);
  if (!play?.frames.length) return;

  const { frame, players, ball, index } = sample(play, time);
  if (selected || valve) drawTrail(ctx, play, index, selected || valve);
  for (const player of players) {
    if (!player.point) continue;
    const offense = player.team === play.offense;
    const qb = player.id === play.qb;
    const marked = player.id === selected;
    const isValve = player.id === valve;
    const best = frame.pressure && player.id === frame.best;
    if (qb && frame.pressure) circle(ctx, player.point, 2.45, "#f38b4536", "#ff9c64", 0.29);
    circle(ctx, player.point, qb ? 1.48 : 1.33, offense ? "#67c8fd" : "#f28378", "#0f2523", 0.22);
    if (best) circle(ctx, player.point, isValve ? 2.98 : 2.38, null, AQUA, 0.31, [0.65, 0.32]);
    if (isValve) circle(ctx, player.point, 2.1, null, GOLD, 0.40);
    if (marked && !isValve && !best) circle(ctx, player.point, 2.06, null, "#c0dafa", 0.32);
    ctx.fillStyle = "#112324";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "800 1.48px system-ui, sans-serif";
    if (player.number != null && player.number !== "") ctx.fillText(player.number, player.point[0], player.point[1] + 0.08);
  }
  if (ball) circle(ctx, ball, 0.67, "#f9e0a8", "#543a2c", 0.18);

  const valvePlayer = players.find((p) => p.id === valve && p.point);
  const bestPlayer = frame.pressure ? players.find((p) => p.id === frame.best && p.point) : null;
  if (valvePlayer) labelPlayer(ctx, valvePlayer.point, nameOf?.(valvePlayer.id) || "Safety valve", GOLD);
  if (bestPlayer && bestPlayer.id !== valve) {
    const close = valvePlayer && Math.abs(valvePlayer.point[1] - bestPlayer.point[1]) < 6;
    labelPlayer(ctx, bestPlayer.point, nameOf?.(bestPlayer.id) || "Best outlet", AQUA, close);
  }
}

export function pickPlayer(canvas, play, time, event) {
  if (!play?.frames.length) return null;
  const bounds = canvas.getBoundingClientRect();
  const x = WIDTH * (event.clientX - bounds.left) / bounds.width;
  const y = HEIGHT * (event.clientY - bounds.top) / bounds.height;
  let nearest = null;
  let bestDistance = 2.5;
  for (const player of sample(play, time).players) {
    if (!player.point) continue;
    const distance = Math.hypot(x - player.point[0], y - player.point[1]);
    if (distance < bestDistance) {
      nearest = player.id;
      bestDistance = distance;
    }
  }
  return nearest;
}
