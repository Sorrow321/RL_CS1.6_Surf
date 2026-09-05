/* RL_Surf viewer — map mesh + trajectory playback + waypoint editor.
 * Plain three.js (r147 UMD, vendored), no build step. See docs/04-visualizer.md.
 *
 * COORDINATES (the only place this is explained): GoldSrc is Z-up right-handed,
 * three.js is Y-up. We convert every incoming GoldSrc point/vector ONCE at load
 * time via (x,y,z) -> (x, z, -y), and convert back on waypoint export via
 * (X,Y,Z) -> (X, -Z, Y). Everything in the scene graph is three.js frame;
 * everything in files (mesh JSON, traj JSONL, waypoints JSON) is GoldSrc frame.
 */
'use strict';

function g2t(x, y, z) { return new THREE.Vector3(x, z, -y); }
function t2g(v) { return [v.x, -v.z, v.y]; }

// ---------------------------------------------------------------- scene setup
var container = document.getElementById('view');
var renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

var scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161a);

var camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 1, 60000);
camera.position.set(1500, 1200, 1500);

var controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.AmbientLight(0xffffff, 0.45));
var sun1 = new THREE.DirectionalLight(0xffffff, 0.75);
sun1.position.set(0.4, 1, 0.6);
scene.add(sun1);
var sun2 = new THREE.DirectionalLight(0x99aacc, 0.3);
sun2.position.set(-0.6, 0.4, -0.5);
scene.add(sun2);

window.addEventListener('resize', function () {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ------------------------------------------------------------------ map state
var mapName = null;
var mapGroup = null;      // everything from the mesh JSON
var worldMesh = null;     // raycast target for the waypoint editor
var layerGroups = {};     // classname -> THREE.Group (brush overlays, markers)

var BRUSH_STYLE = {       // classname -> [color, opacity]
  trigger_teleport: [0x3f6cff, 0.25],
  trigger_push:     [0xffd23d, 0.35],
  trigger_hurt:     [0xff4040, 0.35],
  func_water:       [0x35c8d8, 0.30],
  func_illusionary: [0xb070ff, 0.15]
};
var BRUSH_DEFAULT = [0x9aa0aa, 0.20];

// World face tint by GoldSrc normal.z — surf ramps must pop (docs/04).
function faceColor(nz) {
  if (nz >= 0.7)    return [0.63, 0.72, 0.58];  // walkable: light grey-green
  if (nz > 0.001)   return [0.95, 0.63, 0.18];  // 0 < nz < 0.7: SURF RAMP amber
  if (nz > -0.001)  return [0.50, 0.51, 0.55];  // vertical wall: mid grey
  return [0.30, 0.31, 0.34];                    // ceiling: darker
}

function buildGeometry(positions, normals, indices, colorByNormal) {
  var n = positions.length / 3;
  var pos = new Float32Array(n * 3);
  var nor = new Float32Array(n * 3);
  var col = colorByNormal ? new Float32Array(n * 3) : null;
  for (var i = 0; i < n; i++) {
    var gx = positions[3 * i], gy = positions[3 * i + 1], gz = positions[3 * i + 2];
    var nx = normals[3 * i], ny = normals[3 * i + 1], nz = normals[3 * i + 2];
    pos[3 * i] = gx;  pos[3 * i + 1] = gz;  pos[3 * i + 2] = -gy;
    nor[3 * i] = nx;  nor[3 * i + 1] = nz;  nor[3 * i + 2] = -ny;
    if (col) {
      var c = faceColor(nz);
      col[3 * i] = c[0]; col[3 * i + 1] = c[1]; col[3 * i + 2] = c[2];
    }
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  if (col) geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setIndex(indices);
  return geo;
}

function getLayer(name) {
  if (!layerGroups[name]) {
    var g = new THREE.Group();
    layerGroups[name] = g;
    mapGroup.add(g);
  }
  return layerGroups[name];
}

function rebuildLayerPanel() {
  var list = document.getElementById('layerList');
  list.innerHTML = '';
  Object.keys(layerGroups).sort().forEach(function (name) {
    var grp = layerGroups[name];
    var label = document.createElement('label');
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = grp.visible;
    cb.addEventListener('change', function () { grp.visible = cb.checked; });
    label.appendChild(cb);
    var sw = document.createElement('span');
    sw.className = 'swatch';
    var style = BRUSH_STYLE[name] || BRUSH_DEFAULT;
    sw.style.background = '#' + new THREE.Color(style[0]).getHexString();
    label.appendChild(sw);
    label.appendChild(document.createTextNode(name));
    list.appendChild(label);
  });
}

function loadMesh(json) {
  if (mapGroup) { scene.remove(mapGroup); }
  mapGroup = new THREE.Group();
  layerGroups = {};
  worldMesh = null;
  scene.add(mapGroup);
  mapName = json.map || 'map';
  document.getElementById('mapName').textContent = mapName;

  // world
  var w = json.world;
  var worldGeo = buildGeometry(w.positions, w.normals, w.indices, true);
  worldMesh = new THREE.Mesh(worldGeo, new THREE.MeshLambertMaterial({
    vertexColors: true, side: THREE.DoubleSide }));
  mapGroup.add(worldMesh);

  // brush entity overlays, translucent, grouped by classname for toggling
  (json.brushes || []).forEach(function (b) {
    var style = BRUSH_STYLE[b.classname] || BRUSH_DEFAULT;
    var geo = buildGeometry(b.positions, b.normals, b.indices, false);
    var mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color: style[0], transparent: true, opacity: style[1],
      depthWrite: false, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2 }));
    mesh.userData = { classname: b.classname, model: b.model,
                      targetname: b.targetname, target: b.target };
    getLayer(b.classname).add(mesh);
  });

  // point markers: spawns = cones along yaw, teleport destinations = rings
  var coneGeo = new THREE.ConeGeometry(10, 28, 10);
  coneGeo.rotateX(Math.PI / 2);            // cone now points +Z (its "forward")
  var ringGeo = new THREE.TorusGeometry(20, 3, 8, 28);
  ringGeo.rotateX(Math.PI / 2);            // ring now lies flat (axis = up)
  (json.markers || []).forEach(function (mk) {
    var o = g2t(mk.origin[0], mk.origin[1], mk.origin[2]);
    var mesh;
    if (mk.classname === 'info_player_start' || mk.classname === 'info_player_deathmatch') {
      var color = mk.classname === 'info_player_start' ? 0x4fd07f : 0xd0894f;
      mesh = new THREE.Mesh(coneGeo, new THREE.MeshBasicMaterial({ color: color }));
      var yawRad = mk.yaw * Math.PI / 180;
      // GoldSrc yaw 0 = +X, 90 = +Y(GoldSrc) = -Z(three)
      var dir = new THREE.Vector3(Math.cos(yawRad), 0, -Math.sin(yawRad));
      mesh.rotation.y = Math.atan2(dir.x, dir.z);
    } else {  // info_teleport_destination / info_target
      mesh = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: 0x66aaff }));
    }
    mesh.position.copy(o);
    mesh.userData = mk;
    getLayer('markers').add(mesh);
  });

  rebuildLayerPanel();

  // frame the camera on model-0 bounds
  if (json.bounds) {
    var box = new THREE.Box3();
    box.expandByPoint(g2t(json.bounds.mins[0], json.bounds.mins[1], json.bounds.mins[2]));
    box.expandByPoint(g2t(json.bounds.maxs[0], json.bounds.maxs[1], json.bounds.maxs[2]));
    var center = box.getCenter(new THREE.Vector3());
    var size = box.getSize(new THREE.Vector3()).length();
    camera.position.copy(center).add(new THREE.Vector3(size * 0.35, size * 0.3, size * 0.35));
    controls.target.copy(center);
    controls.update();
  }
}

// ----------------------------------------------------------- trajectory state
var traj = null;   // { episodes: [{header, ticks[[t,x,y,z,vx,vy,vz,yaw,btn,og,prog,rew]], end}] }
var trailGroup = null;
var curEp = 0;
var playTime = 0;        // fractional tick index within current episode
var playing = false;
var playSpeed = 1;

// ---------------------------------------------------------------- zone clock
// The leaderboard's clock: it starts when the player crosses the start
// curtain (maps/<map>.zones.json "start", a thin trigger a few hundred units
// in front of spawn) and stops at the finish curtain ("end"). Our recordings
// run on the spawn clock, which is ~1 s longer. Crossings are found on the
// position track with sub-tick interpolation, like tools/demo/compare_wr.py.
var zoneData = null;              // {start:{mins,maxs}, end:{mins,maxs}}
var WR_TIMES = { surf_src_cannonball: 68.60 };   // human record, zone clock
function fmtClock(sec) {
  if (sec == null || !isFinite(sec)) return '-';
  var m = Math.floor(sec / 60), r = sec - 60 * m;
  return m + ':' + (r < 10 ? '0' : '') + r.toFixed(3);
}
function _crossingTick(ticks, zone, tickMs) {
  // first tick pair whose y straddles the curtain plane while x/z lie inside
  // the (slightly padded) curtain rectangle; returns a fractional tick index
  if (!zone) return null;
  var mn = zone.mins, mx = zone.maxs;
  var yc = 0.5 * (mn[1] + mx[1]), pad = 64;
  for (var i = 1; i < ticks.length; i++) {
    var a = ticks[i - 1], b = ticks[i];
    var y0 = a[2] - yc, y1 = b[2] - yc;
    if ((y0 < 0 && y1 >= 0) || (y0 > 0 && y1 <= 0)) {
      var f = y0 / (y0 - y1);
      var x = a[1] + f * (b[1] - a[1]), z = a[3] + f * (b[3] - a[3]);
      if (x >= mn[0] - pad && x <= mx[0] + pad && z >= mn[2] - pad && z <= mx[2] + pad) return i - 1 + f;
    }
  }
  return null;
}
function epZone(ep) {
  if (!ep || !zoneData) return null;
  if (ep.zone === undefined) {
    var tickMs = ep.header.tick_ms || 10;
    var t0 = _crossingTick(ep.ticks, zoneData.start, tickMs);
    var t1 = null;
    if (t0 != null) {
      var rest = ep.ticks.slice(Math.ceil(t0));
      var t1r = _crossingTick(rest, zoneData.end, tickMs);
      if (t1r != null) t1 = Math.ceil(t0) + t1r;
    }
    ep.zone = { t0: t0, t1: t1, tickMs: tickMs,
                time: (t0 != null && t1 != null) ? (t1 - t0) * tickMs / 1000 : null };
  }
  return ep.zone;
}
function zoneClockAt(ep, playTick) {
  // running zone-clock seconds at a (fractional) tick of the episode
  var z = epZone(ep);
  if (!z || z.t0 == null) return { running: null, final: null, state: 'no-zones' };
  if (playTick < z.t0) return { running: 0, final: z.time, state: 'pre' };
  if (z.t1 != null && playTick >= z.t1) return { running: z.time, final: z.time, state: 'done' };
  return { running: (playTick - z.t0) * z.tickMs / 1000, final: z.time, state: 'run' };
}

var playerGroup = new THREE.Group();
playerGroup.visible = false;
scene.add(playerGroup);
// player collision hull: 32x32x72 GoldSrc units, origin at hull center
var hullBox = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(32, 72, 32)),
  new THREE.LineBasicMaterial({ color: 0xffffff }));
playerGroup.add(hullBox);
var velArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(0, 0, 0), 50, 0x35d8a0, 14, 8);
playerGroup.add(velArrow);

function parseTraj(text) {
  var episodes = [];
  var cur = null;
  var lines = text.split('\n');
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i].trim();
    if (!line) continue;
    var obj;
    try { obj = JSON.parse(line); } catch (e) { continue; }
    if (Array.isArray(obj)) {
      if (!cur) { cur = { header: { tick_ms: 10 }, ticks: [], end: null }; episodes.push(cur); }
      cur.ticks.push(obj);
    } else if (obj && obj.end !== undefined) {
      if (cur) cur.end = obj;
      cur = null;
    } else if (obj) {                      // header (restarts each episode)
      cur = { header: obj, ticks: [], end: null };
      episodes.push(cur);
    }
  }
  return episodes.filter(function (e) { return e.ticks.length > 0; });
}

// training-reward reconstruction: both reward families are pure functions of
// the position track, so the replay can show exactly what the trainer paid.
// Mirrors train_fast.episode_stats / surfgym.rewards incl. the >50u teleport
// filter (a teleport tick pays nothing under either reward).
var runCfg = null;      // run.json "config" of the deep-linked run, if any
var trajStep = null;    // global step parsed from traj_<step>.jsonl, if any

function computeEpRewards(ep) {
  var n = ep.ticks.length;
  var yaw0 = ep.ticks[0][7] * Math.PI / 180;
  var cx = Math.cos(yaw0), sx = Math.sin(yaw0);
  var rewF = new Float32Array(n), rewP = new Float32Array(n);
  var cum = 0, best = 0, path = 0;
  for (var i = 1; i < n; i++) {
    var dx = ep.ticks[i][1] - ep.ticks[i - 1][1];
    var dy = ep.ticks[i][2] - ep.ticks[i - 1][2];
    var d = Math.hypot(dx, dy);
    if (d <= 50) {                              // teleport filter
      path += d;
      cum += dx * cx + dy * sx;
      if (cum > best) best = cum;
    }
    rewF[i] = 0.01 * best;
    rewP[i] = 0.01 * path;
  }
  ep.rewF = rewF; ep.rewP = rewP;
  ep.stats = { fwdMax: best, path: path };
}

// curriculum weight at this recording's step: 0 = pure forward, 1 = pure path,
// null = unknown (no run.json / unparseable filename) -> show both components
function blendW() {
  if (!runCfg) return null;
  if (runCfg.reward === 'forward') return 0;
  if (runCfg.reward === 'path') return 1;
  if (runCfg.reward === 'blend' && runCfg.blend && trajStep !== null) {
    var t0 = runCfg.blend[0], t1 = runCfg.blend[1];
    return Math.min(1, Math.max(0, (trajStep - t0) / (t1 - t0)));
  }
  return null;
}

function epRewardAt(ep, i) {
  var w = blendW();
  if (w === null || !ep.rewF) return null;
  var last = ep.ticks.length - 1;
  var mix = function (j) { return (1 - w) * ep.rewF[j] + w * ep.rewP[j]; };
  return { cur: mix(Math.min(Math.max(i, 0), last)), total: mix(last) };
}

function fmtBoth(ep) {
  // fallback when the reward mode is unknown or not reconstructable in JS
  // (race: the geodesic field lives server-side) — show DISTANCES, clearly
  // unit-labeled, so nobody reads them as training reward
  var last = ep.ticks.length - 1;
  return 'fwd ' + Math.round(ep.rewF[last] * 100) + 'u · path ' +
         Math.round(ep.rewP[last] * 100) + 'u';
}

function updateRewardAvg() {
  var el = document.getElementById('hudRewAvg');
  if (!traj) { el.textContent = '-'; return; }
  var w = blendW(), eps = traj.episodes;
  if (w === null) {
    var f = 0, p = 0;
    eps.forEach(function (e) {
      f += e.rewF[e.ticks.length - 1]; p += e.rewP[e.ticks.length - 1];
    });
    el.textContent = 'fwd ' + Math.round(100 * f / eps.length) + 'u · path ' +
                     Math.round(100 * p / eps.length) + 'u' +
                     (runCfg && runCfg.reward === 'race'
                       ? ' (distances — race reward is on the dashboard)' : '');
  } else {
    var s = 0;
    eps.forEach(function (e) { s += epRewardAt(e, e.ticks.length - 1).total; });
    el.textContent = (s / eps.length).toFixed(1);
  }
}

function loadTraj(text) {
  var episodes = parseTraj(text);
  if (!episodes.length) { console.warn('trajectory file had no ticks'); return; }
  episodes.forEach(computeEpRewards);
  traj = { episodes: episodes };
  updateRewardAvg();
  if (goalGroup) { scene.remove(goalGroup); goalGroup = null; }
  if (trailGroup) scene.remove(trailGroup);
  trailGroup = new THREE.Group();
  scene.add(trailGroup);

  // persistent per-episode trail, vertex-colored by horizontal speed (0=blue, 2000=red)
  episodes.forEach(function (ep) {
    var n = ep.ticks.length;
    var pos = new Float32Array(n * 3);
    var col = new Float32Array(n * 3);
    var c = new THREE.Color();
    for (var i = 0; i < n; i++) {
      var tk = ep.ticks[i];
      pos[3 * i] = tk[1];  pos[3 * i + 1] = tk[3];  pos[3 * i + 2] = -tk[2];
      var speed = Math.hypot(tk[4], tk[5]);
      var f = Math.min(speed / 2000, 1);
      c.setHSL(0.66 * (1 - f), 0.9, 0.55);
      col[3 * i] = c.r; col[3 * i + 1] = c.g; col[3 * i + 2] = c.b;
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    trailGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true })));
  });

  playerGroup.visible = true;
  setEpisode(0);
  setPlaying(true);
}

// ------------------------------------------------------------------ transport
var scrub = document.getElementById('scrub');
var btnPlay = document.getElementById('btnPlay');

function episode() { return traj ? traj.episodes[curEp] : null; }

function setEpisode(i) {
  if (!traj) return;
  curEp = Math.max(0, Math.min(traj.episodes.length - 1, i));
  playTime = 0;
  scrub.max = episode().ticks.length - 1;
  scrub.value = 0;
  document.getElementById('epLabel').textContent =
    'ep ' + (curEp + 1) + '/' + traj.episodes.length;
  buildGoalOverlay(episode());
}

// ------------------------------------------------------------ episode goal
// Goal-conditioned recordings carry, in each episode's header line,
//   goal: { center: [x, y, z], radius: r }   where this episode had to get
//   line: [[x, y, z], ...]                   the reference line it was shown
// (surfgym.record.record_rollout(episode_meta=...)). Drawn for the episode
// being played and rebuilt on every episode change; the map's own race
// zones (addZoneBoxes) stay as they are. World (x, y, z) -> three (x, z, -y).
var goalGroup = null;

function buildGoalOverlay(ep) {
  if (goalGroup) { scene.remove(goalGroup); goalGroup = null; }
  if (!ep || !ep.header) return;
  var g = ep.header.goal, ln = ep.header.line;
  if (!g && !ln) return;
  goalGroup = new THREE.Group();
  if (g && g.center) {
    var r = Math.max(8, Number(g.radius) || 192);
    var sphere = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16),
      new THREE.MeshBasicMaterial({ color: 0x4de07a, transparent: true,
                                    opacity: 0.28, depthWrite: false }));
    sphere.position.set(g.center[0], g.center[2], -g.center[1]);
    goalGroup.add(sphere);
    var ring = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16),
      new THREE.MeshBasicMaterial({ color: 0x4de07a, wireframe: true,
                                    transparent: true, opacity: 0.5 }));
    ring.position.copy(sphere.position);
    goalGroup.add(ring);
  }
  if (ln && ln.length > 1) {
    var pts = ln.map(function (p) { return new THREE.Vector3(p[0], p[2], -p[1]); });
    goalGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0x4de07a })));
  }
  scene.add(goalGroup);
}

function setPlaying(p) {
  playing = p && !!traj;
  // pressing play while parked at the very end = replay from the top
  if (playing && traj) {
    var atEndOfEp = playTime >= episode().ticks.length - 1.001;
    if (atEndOfEp && curEp >= traj.episodes.length - 1) setEpisode(0);
    else if (atEndOfEp) setEpisode(curEp + 1);
  }
  btnPlay.textContent = playing ? '❚❚' : '▶';
}

function stepTick(d) {
  if (!traj) return;
  setPlaying(false);
  playTime = Math.max(0, Math.min(episode().ticks.length - 1, Math.round(playTime) + d));
}

btnPlay.addEventListener('click', function () { setPlaying(!playing); });
document.getElementById('btnStepB').addEventListener('click', function () { stepTick(-1); });
document.getElementById('btnStepF').addEventListener('click', function () { stepTick(1); });
document.getElementById('btnPrevEp').addEventListener('click', function () { setEpisode(curEp - 1); });
document.getElementById('btnNextEp').addEventListener('click', function () { setEpisode(curEp + 1); });
scrub.addEventListener('input', function () { playTime = +scrub.value; });
document.getElementById('speedSel').addEventListener('change', function (e) {
  playSpeed = +e.target.value;
});

function lerpAngleDeg(a, b, f) {
  var d = ((b - a + 540) % 360) - 180;      // shortest arc
  return a + d * f;
}

// interpolate playback state at fractional tick index
function sampleState() {
  var ep = episode();
  if (!ep) return null;
  var n = ep.ticks.length;
  var i0 = Math.min(Math.floor(playTime), n - 1);
  var i1 = Math.min(i0 + 1, n - 1);
  var f = Math.min(Math.max(playTime - i0, 0), 1);
  var a = ep.ticks[i0], b = ep.ticks[i1];
  function L(k) { return a[k] + (b[k] - a[k]) * f; }
  return {
    tick: a[0], x: L(1), y: L(2), z: L(3),
    vx: L(4), vy: L(5), vz: L(6),
    yaw: lerpAngleDeg(a[7], b[7], f),
    buttons: a[8], onground: a[9], progress: L(10), reward: a[11]
  };
}

var tickLabel = document.getElementById('tickLabel');
var hudRew = document.getElementById('hudRew');
var hudSpeed = document.getElementById('hudSpeed');
var hudVz = document.getElementById('hudVz');
var hudTick = document.getElementById('hudTick');
var hudEp = document.getElementById('hudEp');
var hudProg = document.getElementById('hudProg');

function updatePlayer(st) {
  var p = g2t(st.x, st.y, st.z);
  playerGroup.position.copy(p);
  var hs = Math.hypot(st.vx, st.vy);
  if (hs > 1) {
    velArrow.visible = true;
    velArrow.setDirection(new THREE.Vector3(st.vx, 0, -st.vy).normalize());
    velArrow.setLength(Math.max(30, Math.min(hs * 0.2, 400)), 16, 9);
  } else {
    velArrow.visible = false;
  }
  hudSpeed.textContent = Math.round(hs);
  hudVz.textContent = Math.round(st.vz);
  hudTick.textContent = st.tick;
  hudEp.textContent = (curEp + 1) + '/' + traj.episodes.length +
    (episode().end ? ' [' + episode().end.end + ']' : '');
  hudProg.textContent = st.progress.toFixed(1);
  var er = epRewardAt(episode(), Math.round(playTime));
  hudRew.textContent = er ? er.cur.toFixed(1) + ' / ' + er.total.toFixed(1)
                          : fmtBoth(episode());
  tickLabel.textContent = Math.round(playTime) + '/' + (episode().ticks.length - 1);
  var zc = zoneClockAt(episode(), playTime);
  var hz = document.getElementById('hudZone');
  if (hz) {
    if (zc.state === 'no-zones') hz.textContent = '-';
    else hz.textContent = fmtClock(zc.running) + (zc.final != null ? '  (final ' + fmtClock(zc.final) + ')' : '');
  }
  lastClock = zc;
}
var lastClock = null;

// -------------------------------------------------------------------- cameras
var followMode = false;
var lastFollowDir = new THREE.Vector3(1, 0, 0);

function setFollow(on) {
  followMode = on && !!traj;
  controls.enabled = !followMode;
}

function updateFollowCam(st, dt) {
  var p = g2t(st.x, st.y, st.z);
  var hs = Math.hypot(st.vx, st.vy);
  if (hs > 10) lastFollowDir.set(st.vx, 0, -st.vy).normalize();
  // behind the velocity direction, +60 GoldSrc units up, lerped
  var desired = p.clone().addScaledVector(lastFollowDir, -260)
                 .add(new THREE.Vector3(0, 60, 0));
  var alpha = 1 - Math.exp(-6 * dt);
  camera.position.lerp(desired, alpha);
  camera.lookAt(p);
  controls.target.copy(p);   // so orbiting resumes from the player when toggled off
}

// ------------------------------------------------------------ waypoint editor
var editMode = false;
var waypoints = [];          // [{ mesh, sprite }]; position lives on mesh.position
var wpGroup = new THREE.Group();
scene.add(wpGroup);
var wpLine = null;
var wpSphereGeo = new THREE.SphereGeometry(10, 16, 12);
var raycaster = new THREE.Raycaster();
var pointerNdc = new THREE.Vector2();
var dragWp = null;
var dragPlane = new THREE.Plane();
var downPos = { x: 0, y: 0 };

function setEdit(on) {
  editMode = on;
  document.getElementById('editor').classList.toggle('on', on);
}

function makeLabelTexture(num) {
  var cv = document.createElement('canvas');
  cv.width = cv.height = 64;
  var ctx = cv.getContext('2d');
  ctx.font = 'bold 40px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.strokeStyle = '#14161a';
  ctx.lineWidth = 7;
  ctx.strokeText(String(num), 32, 34);
  ctx.fillStyle = '#ffd76e';
  ctx.fillText(String(num), 32, 34);
  return new THREE.CanvasTexture(cv);
}

function renumberWaypoints() {
  waypoints.forEach(function (wp, i) {
    if (wp.sprite.material.map) wp.sprite.material.map.dispose();
    wp.sprite.material.map = makeLabelTexture(i + 1);
    wp.sprite.material.needsUpdate = true;
  });
  document.getElementById('wpCount').textContent = waypoints.length;
}

function rebuildWpLine() {
  if (wpLine) { wpGroup.remove(wpLine); wpLine.geometry.dispose(); }
  wpLine = null;
  if (waypoints.length < 2) return;
  var pts = waypoints.map(function (wp) { return wp.mesh.position; });
  wpLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0xe8a33d }));
  wpGroup.add(wpLine);
}

function addWaypoint(posThree) {
  var mesh = new THREE.Mesh(wpSphereGeo,
    new THREE.MeshBasicMaterial({ color: 0xe8a33d }));
  mesh.position.copy(posThree);
  var sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: makeLabelTexture(waypoints.length + 1), depthTest: false }));
  sprite.scale.set(26, 26, 1);
  sprite.position.set(0, 24, 0);
  mesh.add(sprite);
  wpGroup.add(mesh);
  waypoints.push({ mesh: mesh, sprite: sprite });
  renumberWaypoints();
  rebuildWpLine();
}

function removeWaypoint(wp) {
  var i = waypoints.indexOf(wp);
  if (i < 0) return;
  wpGroup.remove(wp.mesh);
  waypoints.splice(i, 1);
  renumberWaypoints();
  rebuildWpLine();
}

function clearWaypoints() {
  waypoints.slice().forEach(removeWaypoint);
}

function importWaypoints(json) {
  clearWaypoints();
  (json.points || []).forEach(function (p) {
    addWaypoint(g2t(p[0], p[1], p[2]));
  });
  setEdit(true);
}

document.getElementById('wpClear').addEventListener('click', clearWaypoints);
document.getElementById('wpExport').addEventListener('click', function () {
  var out = {
    map: mapName || 'map',
    points: waypoints.map(function (wp) {
      return t2g(wp.mesh.position).map(function (v) { return Math.round(v * 100) / 100; });
    })
  };
  var blob = new Blob([JSON.stringify(out, null, 1)], { type: 'application/json' });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (mapName || 'map') + '.waypoints.json';
  a.click();
  setTimeout(function () { URL.revokeObjectURL(a.href); }, 5000);
});

function setRayFromEvent(ev) {
  var rect = renderer.domElement.getBoundingClientRect();
  pointerNdc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointerNdc.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointerNdc, camera);
}

function pickWaypoint(ev) {
  setRayFromEvent(ev);
  var meshes = waypoints.map(function (wp) { return wp.mesh; });
  var hits = raycaster.intersectObjects(meshes, false);
  if (!hits.length) return null;
  var m = hits[0].object;
  for (var i = 0; i < waypoints.length; i++)
    if (waypoints[i].mesh === m) return waypoints[i];
  return null;
}

renderer.domElement.addEventListener('pointerdown', function (ev) {
  downPos.x = ev.clientX; downPos.y = ev.clientY;
  if (!editMode) return;
  if (ev.button === 0) {
    var wp = pickWaypoint(ev);
    if (wp) {
      dragWp = wp;
      controls.enabled = false;
      // drag on the camera-parallel plane through the waypoint
      var n = camera.getWorldDirection(new THREE.Vector3());
      dragPlane.setFromNormalAndCoplanarPoint(n, wp.mesh.position);
      renderer.domElement.setPointerCapture(ev.pointerId);
      ev.preventDefault();
    }
  } else if (ev.button === 2) {
    var wp2 = pickWaypoint(ev);
    if (wp2) removeWaypoint(wp2);
  }
});

renderer.domElement.addEventListener('pointermove', function (ev) {
  if (!dragWp) return;
  setRayFromEvent(ev);
  var hit = new THREE.Vector3();
  if (raycaster.ray.intersectPlane(dragPlane, hit)) {
    dragWp.mesh.position.copy(hit);
    rebuildWpLine();
  }
});

renderer.domElement.addEventListener('pointerup', function (ev) {
  if (dragWp) {
    dragWp = null;
    controls.enabled = !followMode;
    return;
  }
  if (!editMode || ev.button !== 0 || !worldMesh) return;
  var moved = Math.hypot(ev.clientX - downPos.x, ev.clientY - downPos.y);
  if (moved > 5) return;                       // was an orbit drag, not a click
  setRayFromEvent(ev);
  var hits = raycaster.intersectObject(worldMesh, false);
  if (hits.length) {
    // 40 GoldSrc units above the hit = +40 on three.js Y
    addWaypoint(hits[0].point.clone().add(new THREE.Vector3(0, 40, 0)));
  }
});

renderer.domElement.addEventListener('contextmenu', function (ev) {
  if (editMode) ev.preventDefault();
});

// ------------------------------------------------------------------- keyboard
window.addEventListener('keydown', function (ev) {
  var tag = (ev.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'select' || tag === 'textarea') {
    if (ev.code !== 'Space') return;
    ev.target.blur();
  }
  switch (ev.code) {
    case 'Space': ev.preventDefault(); setPlaying(!playing); break;
    case 'Comma': stepTick(-1); break;
    case 'Period': stepTick(1); break;
    case 'KeyF': setFollow(!followMode); break;
    case 'KeyE': setEdit(!editMode); break;
  }
});

// ------------------------------------------------------------ file load paths
function handleParsedJson(obj) {
  if (obj && obj.world) loadMesh(obj);
  else if (obj && obj.points) importWaypoints(obj);
  else console.warn('unrecognized JSON file (want .mesh.json or .waypoints.json)');
}

function handleFile(file) {
  var reader = new FileReader();
  reader.onload = function () {
    var text = reader.result;
    if (/\.jsonl$/i.test(file.name)) { loadTraj(text); return; }
    try { handleParsedJson(JSON.parse(text)); }
    catch (e) {                       // maybe a .jsonl saved with a .json name
      if (text.indexOf('\n') > 0) loadTraj(text);
      else console.warn('could not parse', file.name, e);
    }
  };
  reader.readAsText(file);
}

document.getElementById('fileInput').addEventListener('change', function (ev) {
  Array.prototype.forEach.call(ev.target.files, handleFile);
  ev.target.value = '';
});

var dropHint = document.getElementById('dropHint');
window.addEventListener('dragover', function (ev) {
  ev.preventDefault();
  dropHint.classList.add('on');
});
window.addEventListener('dragleave', function (ev) {
  if (!ev.relatedTarget) dropHint.classList.remove('on');
});
window.addEventListener('drop', function (ev) {
  ev.preventDefault();
  dropHint.classList.remove('on');
  Array.prototype.forEach.call(ev.dataTransfer.files, handleFile);
});

// default map, if served over http (silently skipped on file://)
var qs = new URLSearchParams(window.location.search);
var meshRequested = null;   // run-config map takes precedence over the default
fetch(qs.get('mesh') || 'assets/surf_ski_2.mesh.json')
  .then(function (r) { return r.ok ? r.json() : null; })
  .then(function (j) { if (j && !mapName && !meshRequested) loadMesh(j); })
  .catch(function () {});

// race zones (maps/<map>.zones.json): translucent start/finish cuboids
function addZoneBoxes(mapStem) {
  fetch('/maps/' + mapStem + '.zones.json')
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (z) {
      if (!z) return;
      zoneData = z;
      if (traj) traj.episodes.forEach(function (e) { e.zone = undefined; });
      [['start', 0x35d8a0], ['end', 0xff5a5a]].forEach(function (kv) {
        var zone = z[kv[0]];
        if (!zone) return;
        var mn = zone.mins, mx = zone.maxs;
        // pad thin timer curtains so they are visible from the side
        var sx = Math.max(mx[0] - mn[0], 8), sy = Math.max(mx[1] - mn[1], 8),
            sz = Math.max(mx[2] - mn[2], 8);
        var geo = new THREE.BoxGeometry(sx, sz, sy);   // g2t: (x, z, -y)
        var mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
          color: kv[1], transparent: true, opacity: 0.25, depthWrite: false }));
        mesh.position.copy(g2t((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2,
                               (mn[2] + mx[2]) / 2));
        scene.add(mesh);
        var edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo),
          new THREE.LineBasicMaterial({ color: kv[1] }));
        edges.position.copy(mesh.position);
        scene.add(edges);
      });
    })
    .catch(function () {});
}

// ⚡ reward potential field: instanced arrows along the shaping's descent
// direction, colored by normalized distance-to-finish (green near, red far).
// Exported by tools/export_goalfield.py; lazy-loaded on first toggle.
var fieldGroup = null;
var fieldMapStem = null;
function toggleField() {
  var btn = document.getElementById('btnField');
  if (fieldGroup) {
    fieldGroup.visible = !fieldGroup.visible;
    btn.style.opacity = fieldGroup.visible ? 1.0 : 0.55;
    return;
  }
  btn.disabled = true;
  btn.textContent = '⚡ loading…';
  fetch('assets/' + fieldMapStem + '.goalfield.json')
    .then(function (r) { if (!r.ok) throw new Error('none'); return r.json(); })
    .then(function (f) {
      var n = f.pot.length;
      var cone = new THREE.ConeGeometry(9, 42, 5);   // points +Y pre-rotation
      var mat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.8 });
      var inst = new THREE.InstancedMesh(cone, mat, n);
      var m4 = new THREE.Matrix4(), q = new THREE.Quaternion();
      var up = new THREE.Vector3(0, 1, 0), dir = new THREE.Vector3();
      var col = new THREE.Color();
      for (var i = 0; i < n; i++) {
        var p = g2t(f.pos[3 * i], f.pos[3 * i + 1], f.pos[3 * i + 2]);
        dir.copy(g2t(f.dir[3 * i] / 100, f.dir[3 * i + 1] / 100,
                     f.dir[3 * i + 2] / 100)).normalize();
        q.setFromUnitVectors(up, dir);
        m4.compose(p, q, new THREE.Vector3(1, 1, 1));
        inst.setMatrixAt(i, m4);
        // potential colormap: 0 (finish) green -> 0.5 yellow -> 1 (far) red
        var t = f.pot[i] / 1000;
        col.setHSL(0.33 * (1 - t), 0.9, 0.5);
        inst.setColorAt(i, col);
      }
      inst.instanceMatrix.needsUpdate = true;
      if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
      fieldGroup = new THREE.Group();
      fieldGroup.add(inst);
      scene.add(fieldGroup);
      btn.disabled = false;
      btn.textContent = '⚡ field';
      btn.style.opacity = 1.0;
    })
    .catch(function () {
      btn.disabled = false;
      btn.textContent = '⚡ no field (run tools/export_goalfield.py)';
    });
}
document.getElementById('btnField').addEventListener('click', toggleField);

// a trajectory recorded on another map must load THAT map's mesh + zones
function loadMapForRun(cfg) {
  var m = cfg && cfg.map;
  fieldMapStem = m || 'surf_ski_2';
  if (cfg && cfg.reward === 'race') {
    document.getElementById('btnField').style.display = '';
  }
  if (!m || m === 'surf_ski_2') { addZoneBoxes(m || 'surf_ski_2'); return; }
  meshRequested = m;
  fetch('assets/' + m + '.mesh.json')
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (j) {
      if (j) loadMesh(j);
      else console.warn('no mesh for ' + m +
                        ' — run: python tools/export_map.py maps/' + m + '.bsp');
      addZoneBoxes(m);
    })
    .catch(function () {});
}

// 🎥 POV: render (server-side, tools/render_pov.py) and open the first-person
// depth video of the trajectory currently loaded via ?traj=
var btnPov = document.getElementById('btnPov');
function setupPovButton(trajUrl) {
  btnPov.style.display = '';
  btnPov.disabled = false;
  btnPov.textContent = '🎥 POV';
  btnPov.onclick = function () {
    btnPov.disabled = true;
    btnPov.textContent = 'rendering…';
    var api = '/api/render_pov?traj=' + encodeURIComponent(trajUrl);
    (function tick() {
      fetch(api)
        .then(function (r) {
          if (r.status === 400) throw new Error('gone');
          if (!r.ok) throw new Error('server');
          return r.json();
        })
        .then(function (j) {
          if (j.status === 'done') {
            btnPov.disabled = false;
            btnPov.textContent = '🎥 POV';
            var v = document.getElementById('povVideo');
            // the server names the file (a --surf-mask render gets its own
            // name so a stale depth-only mp4 is never shown in its place);
            // fall back to the old guess for older servers
            var src = j.pov || trajUrl.replace(/\.jsonl$/, '.pov.mp4');
            v.src = src + '?t=' + Date.now();
            document.getElementById('povPanel').style.display = '';
            v.play().catch(function () {});
          } else if (j.status === 'started' || j.status === 'rendering') {
            setTimeout(tick, 2000);
          } else {
            btnPov.disabled = false;
            btnPov.textContent = 'POV ✗ retry';
          }
        })
        .catch(function (e) {
          btnPov.disabled = false;
          btnPov.textContent = e.message === 'gone'
            ? 'POV ✗ file moved — reopen from dashboard'
            : 'POV ✗ (server?)';
        });
    })();
  };
}

document.getElementById('povClose').addEventListener('click', function () {
  var v = document.getElementById('povVideo');
  v.pause();
  document.getElementById('povPanel').style.display = 'none';
});

// ?traj=/runs/<run>/traj_X.jsonl — deep link from the runs dashboard
if (qs.get('traj')) {
  var trajUrl = qs.get('traj');
  setupPovButton(trajUrl);
  var stepM = trajUrl.match(/traj_(\d+)/);
  trajStep = stepM ? parseInt(stepM[1], 10) : null;
  // the run's config (reward mode, blend window) lives next to the file
  fetch(trajUrl.replace(/[^\/]*$/, 'run.json'))
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (j) {
      runCfg = j && j.config ? j.config : null;
      return fetch(trajUrl);
    })
    .then(function (r) { return r.ok ? r.text() : null; })
    .then(function (t) {
      if (!t) return;
      // A --maps run trains on several maps, so run.json names only one of
      // them and every trajectory would render against that one. Each file
      // records its OWN map in its header line, so the FILE wins over the
      // config. Resolved before loadMapForRun so only one mesh is ever
      // requested.
      var cfg = {};
      if (runCfg) { for (var k in runCfg) { if (runCfg.hasOwnProperty(k)) cfg[k] = runCfg[k]; } }
      try {
        var nl = t.indexOf(String.fromCharCode(10));
        var hdr = JSON.parse(nl > 0 ? t.slice(0, nl) : t);
        if (hdr && hdr.map) cfg.map = hdr.map;
      } catch (e) { /* keep run.json's map */ }
      loadMapForRun(cfg);
      loadTraj(t); setPlaying(true);
      var epq = parseInt(qs.get('ep') || '', 10);
      if (epq >= 1 && traj && epq <= traj.episodes.length) setEpisode(epq - 1);
      // ?autorec=1: start a recording once the map mesh and zones are in
      if (qs.get('autorec')) setTimeout(function () { startRecording(); }, 4000);
    })
    .catch(function () {});
}

// ------------------------------------------------------------------ main loop
var clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  var dt = Math.min(clock.getDelta(), 0.1);

  if (traj) {
    var ep = episode();
    var tickMs = ep.header.tick_ms || 10;
    if (playing) {
      playTime += (dt * 1000 / tickMs) * playSpeed;
      var last = ep.ticks.length - 1;
      if (playTime >= last) {
        if (curEp < traj.episodes.length - 1) setEpisode(curEp + 1);
        else { playTime = last; setPlaying(false); }
      }
      scrub.value = Math.round(playTime);
    }
    var st = sampleState();
    if (st) {
      updatePlayer(st);
      if (followMode) updateFollowCam(st, dt);
    }
  }

  if (!followMode) controls.update();
  renderer.render(scene, camera);
  drawOverlay();
  if (recorder && recorder.state === 'recording') recordFrame();
}

// ------------------------------------------------- clock overlay + recorder
// The zone clock is also burned into a 2D canvas over the WebGL view so a
// recording carries it; the recorder composites both canvases per frame
// into an offscreen canvas and captures that stream (webm, VP9 if available).
var overlay = document.createElement('canvas');
overlay.id = 'hudCanvas';
overlay.style.cssText = 'position:absolute;left:0;top:0;pointer-events:none;';
container.appendChild(overlay);
var octx = overlay.getContext('2d');
function drawOverlay() {
  var W = renderer.domElement.width, H = renderer.domElement.height;
  if (overlay.width !== W || overlay.height !== H) { overlay.width = W; overlay.height = H; }
  overlay.style.width = renderer.domElement.style.width || (W + 'px');
  overlay.style.height = renderer.domElement.style.height || (H + 'px');
  octx.clearRect(0, 0, W, H);
  if (!traj || !lastClock || lastClock.state === 'no-zones') return;
  var dpr = window.devicePixelRatio || 1;
  var fs = Math.round(34 * dpr), x = Math.round(W / 2), y = Math.round(58 * dpr);   // below the file bar
  var main = fmtClock(lastClock.running);
  var color = lastClock.state === 'done' ? '#4de07a' : (lastClock.state === 'pre' ? '#9aa3ae' : '#ffffff');
  octx.textAlign = 'center'; octx.textBaseline = 'top';
  octx.font = 'bold ' + fs + 'px system-ui, sans-serif';
  octx.lineWidth = Math.max(2, 4 * dpr); octx.strokeStyle = 'rgba(0,0,0,0.85)';
  octx.strokeText(main, x, y); octx.fillStyle = color; octx.fillText(main, x, y);
  var mapKey = (mapName || '').replace(/\.bsp$/, '');
  var wr = WR_TIMES[mapKey];
  var sub = 'zone clock' + (wr ? '   WR ' + fmtClock(wr) : '');
  if (lastClock.state === 'done' && wr) {
    var d = lastClock.final - wr;
    sub += '   ' + (d >= 0 ? '+' : '') + d.toFixed(2) + ' s';
  }
  octx.font = Math.round(15 * dpr) + 'px system-ui, sans-serif';
  octx.lineWidth = Math.max(1, 3 * dpr);
  octx.strokeText(sub, x, y + fs + Math.round(4 * dpr));
  octx.fillStyle = '#e8ecef'; octx.fillText(sub, x, y + fs + Math.round(4 * dpr));
  if (recorder && recorder.state === 'recording') {
    octx.textAlign = 'left';
    octx.fillStyle = '#ff5a5a';
    octx.beginPath(); octx.arc(Math.round(18 * dpr), Math.round(18 * dpr), Math.round(7 * dpr), 0, 6.283); octx.fill();
  }
}
var recorder = null, recCanvas = null, rctx = null, recChunks = [], recStopAt = null;
function recordFrame() {
  var W = renderer.domElement.width, H = renderer.domElement.height;
  if (recCanvas.width !== W || recCanvas.height !== H) { recCanvas.width = W; recCanvas.height = H; }
  rctx.drawImage(renderer.domElement, 0, 0);
  rctx.drawImage(overlay, 0, 0);
  // stop ~1.5 s after the finish crossing, or at the episode's end
  var ep = episode();
  if (ep) {
    var z = epZone(ep);
    var endTick = (z && z.t1 != null) ? z.t1 + 1500 / (z.tickMs || 10) : ep.ticks.length - 1;
    if (playTime >= endTick || !playing) stopRecording();
  }
}
function startRecording() {
  if (!traj || recorder) return;
  if (!recCanvas) { recCanvas = document.createElement('canvas'); rctx = recCanvas.getContext('2d'); }
  recCanvas.width = renderer.domElement.width; recCanvas.height = renderer.domElement.height;
  var stream = recCanvas.captureStream(60);
  var mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].filter(function (m) {
    return window.MediaRecorder && MediaRecorder.isTypeSupported(m); })[0];
  if (!mime) { alert('MediaRecorder / webm not supported in this browser'); return; }
  recChunks = [];
  recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12000000 });
  recorder.ondataavailable = function (e) { if (e.data && e.data.size) recChunks.push(e.data); };
  recorder.onstop = function () {
    var blob = new Blob(recChunks, { type: 'video/webm' });
    var url = URL.createObjectURL(blob);
    var ep = episode(), z = ep ? epZone(ep) : null;
    var nm = ((qs.get('traj') || 'run').split('/').slice(-3).join('_').replace(/\.jsonl$/, '')) +
             '_ep' + (curEp + 1) + (z && z.time != null ? '_zone' + z.time.toFixed(2) + 's' : '') + '.webm';
    var a = document.createElement('a'); a.href = url; a.download = nm; a.textContent = 'save ' + nm;
    a.style.cssText = 'display:block;margin-top:4px;color:#7fd07f';
    var host = document.getElementById('recLinks'); if (host) host.appendChild(a);
    a.click();
    recorder = null;
    var b = document.getElementById('btnRec'); if (b) b.textContent = '\u25cf rec';
  };
  // a watchable take: chase camera, real time, from the top of the episode
  setFollow(true); playSpeed = 1;
  var sel = document.getElementById('speedSel'); if (sel) sel.value = '1';
  playTime = 0; setPlaying(true);
  recorder.start(250);
  var b = document.getElementById('btnRec'); if (b) b.textContent = '\u25a0 stop';
}
function stopRecording() {
  if (recorder && recorder.state === 'recording') recorder.stop();
}
var btnRec = document.getElementById('btnRec');
if (btnRec) btnRec.addEventListener('click', function () {
  if (recorder) stopRecording(); else startRecording();
});
animate();
