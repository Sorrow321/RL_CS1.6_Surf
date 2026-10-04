/* skate_laby agent-vs-WR render page: two first-person views of the map mesh (top: agent,
 * bottom: human WR), a HUD, and window.captureFrame(k) -> JPEG data URL for tools/demo/
 * vs_wr_render.py, which drives this page headless and encodes the frames.
 * Per-frame poses come from tools/demo/vs_wr_frames.py (?frames=<url>).
 * Coordinates: GoldSrc (x, y, z) -> three.js (x, z, -y), exactly as viewer/app.js. */
'use strict';

var W = 1280, SIDE = 640, TW = W + SIDE, PH = 720, BAR = 120, H = 2 * PH + BAR;
var VFOV = 67.4;          // vertical fov for a 100 deg horizontal fov at 16:9
var PITCH = 4.0;          // fixed camera pitch (deg down): pitch has no effect on movement
var EYE_STAND = 17.0, EYE_DUCK = 12.0;   // CS view offsets above the origin
var SKY = 0xaebdcc;

function g2t(x, y, z) { return new THREE.Vector3(x, z, -y); }

var params = new URLSearchParams(location.search);
var framesUrl = params.get('frames') || '/runs/skWR_cap2000/vs_frames.json';
var meshUrl = params.get('mesh') || 'assets/skate_laby.mesh.json';

var wrap = document.getElementById('wrap');
var renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(TW, H);
wrap.appendChild(renderer.domElement);
var hud = document.createElement('canvas');
hud.width = TW; hud.height = H;
wrap.appendChild(hud);
var hctx = hud.getContext('2d');
var comp = document.createElement('canvas');
comp.width = TW; comp.height = H;
var cctx = comp.getContext('2d');

var scene = new THREE.Scene();
scene.background = new THREE.Color(SKY);
scene.fog = new THREE.Fog(SKY, 2500, 15000);
scene.add(new THREE.HemisphereLight(0xe6eeff, 0x6a6258, 0.55));
var sun = new THREE.DirectionalLight(0xffffff, 0.52);
sun.position.set(0.45, 1.0, 0.3);
scene.add(sun);
var fill = new THREE.DirectionalLight(0xbfd0e8, 0.30);
fill.position.set(-0.6, 0.35, -0.7);
scene.add(fill);

function faceColor(nz) {
  if (nz >= 0.7) return [0.76, 0.70, 0.60];     // floor: warm sand
  if (nz > 0.001) return [0.93, 0.66, 0.30];    // ramp
  if (nz > -0.001) return [0.80, 0.83, 0.88];   // wall: light cool grey
  return [0.48, 0.50, 0.55];                    // ceiling
}

function buildGeometry(positions, normals, indices, rgb) {
  var n = positions.length / 3;
  var pos = new Float32Array(n * 3), nor = new Float32Array(n * 3), col = new Float32Array(n * 3);
  for (var i = 0; i < n; i++) {
    var gx = positions[3 * i], gy = positions[3 * i + 1], gz = positions[3 * i + 2];
    var nx = normals[3 * i], ny = normals[3 * i + 1], nz = normals[3 * i + 2];
    pos[3 * i] = gx; pos[3 * i + 1] = gz; pos[3 * i + 2] = -gy;
    nor[3 * i] = nx; nor[3 * i + 1] = nz; nor[3 * i + 2] = -ny;
    var c = rgb || faceColor(nz);
    col[3 * i] = c[0]; col[3 * i + 1] = c[1]; col[3 * i + 2] = c[2];
  }
  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  // BSP faces are wound clockwise seen from their front (Quake convention); three.js treats
  // counter-clockwise as the front and, with DoubleSide, lights a back face with the NEGATED
  // normal - floors came out lit by the hemisphere's ground colour (near black). Reverse the
  // winding so the visible side is the front.
  var idx = new Uint32Array(indices.length);
  for (var t = 0; t < indices.length; t += 3) {
    idx[t] = indices[t]; idx[t + 1] = indices[t + 2]; idx[t + 2] = indices[t + 1];
  }
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  return geo;
}

function addSolid(geo) {
  scene.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({
    vertexColors: true, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 })));
  var edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 28),
    new THREE.LineBasicMaterial({ color: 0x23272e, transparent: true, opacity: 0.30 }));
  scene.add(edges);
}

function addGlass(b, opacity) {
  var geo = buildGeometry(b.positions, b.normals, b.indices, [0.80, 0.88, 0.95]);
  scene.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({
    vertexColors: true, transparent: true, opacity: opacity, depthWrite: false,
    side: THREE.DoubleSide })));
  scene.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo, 28),
    new THREE.LineBasicMaterial({ color: 0xe8f0f8, transparent: true, opacity: 0.55 })));
}

var boosters = [];   // trigger_push AABBs (GoldSrc), grown by the player hull

// a trigger_push: a translucent amber volume with bright edges and chevrons on its floor
// pointing the way it pushes (its "angles" yaw), so a viewer sees why the speed jumps
function addBooster(b) {
  var geo = buildGeometry(b.positions, b.normals, b.indices, [1.0, 0.72, 0.22]);
  scene.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.22, depthWrite: false,
    side: THREE.DoubleSide })));
  scene.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo, 28),
    new THREE.LineBasicMaterial({ color: 0xffc040, transparent: true, opacity: 0.95 })));
  var p = b.positions, lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
  for (var i = 0; i < p.length; i += 3) {
    for (var a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], p[i + a]); hi[a] = Math.max(hi[a], p[i + a]); }
  }
  boosters.push([lo[0] - 16, lo[1] - 16, lo[2] - 36, hi[0] + 16, hi[1] + 16, hi[2] + 36]);
  var yaw = ((b.angles && b.angles[1]) || 0) * Math.PI / 180;
  var fx = Math.cos(yaw), fy = Math.sin(yaw), sx = -fy, sy = fx;     // forward / side (GoldSrc)
  var cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2;
  var len = Math.abs(fx) * (hi[0] - lo[0]) + Math.abs(fy) * (hi[1] - lo[1]);
  var wid = Math.abs(sx) * (hi[0] - lo[0]) + Math.abs(sy) * (hi[1] - lo[1]);
  var z = lo[2] + 1.5, half = Math.min(wid * 0.38, 60), depth = half * 0.8;
  var pts = [];
  for (var c = -2; c <= 2; c++) {
    var ox = cx + fx * c * len / 6, oy = cy + fy * c * len / 6;
    var tip = g2t(ox + fx * depth / 2, oy + fy * depth / 2, z);
    pts.push(g2t(ox - fx * depth / 2 + sx * half, oy - fy * depth / 2 + sy * half, z), tip);
    pts.push(tip, g2t(ox - fx * depth / 2 - sx * half, oy - fy * depth / 2 - sy * half, z));
  }
  var cg = new THREE.BufferGeometry().setFromPoints(pts);
  scene.add(new THREE.LineSegments(cg, new THREE.LineBasicMaterial({ color: 0xffd060 })));
  // a second, slightly raised copy reads as a thicker stroke on the floor
  var pts2 = pts.map(function (v) { return v.clone().add(new THREE.Vector3(0, 0.8, 0)); });
  scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts2),
    new THREE.LineBasicMaterial({ color: 0xffd060 })));
}

var camA = new THREE.PerspectiveCamera(VFOV, W / PH, 2, 40000);
var camH = new THREE.PerspectiveCamera(VFOV, W / PH, 2, 40000);
var data = null;
var mmImg = null;
window.ready = false;
window.loadError = null;

function setCam(cam, f) {
  var eye = f[2] + (f[4] ? EYE_DUCK : EYE_STAND);
  var p = g2t(f[0], f[1], eye);
  var yr = f[3] * Math.PI / 180, pr = PITCH * Math.PI / 180;
  var dx = Math.cos(pr) * Math.cos(yr), dy = Math.cos(pr) * Math.sin(yr), dz = -Math.sin(pr);
  cam.position.copy(p);
  cam.lookAt(p.clone().add(new THREE.Vector3(dx, dz, -dy)));
}

// ------------------------------------------------------------------------------------ HUD
var FONT = '"Segoe UI", Arial, sans-serif';

function txt(s, x, y, size, weight, align, color) {
  hctx.font = (weight || 'bold') + ' ' + size + 'px ' + FONT;
  hctx.textAlign = align || 'left';
  hctx.textBaseline = 'alphabetic';
  hctx.lineJoin = 'round';
  hctx.lineWidth = Math.max(3, size / 7);
  hctx.strokeStyle = 'rgba(0,0,0,0.85)';
  hctx.strokeText(s, x, y);
  hctx.fillStyle = color || '#ffffff';
  hctx.fillText(s, x, y);
}

function key(x, y, w, h, label, on) {
  hctx.lineWidth = 2;
  if (on) {
    hctx.fillStyle = 'rgba(245,245,245,0.95)';
    hctx.fillRect(x, y, w, h);
  } else {
    hctx.fillStyle = 'rgba(0,0,0,0.35)';
    hctx.fillRect(x, y, w, h);
    hctx.strokeStyle = 'rgba(220,220,220,0.75)';
    hctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
  }
  hctx.font = 'bold 20px ' + FONT;
  hctx.textAlign = 'center';
  hctx.textBaseline = 'middle';
  hctx.fillStyle = on ? '#141414' : '#e6e6e6';
  hctx.fillText(label, x + w / 2, y + h / 2 + 1);
}

var curK = 0;

// seconds since the runner was last inside a booster (trigger_push) volume, or -1 (not in
// the last 1.2 s). By position, not by a speed jump: under the 2000 u/s cap the 800 booster
// adds only ~190 u/s to a runner already at 1,810, and a speed rule missed it.
function inBooster(f) {
  for (var i = 0; i < boosters.length; i++) {
    var b = boosters[i];
    if (f[0] >= b[0] && f[0] <= b[3] && f[1] >= b[1] && f[1] <= b[4] && f[2] >= b[2] && f[2] <= b[5]) return true;
  }
  return false;
}

function boostAge(fr, k) {
  var lim = Math.round(1.2 * data.fps), k0 = Math.round((data.pre || 0) * data.fps);
  for (var j = k; j >= Math.max(k0, k - lim); j--) {
    if (inBooster(fr[j])) return (k - j) / data.fps;
  }
  return -1;
}

function panelHud(y0, tr, f, t) {
  var fin = t >= tr.finish;
  // a soft top band so the titles read on bright sky
  var g = hctx.createLinearGradient(0, y0, 0, y0 + 110);
  g.addColorStop(0, 'rgba(0,0,0,0.45)'); g.addColorStop(1, 'rgba(0,0,0,0)');
  hctx.fillStyle = g; hctx.fillRect(0, y0, W, 110);
  txt(tr.title, 28, y0 + 52, 38, 'bold');
  txt(tr.sub, 30, y0 + 88, 24, '600', 'left', '#e8e8e8');
  txt((fin ? tr.finish : Math.max(0, t)).toFixed(2) + ' s', W - 28, y0 + 66, 58, 'bold', 'right');
  if (t < 0) {
    hctx.fillStyle = 'rgba(0,0,0,0.35)';
    hctx.fillRect(0, y0 + 110, W, PH - 110);
    txt(String(Math.ceil(-t)), W / 2, y0 + PH / 2 + 70, 200, 'bold', 'center');
  } else if (t < 0.6) {
    txt('GO!', W / 2, y0 + PH / 2 + 60, 170, 'bold', 'center');
  }
  txt(f[5] + ' u/s', 30, y0 + PH - 30, 34, 'bold');
  var age = boostAge(tr.f, curK);
  if (age >= 0) {
    hctx.save();
    hctx.globalAlpha = Math.max(0, 1 - age / 1.2);
    txt('BOOST!', 30, y0 + PH - 82, 54, 'bold', 'left', '#ffd060');
    hctx.restore();
  }
  var k = 50, gp = 6, x0 = W - 3 * k - 2 * gp - 28, ky = y0 + PH - 2 * k - gp - 26;
  key(x0 + k + gp, ky, k, k, 'W', f[6] === 2);
  key(x0, ky + k + gp, k, k, 'A', f[7] === 0);
  key(x0 + k + gp, ky + k + gp, k, k, 'S', f[6] === 0);
  key(x0 + 2 * (k + gp), ky + k + gp, k, k, 'D', f[7] === 2);
  key(x0 - 124, ky + k + gp, 112, k, 'DUCK', !!f[4]);
  if (fin) {
    hctx.fillStyle = 'rgba(0,0,0,0.55)';
    hctx.fillRect(0, y0 + PH / 2 - 62, W, 104);
    txt('FINISHED   ' + tr.finish.toFixed(2) + ' s', W / 2, y0 + PH / 2 + 12, 64, 'bold', 'center');
  }
}

function barHud(k) {
  var y0 = PH;
  hctx.fillStyle = '#16181c';
  hctx.fillRect(0, y0, W, BAR);
  var x0 = 150, x1 = W - 60, yl = y0 + 82;
  hctx.strokeStyle = '#9aa0a8'; hctx.lineWidth = 3;
  hctx.beginPath(); hctx.moveTo(x0, yl); hctx.lineTo(x1, yl); hctx.stroke();
  hctx.lineWidth = 2;
  for (var q = 0; q <= 10; q++) {
    var xq = x0 + (x1 - x0) * q / 10;
    hctx.beginPath(); hctx.moveTo(xq, yl - 7); hctx.lineTo(xq, yl + 7); hctx.stroke();
  }
  txt('route', 24, yl + 8, 22, '600', 'left', '#cfd3d8');
  var pa = data.agent.f[k][8], ph = data.human.f[k][8];
  var xa = x0 + (x1 - x0) * pa, xh = x0 + (x1 - x0) * ph;
  // agent: filled triangle ABOVE the line; human: hollow triangle BELOW (shape, not colour)
  hctx.fillStyle = '#ffffff';
  hctx.beginPath(); hctx.moveTo(xa - 12, yl - 34); hctx.lineTo(xa + 12, yl - 34); hctx.lineTo(xa, yl - 10);
  hctx.closePath(); hctx.fill();
  txt('AGENT', xa - 18, yl - 18, 18, 'bold', 'right');
  hctx.strokeStyle = '#ffffff'; hctx.lineWidth = 3;
  hctx.beginPath(); hctx.moveTo(xh - 12, yl + 34); hctx.lineTo(xh + 12, yl + 34); hctx.lineTo(xh, yl + 10);
  hctx.closePath(); hctx.stroke();
  txt('HUMAN', xh - 18, yl + 32, 18, 'bold', 'right');
  txt(data.gap[k], W / 2, y0 + 38, 30, 'bold', 'center');
}

// ---------------------------------------------------------------------- side column
var MM_X = W + 20, MM_Y = 84;

function mmPt(x, y) {
  var m = data.minimap;
  return [MM_X + (x - m.x0) / (m.x1 - m.x0) * m.w, MM_Y + (m.y1 - y) / (m.y1 - m.y0) * m.h];
}

function trail(fr, k0, k, dashed) {
  hctx.save();
  hctx.lineJoin = 'round'; hctx.lineCap = 'round';
  hctx.setLineDash(dashed ? [9, 7] : []);
  for (var pass = 0; pass < 2; pass++) {       // dark under-stroke, then the line itself
    hctx.lineWidth = pass ? 3 : 6;
    hctx.strokeStyle = pass ? (dashed ? '#e8e8e8' : '#ffffff') : 'rgba(0,0,0,0.75)';
    hctx.beginPath();
    for (var j = k0; j <= k; j += 2) {
      var q = mmPt(fr[j][0], fr[j][1]);
      if (j === k0) hctx.moveTo(q[0], q[1]); else hctx.lineTo(q[0], q[1]);
    }
    var qe = mmPt(fr[k][0], fr[k][1]);
    hctx.lineTo(qe[0], qe[1]);
    hctx.stroke();
  }
  hctx.restore();
}

function marker(q, filled, label) {
  hctx.save();
  hctx.beginPath();
  if (filled) {   // agent: filled triangle pointing down
    hctx.moveTo(q[0] - 11, q[1] - 16); hctx.lineTo(q[0] + 11, q[1] - 16); hctx.lineTo(q[0], q[1] + 3);
  } else {        // human: hollow triangle pointing up
    hctx.moveTo(q[0] - 11, q[1] + 16); hctx.lineTo(q[0] + 11, q[1] + 16); hctx.lineTo(q[0], q[1] - 3);
  }
  hctx.closePath();
  hctx.strokeStyle = '#000000'; hctx.lineWidth = 6; hctx.stroke();
  if (filled) { hctx.fillStyle = '#ffffff'; hctx.fill(); }
  hctx.strokeStyle = '#ffffff'; hctx.lineWidth = 3; hctx.stroke();
  hctx.restore();
  txt(label, q[0] + 16, filled ? q[1] - 8 : q[1] + 22, 20, 'bold', 'left');
}

function sideHud(k, t) {
  hctx.fillStyle = '#121519';
  hctx.fillRect(W, 0, SIDE, H);
  txt('TOP-DOWN MAP', W + 22, 58, 32, 'bold');
  var m = data.minimap;
  hctx.drawImage(mmImg, MM_X, MM_Y, m.w, m.h);
  var qs = mmPt(data.start[0], data.start[1]), qf = mmPt(data.finish[0], data.finish[1]);
  [[qs, 'START', -1], [qf, 'FINISH', -1]].forEach(function (p) {
    hctx.fillStyle = '#ffffff'; hctx.strokeStyle = '#000000'; hctx.lineWidth = 3;
    hctx.strokeRect(p[0][0] - 7, p[0][1] - 7, 14, 14);
    hctx.fillRect(p[0][0] - 5, p[0][1] - 5, 10, 10);
    // START's label sits LEFT of its marker so it never collides with the runners' labels
    txt(p[1], p[0][0] + 12 * p[2], p[0][1] + 7, 18, 'bold', p[2] < 0 ? 'right' : 'left');
  });
  var k0 = Math.round(data.pre * data.fps);
  if (k > k0) {
    trail(data.human.f, k0, k, true);
    trail(data.agent.f, k0, k, false);
  }
  marker(mmPt(data.human.f[k][0], data.human.f[k][1]), false, 'HUMAN');
  marker(mmPt(data.agent.f[k][0], data.agent.f[k][1]), true, 'AGENT');
  var ly = MM_Y + m.h + 44;
  hctx.save(); hctx.lineWidth = 3; hctx.strokeStyle = '#ffffff';
  hctx.setLineDash([]); hctx.beginPath(); hctx.moveTo(W + 24, ly - 8); hctx.lineTo(W + 74, ly - 8); hctx.stroke();
  hctx.setLineDash([9, 7]); hctx.beginPath(); hctx.moveTo(W + 24, ly + 30); hctx.lineTo(W + 74, ly + 30); hctx.stroke();
  hctx.restore();
  txt('AGENT  (filled marker, solid line)', W + 86, ly, 22, '600');
  txt('HUMAN  (hollow marker, dashed line)', W + 86, ly + 38, 22, '600');
  speedChart(k, t, ly + 90);
  resultBox(t, ly + 540);
}

function resultBox(t, y0) {
  txt('RESULT', W + 22, y0, 26, 'bold');
  var a = data.agent, h = data.human;
  [[a, 'AGENT', y0 + 46], [h, 'HUMAN (WR)', y0 + 86]].forEach(function (r) {
    txt(r[1], W + 24, r[2], 26, '600', 'left', '#e8e8e8');
    var done = t >= r[0].finish;
    txt(done ? r[0].finish.toFixed(2) + ' s' : (t < 0 ? '-' : 'racing...'), TW - 30, r[2], 28,
        'bold', 'right', done ? '#ffffff' : '#9aa0a8');
  });
  if (t >= a.finish && t >= h.finish) {
    var d = h.finish - a.finish;
    txt((d >= 0 ? 'AGENT FASTER BY ' : 'HUMAN FASTER BY ') + Math.abs(d).toFixed(2) + ' s',
        W + SIDE / 2, y0 + 150, 34, 'bold', 'center');
  }
}

function speedChart(k, t, y0) {
  txt('SPEED OVER TIME', W + 22, y0 + 10, 26, 'bold');
  var k0 = Math.round(data.pre * data.fps);
  var tEnd = Math.max(data.agent.finish, data.human.finish) + 1;
  // two charts, one per runner, on the same axes: in one chart the traces overlap
  oneChart(data.agent, 'AGENT', false, y0 + 30, k, k0, tEnd, false);
  oneChart(data.human, 'HUMAN', true, y0 + 210, k, k0, tEnd, true);
}

function oneChart(tr, label, dashed, y0, k, k0, tEnd, xLabels) {
  var x0 = W + 92, x1 = TW - 26, yt = y0 + 34, yb = y0 + 160, vLo = 1000, vHi = 2700;
  function X(tt) { return x0 + (x1 - x0) * tt / tEnd; }
  function Y(v) { return yb - (yb - yt) * (Math.max(vLo, Math.min(vHi, v)) - vLo) / (vHi - vLo); }
  hctx.strokeStyle = '#6b717a'; hctx.lineWidth = 1.5;
  hctx.beginPath(); hctx.moveTo(x0, yt); hctx.lineTo(x0, yb); hctx.lineTo(x1, yb); hctx.stroke();
  [1000, 1500, 2000, 2500].forEach(function (v) {
    txt(String(v), x0 - 8, Y(v) + 6, 15, '600', 'right', '#cfd3d8');
  });
  if (xLabels) {
    [0, 20, 40, 60].forEach(function (tt) {
      txt(tt + ' s', X(tt), yb + 22, 15, '600', 'center', '#cfd3d8');
    });
  }
  hctx.save(); hctx.setLineDash([3, 5]); hctx.strokeStyle = '#d8d8d8'; hctx.lineWidth = 2;
  hctx.beginPath(); hctx.moveTo(x0, Y(2000)); hctx.lineTo(x1, Y(2000)); hctx.stroke(); hctx.restore();
  txt('cap 2000', x1, Y(2000) - 6, 14, '600', 'right', '#e0e0e0');
  // the runner's label, with its line style as a key
  hctx.save(); hctx.setLineDash(dashed ? [8, 6] : []); hctx.strokeStyle = '#ffffff'; hctx.lineWidth = 3;
  hctx.beginPath(); hctx.moveTo(W + 22, y0 + 12); hctx.lineTo(W + 70, y0 + 12); hctx.stroke(); hctx.restore();
  txt(label + ' speed', W + 80, y0 + 19, 19, 'bold', 'left');
  if (k > k0) {
    var fr = tr.f, kk = Math.min(k, k0 + Math.round(tr.finish * data.fps));
    hctx.save(); hctx.setLineDash(dashed ? [8, 6] : []); hctx.lineWidth = 2.5;
    hctx.strokeStyle = '#ffffff'; hctx.lineJoin = 'round';
    hctx.beginPath();
    for (var j = k0; j <= kk; j += 2) {
      var tt = j / data.fps - data.pre;
      if (j === k0) hctx.moveTo(X(tt), Y(fr[j][5])); else hctx.lineTo(X(tt), Y(fr[j][5]));
    }
    hctx.stroke(); hctx.restore();
  }
}

// ---------------------------------------------------------------------------- per frame
function renderFrame(k) {
  k = Math.max(0, Math.min(data.n - 1, k));
  var t = k / data.fps - (data.pre || 0);
  var fa = data.agent.f[k], fh = data.human.f[k];
  setCam(camA, fa);
  setCam(camH, fh);
  renderer.setScissorTest(true);
  renderer.setViewport(0, PH + BAR, W, PH);
  renderer.setScissor(0, PH + BAR, W, PH);
  renderer.render(scene, camA);
  renderer.setViewport(0, 0, W, PH);
  renderer.setScissor(0, 0, W, PH);
  renderer.render(scene, camH);
  renderer.setScissorTest(false);
  hctx.clearRect(0, 0, W, H);
  curK = k;
  panelHud(0, data.agent, fa, t);
  barHud(k);
  panelHud(PH + BAR, data.human, fh, t);
  sideHud(k, t);
  return true;
}

window.renderFrame = renderFrame;
window.captureFrame = function (k, quality) {
  renderFrame(k);
  cctx.drawImage(renderer.domElement, 0, 0);
  cctx.drawImage(hud, 0, 0);
  return comp.toDataURL('image/jpeg', quality || 0.93);
};

Promise.all([fetch(meshUrl).then(function (r) { return r.json(); }),
             fetch(framesUrl).then(function (r) { return r.json(); })])
  .then(function (res) {
    var mesh = res[0];
    data = res[1];
    addSolid(buildGeometry(mesh.world.positions, mesh.world.normals, mesh.world.indices, null));
    (mesh.brushes || []).forEach(function (b) {
      if (!b.classname) return;
      if (b.classname === 'trigger_push') { addBooster(b); return; }
      if (b.classname.indexOf('trigger_') === 0) return;
      var rm = b.rendermode || 0, ra = b.renderamt || 0;
      // as the game draws it: rendermode 2 at renderamt 0 is INVISIBLE (an invisible wall),
      // 5 is an additive glow, 4 an alpha-tested cut-out (the windows' frames, func_illusionary
      // *47): drawn opaque, the cut-out filled the window openings with a wall the runners
      // seemed to fly through (the user, 2026-10-04)
      if (rm === 2 && ra === 0) return;
      if (rm === 5) return;
      if (rm === 4) { addGlass(b, 0.14); return; }
      if (rm === 2) {
        // a translucent visual sitting on a booster is drawn by addBooster already
        if (b.classname === 'func_illusionary' && b.speed) return;
        addGlass(b, Math.max(0.12, ra / 255)); return;
      }
      var rgb = b.classname === 'func_button' ? [0.80, 0.30, 0.28] : null;
      addSolid(buildGeometry(b.positions, b.normals, b.indices, rgb));
    });
    return new Promise(function (resolve, reject) {
      mmImg = new Image();
      mmImg.onload = resolve;
      mmImg.onerror = function () { reject('minimap image failed: ' + data.minimap.url); };
      mmImg.src = data.minimap.url;
    });
  })
  .then(function () {
    renderFrame(0);
    window.nFrames = data.n;
    window.ready = true;
  })
  .catch(function (e) { window.loadError = String(e); });
