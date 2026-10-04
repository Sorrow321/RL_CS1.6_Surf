/* skate_laby agent-vs-WR render page: two first-person views of the map mesh (top: agent,
 * bottom: human WR), a HUD, and window.captureFrame(k) -> JPEG data URL for tools/demo/
 * vs_wr_render.py, which drives this page headless and encodes the frames.
 * Per-frame poses come from tools/demo/vs_wr_frames.py (?frames=<url>).
 * Coordinates: GoldSrc (x, y, z) -> three.js (x, z, -y), exactly as viewer/app.js. */
'use strict';

var W = 1280, PH = 720, BAR = 120, H = 2 * PH + BAR;
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
renderer.setSize(W, H);
wrap.appendChild(renderer.domElement);
var hud = document.createElement('canvas');
hud.width = W; hud.height = H;
wrap.appendChild(hud);
var hctx = hud.getContext('2d');
var comp = document.createElement('canvas');
comp.width = W; comp.height = H;
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

var camA = new THREE.PerspectiveCamera(VFOV, W / PH, 2, 40000);
var camH = new THREE.PerspectiveCamera(VFOV, W / PH, 2, 40000);
var data = null;
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

function panelHud(y0, tr, f, t) {
  var fin = t >= tr.finish;
  // a soft top band so the titles read on bright sky
  var g = hctx.createLinearGradient(0, y0, 0, y0 + 110);
  g.addColorStop(0, 'rgba(0,0,0,0.45)'); g.addColorStop(1, 'rgba(0,0,0,0)');
  hctx.fillStyle = g; hctx.fillRect(0, y0, W, 110);
  txt(tr.title, 28, y0 + 52, 38, 'bold');
  txt(tr.sub, 30, y0 + 88, 24, '600', 'left', '#e8e8e8');
  txt((fin ? tr.finish : t).toFixed(2) + ' s', W - 28, y0 + 66, 58, 'bold', 'right');
  txt(f[5] + ' u/s', 30, y0 + PH - 30, 34, 'bold');
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

// ---------------------------------------------------------------------------- per frame
function renderFrame(k) {
  k = Math.max(0, Math.min(data.n - 1, k));
  var t = k / data.fps;
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
  panelHud(0, data.agent, fa, t);
  barHud(k);
  panelHud(PH + BAR, data.human, fh, t);
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
      if (!b.classname || b.classname.indexOf('trigger_') === 0) return;
      var rgb = b.classname === 'func_button' ? [0.80, 0.30, 0.28] : null;
      addSolid(buildGeometry(b.positions, b.normals, b.indices, rgb));
    });
    renderFrame(0);
    window.nFrames = data.n;
    window.ready = true;
  })
  .catch(function (e) { window.loadError = String(e); });
