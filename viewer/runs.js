// runs.js — local W&B-style dashboard over tools/dashboard.py's JSON APIs.
'use strict';

var PREFERRED = [                       // chart order; anything else appends
  'rollout/ep_rew_mean', 'eval/fwd_max', 'eval/path', 'eval/speed_max',
  'train/blend_w', 'rollout/ep_len_mean', 'train/loss', 'train/value_loss',
  'train/entropy_loss', 'train/approx_kl', 'time/fps'
];
var runs = [];
var selected = null;
var side = document.getElementById('side');
var main = document.getElementById('main');

function fmtSteps(n) {
  if (n == null) return '—';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(0) + 'k';
  return '' + n;
}
function fmtDate(iso) {
  if (!iso) return '—';
  return iso.replace('T', ' ').slice(0, 16);
}
function fmtDur(s) {
  if (s == null) return '';
  return s >= 3600 ? (s / 3600).toFixed(1) + 'h' : (s / 60).toFixed(1) + 'min';
}

function renderSide() {
  side.innerHTML = '';
  runs.forEach(function (r) {
    var d = document.createElement('div');
    d.className = 'runcard' + (selected === r.name ? ' sel' : '');
    d.innerHTML =
      '<div class="name">' + r.label +
      '<span class="badge ' + r.status + '">' + r.status + '</span></div>' +
      '<div class="meta">' + fmtDate(r.started) +
      (r.duration_s ? ' · ' + fmtDur(r.duration_s) : '') +
      ' · ' + fmtSteps(r.steps) + ' steps · ' +
      r.trajs.length + ' trajs</div>';
    d.addEventListener('click', function () { select(r.name); });
    side.appendChild(d);
  });
  if (!runs.length) side.innerHTML = '<div class="empty">No runs yet.</div>';
}

function drawChart(canvas, steps, values) {
  var dpr = window.devicePixelRatio || 1;
  var w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  var lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
  if (hi === lo) { hi += 1; lo -= 1; }
  var x0 = steps[0], x1 = steps[steps.length - 1] || 1;
  if (x1 === x0) x1 = x0 + 1;
  var PX = function (s) { return 34 + (s - x0) / (x1 - x0) * (w - 40); };
  var PY = function (v) { return h - 14 - (v - lo) / (hi - lo) * (h - 22); };
  ctx.strokeStyle = '#262a30';
  ctx.beginPath();
  [lo, (lo + hi) / 2, hi].forEach(function (v) {
    ctx.moveTo(34, PY(v)); ctx.lineTo(w - 6, PY(v));
  });
  ctx.stroke();
  ctx.fillStyle = '#6a736a';
  ctx.font = '9px system-ui';
  ctx.fillText(hi.toPrecision(3), 2, PY(hi) + 3);
  ctx.fillText(lo.toPrecision(3), 2, PY(lo) + 3);
  ctx.fillText(fmtSteps(x0), 34, h - 3);
  ctx.fillText(fmtSteps(x1), w - 34, h - 3);
  ctx.strokeStyle = '#7fd07f';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (var i = 0; i < steps.length; i++) {
    var x = PX(steps[i]), y = PY(values[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function renderMain(r, series) {
  var html = '<div id="runhead">' + r.label +
    '<span class="badge ' + r.status + '">' + r.status + '</span>' +
    '<div class="meta">started ' + fmtDate(r.started) +
    (r.finished ? ' · finished ' + fmtDate(r.finished) : '') +
    (r.duration_s ? ' · ' + fmtDur(r.duration_s) : '') +
    ' · ' + fmtSteps(r.steps) + ' steps' +
    (Object.keys(r.config || {}).length
      ? '<br>' + Object.entries(r.config).map(function (kv) {
          return kv[0] + '=' + kv[1];
        }).join('  ') : '') +
    '</div></div>';

  var keys = Object.keys(series);
  keys.sort(function (a, b) {
    var ia = PREFERRED.indexOf(a), ib = PREFERRED.indexOf(b);
    if (ia < 0) ia = 99; if (ib < 0) ib = 99;
    return ia - ib || a.localeCompare(b);
  });
  // A multi-map run emits four series PER MAP (race/map_pct.<map> and
  // friends). At 40+ maps that is 160+ charts and the page is unusable, so
  // per-map series are collapsed behind a <details> and the aggregates
  // (race/map_pct, race/maps_finished, ...) stay in view. Indices come from
  // the FULL sorted key list, so the draw loop below is unchanged.
  var isPerMap = function (k) {
    return /^race\/[A-Za-z0-9_]+\.[A-Za-z0-9_.-]+$/.test(k);
  };
  var cell = function (k, i) {
    var v = series[k].values;
    return '<div class="chart"><div class="t"><span>' + k + '</span><b>' +
      v[v.length - 1].toPrecision(4) + '</b></div>' +
      '<canvas id="ch' + i + '"></canvas></div>';
  };
  var aggHtml = '', perHtml = '', nPer = 0;
  keys.forEach(function (k, i) {
    if (isPerMap(k)) { perHtml += cell(k, i); nPer++; } else { aggHtml += cell(k, i); }
  });

  html += '<h2>Metrics</h2>';
  if (keys.length) {
    if (aggHtml) html += '<div id="charts">' + aggHtml + '</div>';
    if (nPer) {
      html += '<details id="permap"><summary>Per-map series (' + nPer +
        ' charts) - hidden by default</summary>' +
        '<div id="charts-permap" class="charts">' + perHtml + '</div></details>';
    }
  } else {
    html += '<div class="empty">No metrics logged for this run.</div>';
  }

  html += '<h2>Trajectories (greedy policy over training)' +
    ' <button class="rec" data-mode="stoch" data-key="' + recKey(r.name, 'stoch') +
    '" data-label="⏺ record stoch @ latest">⏺ record stoch @ latest</button>' +
    ' <button class="rec" data-mode="greedy" data-key="' + recKey(r.name, 'greedy') +
    '" data-label="⏺ record greedy">⏺ record greedy</button>' +
    (r.config && r.config.reward === 'race'
      // race recordings default to the start line — these sample the
      // scan-based drop pool / the run's actual respawn reservoir instead
      ? ' <button class="rec" data-mode="stoch" data-spawn="mixed" data-key="' +
        recKey(r.name, 'stoch', 'mixed') + '" data-label="⏺ record drop spawns">' +
        '⏺ record drop spawns</button>' +
        (r.config.respawn_frac
          ? ' <button class="rec" data-mode="stoch" data-spawn="reservoir" data-key="' +
            recKey(r.name, 'stoch', 'reservoir') +
            '" data-label="⏺ record frontier">⏺ record frontier</button>'
          : '')
      : '') + '</h2>';
  if (r.trajs.length) {
    html += '<div id="artifacts">' + r.trajs.map(function (t) {
      return '<div class="art"><div><div class="s">@ ' + fmtSteps(t.steps) +
        ' steps · ' + (t.mode || 'greedy') + '</div><div class="kb">' +
        t.kb + ' KB</div></div>' +
        '<button class="watch" data-f="' + t.file + '">▶ watch</button></div>';
    }).join('') + '</div>';
  } else {
    html += '<div class="empty">No trajectory artifacts.</div>';
  }

  if (r.checkpoints.length) {
    html += '<h2>Checkpoints</h2><div id="ckpts">' +
      r.checkpoints.map(function (c) {
        return '<a href="/runs/' + r.name + '/' + c + '" download>' + c + '</a>';
      }).join('') + '</div>';
  }
  main.innerHTML = html;

  // A canvas inside a CLOSED <details> has clientWidth 0, so drawing it now
  // would produce a blank chart that never repaints. Draw the visible ones,
  // and draw the per-map ones the first time the section is opened.
  var drawKeys = function (pred) {
    keys.forEach(function (k, i) {
      if (!pred(k)) return;
      var c = document.getElementById('ch' + i);
      if (c) drawChart(c, series[k].steps, series[k].values);
    });
  };
  drawKeys(function (k) { return !isPerMap(k); });
  var det = document.getElementById('permap');
  if (det) {
    det.addEventListener('toggle', function () {
      if (det.open) drawKeys(isPerMap);
    });
  }
  Array.prototype.forEach.call(document.querySelectorAll('.watch'), function (b) {
    b.addEventListener('click', function () {
      window.open('index.html?traj=' + encodeURIComponent(b.dataset.f), '_blank');
    });
  });
  Array.prototype.forEach.call(document.querySelectorAll('.rec'), function (b) {
    b.addEventListener('click', function () {
      recordRun(b, r.name, b.dataset.mode, b.dataset.spawn);
    });
  });
  applyRecordingState();
}

// spawn a rollout recording from the run's ckpt_latest.pt (tools/record_ckpt)
// and refresh the run view when the new trajectory lands
// In-flight recordings live HERE, not on the button element. poll() runs
// every 4s and, for a live run, calls select() -> renderMain(), which does
// main.innerHTML = html and DESTROYS every button. The old code captured the
// button in a closure, so after the first re-render it wrote "recording... Ns"
// into a detached node while a pristine, enabled button sat in the DOM. The
// job ran to completion on the box and the UI showed nothing the whole time -
// which is indistinguishable from a dead button, and is exactly what it was
// reported as. Keyed state + re-applying it on every render fixes it.
var recording = {};

function recKey(runName, mode, spawn) {
  return runName + '|' + mode + '|' + (spawn || 'default');
}

// re-apply in-flight/error state to the buttons that renderMain just created
function applyRecordingState() {
  Array.prototype.forEach.call(document.querySelectorAll('.rec'), function (b) {
    var st = recording[b.dataset.key];
    if (!st) return;
    if (st.error) {
      b.disabled = false;
      b.textContent = b.dataset.label + ' ✗ ' + (st.error || '');
    } else {
      b.disabled = true;
      // a real percentage from the recorder's own tick counter; falls back to
      // elapsed seconds only until the first progress write lands
      var secs = Math.round((Date.now() - st.t0) / 1000);
      // startup dominates a recording, so show the PHASE, not just a tick
      // percentage that sits at 0 for the first 40s and looks hung
      var ep = (st.ep && st.eps) ? (' ep ' + st.ep + '/' + st.eps) : '';
      var ph = st.phase || 'starting';
      b.textContent = (st.pct === null || st.pct === undefined)
        ? '⏺ ' + ph + '… ' + secs + 's'
        : '⏺ ' + st.pct + '% ' + ph + ep + ' · ' + secs + 's';
    }
  });
}

function recordRun(btn, runName, mode, spawn) {
  var key = recKey(runName, mode, spawn);
  if (recording[key] && !recording[key].error) return;   // already running
  recording[key] = { t0: Date.now(), error: false, pct: null };
  applyRecordingState();
  var url = '/api/record?run=' + encodeURIComponent(runName) + '&mode=' + mode +
    (spawn ? '&spawn=' + spawn : '');
  (function tick() {
    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error('bad');
        return r.json();
      })
      .then(function (j) {
        if (j.status === 'done') {
          delete recording[key];
          if (selected === runName) select(runName);   // pick up the new traj
        } else if (j.status === 'started' || j.status === 'recording') {
          if (recording[key]) {
            recording[key].pct = j.pct;
            recording[key].ep = j.episode;
            recording[key].eps = j.episodes;
            recording[key].phase = j.phase;
          }
          applyRecordingState();                        // survives re-render
          setTimeout(tick, 1000);
        } else {
          // surface the server's reason instead of a bare X
          recording[key].error = (j.error || j.status || 'failed')
            .toString().slice(0, 60);
          applyRecordingState();
        }
      })
      .catch(function () {
        if (recording[key]) { recording[key].error = true; applyRecordingState(); }
      });
  })();
}

function select(name) {
  selected = name;
  renderSide();
  var r = runs.find(function (x) { return x.name === name; });
  if (!r) return;
  fetch('/api/metrics?run=' + encodeURIComponent(name))
    .then(function (resp) { return resp.json(); })
    .then(function (m) { renderMain(r, m.series || {}); })
    .catch(function () { renderMain(r, {}); });
}

function poll() {
  fetch('/api/runs')
    .then(function (r) { return r.json(); })
    .then(function (j) {
      runs = j.runs || [];
      renderSide();
      if (!selected && runs.length) select(runs[0].name);
      else if (selected) {
        var cur = runs.find(function (x) { return x.name === selected; });
        if (cur && cur.status === 'live') select(selected);   // live refresh
      }
      document.getElementById('refresh').textContent =
        'updated ' + new Date().toTimeString().slice(0, 8);
    })
    .catch(function () {});
}
poll();
setInterval(poll, 4000);
