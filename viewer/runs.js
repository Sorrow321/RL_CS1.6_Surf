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

// The run view is rebuilt wholesale every 4 s (setInterval(poll, 4000) ->
// select() -> main.innerHTML = html), which DESTROYS the <details> element
// and with it the open state - so an opened per-map section closed itself a
// second later. Remember it outside the render.
var permapOpen = false;
// which map buckets are open - like permapOpen, this must live outside the
// render because the 4 s poll rebuilds main.innerHTML wholesale
var mapOpen = {};

// The record buttons, for ONE map (tag) or for the run as a whole (null).
// Recording is per-map on a --maps run: /api/record takes &map=, and without
// it record_ckpt.py falls back to whichever map the checkpoint names first.
function recButtons(r, tag) {
  var sfx = tag ? ' ' + tag : '';
  var b = function (mode, spawn, label) {
    return ' <button class="rec" data-mode="' + mode + '"' +
      (spawn ? ' data-spawn="' + spawn + '"' : '') +
      (tag ? ' data-map="' + tag + '"' : '') +
      ' data-key="' + recKey(r.name, mode, spawn || undefined, tag) +
      '" data-label="' + label + '">' + label + '</button>';
  };
  var out = b('stoch', null, '⏺ stoch' + sfx) +
            b('greedy', null, '⏺ greedy' + sfx);
  if (r.config && r.config.reward === 'race') {
    out += b('stoch', 'mixed', '⏺ drop spawns' + sfx);
    if (r.config.respawn_frac) {
      out += b('stoch', 'reservoir', '⏺ frontier' + sfx);
    }
  }
  return out;
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
      html += '<details id="permap"' + (permapOpen ? ' open' : '') +
        '><summary>Per-map series (' + nPer +
        ' charts) - hidden by default</summary>' +
        '<div id="charts-permap" class="charts">' + perHtml + '</div></details>';
    }
  } else {
    html += '<div class="empty">No metrics logged for this run.</div>';
  }

  // On a --maps run the record buttons live inside each map's bucket: a
  // single run-level "record greedy" would silently pick whichever map the
  // checkpoint names first, which is a wrong recording rather than an error.
  var multi = !!(r.config && r.config.maps && r.config.maps.length > 1);
  html += '<h2>Trajectories (greedy policy over training)' +
    (multi ? '<span class="meta"> - record buttons are per map, below</span>'
           : recButtons(r, null)) + '</h2>';
  var artCell = function (t) {
    return '<div class="art"><div><div class="s">@ ' + fmtSteps(t.steps) +
      ' steps · ' + (t.mode || 'greedy') + '</div><div class="kb">' +
      t.kb + ' KB</div></div>' +
      '<button class="watch" data-f="' + t.file + '">▶ watch</button></div>';
  };
  if (r.trajs.length) {
    // A --maps run writes one recording per map per eval, so 40 maps x N
    // evals is an unreadable wall. Bucket by map, collapsed, with THAT
    // map's record buttons inside its own bucket - recording is per-map
    // now, and a single global "record greedy" would silently pick
    // whichever map the checkpoint happens to name first.
    var byMap = {}, plain = [];
    r.trajs.forEach(function (t) {
      if (t.map) { (byMap[t.map] = byMap[t.map] || []).push(t); } else { plain.push(t); }
    });
    var tags = Object.keys(byMap).sort();
    if (plain.length) {
      html += '<div id="artifacts">' + plain.map(artCell).join('') + '</div>';
    }
    tags.forEach(function (tag) {
      var ts = byMap[tag];
      html += '<details class="mapgrp" data-map="' + tag + '"' +
        (mapOpen[tag] ? ' open' : '') + '><summary>' + tag +
        ' <span class="n">(' + ts.length + ')</span></summary>' +
        '<div class="recrow">' + recButtons(r, tag) + '</div>' +
        '<div class="artifacts">' + ts.map(artCell).join('') + '</div></details>';
    });
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
  Array.prototype.forEach.call(document.querySelectorAll('details.mapgrp'),
    function (dg) {
      dg.addEventListener('toggle', function () {
        mapOpen[dg.dataset.map] = dg.open;
      });
    });
  var det = document.getElementById('permap');
  if (det) {
    det.addEventListener('toggle', function () {
      permapOpen = det.open;
      if (det.open) drawKeys(isPerMap);
    });
    // restored open by a refresh: the canvases are new and blank, and no
    // toggle event fires, so draw them here
    if (det.open) drawKeys(isPerMap);
  }
  Array.prototype.forEach.call(document.querySelectorAll('.watch'), function (b) {
    b.addEventListener('click', function () {
      window.open('index.html?traj=' + encodeURIComponent(b.dataset.f), '_blank');
    });
  });
  Array.prototype.forEach.call(document.querySelectorAll('.rec'), function (b) {
    b.addEventListener('click', function () {
      recordRun(b, r.name, b.dataset.mode, b.dataset.spawn, b.dataset.map);
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

function recKey(runName, mode, spawn, map) {
  // the map is part of the identity: recording map A must not mark map B's
  // button as in-flight
  return runName + '|' + mode + '|' + (spawn || 'default') + '|' + (map || 'all');
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

function recordRun(btn, runName, mode, spawn, map) {
  var key = recKey(runName, mode, spawn, map);
  if (recording[key] && !recording[key].error) return;   // already running
  recording[key] = { t0: Date.now(), error: false, pct: null };
  applyRecordingState();
  var url = '/api/record?run=' + encodeURIComponent(runName) + '&mode=' + mode +
    (spawn ? '&spawn=' + spawn : '') +
    (map ? '&map=' + encodeURIComponent(map) : '');
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
