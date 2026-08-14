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
  html += '<h2>Metrics</h2>';
  if (keys.length) {
    html += '<div id="charts">' + keys.map(function (k, i) {
      var v = series[k].values;
      return '<div class="chart"><div class="t"><span>' + k + '</span><b>' +
        v[v.length - 1].toPrecision(4) + '</b></div>' +
        '<canvas id="ch' + i + '"></canvas></div>';
    }).join('') + '</div>';
  } else {
    html += '<div class="empty">No metrics logged for this run.</div>';
  }

  html += '<h2>Trajectories (greedy policy over training)' +
    ' <button class="rec" data-mode="stoch">⏺ record stoch @ latest</button>' +
    ' <button class="rec" data-mode="greedy">⏺ record greedy</button></h2>';
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

  keys.forEach(function (k, i) {
    drawChart(document.getElementById('ch' + i), series[k].steps, series[k].values);
  });
  Array.prototype.forEach.call(document.querySelectorAll('.watch'), function (b) {
    b.addEventListener('click', function () {
      window.open('index.html?traj=' + encodeURIComponent(b.dataset.f), '_blank');
    });
  });
  Array.prototype.forEach.call(document.querySelectorAll('.rec'), function (b) {
    b.addEventListener('click', function () { recordRun(b, r.name, b.dataset.mode); });
  });
}

// spawn a rollout recording from the run's ckpt_latest.pt (tools/record_ckpt)
// and refresh the run view when the new trajectory lands
function recordRun(btn, runName, mode) {
  btn.disabled = true;
  var orig = btn.textContent;
  btn.textContent = '⏺ recording…';
  var url = '/api/record?run=' + encodeURIComponent(runName) + '&mode=' + mode;
  (function tick() {
    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error('bad');
        return r.json();
      })
      .then(function (j) {
        if (j.status === 'done') { select(runName); }
        else if (j.status === 'started' || j.status === 'recording') {
          setTimeout(tick, 3000);
        } else { btn.disabled = false; btn.textContent = orig + ' ✗'; }
      })
      .catch(function () { btn.disabled = false; btn.textContent = orig + ' ✗'; });
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
