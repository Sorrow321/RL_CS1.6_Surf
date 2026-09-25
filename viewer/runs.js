// runs.js - local W&B-style dashboard over tools/dashboard.py's JSON APIs.
//
// Charts are uPlot v1.6.32 (MIT), vendored at viewer/vendor/uPlot.iife.min.js
// and loaded by runs.html. NOT a CDN on purpose: this page has to work on a
// rented box reached through an ssh tunnel, where the browser's outbound
// internet is the box's, and on a workstation with no network at all.
'use strict';

// Source stays pure ASCII (the console this is developed from is cp1251);
// the glyphs the UI has always used come back in as escapes.
var GL = {
  dash: '\u2014', dot: '\u00b7', rec: '\u23fa', play: '\u25b6',
  cross: '\u2717', ell: '\u2026'
};

var PREFERRED = [                       // chart order; anything else appends
  // an expert loop's scoreboard (greedy start-line clock per round) first
  'loop/greedy_best_s', 'loop/greedy_mean_s', 'loop/planner_s', 'loop/finishes_of_9',
  // --goal-planner primlearn: the two questions - does the planner advance us toward the
  // finish, does the executor do what the planner asks
  'plan/eval_finish', 'plan/ep_prog_start', 'plan/finish_start', 'plan/adv_plan',
  'plan/adv_real', 'exec/complete', 'exec/arc_frac',
  'rollout/ep_rew_mean', 'eval/fwd_max', 'eval/path', 'eval/speed_max',
  'train/blend_w', 'rollout/ep_len_mean', 'train/loss', 'train/value_loss',
  'train/entropy_loss', 'train/approx_kl', 'time/fps'
];
// distinct, colour-blind-tolerable, readable on #171a1f
var COLORS = ['#7fd07f', '#6fb2e0', '#e0a35c', '#c98be0', '#e07f7f',
              '#66d0c0', '#c8c46a', '#9aa3ae'];

// --------------------------------------------- what the plots mean --------
// One line per metric family and per metric (base key, the .<map> suffix
// stripped). Shown as the chart title's tooltip and in the "what the plots
// mean" panel. Keep these honest: CLAUDE.md records three metrics that lied.
var GROUP_DESC = {
  race: 'the race reward: TRAINING-side rates (success_rate, finish_s, stall) and the greedy EVAL from the true map start (eval_*, map_pct, maps_finished) - only the eval columns are a verdict',
  rollout: 'the training rollout itself: episode return and length over the spawn pool (90% mid-map spawns)',
  train: 'the PPO optimiser: losses, approximate KL, explained variance',
  time: 'throughput and the step counter',
  dip: 'the setback diagnostic: how deep a loss of progress the policy recovers from, and how deep the one it dies in is (training side)',
  front: 'the forward frontier curriculum (--respawn-frontier): where the start-anchored frontier is, how far ahead spawns are allowed, and where they actually landed',
  gate: '--gate-boxes: did training episodes reach the ramps the dive skips (hit share, per-box episode counts) and what became of them: return / finish / death / length of the hitting vs the missing episodes',
  back: 'the backward curriculum (--respawn-backward): the goal-anchored band widening toward the start, and the far shell\'s finish rate that drives it',
  act: 'action statistics of the rollout: what the policy does with the keys in the air',
  bc: 'behaviour cloning / DAgger losses against the expert rows',
  eval: 'the pre-race eval metrics (forward distance, path, speed) on the greedy policy',
  tail: '--tail-weight: the importance-weight statistics of the tail resampling',
  tick: 'the physics tick under --tick-ms schedules',
  loop: 'an expert loop\'s per-round scoreboard: the planner\'s line and the policy\'s greedy start-line clock',
  held: 'held-out maps (never trained on): the generalisation probe',
  cc: '--curiosity-cond: the temperature buckets of the T-conditioned family',
  exec: '--goal-planner primlearn, the EXECUTOR: does the policy do what the planner asks? Over the primitives the planner chose (closed in this log window)',
  plan: '--goal-planner learned / primlearn, the PLANNER: does it advance us toward the finish? plus its own PPO statistics',
  unstuck: '--unstuck: the plateau-driven sampling temperature T, how long the progress measure has been flat, and the measure itself (the reservoir reach, or under --unstuck-reach alive the deepest point an episode reached and survived --unstuck-hold s later)'
};
var DESC = {
  'exec/complete': 'Share of the planner\'s primitives the executor COMPLETED: covered 90% of the curve inside a 192 u corridor within 1.5x its duration. The direct measure of "does the policy do what the planner asks".',
  'exec/arc_frac': 'Mean fraction of each planner primitive the executor covered inside the corridor (capped at 1). Partial following shows here before it shows in exec/complete.',
  'exec/complete_unif': 'The same completion rate for the UNIFORM random primitives that open some episodes: the executor\'s skill on the whole primitive space, not just on what the planner likes.',
  'plan/adv_plan': 'Progress toward the finish (u, Euclidean) the chosen primitive PROMISES: finish distance at its start minus at its end point, if followed perfectly. Mean per planner primitive. Positive = the planner aims at the finish.',
  'plan/adv_real': 'Progress toward the finish (u) actually made during each planner primitive (never positive on a death). adv_real near adv_plan = the executor delivers what was planned; adv_plan > 0 with adv_real ~0 = plans that cannot be flown.',
  'plan/plan_fwd': 'Share of the planner\'s primitives whose end point is closer to the finish than their start.',
  'plan/death': 'Share of the planner\'s primitives that ended in a death (the episode failed during the primitive).',
  'plan/ep_prog': 'Per TRAINING episode: the closest ALIVE approach to the finish as a fraction of the spawn\'s distance, (d_spawn - d_min) / d_spawn, 1 = finished. Mean over ended episodes (mostly mid-map spawns).',
  'plan/ep_prog_start': 'plan/ep_prog over the episodes spawned at the MAP START only: how far along the map the fleet gets from the real start (Euclidean, 1 = finished).',
  'plan/finish': 'Share of ended training episodes that crossed the finish box.',
  'plan/finish_start': 'Finish share of the training episodes spawned at the map start.',
  'plan/eval_finish': 'The VERDICT: greedy eval from the map start - planner greedy (the mixture\'s heaviest mean) + executor greedy - fraction of episodes that crossed the finish box.',
  'plan/chosen': 'Primitives the planner chose in this log window.',
  'plan/uniform': 'Uniform random primitives drawn to open episodes (--plan-uniform share of episodes).',
  'plan/closed': 'Planner primitives closed (completed, timed out, or the episode ended) - the planner\'s transitions.',
  'plan/reward': 'The planner\'s mean reward per primitive (progress per 1000 u + finish bonus + end-cell novelty).',
  'plan/novelty': 'Mean end-cell novelty bonus per planner primitive (0.5 / sqrt(visits) over 128 u cells).',
  'plan/entropy': 'Mean entropy of the planner\'s mixture at its choices; a collapse to ~0 means it always asks for the same primitive.',
  'plan/cover': '128 u cells the training fleet has visited alive (cumulative).',
  'race/eval_progress': 'Greedy eval from the map start: mean over episodes of (field distance at spawn - the minimum reached), map units. Saturates at the field\'s on-route minimum and is flattered by dives; on a joint run the pooled value is a units mean over maps. Not the verdict.',
  'race/eval_finish_s': 'Mean finish time in seconds (spawn clock) of the greedy eval episodes that FINISHED; empty until the first finish. This is the scoreboard clock.',
  'race/eval_finishes': 'How many of the greedy eval episodes crossed the finish box, the env\'s own test (hull-inflated sweep incl. the unrecorded last tick).',
  'race/map_pct': 'Greedy eval coverage: share of the map\'s own route covered from spawn, mean over episodes; on a joint run the mean over maps. 100% can be reached without finishing (read with maps_finished).',
  'race/map_pct_trigger': 'map_pct restricted to trigger-finish maps (button finishes cannot be pressed by the simulator, so they are weaker evidence).',
  'race/maps_finished': 'Fraction of maps where at least one greedy eval episode crossed the finish box.',
  'race/maps_finished_trigger': 'maps_finished restricted to trigger-finish maps.',
  'race/success_rate': 'TRAINING win rate over the spawn pool, 90% of which are mid-map spawns: it rises when spawns get close to the goal, i.e. it measures the harvest, not the policy. Read it beside reservoir min-depth.',
  'race/finish_s': 'Mean TRAINING finish time from the SPAWN point (mostly mid-map spawns). Not the eval clock.',
  'race/trunc_frac': 'Share of this iteration\'s ended episodes cut by the tick cap.',
  'race/stall_frac': 'Share of this iteration\'s episodes ended by the stall kill (no progress for --stall-secs).',
  'race/crawl_frac': 'Share of ended episodes whose mean horizontal speed was a crawl.',
  'race/surf_paid_frac': '--surf-bonus: fraction of airborne ticks the bonus paid on; near 1.0 means a farm, not surfing.',
  'race/dive_frac': '--surf-bonus: airborne ticks with nothing holding the player up (free fall).',
  'rollout/ep_rew_mean': 'Mean episode return in TRAINING. Not comparable across spawn distributions (a curriculum changes the population it is taken over).',
  'rollout/ep_len_mean': 'Mean training episode length in physics ticks (10 ms). Pinned at ~1,502 means every episode is stall-killed at 15 s.',
  'train/loss': 'Total PPO loss per update.',
  'train/value_loss': 'Critic (value) loss per update.',
  'train/entropy_loss': 'Entropy term of the loss (the entropy bonus, negated).',
  'train/approx_kl': 'Approximate KL between the rollout policy and the updated one; tracks passes over the buffer, not gradient steps.',
  'train/explained_var': '1 - Var(G - V) / Var(G) over the rollout buffer; +1 means the critic explains the returns, <= 0 means it explains nothing.',
  'train/blend_w': 'Blend weight of a scheduled reward switch (--blend).',
  'train/ret_mean': '--ret-norm: running mean of returns.',
  'train/ret_std': '--ret-norm: running std of returns.',
  'time/fps': 'CUMULATIVE steps/s since the process started - includes startup, compile and every eval. Not the current rate; see fps_inst.',
  'time/fps_inst': 'Instantaneous steps/s derived from the cumulative rate, smoothed over 9 rows: the current throughput.',
  'time/total_timesteps': 'Environment steps (physics ticks x envs) since the start of training; the x axis.',
  'tick/tick_ms': 'The physics tick in ms (a --tick-ms schedule moves it).',
  'dip/max_survived_depth': 'The deepest setback (progress lost, in the field\'s units) the policy came BACK from this iteration.',
  'dip/p90_survived_depth': 'The depth the top decile of recovered dips ran to.',
  'dip/max_survived_secs': 'The longest survived dip, in seconds.',
  'dip/survived_per_ep': 'Survived dips per ended episode.',
  'dip/fail_depth': 'Mean depth of the dip an episode DIED in.',
  'dip/fail_secs': 'Mean duration of the dip an episode died in.',
  'dip/fail_frac': 'Share of ended episodes that ended inside a dip (a setback it did not recover from).',
  'dip/p50_term_depth': 'Median terminal dip depth over ended episodes.',
  'dip/p90_term_depth': 'p90 terminal dip depth over ended episodes.',
  'front/pmax': 'The frontier: the deepest field progress an episode that started AT the map start (from rest) reaches, over the window - the p90 of those reaches under --respawn-frontier-quantile. The curriculum cannot inflate it.',
  'front/cap': 'How deep curriculum spawns are allowed: (1 + headroom + grow) x pmax, capped at the map length.',
  'front/grow': 'The plateau term: rises while pmax is stuck so the cap creeps forward; 0 while the frontier keeps moving.',
  'front/pmax_all': 'The same frontier counting EVERY episode, curriculum spawns included; the gap to pmax is what the curriculum bought. A win rate rising while only this moves is the harvest trap.',
  'front/spawn_med': 'Median field progress of where training episodes actually started this iteration.',
  'front/spawn_p90': 'p90 of where training episodes actually started; near the map length means spawns sit next to the goal.',
  'front/harvest_drop': '--respawn-frontier-anchor: share of harvested reservoir snapshots dropped for lying beyond the cap.',
  'gate/hit_frac': 'Share of this iteration\'s ended episodes (spawned before the gate) that entered any gate box - the ramps the dive skips.',
  'gate/n_end': 'Episodes counted this iteration (spawned at or above spawn_d_min).',
  'gate/hit_frac_all': 'The same hit share over EVERY ended episode, spawns on or past the ramp included (the curriculum\'s deep spawns).',
  'gate/hit_ret': 'Mean training return of the episodes that entered a gate box.',
  'gate/miss_ret': 'Mean training return of the episodes that did not.',
  'gate/hit_fin': 'Finish rate of the episodes that entered a gate box.',
  'gate/miss_fin': 'Finish rate of the episodes that did not.',
  'gate/hit_fail': 'Death rate (ended by the core, not finished) of the episodes that entered a gate box.',
  'gate/miss_fail': 'Death rate of the episodes that did not.',
  'gate/hit_len': 'Mean length (ticks) of the episodes that entered a gate box.',
  'gate/miss_len': 'Mean length (ticks) of the episodes that did not.',
  'gate/rise_mean': 'Mean over ended episodes of the largest potential RISE the episode accepted (d minus its running minimum, map units) - does the agent try the dip at all.',
  'gate/rise_p90': 'p90 of that accepted rise over the iteration\'s ended episodes.',
  'gate/dip_frac': 'Share of ended episodes that gave back more than 1,000 u of potential at some point (the dip takers).',
  'gate/vmax_mean': 'Mean top horizontal speed (u/s) reached per episode - the pit is where the record builds its 1,800 u/s.',
  'unstuck/T': 'The sampling temperature the rollout of this iteration ran at (0 = the plain policy; 1 = logits halved on the tempered heads).',
  'unstuck/stuck_steps': 'Env steps since the progress measure last improved by --unstuck-eps.',
  'unstuck/best': 'All-time best of the progress measure (map units of depth).',
  'unstuck/reach': '--unstuck-reach alive: the deepest point (d0 minus d, map units) any episode of the iteration reached and was still alive --unstuck-hold s later; a finish counts as d0, the last seconds of a fall do not count.',
  'unstuck/reach_n': 'How many ended episodes qualified for that reading (all spawns, or only those at d >= --unstuck-reach-spawn-d).',
  'back/W_frac': '--respawn-backward: the band\'s far edge as a fraction of the map; spawns are drawn between the goal and here. 1.0 = the whole path.',
  'back/shell_rate': 'Finish rate of episodes spawned in the band\'s far shell (the hardest part); reaching --respawn-backward-rate widens the band.',
  'back/shell_n': 'How many far-shell episodes that rate is over.',
  'back/n_adv': 'Advances (band widenings) so far.',
  'back/spawn_med': 'Median field progress of where backward spawns landed.',
  'back/spawn_p90': 'p90 of where backward spawns landed.',
  'act/duck_air': 'Share of airborne decisions holding duck.',
  'act/fwd_air': 'Share of airborne decisions holding forward.',
  'act/jump_air': 'Share of airborne decisions pressing jump.',
  'act/strafe_flip': 'A<->D strafe flips per second (the human record does ~0.4).',
  'act/yaw_side_agree': 'Share of decisions where the yaw offset agrees with the strafe key\'s side.',
  'bc/ce_dist': 'Behaviour cloning: cross-entropy of the policy against the expert action distribution.',
  'bc/head_acc': 'Behaviour cloning: per-head accuracy against the expert rows.',
  'bc/joint_acc': 'Behaviour cloning: joint (all heads) accuracy against the expert rows.',
  'bc/value_mse': 'Behaviour cloning: value-head MSE against the expert returns.',
  'eval/fwd_max': 'Greedy eval: maximum forward distance reached (pre-race metric).',
  'eval/path': 'Greedy eval: path length (pre-race metric).',
  'eval/speed_max': 'Greedy eval: maximum speed reached (pre-race metric).',
  'tail/cov': '--tail-weight: coverage of the tail groups.', 'tail/ess': '--tail-weight: effective sample size of the weights.',
  'tail/groups': '--tail-weight: number of tail groups.', 'tail/n_med': '--tail-weight: median group size.',
  'tail/p50': '--tail-weight: median weight.', 'tail/p75': '--tail-weight: p75 weight.', 'tail/p90': '--tail-weight: p90 weight.',
  'tail/w_max': '--tail-weight: maximum weight.', 'tail/w_p90': '--tail-weight: p90 weight.',
  'loop/greedy_best_s': 'Expert loop: best greedy finish time from the start line this round (s).',
  'loop/greedy_mean_s': 'Expert loop: mean greedy finish time from the start line this round (s).',
  'loop/planner_s': 'Expert loop: the planner\'s line time at round start (s).',
  'loop/finishes_of_9': 'Expert loop: greedy finishes out of 9 start-line episodes.'
};
function baseKey(k) { return k.replace(/\.[A-Za-z0-9_.-]+$/, ''); }
function groupOf(k) { var i = k.indexOf('/'); return i < 0 ? k : k.slice(0, i); }
function tagOf(k) { var m = k.match(/\.([A-Za-z0-9_.-]+)$/); return m ? m[1] : null; }
function descOf(k) {
  var b = baseKey(k);
  if (DESC[b]) return DESC[b];
  if (isPerMap(k) && DESC[b] === undefined) return GROUP_DESC[groupOf(k)] || '';
  return GROUP_DESC[groupOf(k)] || '';
}
function escAttr(s) { return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;'); }

var runs = [];
var selected = null;
var side = document.getElementById('side');
var main = document.getElementById('main');
var UP = window.uPlot || null;          // null = vendor file failed to load
var SYNC = UP ? UP.sync('rlsurf-x') : null;

// ---------------------------------------------------------------- state ----
// Everything the user "set" survives a reload; everything they are LOOKING at
// (zoom) survives the 4 s poll. The two are different lifetimes on purpose:
// a zoom is about the current question, the controls are a preference.
var LS = 'rlsurf.dashboard.v1';
var st = {
  xaxis: 'steps', smooth: 0, log: {}, cmp: [], hidden: {},
  permap: false, mapOpen: {}, hideGroup: {}, mapq: '', legend: false
};
try {
  var saved = JSON.parse(localStorage.getItem(LS) || '{}');
  Object.keys(st).forEach(function (k) {
    if (saved[k] !== undefined && saved[k] !== null) st[k] = saved[k];
  });
} catch (e) { /* private mode / corrupt value: defaults are fine */ }
function saveState() {
  try { localStorage.setItem(LS, JSON.stringify(st)); } catch (e) {}
}

// ----------------------------------------------------------- formatting ----
function fmtSteps(n) {
  if (n == null) return GL.dash;
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(0) + 'k';
  return '' + n;
}
function fmtDate(iso) {
  if (!iso) return GL.dash;
  return iso.replace('T', ' ').slice(0, 16);
}
function fmtDur(s) {
  if (s == null) return '';
  return s >= 3600 ? (s / 3600).toFixed(1) + 'h' : (s / 60).toFixed(1) + 'min';
}
// A whole axis of ticks. The unit is picked ONCE, from the largest tick,
// because a column reading 8.4M / 749.5M / 1.5B is unreadable next to
// itself; the decimals come from the tick SPACING, so a 0.5B increment
// reads 1.0B / 1.5B and a 1B increment reads 1B / 2B. `plain` keeps a
// count (the iteration axis) as an integer instead of scaling it to k.
function tickList(splits, incr, suffix, plain) {
  var unit = 1, suf = suffix || '', mx = 0;
  splits.forEach(function (v) { mx = Math.max(mx, Math.abs(v)); });
  if (!suffix && !plain) {
    if (mx >= 1e9) { unit = 1e9; suf = 'B'; }
    else if (mx >= 1e6) { unit = 1e6; suf = 'M'; }
    else if (mx >= 1e3) { unit = 1e3; suf = 'k'; }
    else if (mx > 0 && mx < 1e-3) {
      return splits.map(function (v) { return v.toExponential(1); });
    }
  }
  var d = incr > 0 ? Math.ceil(-Math.log10(incr / unit)) : 0;
  d = Math.max(unit > 1 ? 1 : 0, Math.min(3, d));
  return splits.map(function (v) { return (v / unit).toFixed(d) + suf; });
}
// A value in a tooltip or a header: 3-4 significant digits, times as seconds.
function fmtVal(v, key) {
  if (v == null || !isFinite(v)) return GL.dash;
  var a = Math.abs(v), s;
  if (a === 0) s = '0';
  else if (a >= 1e9) s = (v / 1e9).toFixed(3) + 'B';
  else if (a >= 1e6) s = (v / 1e6).toFixed(3) + 'M';
  else if (a >= 1e5) s = (v / 1e3).toFixed(2) + 'k';
  else if (a < 1e-4) s = v.toExponential(2);
  else s = String(Number(v.toPrecision(4)));
  if (key && /_s$/.test(key)) s += ' s';        // race/finish_s and friends
  return s;
}

// The X axes. "wall" is derived server-side from time/fps (progress.csv has
// no timestamp column at all - see _wall_hours); "rel" is that, normalised
// per run, which is the only axis on which two arms of different speed line
// up point for point.
var AXES = {
  steps: {
    label: 'env steps', needs: null, suffix: '',
    val: function (v) { return fmtSteps(v) + ' steps'; }
  },
  iter: {
    label: 'iteration', needs: 'iter', suffix: '', plain: true,
    val: function (v) { return 'iter ' + Math.round(v); }
  },
  wall: {
    label: 'wall-clock (h)', needs: 'wall', suffix: 'h',
    val: function (v) { return v.toFixed(3) + ' h'; }
  },
  rel: {
    label: 'relative time (%)', needs: 'wall', suffix: '%',
    val: function (v) { return v.toFixed(1) + '% of run'; }
  }
};
function axisOK(id, names) {           // available for EVERY plotted run?
  var need = AXES[id].needs;
  if (!need) return true;
  return names.every(function (n) {
    var m = metricsCache[n];
    return m && m.axes && m.axes.indexOf(need) >= 0;
  });
}
function activeAxis(names) {
  return axisOK(st.xaxis, names) ? st.xaxis : 'steps';
}

// -------------------------------------------------------------- sidebar ----
function renderSide() {
  side.innerHTML = '';
  runs.forEach(function (r) {
    var d = document.createElement('div');
    d.className = 'runcard' + (selected === r.name ? ' sel' : '');
    var on = st.cmp.indexOf(r.name) >= 0;
    d.innerHTML =
      '<div class="name"><input type="checkbox" class="cmp" title="overlay ' +
      'this run on every chart"' + (on ? ' checked' : '') + '>' + r.label +
      '<span class="badge ' + r.status + '">' + r.status + '</span></div>' +
      '<div class="meta">' + fmtDate(r.started) +
      (r.duration_s ? ' ' + GL.dot + ' ' + fmtDur(r.duration_s) : '') +
      ' ' + GL.dot + ' ' + fmtSteps(r.steps) + ' steps ' + GL.dot + ' ' +
      r.trajs.length + ' trajs</div>';
    d.addEventListener('click', function () { select(r.name); });
    var cb = d.querySelector('.cmp');
    cb.addEventListener('click', function (ev) { ev.stopPropagation(); });
    cb.addEventListener('change', function () {
      var i = st.cmp.indexOf(r.name);
      if (cb.checked && i < 0) st.cmp.push(r.name);
      else if (!cb.checked && i >= 0) st.cmp.splice(i, 1);
      saveState();
      if (selected) select(selected);
    });
    side.appendChild(d);
  });
  if (!runs.length) side.innerHTML = '<div class="empty">No runs yet.</div>';
}

// ------------------------------------------------------------- plotting ----
var CH = 132;                    // plot height, px (legend/axes are extra)
var charts = {};                 // metric key -> chart record
var chartSig = '';               // layout identity; a change means rebuild
var zoom = null;                 // {min,max} in CURRENT x units, or null
var applyingZoom = false;        // re-entrancy guard for the sync broadcast
var metricsCache = {};           // run -> {series, axes, maxWall}

function eachChart(fn) {
  Object.keys(charts).forEach(function (k) {
    if (charts[k] && charts[k].u) fn(charts[k].u, charts[k]);
  });
}
function destroyCharts() {
  eachChart(function (u) { try { u.destroy(); } catch (e) {} });
  charts = {};
  chartSig = '';
}

// x values for one run's series, on the axis in force
function xOf(s, run, axis) {
  if (axis === 'iter' && s.iter) return s.iter;
  if (axis === 'wall' && s.wall) return s.wall;
  if (axis === 'rel' && s.wall) {
    var m = (metricsCache[run] && metricsCache[run].maxWall) || 0;
    if (m > 0) {
      return s.wall.map(function (w) { return 100 * w / m; });
    }
  }
  return s.steps;
}

// wandb's smoothing: a plain EMA over the series' own samples, applied
// BEFORE any cross-run alignment so a run is never smoothed through
// another run's sample grid.
function ema(ys, w) {
  if (!(w > 0)) return ys;
  var out = new Array(ys.length), last = null;
  for (var i = 0; i < ys.length; i++) {
    var v = ys[i];
    if (v == null) { out[i] = null; continue; }
    last = (last === null) ? v : last * w + v * (1 - w);
    out[i] = last;
  }
  return out;
}

// Two runs never sample the same x. Build the union and put each run on it,
// with null where it has no sample; spanGaps then draws one continuous line
// per run. (uPlot needs one shared x array - this is that join.)
function unionX(lists) {
  if (lists.length === 1) return lists[0];
  var seen = Object.create(null), out = [];
  lists.forEach(function (l) {
    for (var i = 0; i < l.length; i++) {
      if (seen[l[i]] === undefined) { seen[l[i]] = 1; out.push(l[i]); }
    }
  });
  out.sort(function (a, b) { return a - b; });
  return out;
}
function alignY(xu, xs, ys) {
  if (xu === xs) return ys;
  var out = new Array(xu.length), j = 0;
  for (var i = 0; i < xu.length; i++) {
    while (j < xs.length && xs[j] < xu[i]) j++;
    out[i] = (j < xs.length && xs[j] === xu[i]) ? ys[j] : null;
  }
  return out;
}

// Everything one chart needs: the joined data, the raw (unsmoothed) copy
// drawn faintly behind it, and the per-run labels/colours.
function chartData(key, names, axis) {
  var xs = [], ysRaw = [], labels = [], colors = [];
  names.forEach(function (n, i) {
    var m = metricsCache[n];
    var s = m && m.series && m.series[key];
    if (!s) return;
    xs.push(xOf(s, n, axis));
    ysRaw.push(s.values);
    labels.push(n);
    colors.push(COLORS[i % COLORS.length]);
  });
  if (!xs.length) return null;
  var xu = unionX(xs);
  var w = st.smooth;
  var data = [xu], raw = [null];
  for (var i = 0; i < xs.length; i++) {
    data.push(alignY(xu, xs[i], ema(ysRaw[i], w)));
    raw.push(w > 0 ? alignY(xu, xs[i], ysRaw[i]) : null);
  }
  var xmin = xu[0], xmax = xu[xu.length - 1];
  if (xmin === xmax) {
    // a single sample (an expert loop's first planner point, a run with
    // one eval) would make uPlot pad the scale to 0..2x, and the synced
    // x-axis then drags every other chart to that range. Span the run's
    // whole x-range instead.
    var gmin = Infinity, gmax = -Infinity;
    names.forEach(function (n) {
      var m = metricsCache[n];
      if (!m || !m.series) return;
      Object.keys(m.series).forEach(function (k) {
        var xx = xOf(m.series[k], n, axis);
        if (xx && xx.length) {
          gmin = Math.min(gmin, xx[0]); gmax = Math.max(gmax, xx[xx.length - 1]);
        }
      });
    });
    if (gmin < gmax) { xmin = gmin; xmax = gmax; }
  }
  return {
    data: data, raw: raw, labels: labels, colors: colors,
    xmin: xmin, xmax: xmax,
    last: ysRaw[0][ysRaw[0].length - 1]
  };
}

// A log y-scale cannot show a non-positive sample; blank those points
// rather than let uPlot fail, and fall back to linear if nothing is left.
function logView(cd) {
  var any = false, out = [cd.data[0]];
  for (var i = 1; i < cd.data.length; i++) {
    out.push(cd.data[i].map(function (v) {
      if (v != null && v > 0) { any = true; return v; }
      return null;
    }));
  }
  return any ? out : null;
}

function tooltipPlugin(key, getAxis) {
  var el = null, over = null, inside = false;
  function near(arr, i) {                 // nearest sample that HAS a value
    if (arr[i] != null) return i;
    var lim = Math.min(arr.length, 2000);
    for (var d = 1; d < lim; d++) {
      if (i - d >= 0 && arr[i - d] != null) return i - d;
      if (i + d < arr.length && arr[i + d] != null) return i + d;
    }
    return -1;
  }
  return {
    hooks: {
      init: function (u) {
        el = document.createElement('div');
        el.className = 'u-tip';
        el.style.display = 'none';
        u.over.appendChild(el);
        over = u.over;
        over.addEventListener('mouseenter', function () { inside = true; });
        over.addEventListener('mouseleave', function () {
          inside = false;
          if (el) el.style.display = 'none';
        });
      },
      setCursor: function (u) {
        var i = u.cursor.idx;
        // the cursor is SYNCED across every chart of the run, so without
        // this every chart would pop a tooltip at once
        if (!inside || i == null) { el.style.display = 'none'; return; }
        var ax = AXES[getAxis()] || AXES.steps;
        var html = '<div class="x">' + ax.val(u.data[0][i]) + '</div>';
        var n = 0;
        for (var s = 1; s < u.series.length; s++) {
          if (!u.series[s].show) continue;
          var j = near(u.data[s], i);
          if (j < 0) continue;
          html += '<div class="r"><i style="background:' +
            u.series[s]._color + '"></i><span>' + u.series[s].label +
            '</span><b>' + fmtVal(u.data[s][j], key) + '</b></div>';
          n++;
        }
        if (!n) { el.style.display = 'none'; return; }
        el.innerHTML = html;
        el.style.display = 'block';
        var w = el.offsetWidth, h = el.offsetHeight;
        var x = u.cursor.left + 14, y = u.cursor.top + 12;
        if (x + w > u.bbox.width / devicePixelRatio) x = u.cursor.left - w - 14;
        if (y + h > u.bbox.height / devicePixelRatio) y = Math.max(0, u.cursor.top - h - 8);
        el.style.left = Math.max(0, x) + 'px';
        el.style.top = Math.max(0, y) + 'px';
      }
    }
  };
}

// The raw series, faint, under the smoothed one - so a smoothing setting can
// never hide what the data actually did.
function rawPlugin(rec) {
  return {
    hooks: {
      draw: function (u) {
        var raw = rec.raw;
        if (!raw) return;
        var ctx = u.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.rect(u.bbox.left, u.bbox.top, u.bbox.width, u.bbox.height);
        ctx.clip();
        ctx.globalAlpha = 0.28;
        ctx.lineWidth = 1;
        for (var s = 1; s < u.series.length; s++) {
          var ys = raw[s];
          if (!ys || !u.series[s].show) continue;
          ctx.strokeStyle = u.series[s]._color;
          ctx.beginPath();
          var pen = false;
          for (var i = 0; i < ys.length; i++) {
            var v = ys[i];
            if (v == null || (u.scales.y.distr === 3 && v <= 0)) { pen = false; continue; }
            var px = u.valToPos(u.data[0][i], 'x', true);
            var py = u.valToPos(v, 'y', true);
            if (!pen) { ctx.moveTo(px, py); pen = true; } else ctx.lineTo(px, py);
          }
          ctx.stroke();
        }
        ctx.restore();
      }
    }
  };
}

function setXRange(min, max) {
  zoom = (min == null || !isFinite(min)) ? null : {min: min, max: max};
  applyingZoom = true;
  eachChart(function (u) {
    if (zoom) u.setScale('x', {min: zoom.min, max: zoom.max});
    else if (u._xmin != null) u.setScale('x', {min: u._xmin, max: u._xmax});
  });
  applyingZoom = false;
  var z = document.getElementById('zst');
  if (z) {
    var ax = AXES[activeAxis(plotRuns())] || AXES.steps;
    z.textContent = zoom
      ? 'zoom ' + tickList([zoom.min, zoom.max], (zoom.max - zoom.min) / 4,
                           ax.suffix, ax.plain).join(' .. ')
      : '';
  }
}

function buildChart(rec, key, cd, axis) {
  var logOn = !!st.log[key];
  var data = cd.data;
  if (logOn) {
    var lv = logView(cd);
    if (lv) data = lv; else logOn = false;
  }
  var ax = AXES[axis] || AXES.steps;
  var series = [{}];
  cd.labels.forEach(function (lab, i) {
    series.push({
      label: lab, stroke: cd.colors[i], width: 1.6, spanGaps: true,
      _color: cd.colors[i],
      show: !st.hidden[key + '|' + lab],
      value: function (u, v) { return fmtVal(v, key); }
    });
  });
  var opts = {
    width: Math.max(140, rec.wrap.clientWidth),
    height: CH,
    padding: [10, 10, 0, 0],
    cursor: {
      // synced hover across every chart of the run, wandb-style
      sync: {key: SYNC.key, scales: ['x', null], setSeries: false},
      drag: {x: true, y: false, dist: 4},
      focus: {prox: 30}
    },
    legend: {show: cd.labels.length > 1, live: false},
    scales: {x: {time: false, range: function (u, mn, mx) {
      // the declared bounds, never uPlot's padding of a one-sample series:
      // that padding (0..2x) would be published through the x-sync and
      // read by every other chart as a zoom
      if (zoom) return [zoom.min, zoom.max];
      var r = rec._xr;
      if (r && r[1] > r[0]) return r;
      return mn === mx ? [mn - 1, mx + 1] : [mn, mx];
    }}, y: {distr: logOn ? 3 : 1}},
    axes: [
      {stroke: '#7a8380', grid: {stroke: '#23272d', width: 1},
       ticks: {stroke: '#23272d', size: 3}, font: '10px system-ui', size: 26,
       values: function (u, sp, ai, sc, incr) {
         return tickList(sp, incr, ax.suffix, ax.plain);
       }},
      {stroke: '#7a8380', grid: {stroke: '#23272d', width: 1},
       ticks: {stroke: '#23272d', size: 3}, font: '10px system-ui', size: 48,
       values: function (u, sp, ai, sc, incr) { return tickList(sp, incr); }}
    ],
    series: series,
    hooks: {
      setScale: [function (u, sk) {
        if (sk !== 'x' || applyingZoom || !u._ready) return;
        var mn = u.scales.x.min, mx = u.scales.x.max;
        var span = (u._xmax - u._xmin) || 1;
        // uPlot's own double-click reset lands exactly on the data bounds;
        // treat that as "no zoom" so every OTHER chart un-zooms too.
        // a scale that covers this chart's whole data range is not a zoom
        // (double-click reset, or a synced neighbour's wider bounds)
        if (mn <= u._xmin + span * 1e-6 && mx >= u._xmax - span * 1e-6) setXRange(null);
        else setXRange(mn, mx);
      }],
      setSeries: [function (u, i, o) {
        if (i == null || o == null || o.show === undefined) return;
        var lab = u.series[i].label;
        if (o.show) delete st.hidden[key + '|' + lab];
        else st.hidden[key + '|' + lab] = 1;
        saveState();
      }]
    },
    plugins: [tooltipPlugin(key, function () { return rec.axis; }),
              rawPlugin(rec)]
  };
  if (zoom) { opts.scales.x.min = zoom.min; opts.scales.x.max = zoom.max; }
  rec._xr = [cd.xmin, cd.xmax];
  var u = new UP(opts, data, rec.wrap);
  u._xmin = cd.xmin; u._xmax = cd.xmax; u._ready = true;
  rec.u = u; rec.axis = axis; rec.raw = cd.raw; rec.log = logOn;
  // Wheel zoom about the pointer, applied to every chart at once - but
  // ONLY with ctrl/shift held. The charts cover the whole page, so
  // swallowing a bare wheel would kill scrolling everywhere.
  u.over.addEventListener('wheel', function (e) {
    if (!e.deltaY || !(e.ctrlKey || e.shiftKey)) return;
    e.preventDefault();
    var rect = u.over.getBoundingClientRect();
    var at = u.posToVal(e.clientX - rect.left, 'x');
    var mn = u.scales.x.min, mx = u.scales.x.max;
    var f = e.deltaY < 0 ? 0.75 : 1 / 0.75;
    var nmn = at - (at - mn) * f, nmx = at + (mx - at) * f;
    if (nmn <= u._xmin && nmx >= u._xmax) setXRange(null);
    else setXRange(Math.max(nmn, u._xmin), Math.min(nmx, u._xmax));
  }, {passive: false});
  return u;
}

function updateChart(rec, key, cd, axis) {
  var u = rec.u;
  var data = cd.data;
  if (rec.log) {
    var lv = logView(cd);
    if (lv) data = lv;
  }
  rec.raw = cd.raw; rec.axis = axis;
  u._xmin = cd.xmin; u._xmax = cd.xmax; rec._xr = [cd.xmin, cd.xmax];
  u._ready = false;                 // a data-driven re-range is not a zoom
  u.setData(data, zoom == null);
  if (zoom) u.setScale('x', {min: zoom.min, max: zoom.max});
  u._ready = true;
}

// race/, front/, back/ and held/ series suffixed .<map> are per-map: on a
// 103-map pool run the frontier block alone is 700 charts (2026-09-12)
var PERMAP_RE = /^(race|front|back|held|gate)\/[A-Za-z0-9_]+\.[A-Za-z0-9_.-]+$/;
function isPerMap(k) { return PERMAP_RE.test(k); }

function plotRuns() {
  var names = selected ? [selected] : [];
  st.cmp.forEach(function (n) {
    if (names.indexOf(n) < 0 && metricsCache[n]) names.push(n);
  });
  return names;
}

function cell(key) {
  return '<div class="chart" data-k="' + key + '">' +
    '<div class="t"><span class="k" title="' + escAttr(descOf(key)) + '">' + key + '</span><span class="tr">' +
    '<button class="lg" data-k="' + key + '" title="log y-scale">log</button>' +
    '<b></b></span></div><div class="cwrap"></div></div>';
}

function renderCharts() {
  var host = document.getElementById('chartsec');
  if (!host) return;
  var names = plotRuns();
  if (!UP) {
    host.innerHTML = '<div class="empty">viewer/vendor/uPlot.iife.min.js ' +
      'did not load - charts unavailable.</div>';
    return;
  }
  var keyset = Object.create(null);
  names.forEach(function (n) {
    var m = metricsCache[n];
    if (m && m.series) Object.keys(m.series).forEach(function (k) { keyset[k] = 1; });
  });
  var keys = Object.keys(keyset);
  keys.sort(function (a, b) {
    var ia = PREFERRED.indexOf(a), ib = PREFERRED.indexOf(b);
    if (ia < 0) ia = 99; if (ib < 0) ib = 99;
    return ia - ib || a.localeCompare(b);
  });
  var axis = activeAxis(names);
  renderChips(keys);
  keys = keys.filter(function (k) {
    if (st.hideGroup[groupOf(k)]) return false;
    if (isPerMap(k) && !mapMatch(tagOf(k))) return false;
    return true;
  });
  renderLegend(keys);
  var agg = keys.filter(function (k) { return !isPerMap(k); });
  var per = keys.filter(isPerMap);
  var sig = axis + '#' + names.join(',') + '#' + keys.join(',');

  if (sig !== chartSig) {
    destroyCharts();
    chartSig = sig;
    if (!keys.length) {
      host.innerHTML = '<div class="empty">No metrics logged for this run.</div>';
      return;
    }
    host.innerHTML =
      (agg.length ? '<div id="charts">' + agg.map(cell).join('') + '</div>' : '') +
      (per.length
        ? '<details id="permap"' +
          ((st.permap === undefined ? per.length <= 12 : st.permap) ? ' open' : '') +
          '><summary>Per-map series (' + per.length +
          ' charts)' + (per.length > 12 ? ' - collapsed for a fleet' : '') +
          '</summary><div id="charts-permap" ' +
          'class="charts">' + per.map(cell).join('') + '</div></details>'
        : '');
    Array.prototype.forEach.call(host.querySelectorAll('.chart'), function (el) {
      charts[el.dataset.k] = {el: el, wrap: el.querySelector('.cwrap'),
                              u: null, raw: null, axis: axis, log: false};
    });
    Array.prototype.forEach.call(host.querySelectorAll('button.lg'), function (b) {
      b.addEventListener('click', function () {
        var k = b.dataset.k;
        st.log[k] = !st.log[k];
        saveState();
        b.classList.toggle('on', !!st.log[k]);
        drawInto(charts[k], k, names, axis);
      });
    });
    var det = document.getElementById('permap');
    if (det) {
      det.addEventListener('toggle', function () {
        st.permap = det.open; saveState();
        // a chart built inside a CLOSED <details> has clientWidth 0 and
        // would render as a 0-wide sliver that never repaints
        if (det.open) per.forEach(function (k) {
          drawInto(charts[k], k, names, axis);
        });
      });
    }
  }

  var open = !document.getElementById('permap') ||
             document.getElementById('permap').open;
  keys.forEach(function (k) {
    if (isPerMap(k) && !open) return;
    drawInto(charts[k], k, names, axis);
  });
  Array.prototype.forEach.call(host.querySelectorAll('button.lg'), function (b) {
    b.classList.toggle('on', !!st.log[b.dataset.k]);
  });
}

function drawInto(rec, key, names, axis) {
  if (!rec) return;
  var cd = chartData(key, names, axis);
  if (!cd) return;
  var b = rec.el.querySelector('.t b');
  if (b) b.textContent = fmtVal(cd.last, key);
  if (!rec.wrap.clientWidth) return;         // inside a closed <details>
  // The effective log state can differ from the requested one (a series
  // with no positive sample cannot be drawn on a log axis), and it can
  // change as data arrives - so settle it here and rebuild on a flip.
  var wantLog = !!st.log[key] && logView(cd) != null;
  if (rec.u && rec.log !== wantLog) { rec.u.destroy(); rec.u = null; }
  if (!rec.u) buildChart(rec, key, cd, axis);
  else updateChart(rec, key, cd, axis);
}

var resizeT = null;
window.addEventListener('resize', function () {
  clearTimeout(resizeT);
  resizeT = setTimeout(function () {
    eachChart(function (u, rec) {
      if (rec.wrap.clientWidth) u.setSize({width: rec.wrap.clientWidth, height: CH});
    });
  }, 150);
});

// --------------------------------------------------------- control bar ----
function ctlHTML() {
  return '<div id="ctl">' +
    '<label>x-axis <select id="xax"></select></label>' +
    '<label>smoothing <input id="sm" type="range" min="0" max="99" step="1">' +
    '<span id="smv"></span></label>' +
    '<button id="zrst" title="also: double-click any chart">reset zoom</button>' +
    '<span class="hint">drag to zoom ' + GL.dot + ' dbl-click resets ' +
    GL.dot + ' ctrl+wheel zooms</span>' +
    '<span id="zst" class="hint"></span>' +
    '<span id="cmpinfo" class="hint"></span>' +
    '<span id="grp" class="chips" title="click a family to hide / show its charts"></span>' +
    '<label>maps <input id="mapq" type="text" size="18" placeholder="petrus, cannonball"' +
    ' title="per-map series only for maps matching these comma-separated substrings; empty = all"></label>' +
    '<button id="legend" title="what each plot means">?</button></div>' +
    '<div id="legendbox" hidden></div>';
}

// The chart-family chips and the map filter (2026-09-12): a 103-map pool run
// writes 7 frontier + 4 race series PER MAP, over a thousand charts. Hidden
// families and the map filter are preferences (localStorage), like the axis.
var chipSig = '';
function renderChips(keys) {
  var host = document.getElementById('grp');
  if (!host) return;
  var counts = {}, order = [];
  keys.forEach(function (k) {
    var g = groupOf(k);
    if (!(g in counts)) { counts[g] = 0; order.push(g); }
    counts[g] += 1;
  });
  var sig = order.map(function (g) { return g + ':' + counts[g] + ':' + (st.hideGroup[g] ? 0 : 1); }).join(',');
  if (sig === chipSig) return;
  chipSig = sig;
  host.innerHTML = order.map(function (g) {
    return '<button class="chip' + (st.hideGroup[g] ? ' off' : '') + '" data-g="' + g +
      '" title="' + escAttr(GROUP_DESC[g] || '') + '">' + g + ' (' + counts[g] + ')</button>';
  }).join('');
  Array.prototype.forEach.call(host.querySelectorAll('button.chip'), function (b) {
    b.addEventListener('click', function () {
      var g = b.dataset.g;
      st.hideGroup[g] = !st.hideGroup[g];
      saveState();
      renderCharts();
    });
  });
}
function mapMatch(tag) {
  var q = (st.mapq || '').toLowerCase().split(',').map(function (x) { return x.trim(); })
    .filter(function (x) { return x.length; });
  if (!q.length) return true;
  var t = (tag || '').toLowerCase();
  return q.some(function (x) { return t.indexOf(x) >= 0; });
}
function renderLegend(keys) {
  var box = document.getElementById('legendbox');
  var btn = document.getElementById('legend');
  if (!box) return;
  if (btn) btn.classList.toggle('on', !!st.legend);
  if (!st.legend) { box.hidden = true; return; }
  var groups = {}, order = [];
  keys.forEach(function (k) {
    var g = groupOf(k), b = baseKey(k);
    if (!(g in groups)) { groups[g] = {}; order.push(g); }
    groups[g][b] = 1;
  });
  box.innerHTML = order.map(function (g) {
    var items = Object.keys(groups[g]).sort().map(function (b) {
      return '<dt>' + b + '</dt><dd>' + (DESC[b] || '(no description yet)') + '</dd>';
    }).join('');
    return '<h4>' + g + '/</h4><div class="gd">' + (GROUP_DESC[g] || '') + '</div><dl>' + items + '</dl>';
  }).join('') || '<div class="gd">no charts</div>';
  box.hidden = false;
}
function wireCtl() {
  var xs = document.getElementById('xax');
  Object.keys(AXES).forEach(function (id) {
    var o = document.createElement('option');
    o.value = id; o.textContent = AXES[id].label;
    xs.appendChild(o);
  });
  xs.value = st.xaxis;
  xs.addEventListener('change', function () {
    st.xaxis = xs.value; saveState();
    setXRange(null);                 // a zoom in steps means nothing in hours
    renderCharts();
  });
  var sm = document.getElementById('sm');
  sm.value = Math.round(st.smooth * 100);
  document.getElementById('smv').textContent = st.smooth.toFixed(2);
  sm.addEventListener('input', function () {
    st.smooth = Math.min(0.99, sm.value / 100);
    document.getElementById('smv').textContent = st.smooth.toFixed(2);
    saveState();
    renderCharts();
  });
  document.getElementById('zrst').addEventListener('click', function () {
    setXRange(null);
  });
  var mq = document.getElementById('mapq');
  mq.value = st.mapq || '';
  mq.addEventListener('input', function () {
    st.mapq = mq.value; saveState();
    renderCharts();
  });
  document.getElementById('legend').addEventListener('click', function () {
    st.legend = !st.legend; saveState();
    renderCharts();
  });
}
function updateCtl() {
  var names = plotRuns();
  var xs = document.getElementById('xax');
  if (xs) {
    Array.prototype.forEach.call(xs.options, function (o) {
      var ok = axisOK(o.value, names);
      o.disabled = !ok;
      o.textContent = AXES[o.value].label + (ok ? '' : ' (n/a)');
    });
    xs.value = activeAxis(names);
  }
  var c = document.getElementById('cmpinfo');
  if (c) {
    c.textContent = names.length > 1
      ? GL.dot + ' overlaying ' + names.length + ' runs (tick runs at left)'
      : GL.dot + ' tick runs at left to overlay them';
  }
}

// ----------------------------------------------------------- run view -----
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
  var out = b('stoch', null, GL.rec + ' stoch' + sfx) +
            b('greedy', null, GL.rec + ' greedy' + sfx);
  if (r.config && r.config.reward === 'race') {
    out += b('stoch', 'mixed', GL.rec + ' drop spawns' + sfx);
    if (r.config.respawn_frac) {
      out += b('stoch', 'reservoir', GL.rec + ' frontier' + sfx);
    }
  }
  return out;
}

// The <details> open state, and now the charts themselves, must survive the
// 4 s poll. Only the head and the artifact list are re-rendered wholesale;
// #chartsec is owned by renderCharts and its uPlot instances are kept alive
// across refreshes, which is what preserves zoom, hover and legend toggles.
function ensureSkeleton() {
  if (document.getElementById('chartsec')) return;
  destroyCharts();
  main.innerHTML = '<div id="runhead"></div>' + ctlHTML() +
    '<h2>Metrics</h2><div id="chartsec"></div><div id="artsec"></div>';
  wireCtl();
}

function renderMain(r) {
  ensureSkeleton();
  document.getElementById('runhead').innerHTML = r.label +
    '<span class="badge ' + r.status + '">' + r.status + '</span>' +
    '<div class="meta">started ' + fmtDate(r.started) +
    (r.finished ? ' ' + GL.dot + ' finished ' + fmtDate(r.finished) : '') +
    (r.duration_s ? ' ' + GL.dot + ' ' + fmtDur(r.duration_s) : '') +
    ' ' + GL.dot + ' ' + fmtSteps(r.steps) + ' steps' +
    (Object.keys(r.config || {}).length
      ? '<br>' + Object.entries(r.config).map(function (kv) {
          return kv[0] + '=' + kv[1];
        }).join('  ') : '') + '</div>';
  updateCtl();
  renderCharts();

  // On a --maps run the record buttons live inside each map's bucket: a
  // single run-level "record greedy" would silently pick whichever map the
  // checkpoint names first, which is a wrong recording rather than an error.
  var multi = !!(r.config && r.config.maps && r.config.maps.length > 1);
  var html = '<h2>Trajectories (greedy policy over training)' +
    (multi ? '<span class="meta"> - record buttons are per map, below</span>'
           : recButtons(r, null)) + '</h2>';
  var artCell = function (t) {
    return '<div class="art"><div><div class="s">@ ' + fmtSteps(t.steps) +
      ' steps ' + GL.dot + ' ' + (t.mode || 'greedy') + '</div><div class="kb">' +
      t.kb + ' KB</div></div>' +
      '<button class="watch" data-f="' + t.file + '">' + GL.play +
      ' watch</button></div>';
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
        (st.mapOpen[tag] ? ' open' : '') + '><summary>' + tag +
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
  document.getElementById('artsec').innerHTML = html;

  Array.prototype.forEach.call(document.querySelectorAll('details.mapgrp'),
    function (dg) {
      dg.addEventListener('toggle', function () {
        st.mapOpen[dg.dataset.map] = dg.open; saveState();
      });
    });
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
// every 4s and, for a live run, calls select() -> renderMain(), which
// rebuilds the artifact section and DESTROYS every button. The old code
// captured the button in a closure, so after the first re-render it wrote
// "recording... Ns" into a detached node while a pristine, enabled button
// sat in the DOM. The job ran to completion on the box and the UI showed
// nothing the whole time - which is indistinguishable from a dead button,
// and is exactly what it was reported as. Keyed state + re-applying it on
// every render fixes it.
var recording = {};

function recKey(runName, mode, spawn, map) {
  // the map is part of the identity: recording map A must not mark map B's
  // button as in-flight
  return runName + '|' + mode + '|' + (spawn || 'default') + '|' + (map || 'all');
}

// re-apply in-flight/error state to the buttons that renderMain just created
function applyRecordingState() {
  Array.prototype.forEach.call(document.querySelectorAll('.rec'), function (b) {
    var stt = recording[b.dataset.key];
    if (!stt) return;
    if (stt.error) {
      b.disabled = false;
      b.textContent = b.dataset.label + ' ' + GL.cross + ' ' + (stt.error || '');
    } else {
      b.disabled = true;
      // a real percentage from the recorder's own tick counter; falls back to
      // elapsed seconds only until the first progress write lands
      var secs = Math.round((Date.now() - stt.t0) / 1000);
      // startup dominates a recording, so show the PHASE, not just a tick
      // percentage that sits at 0 for the first 40s and looks hung
      var ep = (stt.ep && stt.eps) ? (' ep ' + stt.ep + '/' + stt.eps) : '';
      var ph = stt.phase || 'starting';
      b.textContent = (stt.pct === null || stt.pct === undefined)
        ? GL.rec + ' ' + ph + GL.ell + ' ' + secs + 's'
        : GL.rec + ' ' + stt.pct + '% ' + ph + ep + ' ' + GL.dot + ' ' + secs + 's';
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
        // a 400 carries the server's reason as JSON too ("no bsp for ...",
        // "map 'x' not in this run"); it used to be thrown away and the
        // button read "X true", which is what the user saw on celestial
        return r.json().catch(function () { return { error: 'http ' + r.status }; });
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

// ------------------------------------------------------------- fetching ----
function isLive(name) {
  var r = runs.find(function (x) { return x.name === name; });
  return !!r && r.status === 'live';
}
function fetchMetrics(name) {
  // a finished run's curves never change; only re-poll the live ones.
  // But a fetch that FAILED (the dashboard restarting, a dropped request)
  // leaves an empty entry, and an empty entry on a finished run used to be
  // final - "No metrics logged" until a page reload. Re-fetch those.
  var cached = metricsCache[name];
  if (cached && !isLive(name) &&
      Object.keys(cached.series || {}).length) return Promise.resolve();
  return fetch('/api/metrics?run=' + encodeURIComponent(name))
    .then(function (resp) { return resp.json(); })
    .then(function (m) {
      var series = m.series || {}, maxWall = 0;
      Object.keys(series).forEach(function (k) {
        var w = series[k].wall;
        if (w && w.length && w[w.length - 1] > maxWall) maxWall = w[w.length - 1];
      });
      metricsCache[name] = {series: series, axes: m.axes || ['steps'],
                            maxWall: maxWall};
    })
    .catch(function () {
      if (!metricsCache[name]) {
        metricsCache[name] = {series: {}, axes: ['steps'], maxWall: 0};
      }
    });
}

function select(name) {
  selected = name;
  // deep link: /viewer/runs.html#run=<name> reopens this run (no history spam)
  try { history.replaceState(null, '', '#run=' + encodeURIComponent(name)); } catch (e) {}
  renderSide();
  var r = runs.find(function (x) { return x.name === name; });
  if (!r) return;
  var want = [name].concat(st.cmp.filter(function (n) { return n !== name; }));
  Promise.all(want.map(fetchMetrics)).then(function () {
    if (selected !== name) return;          // the user moved on mid-fetch
    renderMain(r);
  });
}

function poll() {
  fetch('/api/runs')
    .then(function (r) { return r.json(); })
    .then(function (j) {
      runs = j.runs || [];
      renderSide();
      if (!selected && runs.length) {
        var want = null;
        try { want = decodeURIComponent((location.hash.match(/run=([^&]+)/) || [])[1] || ''); } catch (e) {}
        select(want && runs.some(function (x) { return x.name === want; }) ? want : runs[0].name);
      }
      else if (selected) {
        var cur = runs.find(function (x) { return x.name === selected; });
        var anyLive = cur && cur.status === 'live';
        st.cmp.forEach(function (n) { if (isLive(n)) anyLive = true; });
        if (anyLive) select(selected);                  // live refresh
      }
      document.getElementById('refresh').textContent =
        'updated ' + new Date().toTimeString().slice(0, 8);
    })
    .catch(function () {});
}
poll();
setInterval(poll, 4000);
