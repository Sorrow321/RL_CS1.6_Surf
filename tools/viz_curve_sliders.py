"""viz_curve_sliders.py - one motion primitive, driven by sliders (an interactive HTML page).

The side question it illustrates: a smooth 3-D curve that starts at the agent ALONG its velocity
is fully described by how fast its heading turns over time - a horizontal turn-rate function and
a vertical turn-rate function (Frenet-Serret: curvature and torsion fix a curve up to position and
rotation; the start point and the start direction fix those). Each slider is one coefficient of
those two functions; the lower plot shows the functions, the 3-D plot the curve they produce.

    turn rate:     0 until DELAY, then ramps in (smoothstep, 0.2 s) and moves linearly from
                   START RATE to END RATE by the end of the curve
                   (equal -> an arc / a helix; different -> a spiral; opposite signs -> an S-curve)
    vertical rate: 0 until DELAY, then ramps in to a constant VERTICAL RATE
    length:        DURATION seconds at SPEED (u/s)

    python tools/viz_curve_sliders.py [out.html]
"""
import sys
from pathlib import Path

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Curve sliders</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body { margin: 0; font: 13px sans-serif; background: #fff; color: #111; display: flex; height: 100vh; }
 #side { width: 340px; padding: 12px; overflow-y: auto; border-right: 1px solid #ccc; box-sizing: border-box; }
 #plots { flex: 1; display: flex; flex-direction: column; }
 #p3d { flex: 3; } #p2d { flex: 1.3; }
 .row { margin: 10px 0; } .row b { display: inline-block; width: 190px; }
 input[type=range] { width: 300px; }
 .note { color: #444; font-size: 12px; line-height: 1.45; }
 button { margin: 3px 4px 3px 0; }
</style></head><body>
<div id="side">
 <b>One motion primitive</b>
 <p class="note">The curve starts at the agent along its velocity (black arrow). Its whole shape is
 the two functions in the lower plot: how fast the heading turns sideways and how fast it turns
 up or down, over time. Each slider is one number of those functions.</p>
 <div id="sliders"></div>
 <p class="note">Presets:</p>
 <button data-p="arc">arc (constant turn)</button><button data-p="spiral">spiral (turn tightens)</button>
 <button data-p="s">S-curve (left then right)</button><button data-p="uturn">U-turn</button>
 <button data-p="helix">descending helix</button><button data-p="late">late take-off turn</button>
 <button data-p="straight">straight</button>
 <p class="note" id="stats"></p>
 <label><input type="checkbox" id="keep"> keep previous curves (grey) for comparison</label>
</div>
<div id="plots"><div id="p3d"></div><div id="p2d"></div></div>
<script>
const S = [
 ["h1", "start turn rate (deg/s, + left)", -180, 180, 5, 60],
 ["h2", "end turn rate (deg/s, + left)", -180, 180, 5, 60],
 ["vr", "vertical rate (deg/s, + up)", -90, 90, 5, 0],
 ["delay", "delay before turning (s)", 0, 1.5, 0.05, 0],
 ["dur", "duration (s)", 0.5, 4, 0.1, 2],
 ["speed", "speed (u/s)", 200, 3000, 50, 800],
 ["yaw0", "velocity heading (deg)", -180, 180, 5, 0],
 ["pitch0", "velocity pitch (deg, + up)", -60, 60, 5, 0],
];
const P = {};
const box = document.getElementById("sliders");
for (const [k, name, lo, hi, st, v] of S) {
  P[k] = v;
  box.insertAdjacentHTML("beforeend",
    `<div class="row"><b>${name}</b> <span id="v_${k}">${v}</span><br>
     <input type="range" id="${k}" min="${lo}" max="${hi}" step="${st}" value="${v}"></div>`);
}
const PRE = {
  arc: {h1: 60, h2: 60, vr: 0, delay: 0}, spiral: {h1: 20, h2: 160, vr: 0, delay: 0},
  s: {h1: 90, h2: -90, vr: 0, delay: 0}, uturn: {h1: 110, h2: 110, vr: 0, delay: 0.2, dur: 2},
  helix: {h1: 120, h2: 120, vr: -15, delay: 0, dur: 3, pitch0: -10},
  late: {h1: -70, h2: -70, vr: 20, delay: 1.0}, straight: {h1: 0, h2: 0, vr: 0, delay: 0},
};
const ghosts = [];
function curve() {
  const T = P.dur, dt = 0.005, r = 0.2, d2r = Math.PI / 180;
  let yaw = P.yaw0 * d2r, pitch = P.pitch0 * d2r;
  const x = [0], y = [0], z = [0], t = [0], wh = [0], wv = [0];
  let px = 0, py = 0, pz = 0;
  for (let i = 1; i * dt <= T + 1e-9; i++) {
    const tt = i * dt;
    let s = Math.min(Math.max((tt - P.delay) / r, 0), 1);
    const w = s * s * (3 - 2 * s);
    const f = T > P.delay ? Math.min(Math.max((tt - P.delay) / (T - P.delay), 0), 1) : 0;
    const h = w * (P.h1 + (P.h2 - P.h1) * f), v = w * P.vr;
    yaw += h * d2r * dt;
    pitch = Math.min(Math.max(pitch + v * d2r * dt, -85 * d2r), 85 * d2r);
    px += P.speed * Math.cos(pitch) * Math.cos(yaw) * dt;
    py += P.speed * Math.cos(pitch) * Math.sin(yaw) * dt;
    pz += P.speed * Math.sin(pitch) * dt;
    x.push(px); y.push(py); z.push(pz); t.push(tt); wh.push(h); wv.push(v);
  }
  return {x, y, z, t, wh, wv, yawEnd: yaw / d2r - P.yaw0};
}
function draw() {
  const c = curve();
  const a = 0.35 * P.speed, d2r = Math.PI / 180;
  const ax = a * Math.cos(P.pitch0 * d2r) * Math.cos(P.yaw0 * d2r),
        ay = a * Math.cos(P.pitch0 * d2r) * Math.sin(P.yaw0 * d2r), az = a * Math.sin(P.pitch0 * d2r);
  const tr = ghosts.map((g, i) => ({type: "scatter3d", mode: "lines", x: g.x, y: g.y, z: g.z,
      line: {color: "#bbbbbb", width: 3, dash: "dot"}, name: "earlier " + (i + 1), hoverinfo: "skip"}));
  tr.push({type: "scatter3d", mode: "lines+text", x: [0, ax], y: [0, ay], z: [0, az],
           text: ["agent", "velocity"], textposition: "top center",
           line: {color: "black", width: 9}, name: "velocity", hoverinfo: "skip"});
  tr.push({type: "scatter3d", mode: "lines+markers", x: c.x, y: c.y, z: c.z, name: "the curve",
           line: {color: "#08519c", width: 6},
           marker: {size: c.x.map((_, i) => i === c.x.length - 1 ? 5 : 0), color: "#08519c"},
           customdata: c.t, hovertemplate: "t=%{customdata:.2f} s<extra></extra>"});
  Plotly.react("p3d", tr, {scene: {aspectmode: "data", xaxis: {title: "x (u)"}, yaxis: {title: "y (u)"},
      zaxis: {title: "z (u)"}}, margin: {l: 0, r: 0, t: 0, b: 0}, uirevision: "keep",
      legend: {x: 0, y: 1}});
  Plotly.react("p2d", [
    {x: c.t, y: c.wh, mode: "lines", name: "sideways turn rate (deg/s, + left)", line: {color: "#08519c", width: 3}},
    {x: c.t, y: c.wv, mode: "lines", name: "vertical turn rate (deg/s, + up)", line: {color: "#a63603", width: 3, dash: "dash"}},
  ], {margin: {l: 55, r: 10, t: 10, b: 35}, xaxis: {title: "time along the curve (s)"},
      yaxis: {title: "deg/s", zeroline: true}, legend: {x: 0.01, y: 0.99}, uirevision: "keep"});
  document.getElementById("stats").textContent =
    `length ${(P.speed * P.dur).toFixed(0)} u, total sideways heading change ${c.yawEnd.toFixed(0)} deg`;
}
function set(k, v) { P[k] = +v; document.getElementById(k).value = v; document.getElementById("v_" + k).textContent = v; }
function snapshot() {
  if (document.getElementById("keep").checked) {
    const c = curve(); ghosts.push({x: c.x, y: c.y, z: c.z}); if (ghosts.length > 8) ghosts.shift();
  }
}
for (const [k] of S) {
  const el = document.getElementById(k);
  el.addEventListener("input", () => { set(k, el.value); draw(); });
  el.addEventListener("mousedown", snapshot);
}
document.querySelectorAll("button[data-p]").forEach(b => b.addEventListener("click", () => {
  snapshot();
  for (const [k] of S) set(k, S.find(s => s[0] === k)[5]);
  for (const [k, v] of Object.entries(PRE[b.dataset.p])) set(k, v);
  draw();
}));
document.getElementById("keep").addEventListener("change", e => { if (!e.target.checked) { ghosts.length = 0; draw(); } });
draw();
</script></body></html>
"""

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/research/viz/curve_sliders.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML, encoding="utf-8")
    print(out)
