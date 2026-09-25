"""viz_primitives.py - the planner's motion PRIMITIVES, drawn in 3-D (an interactive HTML page).

A primitive is a short curve that starts at the agent and leaves ALONG its velocity. Its heading
(yaw, pitch) starts at the velocity's own and then turns at a signed horizontal rate (left > 0)
and a signed vertical rate (climb > 0), after an optional straight DELAY ("take off earlier or
later"). The rates ramp in over RAMP_S with a smoothstep, so position, direction and curvature are
all continuous - at the start and at the end of the delay. The curve is traced at the current
speed for DURATION_S seconds, so its length scales with speed like the lookahead points do.
Horizontal turns rotate the heading about the world vertical, so a descending velocity gives a
descending turn. One fixed grid of parameters, the same on every map; nothing comes from data.

    python tools/viz_primitives.py [out.html]

The page: checkboxes filter the curves by turn rate, vertical rate and delay; the radio buttons
switch the example velocity (level / surfing down a ramp / rising after a take-off). Colour follows
the turn (blue = left, orange = right, grey = straight), and every curve also carries a text label
at its end, a line style per delay and an end-marker shape per vertical rate, so nothing relies on
colour alone.
"""
import json
import sys
from pathlib import Path

import numpy as np

TURNS = (-120, -60, -30, 0, 30, 60, 120)        # deg/s, + = left
CLIMBS = (-40, 0, 40)                            # deg/s of pitch, + = up
DELAYS = (0.0, 0.5, 1.0)                         # s straight before turning
DURATION_S = 2.0
RAMP_S = 0.2
DT = 0.01
EXAMPLES = {                                     # name -> (speed u/s, yaw deg, pitch deg)
    "level, 800 u/s": (800.0, 0.0, 0.0),
    "surfing down a ramp, 1200 u/s, 20 deg down": (1200.0, 30.0, -20.0),
    "rising after a take-off, 600 u/s, 25 deg up": (600.0, -40.0, 25.0),
}


def primitive(speed, yaw0_deg, pitch0_deg, turn, climb, delay):
    t = np.arange(0.0, DURATION_S + 1e-9, DT)
    s = np.clip((t - delay) / RAMP_S, 0.0, 1.0)
    w = s * s * (3.0 - 2.0 * s)                  # smoothstep: the rate ramps in, no curvature jump
    yaw = np.radians(yaw0_deg) + np.concatenate(([0.0], np.cumsum(
        0.5 * (w[1:] + w[:-1]) * np.radians(turn) * DT)))
    pitch = np.radians(pitch0_deg) + np.concatenate(([0.0], np.cumsum(
        0.5 * (w[1:] + w[:-1]) * np.radians(climb) * DT)))
    pitch = np.clip(pitch, np.radians(-85.0), np.radians(85.0))
    d = np.stack([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], 1)
    step = 0.5 * (d[1:] + d[:-1]) * speed * DT
    return np.vstack([np.zeros(3), np.cumsum(step, axis=0)])


def label(turn, climb, delay):
    tt = "S" if turn == 0 else ("L" if turn > 0 else "R") + str(abs(turn))
    cc = {-40: "down", 0: "level", 40: "up"}[climb]
    return f"{tt} {cc} d{delay:g}"


def color(turn):
    if turn == 0:
        return "#555555"
    a = {30: 0, 60: 1, 120: 2}[abs(turn)]
    return (["#9ecae1", "#4292c6", "#08519c"] if turn > 0 else ["#fdae6b", "#f16913", "#a63603"])[a]


def build(out):
    traces, metas = [], []
    for ei, (ename, (speed, yaw0, pitch0)) in enumerate(EXAMPLES.items()):
        v = speed * np.array([np.cos(np.radians(pitch0)) * np.cos(np.radians(yaw0)),
                              np.cos(np.radians(pitch0)) * np.sin(np.radians(yaw0)),
                              np.sin(np.radians(pitch0))])
        arrow = 0.35 * v                        # the velocity, drawn 0.35 s long
        traces.append({"type": "scatter3d", "mode": "lines+text", "x": [0, arrow[0]],
                       "y": [0, arrow[1]], "z": [0, arrow[2]], "text": ["agent", "velocity"],
                       "textposition": "top center", "line": {"color": "black", "width": 9},
                       "name": "velocity", "hoverinfo": "skip", "showlegend": False,
                       "visible": ei == 0})
        metas.append({"ex": ei, "kind": "arrow"})
        traces.append({"type": "cone", "x": [arrow[0]], "y": [arrow[1]], "z": [arrow[2]],
                       "u": [v[0]], "v": [v[1]], "w": [v[2]], "sizemode": "absolute",
                       "sizeref": 90, "anchor": "tail", "showscale": False,
                       "colorscale": [[0, "black"], [1, "black"]], "hoverinfo": "skip",
                       "visible": ei == 0})
        metas.append({"ex": ei, "kind": "arrow"})
        for turn in TURNS:
            for climb in CLIMBS:
                for delay in DELAYS:
                    p = primitive(speed, yaw0, pitch0, turn, climb, delay)
                    lab = label(turn, climb, delay)
                    n = len(p)
                    traces.append({
                        "type": "scatter3d", "mode": "lines+markers+text",
                        "x": np.round(p[:, 0], 1).tolist(), "y": np.round(p[:, 1], 1).tolist(),
                        "z": np.round(p[:, 2], 1).tolist(),
                        "line": {"color": color(turn), "width": 4,
                                 "dash": {0.0: "solid", 0.5: "dash", 1.0: "dot"}[delay]},
                        "marker": {"size": [0] * (n - 1) + [5], "color": color(turn),
                                   "symbol": {-40: "diamond-open", 0: "circle",
                                              40: "square"}[climb]},
                        "text": [""] * (n - 1) + [lab], "textposition": "middle right",
                        "textfont": {"size": 10},
                        "name": lab,
                        "hovertemplate": (f"{lab}<br>turn {turn} deg/s, vertical {climb} deg/s, "
                                          f"delay {delay:g} s<br>t=%{{customdata:.2f}} s"
                                          "<extra></extra>"),
                        "customdata": np.round(np.arange(n) * DT, 2).tolist(),
                        "showlegend": False, "visible": ei == 0})
                    metas.append({"ex": ei, "kind": "prim", "turn": turn, "climb": climb,
                                  "delay": delay})
    layout = {
        "scene": {"aspectmode": "data", "xaxis": {"title": "x (u)"}, "yaxis": {"title": "y (u)"},
                  "zaxis": {"title": "z (u)"}, "camera": {"eye": {"x": -1.2, "y": -1.6, "z": 0.9}}},
        "margin": {"l": 0, "r": 0, "t": 10, "b": 0}, "uirevision": "keep",
    }
    boxes = lambda name, vals, fmt: "".join(
        f'<label><input type="checkbox" class="f" data-k="{name}" value="{v}" checked> {fmt(v)}</label> '
        for v in vals)
    radios = "".join(
        f'<label><input type="radio" name="ex" value="{i}" {"checked" if i == 0 else ""}> {n}</label><br>'
        for i, n in enumerate(EXAMPLES))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Motion primitives</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body {{ margin: 0; font: 13px sans-serif; background: #fff; color: #111; display: flex; height: 100vh; }}
 #side {{ width: 330px; padding: 12px; overflow-y: auto; border-right: 1px solid #ccc; box-sizing: border-box; }}
 #plot {{ flex: 1; }}
 h3 {{ margin: 12px 0 4px; font-size: 13px; }}
 label {{ display: inline-block; margin: 2px 6px 2px 0; }}
 .note {{ color: #444; font-size: 12px; line-height: 1.4; }}
</style></head><body>
<div id="side">
 <b>Motion primitives</b>
 <p class="note">Each curve leaves the agent along its velocity (black arrow) and turns at a
 fixed rate after an optional straight delay; {DURATION_S:g} s long at the current speed.
 Label at the end: L/R = left/right turn rate in deg/s (S = straight), down/level/up = vertical
 rate (&#177;40 deg/s), d = delay in s. Line style = delay (solid 0, dashed 0.5, dotted 1.0);
 end marker = vertical (open diamond down, circle level, square up); colour = turn
 (blue left, orange right, grey straight). Drag to rotate, scroll to zoom, hover a curve for
 its parameters.</p>
 <h3>Example velocity</h3>{radios}
 <h3>Turn rate (deg/s, + = left)</h3>{boxes("turn", TURNS, lambda v: ("S 0" if v == 0 else ("L" if v > 0 else "R") + str(abs(v))))}
 <h3>Vertical rate</h3>{boxes("climb", CLIMBS, lambda v: {-40: "down", 0: "level", 40: "up"}[v])}
 <h3>Delay before turning (s)</h3>{boxes("delay", DELAYS, lambda v: f"{v:g}")}
 <h3>Display</h3><label><input type="checkbox" id="labels" checked> end labels</label>
 <p class="note" id="count"></p>
</div>
<div id="plot"></div>
<script>
const traces = {json.dumps(traces)};
const metas = {json.dumps(metas)};
const layout = {json.dumps(layout)};
Plotly.newPlot('plot', traces, layout, {{responsive: true}});
function update() {{
  const ex = +document.querySelector('input[name=ex]:checked').value;
  const on = {{turn: new Set(), climb: new Set(), delay: new Set()}};
  document.querySelectorAll('input.f:checked').forEach(b => on[b.dataset.k].add(+b.value));
  const lab = document.getElementById('labels').checked;
  let n = 0;
  const vis = metas.map(m => {{
    if (m.ex !== ex) return false;
    if (m.kind === 'arrow') return true;
    const v = on.turn.has(m.turn) && on.climb.has(m.climb) && on.delay.has(m.delay);
    if (v) n++;
    return v;
  }});
  Plotly.restyle('plot', {{visible: vis}});
  const idx = metas.map((m, i) => m.kind === 'prim' ? i : -1).filter(i => i >= 0);
  Plotly.restyle('plot', {{mode: idx.map(() => lab ? 'lines+markers+text' : 'lines+markers')}}, idx);
  document.getElementById('count').textContent = n + ' of {len(TURNS) * len(CLIMBS) * len(DELAYS)} primitives shown';
}}
document.querySelectorAll('input').forEach(b => b.addEventListener('change', update));
update();
</script></body></html>"""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html, encoding="utf-8")
    print(f"{out}: {len(TURNS) * len(CLIMBS) * len(DELAYS)} primitives x {len(EXAMPLES)} example velocities")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "runs/research/viz/primitives_3d.html")
