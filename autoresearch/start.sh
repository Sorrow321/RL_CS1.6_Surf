#!/bin/bash
# autoresearch/start.sh - bring the fork-and-flush fleet up on THIS machine (the agent host).
#
#   bash autoresearch/start.sh          # tmux session "ar": watchdog, orchestrator, lead
#   tmux attach -t ar                   # look (Ctrl-b w lists the windows; Ctrl-b d detaches)
#   python autoresearch/orchestrator.py status | wrapup | stop
#
# Windows of tmux session "ar":
#   watchdog  tools/fleet_watchdog.py --daemon. It enforces box deadlines, readiness, the rental
#             policy (single allowed RTX card under its cap) and the box cap, on every
#             instance of the account.
#   orch      autoresearch/orchestrator.py run. The rounds: fork, flush, score, select.
#             It is restarted after a crash, not after a clean finish.
#   lead      a Claude session with no box. It watches everything and answers you via Remote
#             Control.
#   b1..bK    the branch agents, started and ended by the orchestrator.
#
# Only ONE machine may run the watchdog daemon for this vast account. It destroys every
# instance missing from ITS registry, so a second daemon elsewhere kills this fleet's boxes.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export AR_HOME="${AR_HOME:-$HOME/ar}"
export RLSURF_FLEET_DIR="$AR_HOME/fleet"
PY="${AR_PY:-$AR_HOME/venv/bin/python}"
mkdir -p "$AR_HOME/fleet"
MAXB=$("$PY" -c "import json;print(json.load(open('$REPO/autoresearch/config.json'))['max_boxes'])")
MODEL=$("$PY" -c "import json;print(json.load(open('$REPO/autoresearch/config.json'))['model'])")
EFFORT=$("$PY" -c "import json;print(json.load(open('$REPO/autoresearch/config.json'))['effort'])")

if pgrep -f "fleet_watchdog.py --daemon" >/dev/null; then
  echo "a fleet watchdog daemon is already running here: not starting a second one"
else
  tmux has-session -t ar 2>/dev/null || tmux new-session -d -s ar -n main -c "$REPO"
  tmux new-window -d -t ar: -n watchdog -c "$REPO" \
    "env RLSURF_FLEET_DIR=$AR_HOME/fleet RLSURF_POLICY=$REPO/autoresearch/config.json \
RLSURF_MAX_BOXES=$MAXB RLSURF_READY_S=330 RLSURF_GRACE_S=300 PYTHONIOENCODING=utf-8 \
$PY tools/fleet_watchdog.py --daemon 2>&1 | tee -a $AR_HOME/fleet/daemon.out"
fi
tmux has-session -t ar 2>/dev/null || tmux new-session -d -s ar -n main -c "$REPO"

INC=""
test -f "$AR_HOME/incumbent/INCUMBENT.json" && INC="--incumbent $AR_HOME/incumbent/INCUMBENT.json"
if ! tmux list-windows -t ar -F '#{window_name}' | grep -qx orch; then
  tmux new-window -d -t ar: -n orch -c "$REPO" \
    "export PATH=$AR_HOME/venv/bin:\$PATH; until $PY autoresearch/orchestrator.py run $INC; do \
echo 'orchestrator crashed - restarting in 60 s'; sleep 60; done; echo 'orchestrator finished'; sleep 86400"
fi
if ! tmux list-windows -t ar -F '#{window_name}' | grep -qx lead; then
  tmux new-window -d -t ar: -n lead -c "$REPO" \
    "export PATH=$AR_HOME/venv/bin:\$PATH AR_HOME=$AR_HOME RLSURF_FLEET_DIR=$AR_HOME/fleet; \
nice -n 10 claude -n ar-lead --permission-mode auto --model $MODEL --effort $EFFORT \
\"\$(cat autoresearch/LEAD_PROMPT.txt)\"; sleep 86400"
fi
tmux list-windows -t ar -F '#{window_index} #{window_name}'
echo "up. state: $AR_HOME/status.md   log: $AR_HOME/orchestrator.log"
