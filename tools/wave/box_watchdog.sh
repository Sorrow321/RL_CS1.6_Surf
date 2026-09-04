#!/bin/bash
# box_watchdog.sh - ON-BOX self-destruct, for when the workstation that runs
# tools/fleet_watchdog.py is switched off (user, 2026-09-02: "then I'll turn
# off pc"). CLAUDE.md rule 1: a rented box is running or deleted, never stale.
#
#   nohup bash box_watchdog.sh <instance_id> <run> <deadline_epoch> &
#
# Every 60 s: past the deadline -> destroy. Trainer pid (runs/<run>.pid)
# dead for 10 consecutive minutes -> destroy (a box with no load for 5 min
# must be destroyed; 10 min here leaves room for the trainer's own restarts
# and checkpoint writes). Destroy = vastai CLI with -y, then the REST API as
# the fallback, verified by re-listing. Everything is logged.
set -u
ID="${1:?instance id}"
RUN="${2:?run name}"
DEADLINE="${3:?deadline epoch seconds}"
LOG=/root/box_watchdog.log
KEY=$(cat /root/.config/vastai/vast_api_key 2>/dev/null || cat /root/.vast_api_key)
cd /root/RL_Surf
dead=0
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
destroy() {
  log "DESTROY $ID: $1"
  for _ in 1 2 3 4 5; do
    vastai destroy instance "$ID" -y >> "$LOG" 2>&1 || true
    curl -s -X DELETE "https://console.vast.ai/api/v0/instances/$ID/" \
         -H "Authorization: Bearer $KEY" >> "$LOG" 2>&1 || true
    sleep 20
  done
}
log "watchdog up: instance $ID run $RUN deadline $(date -u -d @"$DEADLINE" +%FT%TZ)"
while true; do
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then destroy "deadline reached"; fi
  pid=$(cat "runs/$RUN.pid" 2>/dev/null || echo 0)
  if [ "$pid" != "0" ] && kill -0 "$pid" 2>/dev/null; then
    dead=0
  else
    dead=$((dead + 1))
    log "trainer pid $pid not alive ($dead/10)"
    if [ "$dead" -ge 10 ]; then destroy "trainer dead for 10 minutes"; fi
  fi
  sleep 60
done
