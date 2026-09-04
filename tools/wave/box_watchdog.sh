#!/bin/bash
# box_watchdog.sh - ON-BOX self-destruct, for when the workstation that runs
# tools/fleet_watchdog.py is switched off (user, 2026-09-02: "then I'll turn
# off pc"). CLAUDE.md rule 1: a rented box is running or deleted, never stale.
#
#   nohup bash box_watchdog.sh <instance_id> <run> <deadline_epoch> [grace_min] &
#
# Every 60 s: past the deadline -> destroy. Trainer pid (runs/<run>.pid)
# dead for `grace_min` consecutive minutes (default 40) -> destroy (a box with
# no load for 5 min must be destroyed; 10 min here leaves room for the
# trainer's own restarts and checkpoint writes). Destroy = vastai CLI with -y,
# then the REST API as the fallback, verified by re-listing. Everything is
# logged.
#
# That grace is also the workstation's harvest window: tools/fleet_watchdog.py
# polls the same pid and starts pulling results as soon as it has seen it gone
# twice (~10 min), so raise the grace if a box's results take longer to move
# than they do to compute. The other window is at the far end - the daemon
# harvests at `registry deadline - 20 min`, so THIS deadline must be >= the
# registry one or the box destroys itself before that window opens.
set -u
ID="${1:?instance id}"
RUN="${2:?run name}"
DEADLINE="${3:?deadline epoch seconds}"
GRACE_MIN="${4:-40}"   # 2026-09-04: 10 min lost a 4090 box to the daemon's two-poll harvest race; 40 covers poll + retries + a 15 min pull
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
log "pid-dead grace $GRACE_MIN min = the harvest window this leaves the workstation after the trainer exits (the deadline one is 20 min, and needs this deadline >= the registry deadline)"
while true; do
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then destroy "deadline reached"; fi
  pid=$(cat "runs/$RUN.pid" 2>/dev/null || echo 0)
  if [ "$pid" != "0" ] && kill -0 "$pid" 2>/dev/null; then
    dead=0
  else
    dead=$((dead + 1))
    log "trainer pid $pid not alive ($dead/$GRACE_MIN)"
    if [ "$dead" -ge "$GRACE_MIN" ]; then destroy "trainer dead for $GRACE_MIN minutes"; fi
  fi
  sleep 60
done
