#!/bin/bash
# tunnel.sh - self-healing dashboard tunnel to a rented box.
#
#   bash tools/tunnel.sh 8601 32054 ssh6.vast.ai [8600]
#
# The dashboard binds 127.0.0.1 on the box (tools/dashboard.py), so the only
# way to see it is a forward. Rented boxes drop ssh sessions, so a bare
# `ssh -L` dies silently after an hour and the user finds a dead tab; this
# reconnects forever with keepalives and logs every drop.
#
# Local port convention: 8601..8604, one per box in the fleet.
set -uo pipefail

LOCAL="${1:?usage: tunnel.sh <local_port> <ssh_port> <host> [remote_port]}"
PORT="${2:?usage: tunnel.sh <local_port> <ssh_port> <host> [remote_port]}"
HOST="${3:?usage: tunnel.sh <local_port> <ssh_port> <host> [remote_port]}"
REMOTE="${4:-8600}"

echo "tunnel: http://localhost:$LOCAL/  ->  $HOST:$PORT (remote $REMOTE)"
while true; do
  ssh -N -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 -o ConnectTimeout=15 \
      -L "$LOCAL:localhost:$REMOTE" -p "$PORT" "root@$HOST"
  echo "$(date -u +%H:%M:%S) tunnel dropped (rc=$?), reconnecting in 5s"
  sleep 5
done
