#!/bin/bash
# pull_split.sh - fetch a large file from a box over a slow link in parallel chunks (pbr9b3, 2026-10-27).
#
# A single scp stream from some vast hosts runs at ~20-30 KB/s (a 63 MB checkpoint: 30-50 min, past box.py
# pull's 30-min scp timeout, which then keeps nothing). The per-connection rate is the limit, not the link:
# 16 parallel 4 MB chunks fetched the same checkpoint in ~3 min. The chunks are split on the box, fetched
# at most $PAR at a time (sshd refuses too many simultaneous logins - 2 of 16 failed silently at 16),
# retried until every chunk's md5 matches the box's, joined and md5-checked against the box's file.
#
#   bash tools/pull_split.sh <port> <host> <remote file> <local file> [par=8]
set -u
PORT=$1; HOST=$2; REMOTE=$3; LOCAL=$4; PAR=${5:-8}
SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p "$PORT" "root@$HOST")
TMPD="$(dirname "$LOCAL")/.split_$(basename "$LOCAL")"
RD="/root/.split_$(basename "$REMOTE")_$$"
mkdir -p "$TMPD"
"${SSH[@]}" "mkdir -p $RD && cd $RD && split -b 4M -d -a 3 '$REMOTE' part_ && md5sum part_* > parts.md5 && md5sum '$REMOTE' | cut -d' ' -f1 > whole.md5" || { echo "!! split failed"; exit 1; }
scp -q -o StrictHostKeyChecking=no -P "$PORT" "root@$HOST:$RD/parts.md5" "root@$HOST:$RD/whole.md5" "$TMPD/" || exit 1
for round in 1 2 3 4 5; do
  need=$(cd "$TMPD" && md5sum -c parts.md5 2>/dev/null | grep -v ': OK$' | cut -d: -f1)
  [ -z "$need" ] && break
  echo "round $round: $(echo "$need" | wc -w) chunk(s) to fetch"
  printf '%s\n' $need | xargs -P "$PAR" -I{} timeout 1800 scp -q -o StrictHostKeyChecking=no -P "$PORT" "root@$HOST:$RD/{}" "$TMPD/{}"
done
(cd "$TMPD" && md5sum -c --quiet parts.md5) || { echo "!! chunks still missing after 5 rounds"; exit 1; }
cat "$TMPD"/part_* > "$LOCAL.part"
if [ "$(md5sum "$LOCAL.part" | cut -d' ' -f1)" = "$(cat "$TMPD/whole.md5")" ]; then
  mv "$LOCAL.part" "$LOCAL"; rm -rf "$TMPD"; "${SSH[@]}" "rm -rf $RD"
  echo "ok $(md5sum "$LOCAL")"
else
  echo "!! joined md5 differs from the box's"; exit 1
fi
