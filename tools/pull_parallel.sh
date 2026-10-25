#!/bin/bash
# pull_parallel.sh RUN FILE [DST_NAME] - fetch runs/RUN/FILE from this agent's box in parallel
# 2 MB chunks (pbr3b1, 2026-10-25). A box whose single scp runs at ~20 KB/s (57580595, Thailand)
# cannot deliver a 63 MB checkpoint inside box.py pull's 30-minute scp timeout; 8 parallel
# connections ran ~6x faster. The file lands as $AR_HOME/runs/$AR_AGENT/RUN/DST_NAME only when
# its md5 matches the box's, so the round end's handon_ckpt sees a complete file or none.
set -euo pipefail
RUN=$1; FILE=$2; DST=${3:-$2}
REC=$AR_HOME/boxes/$AR_AGENT.json
H=$(python3 -c "import json;print(json.load(open('$REC'))['host'])")
P=$(python3 -c "import json;print(json.load(open('$REC'))['port'])")
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p $P root@$H"
TAG=$(echo "$RUN.$FILE" | tr '/' '_')
WANT=$($SSH "cd /root/RL_Surf/runs/$RUN && md5sum $FILE | cut -d' ' -f1 && rm -rf /root/pt/$TAG && mkdir -p /root/pt/$TAG && split -b 2000000 -d -a 3 $FILE /root/pt/$TAG/c_" 2>/dev/null | tail -1)
LIST=$($SSH "ls /root/pt/$TAG" 2>/dev/null)
OUT=$AR_HOME/runs/$AR_AGENT/$RUN; mkdir -p "$OUT/.chunks_$TAG"
echo "$LIST" | xargs -P 8 -I{} sh -c "for k in 1 2 3 4; do scp -q -o BatchMode=yes -o StrictHostKeyChecking=no -P $P root@$H:/root/pt/$TAG/{} '$OUT/.chunks_$TAG/{}' && exit 0; sleep 5; done; exit 1"
cat $(echo "$LIST" | sed "s#^#$OUT/.chunks_$TAG/#") > "$OUT/$DST.part"
GOT=$(md5sum "$OUT/$DST.part" | cut -d' ' -f1)
if [ "$GOT" = "$WANT" ]; then mv "$OUT/$DST.part" "$OUT/$DST"; rm -rf "$OUT/.chunks_$TAG"; $SSH "rm -rf /root/pt/$TAG" 2>/dev/null || true; echo "pulled $RUN/$FILE -> $OUT/$DST (md5 $WANT)"; else echo "!! md5 $GOT != $WANT, kept $OUT/$DST.part"; exit 1; fi
