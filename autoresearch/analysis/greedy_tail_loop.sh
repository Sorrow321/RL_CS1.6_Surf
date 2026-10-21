#!/bin/bash
# greedy_tail_loop.sh RUN TARGET PERIOD_S [HOURS] - keep TARGET (a path under /root/RL_Surf on the
# box, the run's --spawn-states file) cut from the
# run's OWN latest greedy line (CLAUDE.md section 0: the policy's own recording, no demo):
#   ckpt_latest -> snapshot -> local record_ckpt --dump-states (9 greedy episodes from the true
#   start) -> tools/greedy_tail_states.py (last 3 s before each failure) -> scp + atomic mv.
# The trainer re-reads --spawn-states when the file's mtime changes (every 20 iterations).
RUN=$1; TARGET=$2; PERIOD=${3:-1200}; HOURS=${4:-6}
cd "$(dirname "$0")/../.."
source /home/steam/ar/venv/bin/activate
export AR_AGENT=${AR_AGENT:-r3b2} AR_HOME=${AR_HOME:-/home/steam/ar} PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1
REC=$AR_HOME/boxes/$AR_AGENT.json
HOST=$(python3 -c "import json;print(json.load(open('$REC'))['host'])")
PORT=$(python3 -c "import json;print(json.load(open('$REC'))['port'])")
W=${GT_DIR:-/tmp/gt_$RUN}; mkdir -p "$W"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p $PORT root@$HOST"
end=$(( $(date +%s) + HOURS*3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  t0=$(date +%s)
  echo "== $(date -u +%H:%M:%S) refresh $RUN"
  if ! $SSH "cd /root/RL_Surf && kill -0 \$(cat runs/$RUN.pid) 2>/dev/null" 2>/dev/null; then
    echo "   trainer gone: stop"; exit 0; fi
  $SSH "cd /root/RL_Surf && cp runs/$RUN/ckpt_latest.pt runs/$RUN/ckpt_gtsnap.pt && md5sum runs/$RUN/ckpt_gtsnap.pt" 2>/dev/null | grep -v Welcome > "$W/md5.txt"
  scp -q -o BatchMode=yes -o StrictHostKeyChecking=no -P $PORT root@$HOST:/root/RL_Surf/runs/$RUN/ckpt_gtsnap.pt "$W/snap.pt"
  want=$(cut -d' ' -f1 "$W/md5.txt"); got=$(md5sum "$W/snap.pt" | cut -d' ' -f1)
  if [ -n "$want" ] && [ "$want" = "$got" ]; then
    seed=$(( $(date +%s) % 100000 ))
    if nice -n 15 timeout 900 python tools/record_ckpt.py "$W/snap.pt" --episodes 9 --seed $seed \
         --ep-ticks 6000 --dump-states "$W/g.npz" --out "$W/g.jsonl" > "$W/rec.log" 2>&1 \
       && python tools/greedy_tail_states.py "$W/g.npz" "$W/g.jsonl" "$W/tail.npy" --secs 3 --every 5 --min-speed ${GT_MIN_SPEED:-300} | tail -1; then
      scp -q -o BatchMode=yes -o StrictHostKeyChecking=no -P $PORT "$W/tail.npy" root@$HOST:/root/RL_Surf/$TARGET.new \
        && $SSH "cd /root/RL_Surf && mv $TARGET.new $TARGET && ls -la $TARGET" 2>/dev/null | grep -v Welcome
      python autoresearch/analysis/sid_eval_route.py "$W/g.jsonl" 2>/dev/null | cut -c1-240
    else
      echo "   record/cut failed: the previous file stays"; tail -3 "$W/rec.log"
    fi
  else
    echo "   snapshot md5 mismatch ($want vs $got): skipped"
  fi
  dt=$(( $(date +%s) - t0 )); sl=$(( PERIOD - dt )); [ $sl -gt 30 ] && sleep $sl
done
