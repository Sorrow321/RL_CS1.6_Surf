#!/bin/bash
# finish_chain.sh - on a box beside a training run (zlr3b4, 2026-11-01): the policy's own greedy
# line (tools/beam_tas.py --frontier-last, via finish_search.py) plus a short policy beam search
# from its last seconds, then physics-only shooting around that line (tools/shoot_finish.py
# --evolve); every finishing line found is merged into the run's --spawn-states pool IN PLACE
# (tools/line_pool.py, atomic: the trainer re-reads the file when its mtime changes).
#
#   nohup bash tools/finish_chain.sh runs_ckpt.pt runs/sp3/pool.npy runs/fs1 > runs/fs1.log 2>&1 &
set -u
cd /root/RL_Surf
CK=${1:-runs_ckpt.pt}
POOL=${2:-runs/sp3/pool.npy}
OUT=${3:-runs/fs1}
SHARE=${SHARE:-0.25}
mkdir -p "$OUT"
echo "== $(date -u +%H:%M:%S) greedy line + policy beam searches"
python3 -u tools/finish_search.py "$CK" --out "$OUT" --at=-8:-4:4 --window 12 --seeds 0 \
  --envs 4096 --greedy-envs 512 --view-hold 0.3 12 --branch 2048 --branch-jitter 20 \
  --branch-burst 1.5 --branch-hold 0.4 --branch-protect 4
[ -f "$OUT/greedy/beam_frontier.npz" ] || { echo "no greedy line"; exit 1; }
echo "== $(date -u +%H:%M:%S) shooting around the greedy line"
python3 -u tools/shoot_finish.py "$CK" "$OUT/greedy/beam_frontier.npz" --out "$OUT/shoot" \
  --at=-14:-2:0.5 --envs 4096 --rounds 12 --evolve 64 --mut-sigma 4 --duck-flip 0.1 --threads 8
ls "$OUT"/shoot/*.npz "$OUT"/s*/beam_frontier.npz 2>/dev/null > "$OUT/lines.txt"
echo "== $(date -u +%H:%M:%S) $(wc -l < "$OUT/lines.txt") candidate lines"
if [ -s "$OUT/lines.txt" ]; then
  python3 tools/line_pool.py $(cat "$OUT/lines.txt") --finished-only --secs 3 --stride 1 \
    --merge "$POOL" --share "$SHARE" --out "$POOL" --tick-ms 7 || echo "no finishing line merged"
fi
echo "== $(date -u +%H:%M:%S) done"
