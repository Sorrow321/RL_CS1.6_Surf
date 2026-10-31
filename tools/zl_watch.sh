#!/bin/bash
# zl_watch.sh RUN [RUN ...] - one look at runs on this agent's box (through box.py, the only door):
# the last step lines, the last greedy line, and the newest greedy eval's per-episode energy report.
#   bash tools/zl_watch.sh zlr1b1_en zlr1b1_wen
cd "$(dirname "$0")/.." || exit 1
AG=${AR_AGENT:-zlr1b1}
CMD="cd /root/RL_Surf"
for RUN in "$@"; do
  CMD="$CMD; echo \"=== $RUN \$(date -u +%H:%M:%S)\"; grep -E '^step |greedy:' runs/${RUN}_launch.txt 2>/dev/null | tail -3 | cut -c1-400; \
T=\$(ls runs/$RUN/traj_*.jsonl 2>/dev/null | grep -E 'traj_[0-9]+[.]jsonl\$' | tail -1); \
[ -n \"\$T\" ] && python3 tools/eval_energy_report.py \"\$T\" 2>/dev/null | cut -c1-220"
done
CMD="$CMD; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; uptime"
python autoresearch/box.py ssh --agent "$AG" -- "$CMD"
