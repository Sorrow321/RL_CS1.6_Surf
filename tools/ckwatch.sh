#!/bin/bash
# usage (ON THE BOX, from anywhere): nohup tools/ckwatch.sh <run> <protected-step-or-none> > /root/ckwatch_<run>.log 2>&1 &
# The trainer keeps only ckpt_latest.pt (+ every --ckpt-every steps); the round end hands on the
# newest ckpt_<step>.pt at/before the best eval, so a best eval between snapshots needs these.
# keep per-step copies of a run's ckpt_latest.pt (newest 15 + protected), until the trainer exits
RUN="$1"; PROTECT="$2"
cd /root/RL_Surf
PID=$(cat runs/${RUN}.pid)
while kill -0 "$PID" 2>/dev/null; do
  STEP=$(python3 -c "import torch,sys;print(torch.load('runs/$RUN/ckpt_latest.pt',map_location='cpu',weights_only=False)['global_step'])" 2>/dev/null)
  if [ -n "$STEP" ] && [ ! -f "runs/$RUN/ckpt_$STEP.pt" ]; then
    cp "runs/$RUN/ckpt_latest.pt" "runs/$RUN/ckpt_$STEP.tmp" && \
    S2=$(python3 -c "import torch;print(torch.load('runs/$RUN/ckpt_$STEP.tmp',map_location='cpu',weights_only=False)['global_step'])" 2>/dev/null) && \
    if [ "$S2" = "$STEP" ]; then mv "runs/$RUN/ckpt_$STEP.tmp" "runs/$RUN/ckpt_$STEP.pt"; else rm -f "runs/$RUN/ckpt_$STEP.tmp"; fi
  fi
  ls -t runs/$RUN/ckpt_[0-9]*.pt 2>/dev/null | tail -n +16 | grep -v "$PROTECT" | xargs -r rm -f
  sleep 60
done
