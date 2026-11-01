#!/bin/bash
cd /home/steam/ar_loop/wt/b2
export AR_HOME=/home/steam/ar_loop AR_AGENT=zlr2b2 AR_ROUND=2 AR_START=1793466558.4978988 RLSURF_FLEET_DIR=/home/steam/ar/fleet PYTHONIOENCODING=utf-8 PATH=/home/steam/ar_loop/venv/bin:/home/steam/.local/bin:/home/steam/ar/venv/bin:/home/steam/ar/venv/bin:/home/steam/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
exec nice -n 10 /home/steam/.local/bin/claude -n ar-zlr2b2 --permission-mode auto --model claude-opus-5-5 --effort max "$(cat .ar_prompt.txt)"
