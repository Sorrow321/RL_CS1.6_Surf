#!/bin/bash
cd /home/steam/ar_az/wt/b3
export AR_HOME=/home/steam/ar_az AR_AGENT=azr1b3 AR_ROUND=1 AR_START=1794256799.7521636 RLSURF_FLEET_DIR=/home/steam/ar/fleet PYTHONIOENCODING=utf-8 PATH=/home/steam/ar_az/venv/bin:/home/steam/.local/bin:/home/steam/ar/venv/bin:/home/steam/ar/venv/bin:/home/steam/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
exec nice -n 10 /home/steam/.local/bin/claude -n ar-azr1b3 --permission-mode auto --model claude-opus-5-5 --effort max "$(cat .ar_prompt.txt)"
