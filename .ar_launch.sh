#!/bin/bash
cd /home/steam/ar_uf2/wt/b2
export AR_HOME=/home/steam/ar_uf2 AR_AGENT=uf2r2b2 AR_ROUND=2 AR_START=1793396406.418926 RLSURF_FLEET_DIR=/home/steam/ar/fleet PYTHONIOENCODING=utf-8 PATH=/home/steam/ar_uf2/venv/bin:/home/steam/.local/bin:/home/steam/ar/venv/bin:/home/steam/ar/venv/bin:/home/steam/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
exec nice -n 10 /home/steam/.local/bin/claude -n ar-uf2r2b2 --permission-mode auto --model claude-opus-5-5 --effort max "$(cat .ar_prompt.txt)"
