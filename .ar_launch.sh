#!/bin/bash
cd /home/steam/ar_pb/wt/b2
export AR_HOME=/home/steam/ar_pb AR_AGENT=pbfr1b2 AR_ROUND=1 AR_START=1793273814.4952023 RLSURF_FLEET_DIR=/home/steam/ar/fleet PYTHONIOENCODING=utf-8 PATH=/home/steam/ar_pb/venv/bin:/home/steam/.local/bin:/home/steam/ar/venv/bin:/home/steam/ar/venv/bin:/home/steam/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
exec nice -n 10 /home/steam/.local/bin/claude -n ar-pbfr1b2 --permission-mode auto --model claude-opus-5-5 --effort max "$(cat .ar_prompt.txt)"
