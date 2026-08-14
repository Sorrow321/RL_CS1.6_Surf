#!/usr/bin/env bash
# build.sh — Linux/gcc build of libsurfcore.so + tests.
# Usage: ./build.sh [--test]
# Requires: gcc (with OpenMP), libc/libm. Float flags mirror docs/03:
# -ffp-contract=off (no FMA contraction — determinism), default IEEE math.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

CFLAGS="-O2 -fPIC -ffp-contract=off -fopenmp -Wall -Wno-unused-function"
SRC="src/bsp.c src/trace.c src/pm.c src/env.c"
PM_SRC="src/bsp.c src/trace.c src/pm.c"

gcc $CFLAGS -DSURFCORE_BUILD -fvisibility=hidden -shared $SRC -o build/libsurfcore.so -lm
gcc $CFLAGS tests/dump_map.c      $PM_SRC -o build/dump_map     -lm
gcc $CFLAGS tests/test_trace.c    $PM_SRC -o build/test_trace   -lm
gcc $CFLAGS tests/test_physics.c  $PM_SRC -o build/test_physics -lm
gcc $CFLAGS -DSURFCORE_BUILD tests/bench.c $SRC -o build/bench  -lm
echo "build OK"

if [[ "${1:-}" == "--test" ]]; then
    ./build/test_trace   maps/surf_ski_2.bsp
    ./build/test_physics maps/surf_ski_2.bsp
    ./build/bench        maps/surf_ski_2.bsp 256 2000
fi
