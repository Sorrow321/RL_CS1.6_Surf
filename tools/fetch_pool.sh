#!/bin/bash
# fetch_pool.sh - pull the map pool onto a box and unpack it in place.
#
#   POOL_ID=<google-drive-file-id> bash tools/fetch_pool.sh
#   POOL_URL=https://... bash tools/fetch_pool.sh          # any direct URL
#
# The archive is a flat `maps/` tree, so it unpacks over the repo root and
# every file lands exactly where the code already looks: zones.load_zones
# and the field cache both resolve <bsp>.parent/<stem>.<ext>. No path flags,
# no code change.
#
# Per map it carries the .bsp, the gated .goal_<cell>.npz, a prebaked
# .sdf_32.npz (worth ~13 minutes of CPU per box - it is rebuilt otherwise),
# .occ_32/.slabocc_32 byproducts, and a .zones.json pinned as
# `source: manual` so load_zones never regenerates it from the BSP.
#
# THE SOURCE LOCATION IS NOT IN GIT. The pool is the user's map corpus plus
# fields derived from it, and a share link is just a pointer to the same
# data, so it is passed in by environment or read from a gitignored file
# (runs/pool_source.txt). Do not hardcode it here.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${POOL_URL:-}" ]; then
  if [ -z "${POOL_ID:-}" ] && [ -f runs/pool_source.txt ]; then
    POOL_ID="$(tr -d ' \r\n' < runs/pool_source.txt)"
  fi
  if [ -z "${POOL_ID:-}" ]; then
    echo "!! set POOL_ID (drive file id) or POOL_URL, or write the id into"
    echo "   runs/pool_source.txt (gitignored)" >&2
    exit 2
  fi
  POOL_URL="https://drive.usercontent.google.com/download?id=${POOL_ID}&export=download&confirm=t"
fi

EXPECT_MD5="${POOL_MD5:-266f0855458cbfb2f4bb61fbaa4d55ea}"
TARBALL="${TARBALL:-/tmp/surf_pool.tar.gz}"

if [ ! -s "$TARBALL" ]; then
  echo "== fetching the pool"
  # --fail so an HTML error page never gets written out as a tarball, and
  # -C - so an interrupted pull resumes instead of restarting 546 MB
  curl -fL -C - --retry 3 --retry-delay 5 "$POOL_URL" -o "$TARBALL"
fi

echo "== verifying"
GOT=$(md5sum "$TARBALL" | cut -d' ' -f1)
if [ "$GOT" != "$EXPECT_MD5" ]; then
  # a truncated transfer that still exits 0 has bitten this project before
  echo "!! md5 mismatch: got $GOT want $EXPECT_MD5" >&2
  echo "   (delete $TARBALL and re-run; a Drive quota page can land here)" >&2
  exit 1
fi

echo "== unpacking into $(pwd)/maps"
tar -xzf "$TARBALL" -C .
N=$(ls maps/*.bsp 2>/dev/null | wc -l)
echo "== $N maps present"
if [ -f manifest.json ]; then
  python3 - <<'PY' || true
import json
d = json.load(open("manifest.json"))
k = {}
for r in d["maps"]:
    k[r["finish_kind"]] = k.get(r["finish_kind"], 0) + 1
print(f"   manifest: {len(d['maps'])} maps, finish kinds {k}")
PY
fi
