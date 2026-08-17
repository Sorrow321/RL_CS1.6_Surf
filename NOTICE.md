# Notices & Attribution

**RL_Surf** is a non-commercial research/hobby project. It is not affiliated
with, endorsed by, or sponsored by Valve Corporation. *Counter-Strike* and
*Half-Life* are trademarks of Valve Corporation, used here descriptively.

## License of this repository

All original code in this repository (`src/`, `python/`, `tools/`, `tests/`,
`viewer/app.js`, `viewer/index.html`) is released under the **GNU GPLv3**
(see [LICENSE](LICENSE)).

GPLv3 is chosen deliberately: the player-movement code in `src/pm.c` is a
faithful transcription of the movement logic in **ReGameDLL_CS** (see below),
and this repository honors that project's GPLv3 licensing.

## Upstream engine sources (fetched, not redistributed)

The physics port was written against — and its docs cite — reference sources
that are **not stored in this repository**. Run
`tools\fetch_third_party.ps1` to download them into `third_party/` from their
public upstream repositories:

| Source | Upstream | License / status |
|---|---|---|
| ReGameDLL_CS (`cs_*.cpp/h`, `weapons.h`) | github.com/rehlds/ReGameDLL_CS | GPLv3 (reverse-engineered CS 1.6 game DLL; underlying game code © Valve) |
| ReHLDS (`engine_*.cpp/h`, `common_*.h`) | github.com/rehlds/ReHLDS | GPLv3 (reverse-engineered HLDS engine; underlying engine code © Valve) |
| Half-Life SDK (`hl_pm_*.c/h`) | github.com/ValveSoftware/halflife | © Valve, Half-Life 1 SDK license |
| hlbsp format spec (`hlbsp_bspdef.h`) | github.com/bernhardmgruber/hlbsp | BSP v30 community spec |
| UltimateSurf.cfg | github.com/tonykaram1993/UltimateSurf | surf server config reference |

## Bundled third-party content

- **`maps/*.bsp` — community-made maps; all rights remain with their
  creators.** Currently bundled: `surf_ski_2.bsp` (a classic community
  Counter-Strike 1.6 surf map) and `surf_src_cannonball.bsp` (a community
  CS 1.6 port of a Counter-Strike: Source surf map — © the original mapper
  and the port's author). They are distributed here as is customary for
  custom CS maps (map archives and server fastdl distribution), solely as
  research inputs for this non-commercial project: the maps ARE the
  benchmark, the way a dataset is to a vision paper. They embed their
  mappers' textures; Valve's `halflife.wad` is referenced but **not**
  included. The cached `.npz`/`.zones.json` files alongside them are
  machine-derived navigation data baked from the maps for training, not
  creative content. If you are one of the authors and want your map
  removed, or credited by name (we would prefer to credit you!), open an
  issue and it will be done promptly.
- **`viewer/vendor/three.min.js`, `OrbitControls.js`** — three.js r147,
  © three.js authors, MIT license.
- No Valve game assets (textures, models, sounds, code binaries) are
  included in this repository.

## Runtime dependencies

pyglet (BSD), numpy (BSD), optionally stable-baselines3 (MIT) /
gymnasium (MIT) — installed by the user, not bundled.
