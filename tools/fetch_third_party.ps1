# fetch_third_party.ps1 — download the upstream engine reference sources into
# third_party/. These files are cited throughout docs/ and were the ground
# truth for the physics port; they are NOT redistributed in this repository
# (see NOTICE.md). Usage:  .\tools\fetch_third_party.ps1 [-Force]
param([switch]$Force)

$dst = Join-Path (Split-Path $PSScriptRoot -Parent) "third_party"
New-Item -ItemType Directory -Force $dst | Out-Null

$CS = "https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master"
$RE = "https://raw.githubusercontent.com/rehlds/ReHLDS/master/rehlds"
$HL = "https://raw.githubusercontent.com/ValveSoftware/halflife/master"

$files = @{
    # ReGameDLL_CS — the CS 1.6 game DLL (authoritative for pm.c)
    "cs_pm_shared.cpp"    = "$CS/regamedll/pm_shared/pm_shared.cpp"
    "cs_pm_shared.h"      = "$CS/regamedll/pm_shared/pm_shared.h"
    "cs_pm_math.cpp"      = "$CS/regamedll/pm_shared/pm_math.cpp"
    "cs_pm_defs.h"        = "$CS/regamedll/pm_shared/pm_defs.h"
    "cs_mathlib.h"        = "$CS/regamedll/common/mathlib.h"
    "cs_usercmd.h"        = "$CS/regamedll/common/usercmd.h"
    "cs_util.h"           = "$CS/regamedll/dlls/util.h"
    "cs_triggers.cpp"     = "$CS/regamedll/dlls/triggers.cpp"
    "cs_subs.cpp"         = "$CS/regamedll/dlls/subs.cpp"
    "cs_doors.cpp"        = "$CS/regamedll/dlls/doors.cpp"
    "cs_bmodels.cpp"      = "$CS/regamedll/dlls/bmodels.cpp"
    "weapons.h"           = "$CS/regamedll/dlls/weapons.h"
    # ReHLDS — engine (trace/hull/wrapper code, cvar defaults)
    "engine_world.cpp"    = "$RE/engine/world.cpp"
    "engine_world.h"      = "$RE/engine/world.h"
    "engine_pmovetst.cpp" = "$RE/engine/pmovetst.cpp"
    "engine_pmove.cpp"    = "$RE/engine/pmove.cpp"
    "engine_model.cpp"    = "$RE/engine/model.cpp"
    "engine_sv_user.cpp"  = "$RE/engine/sv_user.cpp"
    "engine_sv_phys.cpp"  = "$RE/engine/sv_phys.cpp"
    "engine_sv_main.cpp"  = "$RE/engine/sv_main.cpp"
    "engine_host.cpp"     = "$RE/engine/host.cpp"
    "engine_bspfile.h"    = "$RE/public/rehlds/bspfile.h"
    "common_const.h"      = "$RE/common/const.h"
    "common_com_model.h"  = "$RE/common/com_model.h"
    # Half-Life SDK (Valve) — HL variant of pm_shared, for cross-reference
    "hl_pm_shared.c"      = "$HL/pm_shared/pm_shared.c"
    "hl_pm_math.c"        = "$HL/pm_shared/pm_math.c"
    "hl_pm_defs.h"        = "$HL/pm_shared/pm_defs.h"
    "hl_pm_movevars.h"    = "$HL/pm_shared/pm_movevars.h"
    # community references
    "hlbsp_bspdef.h"      = "https://raw.githubusercontent.com/bernhardmgruber/hlbsp/master/src/bspdef.h"
    "UltimateSurf.cfg"    = "https://raw.githubusercontent.com/tonykaram1993/UltimateSurf/master/configs/UltimateSurf.cfg"
}

$ok = 0; $skip = 0; $fail = 0
foreach ($name in $files.Keys | Sort-Object) {
    $path = Join-Path $dst $name
    if ((Test-Path $path) -and -not $Force) { $skip++; continue }
    try {
        Invoke-WebRequest -Uri $files[$name] -OutFile $path -UseBasicParsing -ErrorAction Stop
        $len = (Get-Item $path).Length
        if ($len -lt 100) { throw "suspiciously small ($len bytes)" }
        Write-Host ("  ok    {0}  ({1:N0} bytes)" -f $name, $len)
        $ok++
    } catch {
        Write-Warning "FAILED $name : $_"
        $fail++
    }
}
Write-Host "fetched $ok, skipped $skip (already present), failed $fail"
if ($fail -gt 0) { exit 1 }
