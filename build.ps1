# build.ps1 — build surfcore.dll + test executables with MSVC.
# Flags per docs/03: /O2 /fp:precise (no FMA contraction/reassociation), /openmp.
param([switch]$TestsOnly)

$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { Write-Error "vcvars64.bat not found"; exit 1 }
New-Item -ItemType Directory -Force build | Out-Null

$src = "src\bsp.c src\trace.c src\pm.c src\env.c"
$flags = "/nologo /O2 /fp:precise /openmp /W3 /DSURFCORE_BUILD"

$cmds = @()
if (-not $TestsOnly) {
    $cmds += "cl $flags /LD $src /Fe:build\surfcore.dll /Fo:build\ /link /IMPLIB:build\surfcore.lib"
}
$cmds += "cl $flags tests\dump_map.c src\bsp.c src\trace.c src\pm.c /Fe:build\dump_map.exe /Fo:build\"
$cmds += "cl $flags tests\test_trace.c src\bsp.c src\trace.c src\pm.c /Fe:build\test_trace.exe /Fo:build\"
$cmds += "cl $flags tests\test_physics.c src\bsp.c src\trace.c src\pm.c /Fe:build\test_physics.exe /Fo:build\"
$cmds += "cl $flags tests\bench.c $src /Fe:build\bench.exe /Fo:build\"

foreach ($c in $cmds) {
    Write-Host ">> $c"
    cmd /c "call `"$vcvars`" >nul 2>&1 && $c"
    if ($LASTEXITCODE -ne 0) { Write-Error "build failed: $c"; exit 1 }
}
Write-Host "build OK"
