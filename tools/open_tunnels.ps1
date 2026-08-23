# open_tunnels.ps1 - one dashboard tunnel per rented box, from the live
# vast instance list. Idempotent: re-running kills the old tunnels first, so
# it can be used to refresh after the fleet changes.
#
#   powershell -File tools\open_tunnels.ps1
#
# Local 8600 is this machine's own dashboard and is left alone; rented boxes
# get 8601 upward in instance-id order. Each box's dashboard is started
# remotely if it is not already up.
# NOT "Stop": PowerShell 5.1 wraps a native exe's stderr in an
# ErrorRecord, and vast.ai's ssh prints a welcome banner there on
# every connection - which aborts the script on box one.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"

$raw = & vastai show instances --raw 2>$null | Out-String
$inst = $raw | ConvertFrom-Json
if (-not $inst) { Write-Host "no instances"; exit 0 }

# drop any tunnels we opened before (ssh -N -L 86xx)
Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" |
    Where-Object { $_.CommandLine -match '-L\s*86\d\d:' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

$reg = @{}
if (Test-Path "runs\fleet.json") {
    $reg = Get-Content "runs\fleet.json" -Raw | ConvertFrom-Json
}

$port = 8601
$rows = @()
foreach ($i in ($inst | Sort-Object id)) {
    $h = $i.ssh_host; $p = $i.ssh_port; $id = $i.id
    if (-not $h) { continue }
    $label = "?"
    if ($reg.PSObject.Properties.Name -contains "$id") { $label = $reg."$id".label }

    # start the dashboard on the box if it is not already listening
    # The detached form matters: `nohup cmd &` alone leaves ssh waiting on the
    # channel and the call HANGS (it ate a 300 s budget before this was fixed).
    # A subshell with every fd redirected, plus -n, returns immediately. If a
    # dashboard is already up the new one just loses the port bind, which is
    # harmless and is why this is safe to re-run.
    $remote = "cd /root/RL_Surf 2>/dev/null || cd /workspace/RL_Surf; " +
              "(setsid python3 tools/dashboard.py --port 8600 " +
              "</dev/null >/tmp/dash.log 2>&1 &) ; exit 0"
    & ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=12 -p $p "root@$h" $remote *> $null
    $up = ""


    Start-Process ssh -ArgumentList @(
        "-o","StrictHostKeyChecking=no","-o","ServerAliveInterval=30",
        "-o","ExitOnForwardFailure=yes","-N",
        "-L","${port}:localhost:8600","-p","$p","root@$h") -WindowStyle Hidden
    $rows += [pscustomobject]@{
        instance = $id; arm = $label; url = "http://localhost:$port/"
        dash_procs = ($up -join " ").Trim()
    }
    $port++
}
Start-Sleep -Seconds 4
$rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
Write-Host "local dashboard (this machine): http://localhost:8600/"
