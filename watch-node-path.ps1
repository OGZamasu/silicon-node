# Silicon Node - network path watchdog (a scheduled task, every 5 minutes).
# The node has three legs that can each fail independently while the service
# itself stays healthy (seen 2026-08-22, hub request 135):
#   1. the service inside WSL            (:8790 in the SiliconNode distro)
#   2. the Windows forwarding leg        (netsh portproxy via iphlpsvc)
#   3. the tailnet leg                   (tailscale serve accepts TCP first,
#                                         then dials the backend - a wedged
#                                         dial looks like "TCP connects, HTTP
#                                         times out" to remote peers)
# This script probes each leg with short timeouts and repairs only the leg
# that failed. Quiet when healthy; appends to the log only on trouble.
#
# Install it with register-path-watchdog.ps1 (one elevated run). That copies
# this file to %ProgramData%\SiliconNode, where only administrators can
# change it, and runs it from there as the owner's account - not from the
# checkout, which ordinary users and the node itself can write (hub 153).
# Pure ASCII on purpose: Windows PowerShell 5.1 reads BOM-less UTF-8 as ANSI.

$ErrorActionPreference = "SilentlyContinue"
$LogFile = Join-Path $PSScriptRoot "node-path-watchdog.log"

function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $LogFile -Value $line -Encoding ASCII
}

# /health is the node's open probe: every /v1/ route needs a bearer token
# from anywhere but real loopback, so probing /v1/node without one reads
# 401 - and counted as "down", that made every run a false alarm.
function Probe($url) {
    $code = & curl.exe -s -o NUL -w "%{http_code}" -m 6 $url
    return ($code -eq "200")
}

function GuestProbe {
    $code = & wsl.exe -d SiliconNode --exec curl -s -m 6 -o /dev/null -w "%{http_code}" http://127.0.0.1:8790/health
    return ("$code" -match "200")
}

function DistroRunning {
    $list = (& wsl.exe --list --running | Out-String) -replace "`0", ""
    return ($list -match "SiliconNode")
}

function IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    return (New-Object Security.Principal.WindowsPrincipal $id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

# The distro lives only while some wsl.exe session holds it: without the
# keepalive, WSL shuts SiliconNode down about two minutes after the last
# session ends, taking the service and ninfer with it (seen 2026-09-24).
# The keepalive task has only a logon trigger, so a keepalive that dies is
# not replaced until the next sign-in - replace it here, before the node
# goes down rather than after.
$keep = Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" |
    Where-Object { $_.CommandLine -match "SiliconNode" -and $_.CommandLine -match "sleep" }
if (-not $keep) {
    Log "WSL keepalive missing - restarting it"
    Start-ScheduledTask -TaskName "SiliconNode Keepalive"
    if (-not $?) {
        Start-Process -FilePath "wsl.exe" -ArgumentList "-d","SiliconNode","--exec","sleep","infinity" -WindowStyle Hidden
    }
}

$localUrl = "http://127.0.0.1:8790/health"
$tsExe = "C:\Program Files\Tailscale\tailscale.exe"
$tsIp = $null
if (Test-Path $tsExe) { $tsIp = ((& $tsExe ip -4 | Out-String).Trim() -split "\s+")[0] }

$localOk = Probe $localUrl
$tsOk = $true
if ($tsIp) { $tsOk = Probe "http://${tsIp}:8790/health" }

if ($localOk -and $tsOk) { exit 0 }   # all legs healthy - stay quiet

# Legs 2 and 3 are repaired with netsh and service restarts, which need an
# elevated token. Without one, say what would have been done instead of
# logging a repair that silently failed.
$admin = IsAdmin

# --- leg 1 + 2: local path (Windows loopback -> portproxy -> WSL) ---
if (-not $localOk) {
    Log "local path down ($localUrl)"
    if (-not (DistroRunning)) {
        Log "distro not running - starting keepalive"
        Start-Process -FilePath "wsl.exe" -ArgumentList "-d","SiliconNode","--exec","sleep","infinity" -WindowStyle Hidden
        Start-Sleep -Seconds 30
    }
    if (GuestProbe) {
        # Service is fine inside the guest - the forwarding leg is broken.
        if (-not $admin) {
            Log "forwarding leg broken, but this task is not elevated - re-run register-path-watchdog.ps1 as administrator"
        } else {
            $wslIp = ((& wsl.exe -d SiliconNode -- hostname -I | Out-String).Trim() -split "\s+")[0]
            if ($wslIp) {
                netsh interface portproxy delete v4tov4 listenport=8790 listenaddress=0.0.0.0 | Out-Null
                netsh interface portproxy add v4tov4 listenport=8790 listenaddress=0.0.0.0 connectport=8790 connectaddress=$wslIp | Out-Null
                if ($LASTEXITCODE -eq 0) { Log "portproxy refreshed to ${wslIp}:8790" }
                else { Log "portproxy refresh FAILED (netsh exit $LASTEXITCODE)" }
            }
            if (-not (Probe $localUrl)) {
                Log "still down after portproxy refresh - restarting iphlpsvc"
                Restart-Service iphlpsvc -Force
                Start-Sleep -Seconds 10
            }
        }
    } else {
        Log "service down inside the guest - restarting silicon-node"
        & wsl.exe -d SiliconNode --exec systemctl restart silicon-node | Out-Null
        Start-Sleep -Seconds 20
    }
    $localOk = Probe $localUrl
    Log "local path after repair: $(if ($localOk) { 'UP' } else { 'STILL DOWN' })"
}

# --- leg 3: tailnet (tailscale serve) ---
if ($localOk -and $tsIp -and -not (Probe "http://${tsIp}:8790/health")) {
    if (-not $admin) {
        Log "tailnet leg down (local OK), but this task is not elevated - cannot restart Tailscale"
        exit 0
    }
    Log "tailnet leg down (local OK) - restarting the Tailscale service"
    Restart-Service Tailscale -Force
    Start-Sleep -Seconds 20
    $tsOk = Probe "http://${tsIp}:8790/health"
    Log "tailnet leg after restart: $(if ($tsOk) { 'UP' } else { 'STILL DOWN' })"
}
