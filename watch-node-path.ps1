# Silicon Node - network path watchdog (runs as SYSTEM via scheduled task).
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
# Register with register-path-watchdog.ps1 (one elevated run).

$ErrorActionPreference = "SilentlyContinue"
$LogFile = Join-Path (Split-Path $PSScriptRoot -Parent) "node-path-watchdog.log"

function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $LogFile -Value $line -Encoding ASCII
}

function Probe($url) {
    $code = & curl.exe -s -o NUL -w "%{http_code}" -m 6 $url
    return ($code -eq "200")
}

function GuestProbe {
    $code = & wsl.exe -d SiliconNode --exec curl -s -m 6 -o /dev/null -w "%{http_code}" http://127.0.0.1:8790/v1/node
    return ("$code" -match "200")
}

function DistroRunning {
    $list = (& wsl.exe --list --running | Out-String) -replace "`0", ""
    return ($list -match "SiliconNode")
}

$localUrl = "http://127.0.0.1:8790/v1/node"
$tsExe = "C:\Program Files\Tailscale\tailscale.exe"
$tsIp = $null
if (Test-Path $tsExe) { $tsIp = ((& $tsExe ip -4 | Out-String).Trim() -split "\s+")[0] }

$localOk = Probe $localUrl
$tsOk = $true
if ($tsIp) { $tsOk = Probe "http://${tsIp}:8790/v1/node" }

if ($localOk -and $tsOk) { exit 0 }   # all legs healthy - stay quiet

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
        $wslIp = ((& wsl.exe -d SiliconNode -- hostname -I | Out-String).Trim() -split "\s+")[0]
        if ($wslIp) {
            netsh interface portproxy delete v4tov4 listenport=8790 listenaddress=0.0.0.0 | Out-Null
            netsh interface portproxy add v4tov4 listenport=8790 listenaddress=0.0.0.0 connectport=8790 connectaddress=$wslIp | Out-Null
            Log "portproxy refreshed to ${wslIp}:8790"
        }
        if (-not (Probe $localUrl)) {
            Log "still down after portproxy refresh - restarting iphlpsvc"
            Restart-Service iphlpsvc -Force
            Start-Sleep -Seconds 10
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
if ($localOk -and $tsIp -and -not (Probe "http://${tsIp}:8790/v1/node")) {
    Log "tailnet leg down (local OK) - restarting the Tailscale service"
    Restart-Service Tailscale -Force
    Start-Sleep -Seconds 20
    $tsOk = Probe "http://${tsIp}:8790/v1/node"
    Log "tailnet leg after restart: $(if ($tsOk) { 'UP' } else { 'STILL DOWN' })"
}
