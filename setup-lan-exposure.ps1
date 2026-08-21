# Silicon Node — LAN exposure (run as Administrator)
# Forwards Windows :8790 -> WSL2 (SiliconNode distro) :8790 and opens the
# firewall for the LAN. Re-run after a reboot if the WSL IP changed
# (portproxy is updated idempotently).

$ErrorActionPreference = "Stop"

$wslIp = (wsl -d SiliconNode -- hostname -I).Trim().Split(" ")[0]
if (-not $wslIp) { throw "Could not determine the SiliconNode WSL IP. Is the distro running?" }
Write-Host "SiliconNode WSL IP: $wslIp"

netsh interface portproxy delete v4tov4 listenport=8790 listenaddress=0.0.0.0 2>$null
netsh interface portproxy add v4tov4 listenport=8790 listenaddress=0.0.0.0 connectport=8790 connectaddress=$wslIp
Write-Host "portproxy 0.0.0.0:8790 -> ${wslIp}:8790"

if (-not (Get-NetFirewallRule -DisplayName "Silicon Node 8790" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName "Silicon Node 8790" -Direction Inbound `
        -Action Allow -Protocol TCP -LocalPort 8790 -Profile Private | Out-Null
    Write-Host "firewall rule created (inbound TCP 8790, Private profile)"
} else {
    Write-Host "firewall rule already present"
}

# Keep the SiliconNode distro (and its systemd service) alive across logins.
schtasks /create /tn "SiliconNode Keepalive" /tr "wsl.exe -d SiliconNode --exec sleep infinity" /sc onlogon /f | Out-Null
Start-Process -FilePath "wsl.exe" -ArgumentList "-d","SiliconNode","--exec","sleep","infinity" -WindowStyle Hidden
Write-Host "keepalive scheduled task created"

$lan = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.InterfaceAlias -notmatch 'WSL|Loopback|vEthernet|Tailscale' -and $_.PrefixOrigin -in 'Dhcp','Manual' } |
    Select-Object -First 1).IPAddress
Write-Host ""
Write-Host "Done. The Mac can now use:  http://${lan}:8790"
