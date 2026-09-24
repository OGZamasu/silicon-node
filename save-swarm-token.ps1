# Save the swarm token (from the Mac's swarm.json) onto this node.
# Writes \\wsl$\SiliconNode\opt\silicon\swarm.json in the same shape the
# Mac uses: the shared token plus the peer registry. Hidden prompt, so the
# token never lands in command history.
#
#   .\save-swarm-token.ps1 -PeerUrl http://100.64.0.9:8788
#
# -PeerUrl is your Mac's swarm address (Silicon Optimizer -> Swarm tab).
# Without it, the peer already saved in swarm.json is kept; with neither,
# the script stops: the node sends the swarm token to that peer, so it must
# be one you named, never a default.
# Pure ASCII on purpose: Windows PowerShell 5.1 reads BOM-less UTF-8 as ANSI.

param(
    [string]$PeerUrl,
    [string]$PeerName = "silicon-optimizer-mac"
)

$path = "\\wsl$\SiliconNode\opt\silicon\swarm.json"

$peers = @()
if ($PeerUrl) {
    $uri = $null
    if (-not [Uri]::TryCreate($PeerUrl, [UriKind]::Absolute, [ref]$uri) -or
        ($uri.Scheme -ne "http" -and $uri.Scheme -ne "https")) {
        Write-Error "-PeerUrl must be an http(s) address such as http://100.64.0.9:8788"
        exit 1
    }
    $peers = @([ordered]@{ name = $PeerName; base_url = $PeerUrl.TrimEnd("/") })
} elseif (Test-Path $path) {
    try {
        $saved = Get-Content $path -Raw | ConvertFrom-Json
        $peers = @($saved.peers | Where-Object { $_.base_url })
    } catch { $peers = @() }
}
if ($peers.Count -eq 0) {
    Write-Error ("No swarm peer to save. Run again with -PeerUrl <your Mac's " +
                 "swarm address>, e.g. .\save-swarm-token.ps1 -PeerUrl http://100.64.0.9:8788")
    exit 1
}

$sec = Read-Host "Paste the swarm token" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
try { $token = [Runtime.InteropServices.Marshal]::PtrToStringAuto($ptr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
if (-not $token) { Write-Error "No token entered."; exit 1 }

$config = [ordered]@{
    swarm_token = $token
    peers       = $peers
} | ConvertTo-Json -Depth 4

[IO.File]::WriteAllText($path, $config + "`n")
Write-Host "OK: wrote $path (token + peer $($peers[0].base_url))" -ForegroundColor Green
Write-Host "Restart the service to pick it up: wsl -d SiliconNode -- systemctl restart silicon-node"
