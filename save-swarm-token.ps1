# Save the swarm token (from the Mac's swarm.json) onto this node.
# Writes \\wsl$\SiliconNode\opt\silicon\swarm.json in the same shape the
# Mac uses: the shared token plus the peer registry. Hidden prompt, so the
# token never lands in command history.

$sec = Read-Host "Paste the swarm token" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
try { $token = [Runtime.InteropServices.Marshal]::PtrToStringAuto($ptr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
if (-not $token) { Write-Error "No token entered."; exit 1 }

$config = [ordered]@{
    swarm_token = $token
    peers       = @(
        [ordered]@{
            name     = "silicon-optimizer-mac"
            base_url = "http://100.99.135.89:8788"
        }
    )
} | ConvertTo-Json -Depth 4

$path = "\\wsl$\SiliconNode\opt\silicon\swarm.json"
[IO.File]::WriteAllText($path, $config + "`n")
Write-Host "OK: wrote $path (token + Mac peer registry)" -ForegroundColor Green
Write-Host "Tell Claude it is saved - the node picks it up on next service restart."
