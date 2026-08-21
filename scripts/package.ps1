# Build a versioned release zip + SHA256SUMS from the committed tree.
# Run from the repo root: powershell -File scripts\package.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $root
$version = (Get-Content (Join-Path $root "VERSION")).Trim()
$rel = Join-Path $root "release"
New-Item -ItemType Directory -Force $rel | Out-Null
$stage = Join-Path $rel "silicon-node"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }

# git archive exports exactly the committed tree - no venvs, no logs,
# no secrets, by construction.
git archive --format=tar --prefix=silicon-node/ HEAD -o (Join-Path $rel "src.tar")
tar -xf (Join-Path $rel "src.tar") -C $rel
Remove-Item (Join-Path $rel "src.tar")

$zip = Join-Path $rel "silicon-node-v$version.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $stage -DestinationPath $zip
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
"$hash  silicon-node-v$version.zip" |
    Out-File -Encoding ascii (Join-Path $rel "SHA256SUMS.txt")

# Pin the hash into the scoop manifest so the bucket always matches.
$manifest = Join-Path $root "bucket\silicon-node.json"
$m = Get-Content $manifest -Raw | ConvertFrom-Json
$m.version = $version
$m.hash = $hash
$m.url = "https://github.com/OGZamasu/silicon-node/releases/download/v$version/silicon-node-v$version.zip"
$m | ConvertTo-Json -Depth 6 | Out-File -Encoding ascii $manifest

Write-Host "release/silicon-node-v$version.zip"
Write-Host "sha256: $hash"
Write-Host "Scoop manifest updated - commit it, tag v$version, upload the zip + SHA256SUMS.txt to the release."
Pop-Location
