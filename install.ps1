# Silicon Node — Windows-side install: tray app shortcuts + autostart.
# The model stack is provisioned separately: see docs/PROVISIONING.md.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

# The tray app is Electron; install its dependency once if Node.js exists.
$electronDir = Join-Path $root "gui\electron"
$electronExe = Join-Path $electronDir "node_modules\electron\dist\electron.exe"
if (-not (Test-Path $electronExe)) {
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if ($npm) {
        Push-Location $electronDir
        npm install --no-audit --no-fund
        if (-not (Test-Path $electronExe)) {
            node (Join-Path $electronDir "node_modules\electron\install.js")
        }
        Pop-Location
    } else {
        Write-Warning "Node.js not found - skipping the tray app. The dashboard still works in any browser at http://127.0.0.1:8790/ui"
    }
}

if (Test-Path $electronExe) {
    $ico = Join-Path $electronDir "icon.ico"
    $ws = New-Object -ComObject WScript.Shell
    foreach ($dest in @(
        "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Silicon Node.lnk",
        "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Silicon Node.lnk")) {
        $s = $ws.CreateShortcut($dest)
        $s.TargetPath = $electronExe
        $s.Arguments = ('"{0}"' -f $electronDir)
        $s.WorkingDirectory = $electronDir
        if (Test-Path $ico) { $s.IconLocation = $ico }
        $s.Description = "Silicon Node - GPU job service dashboard"
        $s.Save()
        Write-Host "created: $dest"
    }
    Write-Host "Silicon Node installed. Find it in the Start menu; it starts with Windows." -ForegroundColor Green
}
Write-Host "Next: docs/PROVISIONING.md sets up the WSL distro and models."
