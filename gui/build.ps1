# Rebuild the Silicon Node tray shell after editing tray.py.
# (The dashboard itself is web — server/ui/index.html — and deploys with
# the service; no build step.)
Set-Location $PSScriptRoot
& ".venv\Scripts\pyinstaller.exe" --noconfirm --windowed --onefile `
    --name "SiliconNode" --icon "silicon-node.ico" tray.py
Write-Host "Built: $PSScriptRoot\dist\SiliconNode.exe"
