# Registers the Silicon Node path watchdog as a SYSTEM scheduled task that
# runs every 5 minutes. RUN AS ADMINISTRATOR, once. The watchdog itself
# (watch-node-path.ps1) is quiet when healthy and logs repairs to
# node-path-watchdog.log next to the repo.

$ErrorActionPreference = "Stop"

$script = Join-Path $PSScriptRoot "watch-node-path.ps1"
if (-not (Test-Path $script)) { throw "watch-node-path.ps1 not found next to this script." }

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" `
    -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 4) -StartWhenAvailable

Register-ScheduledTask -TaskName "SiliconNode Path Watchdog" -Action $action `
    -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null

# Retire the unelevated fallback task if one was registered before this.
Unregister-ScheduledTask -TaskName "SiliconNode Path Watchdog (user)" `
    -Confirm:$false -ErrorAction SilentlyContinue

Write-Host "Registered 'SiliconNode Path Watchdog' (SYSTEM, every 5 min)."
Write-Host "Log: $(Join-Path (Split-Path $PSScriptRoot -Parent) 'node-path-watchdog.log')"
