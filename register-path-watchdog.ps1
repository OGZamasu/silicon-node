# Registers the Silicon Node path watchdog: a scheduled task that runs
# watch-node-path.ps1 every 5 minutes. RUN AS ADMINISTRATOR, once - and
# again after changing watch-node-path.ps1, because the task never runs the
# checkout's copy (see below).
#
# Design (hub 153):
#  - It runs as the account that registers it (the owner), elevated, and
#    only while that account is logged on - the same condition the node
#    itself has, since the WSL keepalive is a logon task. Not as SYSTEM:
#    WSL distros are per-user, so a SYSTEM task cannot see SiliconNode at
#    all, and could never probe or restart the service inside it.
#  - Elevated because two of its three repairs need it: refreshing the
#    netsh portproxy and restarting the IP Helper / Tailscale services.
#  - It runs a COPY in %ProgramData%\SiliconNode, which only SYSTEM and
#    Administrators may change. The checkout is writable by ordinary users
#    and by the node's own process through /mnt/<drive>; an elevated task running
#    a script from there would run whatever last edited it. The log lives
#    in the same protected folder.
# Pure ASCII on purpose: Windows PowerShell 5.1 reads BOM-less UTF-8 as ANSI.

$ErrorActionPreference = "Stop"

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = (New-Object Security.Principal.WindowsPrincipal $id).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    throw "Run this from an elevated PowerShell (Run as administrator)."
}

$source = Join-Path $PSScriptRoot "watch-node-path.ps1"
if (-not (Test-Path $source)) { throw "watch-node-path.ps1 not found next to this script." }

# --- the protected copy ---------------------------------------------------
$dest = Join-Path $env:ProgramData "SiliconNode"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item -Path $source -Destination (Join-Path $dest "watch-node-path.ps1") -Force

# SIDs, not names, so this works on any Windows display language:
#   S-1-5-18 SYSTEM, S-1-5-32-544 Administrators, S-1-5-32-545 Users.
# Inheritance off; Administrators own the folder so no user can rewrite
# its ACL; users may read and run, never write.
& icacls.exe $dest /setowner "*S-1-5-32-544" /T /C /Q | Out-Null
& icacls.exe $dest /inheritance:r /grant:r "*S-1-5-18:(OI)(CI)F" "*S-1-5-32-544:(OI)(CI)F" "*S-1-5-32-545:(OI)(CI)RX" /T /C /Q | Out-Null
if ($LASTEXITCODE -ne 0) { throw "icacls could not lock down $dest (exit $LASTEXITCODE)." }

# Verify rather than trust: nobody but SYSTEM and Administrators may write.
# Only the bits that change something - WriteData, AppendData, WriteEA,
# DeleteChild, WriteAttributes, Delete, WRITE_DAC, WRITE_OWNER, plus
# GENERIC_WRITE/GENERIC_ALL - since read-and-execute shares bits (e.g.
# Synchronize) with FullControl.
$writeMask = 0x500D0156
$allowed = @("S-1-5-18", "S-1-5-32-544")
foreach ($item in @((Get-Item $dest)) + @(Get-ChildItem $dest -Recurse)) {
    $acl = Get-Acl $item.FullName
    foreach ($rule in $acl.Access) {
        $sid = $rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
        $writes = ([int64]$rule.FileSystemRights -band $writeMask) -ne 0
        if ($rule.AccessControlType -eq "Allow" -and $writes -and ($allowed -notcontains $sid)) {
            throw "Unexpected write access on $($item.FullName) for $($rule.IdentityReference)."
        }
    }
    $owner = (New-Object Security.Principal.NTAccount $acl.Owner).Translate([Security.Principal.SecurityIdentifier]).Value
    if ($allowed -notcontains $owner) {
        throw "$($item.FullName) is owned by $($acl.Owner), who could rewrite its ACL."
    }
}

# --- the task --------------------------------------------------------------
$script = Join-Path $dest "watch-node-path.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId $id.Name `
    -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 4) -StartWhenAvailable

# Same name as the old SYSTEM task, so -Force replaces it in place.
Register-ScheduledTask -TaskName "SiliconNode Path Watchdog" -Action $action `
    -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null

# Retire the unelevated fallback task if one was registered before this:
# it ran the checkout's copy and could not perform any repair.
Unregister-ScheduledTask -TaskName "SiliconNode Path Watchdog (user)" `
    -Confirm:$false -ErrorAction SilentlyContinue

$t = Get-ScheduledTask -TaskName "SiliconNode Path Watchdog"
Write-Host "Registered 'SiliconNode Path Watchdog': $($t.Principal.UserId), $($t.Principal.LogonType), $($t.Principal.RunLevel), every 5 min."
Write-Host "Runs: $script"
Write-Host "Log:  $(Join-Path $dest 'node-path-watchdog.log')"
