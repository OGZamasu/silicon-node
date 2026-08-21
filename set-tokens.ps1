# One-time token setup for the Silicon Node + Memories hub.
# Prompts are hidden; tokens never appear in command history or this file.
# Press Enter on an empty prompt to skip that token.

function Read-Plain([string]$label) {
    $sec = Read-Host $label -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
    try { [Runtime.InteropServices.Marshal]::PtrToStringAuto($ptr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
}

# --- HuggingFace token -> WSL secrets.env (unblocks image-to-mesh) ---------
$hf = Read-Plain "HuggingFace token (hf_..., Enter to skip)"
if ($hf) {
    if (-not $hf.StartsWith("hf_")) { Write-Warning "That doesn't look like an hf_ token; writing it anyway." }
    $path = "\\wsl$\SiliconNode\opt\silicon\secrets.env"
    [IO.File]::WriteAllText($path, "HF_TOKEN=$hf`n")
    Write-Host "OK: wrote $path" -ForegroundColor Green
} else {
    Write-Host "Skipped HF token."
}

# --- Memories hub token -> user environment variable -----------------------
$mem = Read-Plain "Memories hub token (Enter to skip)"
if ($mem) {
    [Environment]::SetEnvironmentVariable("MEMORIES_TOKEN", $mem, "User")
    Write-Host "OK: MEMORIES_TOKEN set (user scope). Restart the Claude app to pick it up." -ForegroundColor Green
} else {
    Write-Host "Skipped Memories token."
}

Write-Host ""
Write-Host "Done. After restarting the app, tell Claude the tokens are in."
