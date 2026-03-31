# run_forever.ps1
# Usage: ./run_forever.ps1 "python src/run_live_forward.py transformer"
param(
    [string]$Command
)

Write-Host "Starting Auto-Restart Loop for: $Command" -ForegroundColor Green

while ($true) {
    try {
        # Run the command and wait for it to finish (or crash)
        Invoke-Expression $Command
    }
    catch {
        Write-Host "Error occurred: $_" -ForegroundColor Red
    }

    Write-Host "Process exited. Restarting in 10 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
}
