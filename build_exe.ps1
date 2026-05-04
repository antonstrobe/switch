$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

python -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --windowed `
  --name SwitchVisionMonitor `
  --distpath . `
  --workpath build\pyinstaller `
  --specpath build `
  --collect-all mediapipe `
  app.pyw

Write-Host "Built: $PSScriptRoot\SwitchVisionMonitor.exe"
Write-Host "Keep the exe next to models\ and bin\ so it can use the local Gemma files."
