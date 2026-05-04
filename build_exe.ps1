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
  --collect-all kagglehub `
  --collect-all kagglesdk `
  --collect-data transformers `
  --collect-submodules transformers.models.gemma4 `
  --collect-submodules transformers.pipelines `
  --collect-submodules tokenizers `
  --collect-binaries safetensors `
  --collect-binaries torch `
  --hidden-import torch `
  --hidden-import torchvision `
  --hidden-import accelerate `
  --hidden-import huggingface_hub `
  --hidden-import transformers `
  --hidden-import transformers.models.auto `
  --hidden-import transformers.image_utils `
  --hidden-import transformers.processing_utils `
  --hidden-import transformers.models.gemma4 `
  app.pyw

Write-Host "Built: $PSScriptRoot\SwitchVisionMonitor.exe"
Write-Host "The exe uses KaggleHub competition files and the official Google Gemma runtime."
