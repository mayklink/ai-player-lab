$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Instale Git antes de preparar a base.' }
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 não encontrado. Instale-o ou crie .venv com Python 3.11+.' }
}
& .\.venv\Scripts\python.exe -m pip install -e '.[desktop,dev]'
if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar dependências.' }
& .\.venv\Scripts\python.exe scripts\bootstrap_upstream.py
if ($LASTEXITCODE -ne 0) { throw 'Falha ao preparar Pixel-to-Action. Nada foi publicado.' }
& .\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Testes falharam. Não habilite controle real.' }
& .\.venv\Scripts\python.exe -m ai_player_lab doctor
if ($LASTEXITCODE -ne 0) { throw 'Falha no diagnóstico de hardware.' }
& .\.venv\Scripts\python.exe -m ai_player_lab benchmark --frames 180
if ($LASTEXITCODE -ne 0) { throw 'Falha no benchmark sintético.' }
Write-Host 'Preparação concluída. Próximo passo: .\.venv\Scripts\python.exe -m ai_player_lab calibrate'
