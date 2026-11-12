param(
    [string]$ApiKey = "sk-local-change-me",
    [int]$Port = 18789
)

$ErrorActionPreference = "Stop"
$env:DOUBAO_WORK_API_KEY = $ApiKey
$env:DOUBAO_WORK_PORT = "$Port"
python -m doubao_work2api
