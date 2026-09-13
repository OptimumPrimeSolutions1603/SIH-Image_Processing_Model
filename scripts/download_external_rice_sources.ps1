$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$rawRoot = Join-Path $projectRoot 'data\external_rice\raw_sources'

$uciRoot = Join-Path $rawRoot 'uci_rice_leaf_diseases'
$dhanRoot = Join-Path $rawRoot 'dhan_shomadhan'
New-Item -ItemType Directory -Force -Path $uciRoot, $dhanRoot | Out-Null

$uciZip = Join-Path $uciRoot 'source.zip'
curl.exe -L 'https://archive.ics.uci.edu/static/public/486/rice%2Bleaf%2Bdiseases.zip' -o $uciZip
if ($LASTEXITCODE -ne 0) { throw 'UCI download failed.' }

$metadataPath = Join-Path $dhanRoot 'metadata.json'
curl.exe -L 'https://api.data.mendeley.com/datasets/znsxdctwtt/versions/1' -o $metadataPath
if ($LASTEXITCODE -ne 0) { throw 'Mendeley metadata download failed.' }

Write-Host "Downloaded UCI archive: $uciZip"
Write-Host "Downloaded Dhan-Shomadhan metadata: $metadataPath"
