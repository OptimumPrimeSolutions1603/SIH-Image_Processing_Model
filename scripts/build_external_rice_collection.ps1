$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$externalRoot = Join-Path $projectRoot 'data\external_rice'
$rawRoot = Join-Path $externalRoot 'raw_sources'
$manifestPath = Join-Path $externalRoot 'manifest.csv'
$split = 'external_val'
$perClass = 60

$sources = @(
    @{
        Label = 'bacterial_leaf_blight'
        SourceGroup = 'dhan_shomadhan_v1'
        SourceTitle = 'Dhan-Shomadhan'
        SourceUrl = 'https://data.mendeley.com/datasets/znsxdctwtt/1'
        License = 'CC BY 4.0'
        Folder = Join-Path $rawRoot 'dhan_shomadhan\extracted\Rice___bacterial_blight'
    },
    @{
        Label = 'leaf_blast'
        SourceGroup = 'dhan_shomadhan_v1'
        SourceTitle = 'Dhan-Shomadhan'
        SourceUrl = 'https://data.mendeley.com/datasets/znsxdctwtt/1'
        License = 'CC BY 4.0'
        Folder = Join-Path $rawRoot 'dhan_shomadhan\extracted\Rice___blast'
    },
    @{
        Label = 'brown_spot'
        SourceGroup = 'dhan_shomadhan_v1'
        SourceTitle = 'Dhan-Shomadhan'
        SourceUrl = 'https://data.mendeley.com/datasets/znsxdctwtt/1'
        License = 'CC BY 4.0'
        Folder = Join-Path $rawRoot 'dhan_shomadhan\extracted\Rice___brown_spot'
    },
    @{
        Label = 'bacterial_leaf_streak'
        SourceGroup = 'banglariceleaf_v1'
        SourceTitle = 'BanglaRiceLeaf'
        SourceUrl = 'https://doi.org/10.7910/DVN/XAOBYW'
        License = 'CC0 1.0'
        Folder = Join-Path $rawRoot 'bangla_rice_leaf\extracted_bls\Bacterial Leaf Streak (BLS)'
    },
    @{
        Label = 'healthy'
        SourceGroup = 'banglariceleaf_v1'
        SourceTitle = 'BanglaRiceLeaf'
        SourceUrl = 'https://doi.org/10.7910/DVN/XAOBYW'
        License = 'CC0 1.0'
        Folder = Join-Path $rawRoot 'bangla_rice_leaf\extracted_healthy\Healthy Leaf'
    }
)

$managedGroups = $sources.SourceGroup | Select-Object -Unique
$rows = @()
if (Test-Path -LiteralPath $manifestPath) {
    $rows = @(Import-Csv -LiteralPath $manifestPath | Where-Object { $_.source_group -notin $managedGroups })
}

foreach ($source in $sources) {
    if (-not (Test-Path -LiteralPath $source.Folder)) {
        throw "Missing source folder: $($source.Folder)"
    }

    $destination = Join-Path $externalRoot "$split\$($source.Label)"
    New-Item -ItemType Directory -Force -Path $destination | Out-Null

    $files = @(Get-ChildItem -LiteralPath $source.Folder -File |
        Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|webp)$' } |
        Sort-Object Name |
        Select-Object -First $perClass)
    if ($files.Count -lt $perClass) {
        throw "$($source.SourceTitle)/$($source.Label) only has $($files.Count) usable images."
    }

    $index = 0
    foreach ($file in $files) {
        $index++
        $extension = $file.Extension.ToLowerInvariant()
        $imageId = '{0}_{1}_{2:d3}' -f $source.SourceGroup, $source.Label, $index
        $filename = "$imageId$extension"
        $target = Join-Path $destination $filename
        Copy-Item -LiteralPath $file.FullName -Destination $target -Force
        $relativePath = "$split/$($source.Label)/$filename"
        $rows += [PSCustomObject]@{
            image_id = $imageId
            relative_path = $relativePath
            label = $source.Label
            split = $split
            source_group = $source.SourceGroup
            source_title = $source.SourceTitle
            source_url = $source.SourceUrl
            license = $source.License
            verification_status = 'dataset_expert_label'
            verified_by = $source.SourceTitle
            capture_device = ''
            rice_variety = ''
            location = 'Bangladesh'
            capture_date = ''
            notes = 'Deterministic filename-sorted sample; keep source group isolated from adaptation training.'
        }
    }
}

$rows | Sort-Object split,label,image_id | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding utf8
$rows | Group-Object split,label | Sort-Object Name | ForEach-Object {
    [PSCustomObject]@{Group=$_.Name; Images=$_.Count}
}
