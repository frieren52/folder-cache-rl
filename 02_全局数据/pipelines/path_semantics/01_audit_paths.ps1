param(
    [Parameter(Mandatory = $false)]
    [string]$InputFile = "..\recovered_attachments_merged\合并完整数据.txt",

    [Parameter(Mandatory = $false)]
    [string]$OutputDir = ".\02_全局数据\processed\work\path_semantics\01_audit"
)

$ErrorActionPreference = "Stop"

function Get-TokenClass {
    param([string]$Token)

    if ($Token -match '^20\d{6}$') {
        return 'date_yyyymmdd'
    }
    if ($Token -match '^(19|20)\d{2}$') {
        return 'year'
    }
    if ($Token -match '^(\d{3,4}M|\d{2,3}KM|GEO1K|GEOQK)$') {
        return 'resolution'
    }
    if ($Token -match '^(ASCEND|DESCEND|ASCENDC|DESCENDC)$') {
        return 'orbit_direction'
    }
    if ($Token -match '^(L0|L1|L2|L3|L4|L2L3|1A|1B|L1C|L1D|IMG)$') {
        return 'data_level'
    }
    if ($Token -match '^(FY[34][A-H](CHN|IMG|SIMU)?|GF5A|JPSS1|METOPC|AQUA|TERRA|H09)$') {
        return 'satellite_or_platform'
    }
    if ($Token -match '^(GLL|NIG|NOM|HAM|MLT|MCT|LCC|PSP|SIN)$') {
        return 'projection_candidate'
    }
    if ($Token -match '^\d{7,}$') {
        return 'numeric_id'
    }
    if ($Token -match '^\d+$') {
        return 'other_numeric'
    }
    return 'categorical_or_unknown'
}

function Get-ShapeToken {
    param([string]$Token)

    switch (Get-TokenClass $Token) {
        'date_yyyymmdd' { return '<DATE>' }
        'year' { return '<YEAR>' }
        'resolution' { return '<RESOLUTION>' }
        'orbit_direction' { return '<DIRECTION>' }
        'numeric_id' { return '<NUMERIC_ID>' }
        default { return $Token }
    }
}

function Get-BranchPrefix {
    param(
        [string[]]$Segments,
        [int]$Length = 3
    )

    $take = [Math]::Min($Length, $Segments.Count)
    return ($Segments[0..($take - 1)] -join '/')
}

$resolvedInput = (Resolve-Path -LiteralPath $InputFile).Path
$resolvedOutput = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDir))
New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null

$rows = [System.Collections.Generic.List[object]]::new()
$malformed = [System.Collections.Generic.List[object]]::new()
$lineNumber = 0

foreach ($line in [System.IO.File]::ReadLines($resolvedInput)) {
    $lineNumber++
    if ($line.StartsWith('#') -or [string]::IsNullOrWhiteSpace($line)) {
        continue
    }

    if ($line -notmatch '^(?<access>\d+)\s+(?<path>\S+)\s+(?<size>\d+)\s*$') {
        $malformed.Add([PSCustomObject]@{
            line_number = $lineNumber
            raw_line = $line
        })
        continue
    }

    $path = $Matches.path
    $segments = $path.Trim('/').Split('/', [System.StringSplitOptions]::RemoveEmptyEntries)
    if ($segments.Count -eq 0) {
        $malformed.Add([PSCustomObject]@{
            line_number = $lineNumber
            raw_line = $line
        })
        continue
    }

    $classes = @($segments | ForEach-Object { Get-TokenClass $_ })
    $shapeSegments = @($segments | ForEach-Object { Get-ShapeToken $_ })
    $lastSegment = $segments[-1]
    $containsFilename = [bool]($lastSegment -match '\.(HDF|NC|H5|JPG|JPEG|PNG|TIF|TIFF|BIN|TXT|CSV|ZIP|TAR|L1B|L1C|DAT)(\.|$)')

    $rows.Add([PSCustomObject]@{
        line_number = $lineNumber
        access_count = [int64]$Matches.access
        path = $path
        total_size_bytes = [int64]$Matches.size
        root = $segments[0]
        branch_2 = Get-BranchPrefix $segments 2
        branch_3 = Get-BranchPrefix $segments 3
        branch_4 = Get-BranchPrefix $segments 4
        depth = $segments.Count
        last_segment = $lastSegment
        shape = $shapeSegments -join '/'
        contains_filename = $containsFilename
        segments = $segments
        token_classes = $classes
    })
}

$totalRows = $rows.Count
$totalAccesses = [int64](($rows | Measure-Object access_count -Sum).Sum)
$totalSize = [int64](($rows | Measure-Object total_size_bytes -Sum).Sum)

$rootSummary = @(
    $rows |
        Group-Object root |
        ForEach-Object {
            $rowCount = $_.Count
            [PSCustomObject]@{
                root = $_.Name
                rows = $rowCount
                row_percentage = [Math]::Round(100.0 * $rowCount / $totalRows, 4)
                accesses = [int64](($_.Group | Measure-Object access_count -Sum).Sum)
                total_size_bytes = [int64](($_.Group | Measure-Object total_size_bytes -Sum).Sum)
                min_depth = ($_.Group | Measure-Object depth -Minimum).Minimum
                max_depth = ($_.Group | Measure-Object depth -Maximum).Maximum
            }
        } |
        Sort-Object rows -Descending
)

$branchSummary = @(
    $rows |
        Group-Object branch_3 |
        ForEach-Object {
            $rowCount = $_.Count
            [PSCustomObject]@{
                branch = $_.Name
                rows = $rowCount
                row_percentage = [Math]::Round(100.0 * $rowCount / $totalRows, 4)
                accesses = [int64](($_.Group | Measure-Object access_count -Sum).Sum)
                unique_shapes = @($_.Group | Group-Object shape).Count
                min_depth = ($_.Group | Measure-Object depth -Minimum).Minimum
                max_depth = ($_.Group | Measure-Object depth -Maximum).Maximum
            }
        } |
        Sort-Object rows -Descending
)

$shapeGroups = @($rows | Group-Object shape | Sort-Object Count -Descending)
$runningRows = 0
$shapeSummary = @(
    foreach ($group in $shapeGroups) {
        $runningRows += $group.Count
        [PSCustomObject]@{
            shape = $group.Name
            rows = $group.Count
            row_percentage = [Math]::Round(100.0 * $group.Count / $totalRows, 6)
            cumulative_rows = $runningRows
            cumulative_percentage = [Math]::Round(100.0 * $runningRows / $totalRows, 4)
            example_path = $group.Group[0].path
        }
    }
)

$classNames = @(
    'date_yyyymmdd',
    'year',
    'resolution',
    'orbit_direction',
    'data_level',
    'satellite_or_platform',
    'projection_candidate',
    'numeric_id',
    'other_numeric',
    'categorical_or_unknown'
)

$tokenCoverage = @(
    foreach ($className in $classNames) {
        $rowsWithClass = @($rows | Where-Object { $_.token_classes -contains $className }).Count
        $tokenCount = 0
        foreach ($row in $rows) {
            $tokenCount += @($row.token_classes | Where-Object { $_ -eq $className }).Count
        }
        [PSCustomObject]@{
            token_class = $className
            rows_with_class = $rowsWithClass
            row_coverage_percentage = [Math]::Round(100.0 * $rowsWithClass / $totalRows, 4)
            token_count = $tokenCount
        }
    }
)

$projectionTokens = @(
    foreach ($row in $rows) {
        for ($i = 0; $i -lt $row.segments.Count; $i++) {
            if ($row.token_classes[$i] -eq 'projection_candidate') {
                [PSCustomObject]@{
                    token = $row.segments[$i]
                    path = $row.path
                    depth = $i
                }
            }
        }
    }
)

$majorBranches = @(
    'FYDATAOUTSHARE/DATAIOT/FY3',
    'FYDATAOUTSHARE/DATA/FY3',
    'FYDATAARCH/DSSCACHE/FY3H',
    'FYDATAARCH/DSSCACHE/FY3F',
    'FYDATAARCH/DSSCACHE/FY3G',
    'FYDATAINSHARE/BAKIOT/FY3'
)

$depthTokenSummary = [System.Collections.Generic.List[object]]::new()
foreach ($branch in $majorBranches) {
    $branchRows = @($rows | Where-Object { $_.branch_3 -eq $branch })
    if ($branchRows.Count -eq 0) {
        continue
    }
    $maxDepth = ($branchRows | Measure-Object depth -Maximum).Maximum
    for ($depth = 0; $depth -lt $maxDepth; $depth++) {
        $tokensAtDepth = @(
            $branchRows |
                Where-Object { $_.segments.Count -gt $depth } |
                ForEach-Object { $_.segments[$depth] }
        )
        $denominator = $tokensAtDepth.Count
        $rank = 0
        foreach ($group in @($tokensAtDepth | Group-Object | Sort-Object Count -Descending | Select-Object -First 30)) {
            $rank++
            $depthTokenSummary.Add([PSCustomObject]@{
                branch = $branch
                depth = $depth
                rank = $rank
                token = $group.Name
                token_class = Get-TokenClass $group.Name
                count = $group.Count
                percentage_at_depth = [Math]::Round(100.0 * $group.Count / $denominator, 4)
            })
        }
    }
}

$outshareMainRows = @(
    $rows | Where-Object {
        $_.segments.Count -ge 6 -and
        $_.segments[0] -eq 'FYDATAOUTSHARE' -and
        ($_.segments[1] -eq 'DATA' -or $_.segments[1] -eq 'DATAIOT') -and
        $_.segments[2] -eq 'FY3'
    }
)

$outshareRootRows = @($rows | Where-Object root -eq 'FYDATAOUTSHARE')
$outshareMainCount = $outshareMainRows.Count
$outshareSatelliteAtL3 = @($outshareMainRows | Where-Object { $_.segments[3] -match '^FY3[A-H](CHN|IMG)?$' }).Count
$outshareLevelAtL5 = @($outshareMainRows | Where-Object { $_.segments[5] -match '^(L0|L1|L2|L3|L4|L2L3|1A|1B|L1C|IMG)$' }).Count
$outshareDateAtEnd = @($outshareMainRows | Where-Object { $_.segments[-1] -match '^20\d{6}$' }).Count

$l6TypeSummary = @(
    $outshareMainRows |
        ForEach-Object {
            $token = $_.segments[6]
            $type = switch (Get-TokenClass $token) {
                'resolution' { 'resolution' }
                'orbit_direction' { 'orbit_direction' }
                'year' { 'year' }
                default {
                    if ($token -match '^(ORBIT|GRAN|GBAL|REG|RNC|RNG|DAILY|10DAY)$') {
                        'region_or_period'
                    }
                    else {
                        'product_or_other'
                    }
                }
            }
            [PSCustomObject]@{ type = $type }
        } |
        Group-Object type |
        ForEach-Object {
            [PSCustomObject]@{
                l6_type = $_.Name
                rows = $_.Count
                percentage = [Math]::Round(100.0 * $_.Count / $outshareMainCount, 4)
            }
        } |
        Sort-Object rows -Descending
)

$archRows = @($rows | Where-Object root -eq 'FYDATAARCH')
$archDsscacheRows = @($archRows | Where-Object { $_.segments.Count -gt 1 -and $_.segments[1] -eq 'DSSCACHE' })
$archDirectLevelAtL4 = @(
    $archDsscacheRows |
        Where-Object {
            $_.segments.Count -gt 4 -and
            $_.segments[4] -match '^(L0|L1|L2|L3|L4|L2L3|1A|1B|L1C|IMG)$'
        }
).Count

$filenameRows = @($rows | Where-Object contains_filename).Count
$projectionValueSummary = @(
    $projectionTokens |
        Group-Object token |
        ForEach-Object {
            [PSCustomObject]@{
                token = $_.Name
                rows = $_.Count
            }
        } |
        Sort-Object rows -Descending
)

function Get-TopCoverage {
    param([int]$TopN)
    if ($shapeSummary.Count -eq 0) {
        return 0.0
    }
    $take = [Math]::Min($TopN, $shapeSummary.Count)
    return [Math]::Round(($shapeSummary[$take - 1].cumulative_percentage), 4)
}

$summary = [ordered]@{
    input_file = $resolvedInput
    parsed_rows = $totalRows
    malformed_rows = $malformed.Count
    total_accesses = $totalAccesses
    total_size_bytes = $totalSize
    filename_rows = $filenameRows
    unique_roots = $rootSummary.Count
    unique_branch3 = $branchSummary.Count
    unique_shapes = $shapeSummary.Count
    top_10_shape_coverage = Get-TopCoverage 10
    top_50_shape_coverage = Get-TopCoverage 50
    top_100_shape_coverage = Get-TopCoverage 100
    top_500_shape_coverage = Get-TopCoverage 500
    top_1000_shape_coverage = Get-TopCoverage 1000
    outshare_main_rows = $outshareMainCount
    outshare_main_share = [Math]::Round(100.0 * $outshareMainCount / $outshareRootRows.Count, 4)
    outshare_main_satellite_l3_match = [Math]::Round(100.0 * $outshareSatelliteAtL3 / $outshareMainCount, 4)
    outshare_main_level_l5_match = [Math]::Round(100.0 * $outshareLevelAtL5 / $outshareMainCount, 4)
    outshare_main_date_at_end = [Math]::Round(100.0 * $outshareDateAtEnd / $outshareMainCount, 4)
    arch_dsscache_rows = $archDsscacheRows.Count
    arch_dsscache_share = [Math]::Round(100.0 * $archDsscacheRows.Count / $archRows.Count, 4)
    arch_dsscache_direct_level_l4_match = [Math]::Round(100.0 * $archDirectLevelAtL4 / $archDsscacheRows.Count, 4)
}

$rootSummary | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $resolvedOutput 'root_summary.csv')
$branchSummary | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $resolvedOutput 'branch_summary.csv')
$shapeSummary | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $resolvedOutput 'path_shape_summary.csv')
$tokenCoverage | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $resolvedOutput 'token_pattern_coverage.csv')
$depthTokenSummary | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $resolvedOutput 'major_branch_depth_tokens.csv')
$l6TypeSummary | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $resolvedOutput 'outshare_main_l6_types.csv')
$projectionValueSummary | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $resolvedOutput 'projection_tokens.csv')
$malformed | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $resolvedOutput 'malformed_rows.csv')
$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 -LiteralPath (Join-Path $resolvedOutput 'audit_summary.json')

$coverageTable = ($tokenCoverage | ForEach-Object {
    "| $($_.token_class) | $($_.rows_with_class) | $($_.row_coverage_percentage)% | $($_.token_count) |"
}) -join "`n"

$rootTable = ($rootSummary | ForEach-Object {
    "| $($_.root) | $($_.rows) | $($_.row_percentage)% | $($_.accesses) | $($_.min_depth)～$($_.max_depth) |"
}) -join "`n"

$l6Table = ($l6TypeSummary | ForEach-Object {
    "| $($_.l6_type) | $($_.rows) | $($_.percentage)% |"
}) -join "`n"

$projectionTable = ($projectionValueSummary | ForEach-Object {
    "| $($_.token) | $($_.rows) |"
}) -join "`n"

$report = @"
# 步骤1：路径数据与现有规则审计报告

## 1. 审计范围

- 输入文件：$resolvedInput
- 有效路径记录：$totalRows
- 无法解析的行：$($malformed.Count)
- 聚合访问次数：$totalAccesses
- 检测到的具体文件名记录：$filenameRows
- 唯一路径骨架：$($shapeSummary.Count)

当前输入是目录级聚合数据，不包含具体文件名。因此本步骤只能验证路径规则，不能验证文档中的文件名模板覆盖率。

## 2. 根目录分布

| 根目录 | 记录数 | 记录占比 | 聚合访问次数 | 路径深度范围 |
| --- | ---: | ---: | ---: | --- |
$rootTable

## 3. 可通过强模式识别的Token

| Token类别 | 出现路径数 | 路径覆盖率 | Token数量 |
| --- | ---: | ---: | ---: |
$coverageTable

说明：覆盖率表示路径中出现符合模式的Token，不等价于业务语义已经得到权威确认。

## 4. 路径骨架复杂度

对日期、年份、分辨率、升降轨方向和长数字ID进行占位归一化后：

- 唯一骨架数：$($shapeSummary.Count)
- Top 10骨架覆盖：$(Get-TopCoverage 10)%
- Top 50骨架覆盖：$(Get-TopCoverage 50)%
- Top 100骨架覆盖：$(Get-TopCoverage 100)%
- Top 500骨架覆盖：$(Get-TopCoverage 500)%
- Top 1000骨架覆盖：$(Get-TopCoverage 1000)%

这说明少量主模板可以覆盖主流数据，但无法通过一张固定位置表覆盖所有路径。应采用“分支路由 + Token类型识别 + 状态机 + 未知Token兜底”。

## 5. FYDATAOUTSHARE主分支验证

FYDATAOUTSHARE/DATAIOT/FY3 和 FYDATAOUTSHARE/DATA/FY3 共 $outshareMainCount 条，占 FYDATAOUTSHARE 的 $([Math]::Round(100.0 * $outshareMainCount / $outshareRootRows.Count, 4))%。

- L3满足FY3卫星模式：$([Math]::Round(100.0 * $outshareSatelliteAtL3 / $outshareMainCount, 4))%
- L5满足数据等级模式：$([Math]::Round(100.0 * $outshareLevelAtL5 / $outshareMainCount, 4))%
- 路径末段满足8位日期模式：$([Math]::Round(100.0 * $outshareDateAtEnd / $outshareMainCount, 4))%

说明主干“根目录/子系统/数据域/卫星/仪器/等级”高度稳定。但是L6语义明显漂移：

| L6候选类型 | 记录数 | 占比 |
| --- | ---: | ---: |
$l6Table

因此L6之后不能使用单一固定槽位规则。

## 6. FYDATAARCH验证

- FYDATAARCH记录：$($archRows.Count)
- 其中DSSCACHE：$($archDsscacheRows.Count)，占 $([Math]::Round(100.0 * $archDsscacheRows.Count / $archRows.Count, 4))%
- DSSCACHE路径中L4直接满足等级模式：$archDirectLevelAtL4 条，占 $([Math]::Round(100.0 * $archDirectLevelAtL4 / $archDsscacheRows.Count, 4))%

这验证了DSSCACHE确实是FYDATAARCH的绝对主流，但也说明L4并非始终是等级。例如TEMPWORK分支会在后续层级重新出现仪器和等级。

## 7. 路径中的投影候选Token

| Token | 出现次数 |
| --- | ---: |
$projectionTable

原规则中“投影基本不在路径侧出现”的表述不准确。以上Token仍需结合上下文判断是否确实表示投影，但不能在路径解析阶段直接忽略。

## 8. 已确认的规则问题

1. 文档标题声称“10类别字段 + 4实例字段”，实际Schema表列出13个类别字段和4个实例字段。
2. “6位纯数字段=日期（20\\d{4}）”不准确；实际8位YYYYMMDD才是主要完整日期，6位通常只能表示年月或其他编码。
3. 绝对层级位置会随根目录、子系统和中间业务分支发生漂移，不能跨分支复用同一槽位定义。
4. “投影基本由文件名提供”与真实路径中出现的MLT/GLL/NIG/HAM候选Token不一致。
5. 文档中的文件名覆盖率和多后缀比例无法用当前数据验证，因为当前输入没有文件名。
6. 产品、区域、周期、处理系统等自由Token缺少权威代码表，现阶段只能做候选推断，不能声称语义已确认。

## 9. 步骤1结论

可以从路径中提取有价值的语义，但应区分三档：

- 高置信：归档根、显式卫星/平台、显式等级、8位日期、规范分辨率、显式升降轨。
- 条件置信：仪器、产品、区域、周期、投影，需要分支上下文和代码表。
- 未确认：未知业务Token、处理系统、临时目录和数字ID，必须原样保留并输出解析告警。

下一步应先实现通用Token类型识别器，为每个Token输出候选类型、位置、规则证据和置信度；暂时不强制把所有Token映射为唯一语义字段。

## 10. 产出文件

- audit_summary.json：总体统计。
- root_summary.csv：根目录分布。
- branch_summary.csv：三级分支分布及深度。
- path_shape_summary.csv：全部路径骨架和累计覆盖率。
- token_pattern_coverage.csv：强模式Token覆盖率。
- major_branch_depth_tokens.csv：主要分支各层级Top Token。
- outshare_main_l6_types.csv：OUTSHARE主分支L6语义漂移统计。
- projection_tokens.csv：路径中的投影候选Token。
- malformed_rows.csv：无法解析的原始行。
"@

$report | Set-Content -Encoding utf8 -LiteralPath (Join-Path $resolvedOutput '步骤1_路径规则审计报告.md')

Write-Output "STEP1_AUDIT_COMPLETE"
Write-Output "OUTPUT_DIR=$resolvedOutput"
Write-Output ($summary | ConvertTo-Json -Depth 4)
