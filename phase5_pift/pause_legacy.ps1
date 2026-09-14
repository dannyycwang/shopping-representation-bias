$repoRoot = Split-Path $PSScriptRoot
$worker = Get-Process -Id 22808 -ErrorAction SilentlyContinue
if (-not $worker) { exit }
$workerStart = $worker.StartTime
$pattern = Join-Path $repoRoot 'phase5_mitigation/embeddings/Perm-FT-cons0.1_s42/C2s4_*.json'
while (-not (Get-ChildItem $pattern -ErrorAction SilentlyContinue)) {
    if (-not (Get-Process -Id 22808 -ErrorAction SilentlyContinue)) { exit }
    Start-Sleep -Seconds 2
}
# The array and metadata are now durable; defer the remaining legacy queue.
Stop-Process -Id 22808
$seconds = ((Get-Date) - $workerStart).TotalSeconds
@{name='legacy_consistency_queue_priority_pause';seconds=$seconds;accounting='Worker wall time since explicit continuation start; no completed stages were recorded during this evaluation. Paused after durable C2s4 cache at user request to prioritize PI-FT.'} | ConvertTo-Json -Compress | Add-Content (Join-Path $repoRoot 'phase5_mitigation/cost.jsonl')
@{paused_utc=[DateTime]::UtcNow.ToString('o');reason='User requested source-aligned PI-FT before more consistency';legacy_cache='C2s4 saved; all previous caches retained';accounted_seconds=$seconds} | ConvertTo-Json | Set-Content (Join-Path $PSScriptRoot 'legacy_pause.json') -Encoding utf8
