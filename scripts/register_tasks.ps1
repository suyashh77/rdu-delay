# Registers the two start-now loggers as Windows scheduled tasks (run once, as yourself).
#   rdu-faa-status       every 15 min, free
#   rdu-aeroapi-nightly  22:40 daily, AeroAPI capped at 4.50 USD/month in code
# Tasks run only while you are logged in and the PC is awake. Missed runs start when it wakes.
# Remove later with:  Unregister-ScheduledTask -TaskName "rdu-*" -Confirm:$false
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $py)) { throw "pythonw not found at $py" }
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -MultipleInstances IgnoreNew

$faaAction = New-ScheduledTaskAction -Execute $py -Argument "-m rdu.ingest.live faa" -WorkingDirectory $root
$faaTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName "rdu-faa-status" -Action $faaAction -Trigger $faaTrigger -Settings $settings `
    -Description "RDU delay project: FAA NAS status XML every 15 min (free)" -Force | Out-Null

$nAction = New-ScheduledTaskAction -Execute $py -Argument "-m rdu.ingest.live pipeline" -WorkingDirectory $root
$nTrigger = New-ScheduledTaskTrigger -Daily -At "22:40"
Register-ScheduledTask -TaskName "rdu-aeroapi-nightly" -Action $nAction -Trigger $nTrigger -Settings $settings `
    -Description "RDU delay project: Collect RDU schedule, publish frozen forecasts, and refresh actual departures; capped at 4.50 USD/month" -Force | Out-Null

Get-ScheduledTask -TaskName "rdu-*" | Select-Object TaskName, State,
    @{n = 'NextRun'; e = { (Get-ScheduledTaskInfo $_).NextRunTime } } | Format-Table -AutoSize
