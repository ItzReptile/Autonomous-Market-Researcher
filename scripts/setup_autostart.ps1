<#
.SYNOPSIS
    Stage 4 Auto-Start Configuration for the Live Paper Trading Daemon.

.DESCRIPTION
    Registers a Windows Scheduled Task so the daemon survives reboots and
    restarts itself after a crash, for an unattended 60-90 day run.

    Environment dependencies, stated explicitly:

      * TRIGGER is AtLogOn by default. The task fires when THIS USER logs on,
        not at machine boot. A reboot that stops at the lock screen without a
        login will NOT start the daemon. Use -Trigger AtStartup (requires an
        elevated shell) for genuine boot-time start independent of login.

      * NO NETWORK CONDITION is attached. This is deliberate. The daemon is
        built to start offline, log FEED_UNREACHABLE, and retry on its poll
        interval, which is more robust than letting the scheduler withhold the
        task until Windows decides a network is "available" -- a judgement that
        is unreliable on a new WiFi network or after a move.

      * ExecutionTimeLimit is set to 0 (unlimited). The Windows DEFAULT is 3
        days, which would silently terminate a 60-90 day run mid-flight.

      * The Python path is resolved at registration time and baked into the
        task. Re-run this script after any Python upgrade or reinstall.

.PARAMETER Action
    register | unregister | status
.PARAMETER Trigger
    AtLogOn (default) | AtStartup (needs elevation)
#>

param(
    [ValidateSet("register", "unregister", "status")]
    [string]$Action = "register",

    [ValidateSet("AtLogOn", "AtStartup")]
    [string]$Trigger = "AtLogOn"
)

$TaskName         = "Stage4LivePaperTradingDaemon"
# Non-root task folder: writing to the ROOT folder requires elevation on many
# systems, while a user subfolder does not.
$TaskPath         = "\Stage4\"
$WorkingDirectory = (Resolve-Path "$PSScriptRoot\..").Path
$ScriptPath       = "$WorkingDirectory\src\lab\stage4\live_runner.py"
$LogPath          = "$WorkingDirectory\data\paper_trading\daemon.log"

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) { $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue }
if (-not $pythonCmd) {
    Write-Host "ERROR: python not found on PATH. Cannot register auto-start." -ForegroundColor Red
    exit 1
}
$PythonPath = $pythonCmd.Source

Write-Host ("=" * 78)
Write-Host "STAGE 4 LIVE PAPER TRADING DAEMON: AUTO-START CONFIGURATION"
Write-Host "Task      : $TaskName"
Write-Host "Python    : $PythonPath"
Write-Host "Script    : $ScriptPath"
Write-Host "Trigger   : $Trigger"
Write-Host ("=" * 78)

function Test-Elevated {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

if ($Action -eq "register") {
    if (-not (Test-Path $ScriptPath)) {
        Write-Host "ERROR: daemon script not found at $ScriptPath" -ForegroundColor Red
        exit 1
    }
    if ($Trigger -eq "AtStartup" -and -not (Test-Elevated)) {
        Write-Host "ERROR: -Trigger AtStartup requires an elevated PowerShell session." -ForegroundColor Red
        Write-Host "       Re-run as Administrator, or use -Trigger AtLogOn." -ForegroundColor Yellow
        exit 1
    }

    $null = New-Item -ItemType Directory -Force -Path (Split-Path $LogPath)

    $ActionObj = New-ScheduledTaskAction `
        -Execute $PythonPath `
        -Argument "-u `"$ScriptPath`"" `
        -WorkingDirectory $WorkingDirectory

    if ($Trigger -eq "AtStartup") {
        $TriggerObj = New-ScheduledTaskTrigger -AtStartup
    } else {
        $TriggerObj = New-ScheduledTaskTrigger -AtLogOn
    }

    # ExecutionTimeLimit 0 = run indefinitely. Without this Windows kills the
    # task after 3 days, which would end a 90-day run without warning.
    $SettingsObj = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -RestartCount 999 `
        -RestartInterval (New-TimeSpan -Minutes 2) `
        -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
        -MultipleInstances IgnoreNew `
        -StartWhenAvailable

    $taskRegistered = $false
    try {
        Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $ActionObj `
            -Trigger $TriggerObj `
            -Settings $SettingsObj `
            -Description "Stage 4 forward paper trading daemon (Phase 30 survivor, UNIV_10)." `
            -TaskPath $TaskPath `
            -Force -ErrorAction Stop | Out-Null
        $verify = Get-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue
        if ($verify) {
            $taskRegistered = $true
            Write-Host "  VERIFIED: Scheduled Task present, state = $($verify.State)" -ForegroundColor Green
        }
    } catch {
        Write-Host "  Scheduled Task registration denied: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    # ------------------------------------------------------------------
    # Fallback: user-level Startup entry with a self-restart wrapper.
    # This machine denies non-elevated task registration, so without this the
    # daemon would have no auto-start at all. The wrapper loop restores the
    # crash-restart behaviour that the Scheduled Task would have provided;
    # what it CANNOT do is start without a user logon.
    # ------------------------------------------------------------------
    $StartupFolder = [Environment]::GetFolderPath("Startup")
    $BatPath = Join-Path $StartupFolder "$TaskName.bat"
    $BatLines = @(
        '@echo off',
        'rem Stage 4 live paper trading daemon - auto-start with crash restart.',
        "cd /d `"$WorkingDirectory`"",
        ':run',
        "echo [%date% %time%] starting daemon >> `"$LogPath`"",
        "`"$PythonPath`" -u `"$ScriptPath`" >> `"$LogPath`" 2>&1",
        "echo [%date% %time%] daemon exited with %errorlevel%; restarting in 60s >> `"$LogPath`"",
        'timeout /t 60 /nobreak > nul',
        'goto run'
    )
    [System.IO.File]::WriteAllText($BatPath, ($BatLines -join "`r`n") + "`r`n")
    $batOk = Test-Path $BatPath
    if ($batOk) {
        Write-Host "  VERIFIED: Startup entry written -> $BatPath" -ForegroundColor Green
    } else {
        Write-Host "  ERROR: could not write Startup entry." -ForegroundColor Red
    }

    Write-Host ""
    if ($taskRegistered) {
        Write-Host "AUTO-START: FULL (Scheduled Task + Startup fallback)" -ForegroundColor Green
        exit 0
    } elseif ($batOk) {
        Write-Host "AUTO-START: DEGRADED (Startup entry only)" -ForegroundColor Yellow
        Write-Host "  Works: restarts on crash, starts on user logon." -ForegroundColor Yellow
        Write-Host "  Does NOT: start after a reboot that stops at the lock screen." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  To upgrade to full boot-time start, run ONCE in an ELEVATED PowerShell:" -ForegroundColor Cyan
        Write-Host "    powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Action register -Trigger AtStartup" -ForegroundColor Cyan
        exit 2
    } else {
        Write-Host "AUTO-START: FAILED - no mechanism installed." -ForegroundColor Red
        exit 1
    }

} elseif ($Action -eq "unregister") {
    # This block used to wrap the removal in a try/catch whose catch printed
    # "No scheduled task to remove." and fell through to exit 0. That conflates
    # two completely different outcomes: the task was absent, or the removal
    # FAILED (typically because it needs an elevated prompt). On the one real
    # shutdown this daemon ever had, it reported success while leaving the task
    # armed. Distinguish the two, and verify rather than assume.
    $existing = Get-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue
    if (-not $existing) {
        Write-Host "  No scheduled task to remove."
    } else {
        try {
            Unregister-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -Confirm:$false -ErrorAction Stop
        } catch {
            Write-Host "  FAILED to remove the scheduled task: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "  THE TASK IS STILL ARMED. Re-run this from an elevated prompt." -ForegroundColor Red
            exit 1
        }
        Write-Host "  Scheduled task removed."
    }
    # Verify absence instead of trusting the call that claimed to remove it.
    $still = Get-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue
    if ($still) {
        Write-Host "  FAILED: task still registered after removal. STILL ARMED." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Verified: no scheduled task is registered." -ForegroundColor Green
    # Clean up the legacy Startup-folder .bat from the previous implementation.
    $Startup = [Environment]::GetFolderPath("Startup")
    $Legacy = Join-Path $Startup "$TaskName.bat"
    if (Test-Path $Legacy) { Remove-Item $Legacy -Force; Write-Host "  Legacy startup .bat removed." }
    exit 0

} elseif ($Action -eq "status") {
    $StartupFolder = [Environment]::GetFolderPath("Startup")
    $BatPath = Join-Path $StartupFolder "$TaskName.bat"
    Write-Host "STARTUP_ENTRY: $(Test-Path $BatPath)  ($BatPath)"

    $task = Get-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Host "REGISTERED: False" -ForegroundColor Red
        exit 1
    }
    $info = Get-ScheduledTaskInfo -TaskName $TaskName -TaskPath $TaskPath
    $limit = $task.Settings.ExecutionTimeLimit
    Write-Host "REGISTERED: True" -ForegroundColor Green
    Write-Host "  State              : $($task.State)"
    Write-Host "  Triggers           : $($task.Triggers.CimClass.CimClassName -join ', ')"
    Write-Host "  ExecutionTimeLimit : $limit  (PT0S / empty = unlimited)"
    Write-Host "  RestartCount       : $($task.Settings.RestartCount)"
    Write-Host "  LastRunTime        : $($info.LastRunTime)"
    Write-Host "  LastTaskResult     : $($info.LastTaskResult)"
    Write-Host "  NextRunTime        : $($info.NextRunTime)"
    exit 0
}
