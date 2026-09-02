param(
    [ValidateSet(
        "ablation-smoke",
        "ablation-full"
    )]
    [string]$Task = "ablation-smoke",
    [switch]$NoWait,
    [switch]$DryRun,
    [string]$ExistingJobId
)

$ErrorActionPreference = "Stop"
$remote = "koko2725@olympus.dsv.su.se"
$remoteDirectory = "~/meeting-qa-chunking"
# One table keeps task-specific paths out of the upload/monitoring logic below.
$jobs = @{
    "ablation-smoke" = @{
        Slurm = "src/wormulon/ablation_smoke.slurm"
        LogPrefix = "slurm-ablation-smoke"
        Result = "runs/ablations/smoke"
        ResultIsDirectory = $true
    }
    "ablation-full" = @{
        Slurm = "src/wormulon/ablation_full.slurm"
        LogPrefix = "slurm-ablation-full"
        Result = "runs/ablations/full"
        ResultIsDirectory = $true
        DataManifest = "src/configs/ablation-meetings.txt"
    }
}

function Receive-RemoteDirectory {
    param(
        [string]$RemotePath,
        [string]$LocalPath,
        [string]$TransferTag
    )

    # Download beside the destination, then swap only after transfer succeeds.
    $localParent = Split-Path -Parent $LocalPath
    New-Item -ItemType Directory -Force $localParent | Out-Null
    $directoryName = Split-Path -Leaf $LocalPath
    $stagingPath = Join-Path $localParent ".download-$directoryName-$TransferTag"
    $downloadedPath = Join-Path $stagingPath $directoryName
    $backupPath = "$LocalPath.previous"

    $parentPrefix = [IO.Path]::GetFullPath($localParent) + `
        [IO.Path]::DirectorySeparatorChar
    foreach ($path in @($LocalPath, $stagingPath, $backupPath)) {
        if (-not [IO.Path]::GetFullPath($path).StartsWith($parentPrefix)) {
            throw "Unsafe local transfer path: $path"
        }
    }

    if (Test-Path -LiteralPath $stagingPath) {
        Remove-Item -LiteralPath $stagingPath -Recurse -Force
    }
    New-Item -ItemType Directory -Force $stagingPath | Out-Null

    scp -O -o BatchMode=yes `
        -o ServerAliveInterval=15 `
        -o ServerAliveCountMax=4 `
        -r "${remote}:${RemotePath}" $stagingPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $downloadedPath)) {
        throw "Could not download $RemotePath"
    }

    if (Test-Path -LiteralPath $backupPath) {
        Remove-Item -LiteralPath $backupPath -Recurse -Force
    }
    if (Test-Path -LiteralPath $LocalPath) {
        Move-Item -LiteralPath $LocalPath -Destination $backupPath
    }
    try {
        Move-Item -LiteralPath $downloadedPath -Destination $LocalPath
    }
    catch {
        if (Test-Path -LiteralPath $backupPath) {
            Move-Item -LiteralPath $backupPath -Destination $LocalPath
        }
        throw
    }
    finally {
        if (Test-Path -LiteralPath $stagingPath) {
            Remove-Item -LiteralPath $stagingPath -Recurse -Force
        }
    }
    if (Test-Path -LiteralPath $backupPath) {
        Remove-Item -LiteralPath $backupPath -Recurse -Force
    }
}

Push-Location $PSScriptRoot
try {
    $job = $jobs[$Task]

    if ($DryRun) {
        Write-Host "Task: $Task"
        Write-Host "Upload: src/"
        Write-Host "Slurm: $($job.Slurm)"
        Write-Host "Result: $($job.Result)"
        if ($job.DataManifest) {
            Write-Host "Data: the 20 QMSum files in $($job.DataManifest)"
        }
        if ($job.MeetingIds) {
            Write-Host "Data: $($job.MeetingIds -join ', ')"
        }
        return
    }

    if ($ExistingJobId) {
        $jobId = $ExistingJobId
        Write-Host "Monitoring existing $Task job $jobId..."
    }
    else {
        $remotePathOutput = ssh -o BatchMode=yes $remote `
            "cd $remoteDirectory && pwd"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not resolve the remote repository directory"
        }
        $remotePath = ($remotePathOutput -join "`n").Trim()
        if (-not $remotePath.EndsWith("/meeting-qa-chunking")) {
            throw "Unexpected remote repository directory: $remotePath"
        }
        $remoteSourcePath = "$remotePath/src"

        Write-Host "Replacing remote source snapshot..."
        ssh -o BatchMode=yes $remote "rm -rf -- $remoteSourcePath"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not remove the previous remote source snapshot"
        }

        Write-Host "Uploading src/..."
        scp -o BatchMode=yes -r src "${remote}:${remoteDirectory}/"
        if ($LASTEXITCODE -ne 0) {
            throw "Upload failed"
        }

        if ($job.DataManifest -or $job.MeetingIds) {
            $remoteDataDirectory = "$remoteDirectory/data/raw/qmsum/data/ALL/val"
            ssh -o BatchMode=yes $remote "mkdir -p $remoteDataDirectory"
            if ($LASTEXITCODE -ne 0) {
                throw "Could not create the remote data directory"
            }
            $meetingIds = if ($job.DataManifest) {
                Get-Content -LiteralPath $job.DataManifest
            }
            else {
                $job.MeetingIds
            }
            Write-Host "Uploading $($meetingIds.Count) selected QMSum meetings..."
            foreach ($meetingId in $meetingIds) {
                $localMeeting = Join-Path $PSScriptRoot `
                    "data\raw\qmsum\data\ALL\val\$meetingId.json"
                if (-not (Test-Path -LiteralPath $localMeeting)) {
                    throw "Missing local meeting: $localMeeting"
                }
                scp -o BatchMode=yes $localMeeting `
                    "${remote}:${remoteDataDirectory}/"
                if ($LASTEXITCODE -ne 0) {
                    throw "Could not upload meeting: $meetingId"
                }
            }
        }

        Write-Host "Submitting $Task job..."
        $submission = ssh -o BatchMode=yes $remote `
            "cd $remoteDirectory && sbatch $($job.Slurm)"
        if ($LASTEXITCODE -ne 0) {
            throw "Job submission failed"
        }

        $submissionText = $submission -join "`n"
        Write-Host $submissionText
        if ($submissionText -notmatch "Submitted batch job (\d+)") {
            throw "Could not read the Slurm job ID"
        }
        $jobId = $Matches[1]

        if ($NoWait) {
            Write-Host "Submitted without waiting. Job ID: $jobId"
            return
        }
    }

    Write-Host -NoNewline "Waiting for job $jobId"
    $state = "UNKNOWN"
    $statusFailures = 0
    $activeStates = @(
        "UNKNOWN",
        "CONFIGURING",
        "PENDING",
        "RUNNING",
        "COMPLETING",
        "SUSPENDED",
        "REQUEUED",
        "RESIZING",
        "STAGE_OUT"
    )
    # squeue is authoritative while live; sacct supplies the terminal state.
    while ($state -in $activeStates) {
        Start-Sleep -Seconds 5

        $savedErrorPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $queueOutput = ssh -o BatchMode=yes `
            -o ConnectTimeout=15 `
            -o ServerAliveInterval=15 `
            -o ServerAliveCountMax=2 `
            $remote `
            "squeue -h -j $jobId -o '%T'" 2>$null
        $queueSucceeded = $LASTEXITCODE -eq 0
        $ErrorActionPreference = $savedErrorPreference
        $queueState = @($queueOutput | Where-Object { $_ })[0]
        if ($queueSucceeded -and $queueState) {
            $state = $queueState.Trim()
            $statusFailures = 0
            Write-Host -NoNewline "."
            continue
        }

        $ErrorActionPreference = "Continue"
        $stateOutput = ssh -o BatchMode=yes `
            -o ConnectTimeout=15 `
            -o ServerAliveInterval=15 `
            -o ServerAliveCountMax=2 `
            $remote `
            "sacct -X -j $jobId --format=JobID,State -n -P" 2>$null
        $accountingSucceeded = $LASTEXITCODE -eq 0
        $ErrorActionPreference = $savedErrorPreference
        $stateLine = @(
            $stateOutput | Where-Object { $_ -match "^$jobId\|" }
        )[0]
        if ($stateLine) {
            $state = ($stateLine -split "\|")[1].Trim()
            $statusFailures = 0
        }
        elseif (-not $queueSucceeded -or -not $accountingSucceeded) {
            $statusFailures += 1
            if ($statusFailures -ge 60) {
                throw "Could not read Slurm status for job $jobId after 5 minutes"
            }
            $state = "UNKNOWN"
            Write-Host -NoNewline "?"
            continue
        }

        if ($state -in $activeStates) {
            Write-Host -NoNewline "."
        }
    }
    Write-Host ""

    $remoteLog = "$($job.LogPrefix)-$jobId.out"
    $localLogDirectory = Join-Path $PSScriptRoot "runs\wormulon\logs"
    $localLog = Join-Path $localLogDirectory $remoteLog
    New-Item -ItemType Directory -Force $localLogDirectory | Out-Null
    scp -o BatchMode=yes "${remote}:${remoteDirectory}/${remoteLog}" $localLog
    if ($LASTEXITCODE -ne 0) {
        throw "Could not download the Slurm log"
    }

    Write-Host "Job state: $state"
    Write-Host "Log: $localLog"
    Write-Host "--- last log lines ---"
    Get-Content -LiteralPath $localLog -Tail 30

    if (-not $state.StartsWith("COMPLETED")) {
        throw "Slurm job $jobId did not complete successfully"
    }

    $localResult = Join-Path $PSScriptRoot $job.Result
    if ($job.ResultIsDirectory) {
        Receive-RemoteDirectory `
            "$remoteDirectory/$($job.Result)" `
            $localResult `
            $jobId
    }
    else {
        New-Item -ItemType Directory -Force `
            (Split-Path -Parent $localResult) | Out-Null
        scp -o BatchMode=yes `
            "${remote}:${remoteDirectory}/$($job.Result)" $localResult
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Could not download the result"
    }
    Write-Host "Result: $localResult"

    if ($job.ResultIsDirectory) {
        # Lumber artifacts live outside the ablation result but reports need them.
        $localLumber = Join-Path $PSScriptRoot "runs\lumber\qmsum"
        Receive-RemoteDirectory `
            "$remoteDirectory/runs/lumber/qmsum" `
            $localLumber `
            "lumber-$jobId"
        # Scripts live below src/, so expose the reusable package to Python.
        $previousPythonPath = $env:PYTHONPATH
        $env:PYTHONPATH = Join-Path $PSScriptRoot "src"
        python src/tools/report_ablations.py --root $localResult
        $env:PYTHONPATH = $previousPythonPath
        if ($LASTEXITCODE -ne 0) {
            throw "Could not generate the ablation report"
        }
    }
}
finally {
    Pop-Location
}
