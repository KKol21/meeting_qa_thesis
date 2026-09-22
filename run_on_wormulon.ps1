param(
    [ValidateSet(
        "ablation-smoke",
        "ablation-validation-full",
        "ablation-full",
        "lumber-sweep",
        "lumber-model-check",
        "baseline-sweep",
        "boundary-control",
        "grounded-judge-check",
        "grounded-judge-sample"
    )]
    [string]$Task = "ablation-smoke",
    [switch]$NoWait,
    [switch]$DryRun,
    [string]$ExistingJobId,
    [string]$Meeting,
    [int]$QuestionIndex = -1,
    [string]$Condition
)

$ErrorActionPreference = "Stop"
$remote = "koko2725@olympus.dsv.su.se"
$remoteDirectory = "~/meeting-qa-chunking"
# One table keeps task-specific paths out of the upload/monitoring logic below.
$jobs = @{
    "ablation-smoke" = @{
        Preset = "src/configs/ablation-smoke.toml"
        WallTime = "04:00:00"
    }
    "ablation-full" = @{
        Preset = "src/configs/ablation-full.toml"
        WallTime = "10:00:00"
    }
    "ablation-validation-full" = @{
        Preset = "src/configs/ablation-validation-full.toml"
        WallTime = "10:00:00"
    }
    "lumber-sweep" = @{
        Preset = "src/configs/ablation-validation-full.toml"
        WallTime = "10:00:00"
        Slurm = "src/wormulon/lumber_sweep.slurm"
        JobName = "lumber-sweep"
        LogPrefix = "slurm-lumber-sweep"
        Result = "runs/ablations/lumber-sweep"
    }
    "lumber-model-check" = @{
        Preset = "src/configs/ablation-validation-full.toml"
        WallTime = "10:00:00"
        Slurm = "src/wormulon/lumber_model_check.slurm"
        JobName = "lumber-model-check"
        LogPrefix = "slurm-lumber-model-check"
        Result = "runs/ablations/lumber-model-check"
    }
    "baseline-sweep" = @{
        Preset = "src/configs/ablation-validation-full.toml"
        WallTime = "04:00:00"
        Slurm = "src/wormulon/baseline_sweep.slurm"
        JobName = "baseline-sweep"
        LogPrefix = "slurm-baseline-sweep"
        Result = "runs/ablations/baseline-sweep"
    }
    "boundary-control" = @{
        Preset = "src/configs/ablation-validation-full.toml"
        WallTime = "06:00:00"
        Slurm = "src/wormulon/boundary_control.slurm"
        JobName = "boundary-control"
        LogPrefix = "slurm-boundary-control"
        Result = "runs/ablations/boundary-control"
        SegmentationInput = "runs/ablations/lumber-sweep/segmentation/1000"
    }
    "grounded-judge-check" = @{
        Preset = "src/configs/ablation-full.toml"
        WallTime = "02:00:00"
        Slurm = "src/wormulon/grounded_judge.slurm"
        JobName = "grounded-judge-check"
        LogPrefix = "slurm-grounded-judge-check"
        Result = "runs/ablations/full/grounded-judge"
        MeetingIds = @("education_4")
        Arguments = @(
            "--meeting", "education_4",
            "--question-index", "3",
            "--condition", "lumber__dense__w2048"
        )
    }
    "grounded-judge-sample" = @{
        Preset = "src/configs/ablation-full.toml"
        WallTime = "04:00:00"
        Slurm = "src/wormulon/grounded_judge_sample.slurm"
        JobName = "grounded-judge-sample"
        LogPrefix = "slurm-grounded-judge-sample"
        Result = "runs/ablations/full/grounded-judge"
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
    if ($Task -eq "grounded-judge-check") {
        $targetMeeting = if ($Meeting) { $Meeting } else { "education_4" }
        $targetQuestion = if ($QuestionIndex -ge 0) { $QuestionIndex } else { 3 }
        $targetCondition = if ($Condition) {
            $Condition
        } else {
            "lumber__dense__w2048"
        }
        $job.MeetingIds = @($targetMeeting)
        $job.Arguments = @(
            "--meeting", $targetMeeting,
            "--question-index", "$targetQuestion",
            "--condition", $targetCondition
        )
    }
    $previousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = Join-Path $PSScriptRoot "src"
    $descriptionJson = python -m meeting_qa_chunking.run_preset `
        --preset $job.Preset `
        --describe
    $env:PYTHONPATH = $previousPythonPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read preset: $($job.Preset)"
    }
    $description = ($descriptionJson -join "`n") | ConvertFrom-Json
    $slurm = if ($job.Slurm) { $job.Slurm } else { "src/wormulon/ablation.slurm" }
    $logPrefix = if ($job.LogPrefix) {
        $job.LogPrefix
    } else {
        "slurm-ablation-$($description.name)"
    }
    $result = if ($job.Result) { $job.Result } else { $description.output_root }
    $jobName = if ($job.JobName) { $job.JobName } else { "ablation-$($description.name)" }

    if ($DryRun) {
        Write-Host "Task: $Task"
        Write-Host "Preset: $($job.Preset)"
        Write-Host "Upload: src/"
        Write-Host "Slurm: $slurm ($($job.WallTime))"
        Write-Host "Job name: $jobName"
        Write-Host "Result: $result"
        if ($job.SegmentationInput) {
            Write-Host "Segmentation input: $($job.SegmentationInput)"
        }
        $meetingIds = if ($job.MeetingIds) {
            $job.MeetingIds
        } else {
            $description.meeting_ids
        }
        Write-Host "Data: $($meetingIds -join ', ')"
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

        if ($Task -eq "ablation-full") {
            # The former validation run used this path. Move it once before the
            # held-out test run so old meeting files cannot contaminate it.
            $oldMarker = "$remotePath/runs/ablations/full/retrieval/ES2006a.json"
            $validationResult = "$remotePath/runs/ablations/validation-full"
            $migration = "if [ -f '$oldMarker' ]; then " +
                "if [ -e '$validationResult' ]; then " +
                "echo 'Both legacy full and validation-full exist' >&2; exit 3; " +
                "fi; mv -- '$remotePath/runs/ablations/full' " +
                "'$validationResult'; fi"
            ssh -o BatchMode=yes $remote $migration
            if ($LASTEXITCODE -ne 0) {
                throw "Could not archive the remote validation run"
            }
        }

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

        $remoteDataDirectory = "$remoteDirectory/$($description.data_dir)"
        ssh -o BatchMode=yes $remote "mkdir -p $remoteDataDirectory"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the remote data directory"
        }
        $meetingIds = if ($job.MeetingIds) {
            $job.MeetingIds
        } else {
            $description.meeting_ids
        }
        Write-Host "Uploading $($meetingIds.Count) selected QMSum meetings..."
        foreach ($meetingId in $meetingIds) {
            $localMeeting = Join-Path $PSScriptRoot `
                "$($description.data_dir)\$meetingId.json"
            if (-not (Test-Path -LiteralPath $localMeeting)) {
                throw "Missing local meeting: $localMeeting"
            }
            scp -o BatchMode=yes $localMeeting `
                "${remote}:${remoteDataDirectory}/"
            if ($LASTEXITCODE -ne 0) {
                throw "Could not upload meeting: $meetingId"
            }
        }

        if ($job.SegmentationInput) {
            $remoteInput = "$remoteDirectory/$($job.SegmentationInput)"
            ssh -o BatchMode=yes $remote "mkdir -p $remoteInput"
            if ($LASTEXITCODE -ne 0) {
                throw "Could not create the remote segmentation directory"
            }
            Write-Host "Uploading Lumber segmentations..."
            foreach ($meetingId in $description.meeting_ids) {
                $localInput = Join-Path $PSScriptRoot `
                    "$($job.SegmentationInput)\$meetingId.json"
                if (-not (Test-Path -LiteralPath $localInput)) {
                    throw "Missing segmentation: $localInput"
                }
                scp -o BatchMode=yes $localInput "${remote}:${remoteInput}/"
                if ($LASTEXITCODE -ne 0) {
                    throw "Could not upload segmentation: $meetingId"
                }
            }
        }

        Write-Host "Submitting $Task job..."
        $arguments = if ($job.Arguments) {
            " " + ($job.Arguments -join " ")
        } else {
            ""
        }
        $submission = ssh -o BatchMode=yes $remote `
            "cd $remoteDirectory && sbatch --job-name=$jobName --time=$($job.WallTime) --output=$logPrefix-%j.out $slurm $($job.Preset)$arguments"
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
        else {
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

    $remoteLog = "$logPrefix-$jobId.out"
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

    $localResult = Join-Path $PSScriptRoot $result
    Receive-RemoteDirectory `
        "$remoteDirectory/$result" `
        $localResult `
        $jobId
    Write-Host "Result: $localResult"

    if ($Task -in @(
        "ablation-smoke",
        "ablation-validation-full",
        "ablation-full"
    )) {
        $previousPythonPath = $env:PYTHONPATH
        $env:PYTHONPATH = Join-Path $PSScriptRoot "src"
        python src/tools/report_ablations.py --preset $job.Preset
        $env:PYTHONPATH = $previousPythonPath
        if ($LASTEXITCODE -ne 0) {
            throw "Could not generate the ablation report"
        }
    }
}
finally {
    Pop-Location
}
