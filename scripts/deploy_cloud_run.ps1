# Deploy NeuroBridge to Cloud Run (public URL) with Gemini on Vertex AI. Windows PowerShell version
# of scripts/deploy_cloud_run.sh.
#
# Run from the repo root on a machine OUTSIDE a corporate network that blocks Google Cloud:
#   gcloud auth login
#   powershell -ExecutionPolicy Bypass -File scripts\deploy_cloud_run.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\deploy_cloud_run.ps1 -ProjectId my-project
#
# Safe to re-run. Remove everything with:  ... deploy_cloud_run.ps1 -Teardown
param(
    [string]$ProjectId = "awesome-project-491217",
    [string]$Region = "us-central1",
    [string]$Service = "neurobridge",
    [string]$GeminiModel = "gemini-2.5-flash",
    [switch]$Teardown
)

$ErrorActionPreference = "Stop"
$Repo = "cloud-run-source-deploy"
$RunSa = "neurobridge-run@$ProjectId.iam.gserviceaccount.com"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# Run gcloud; throw on failure unless -AllowFail.
function G {
    param([switch]$AllowFail, [Parameter(ValueFromRemainingArguments = $true)][string[]]$GArgs)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $out = & gcloud "--project=$ProjectId" --quiet @GArgs 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0 -and -not $AllowFail) {
        $out | ForEach-Object { Write-Host $_ }
        throw "gcloud $($GArgs -join ' ') failed (exit $code)"
    }
    return @{ Code = $code; Out = ($out | Out-String).Trim() }
}

Set-Location (Join-Path $PSScriptRoot "..")

if ($Teardown) {
    Step "Deleting Cloud Run service $Service"
    G run services delete $Service "--region=$Region" -AllowFail | Out-Null
    Step "Deleting container images in $Repo"
    G artifacts repositories delete $Repo "--location=$Region" -AllowFail | Out-Null
    Step "Deleting runtime service account"
    G iam service-accounts delete $RunSa -AllowFail | Out-Null
    Write-Host "Teardown complete. Billing stays linked; unlink it in the console if you want."
    exit 0
}

Step "Checks"
foreach ($f in "Dockerfile", "requirements.txt", "artifacts\metrics.json", "artifacts\demo_trials.npz") {
    if (-not (Test-Path $f)) { throw "Missing $f - run from the repo root of a full clone." }
}
$account = (& gcloud config get-value account 2>$null)
if (-not $account) { throw "Not logged in. Run: gcloud auth login" }
Write-Host "Account: $account"
Write-Host "Project: $ProjectId   Region: $Region   Service: $Service"
$billing = (G billing projects describe $ProjectId "--format=value(billingEnabled)" -AllowFail).Out
if ($billing -ne "True") { throw "Billing is not enabled on $ProjectId." }

Step "Enabling APIs (no-op if already enabled)"
G services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com `
    aiplatform.googleapis.com iam.googleapis.com logging.googleapis.com | Out-Null

Step "Runtime service account"
if ((G iam service-accounts describe $RunSa -AllowFail).Code -ne 0) {
    G iam service-accounts create neurobridge-run "--display-name=NeuroBridge Cloud Run runtime" | Out-Null
}
Start-Sleep -Seconds 10  # new service accounts take a moment to become usable in IAM bindings

Step "Granting roles"
# Runtime: call Gemini on Vertex AI and write logs. Nothing else.
foreach ($role in "roles/aiplatform.user", "roles/logging.logWriter") {
    G projects add-iam-policy-binding $ProjectId "--member=serviceAccount:$RunSa" "--role=$role" "--condition=None" | Out-Null
}
# Build: source deploys build with the Compute Engine default service account, which
# needs Cloud Run Builder (https://cloud.google.com/run/docs/deploying-source-code).
$projectNumber = (G projects describe $ProjectId "--format=value(projectNumber)").Out
$computeSa = "$projectNumber-compute@developer.gserviceaccount.com"
G projects add-iam-policy-binding $ProjectId "--member=serviceAccount:$computeSa" "--role=roles/run.builder" "--condition=None" | Out-Null
Write-Host "Waiting 60s for IAM changes to propagate..."
Start-Sleep -Seconds 60

Step "Artifact Registry repository"
if ((G artifacts repositories describe $Repo "--location=$Region" -AllowFail).Code -ne 0) {
    G artifacts repositories create $Repo "--location=$Region" "--repository-format=docker" `
        "--description=NeuroBridge container images" | Out-Null
}

Step "Building and deploying (first build takes ~5-10 minutes)"
# Stream output live for the long build.
& gcloud "--project=$ProjectId" --quiet run deploy $Service `
    --source=. `
    "--region=$Region" `
    "--service-account=$RunSa" `
    --allow-unauthenticated `
    "--set-env-vars=GOOGLE_CLOUD_PROJECT=$ProjectId,GOOGLE_CLOUD_LOCATION=$Region,GEMINI_MODEL=$GeminiModel" `
    --memory=2Gi --cpu=1 `
    --min-instances=0 --max-instances=2 `
    --concurrency=20 --timeout=3600 `
    --session-affinity `
    --port=8080
if ($LASTEXITCODE -ne 0) { throw "Deploy failed. See the build log link above." }

$url = (G run services describe $Service "--region=$Region" "--format=value(status.url)").Out

Step "Smoke test"
foreach ($i in 1..6) {
    try {
        $h = (Invoke-WebRequest "$url/_stcore/health" -UseBasicParsing -TimeoutSec 30).Content
        if ($h -eq "ok") { Write-Host "Health check: ok"; break }
    } catch { }
    Write-Host "Waiting for the service to start ($i/6)..."
    Start-Sleep -Seconds 10
}

Write-Host "`nDeployed: $url" -ForegroundColor Green
Write-Host "Open it in a browser, go to Communicate, replay a trial, confirm, and click 'Suggest phrases'."
Write-Host "The sidebar should show 'Gemini: connected'."
Write-Host "Logs:     gcloud run services logs read $Service --project=$ProjectId --region=$Region --limit=50"
Write-Host "Teardown: powershell -ExecutionPolicy Bypass -File scripts\deploy_cloud_run.ps1 -Teardown"
