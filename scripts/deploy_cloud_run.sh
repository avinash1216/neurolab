#!/usr/bin/env bash
# Deploy NeuroBridge to Cloud Run (public URL) with Gemini on Vertex AI.
#
# Run from the repo root on a machine OUTSIDE a corporate network that blocks Google Cloud:
#   gcloud auth login
#   bash scripts/deploy_cloud_run.sh                 # uses defaults below
#   PROJECT_ID=my-project REGION=us-central1 bash scripts/deploy_cloud_run.sh
#
# Safe to re-run: every step is idempotent. Remove everything with:
#   bash scripts/deploy_cloud_run.sh --teardown
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-awesome-project-491217}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-neurobridge}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}"
REPO="cloud-run-source-deploy"
RUN_SA_NAME="neurobridge-run"
RUN_SA="${RUN_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
G="gcloud --project=${PROJECT_ID} --quiet"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--teardown" ]]; then
  step "Deleting Cloud Run service ${SERVICE}"
  $G run services delete "${SERVICE}" --region="${REGION}" || true
  step "Deleting container images in ${REPO}"
  $G artifacts repositories delete "${REPO}" --location="${REGION}" || true
  step "Deleting runtime service account"
  $G iam service-accounts delete "${RUN_SA}" || true
  echo "Teardown complete. Billing stays linked; unlink it in the console if you want."
  exit 0
fi

step "Checks"
for f in Dockerfile requirements.txt artifacts/metrics.json artifacts/demo_trials.npz; do
  [[ -f "$f" ]] || { echo "Missing $f - run from the repo root of a full clone." >&2; exit 1; }
done
ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
[[ -n "${ACCOUNT}" ]] || { echo "Not logged in. Run: gcloud auth login" >&2; exit 1; }
echo "Account: ${ACCOUNT}"
echo "Project: ${PROJECT_ID}   Region: ${REGION}   Service: ${SERVICE}"
BILLING="$(gcloud billing projects describe "${PROJECT_ID}" --format='value(billingEnabled)' 2>/dev/null || true)"
[[ "${BILLING}" == "True" ]] || { echo "Billing is not enabled on ${PROJECT_ID}." >&2; exit 1; }

step "Enabling APIs (no-op if already enabled)"
$G services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  aiplatform.googleapis.com iam.googleapis.com logging.googleapis.com

step "Runtime service account"
$G iam service-accounts describe "${RUN_SA}" >/dev/null 2>&1 || \
  $G iam service-accounts create "${RUN_SA_NAME}" --display-name="NeuroBridge Cloud Run runtime"
sleep 10  # new service accounts take a moment to become usable in IAM bindings

step "Granting roles"
# Runtime: call Gemini on Vertex AI and write logs. Nothing else.
for role in roles/aiplatform.user roles/logging.logWriter; do
  $G projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${RUN_SA}" \
    --role="${role}" --condition=None >/dev/null
done
# Build: source deploys build with the Compute Engine default service account, which
# needs Cloud Run Builder (https://cloud.google.com/run/docs/deploying-source-code).
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
$G projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${COMPUTE_SA}" \
  --role=roles/run.builder --condition=None >/dev/null
echo "Waiting 60s for IAM changes to propagate..."
sleep 60

step "Artifact Registry repository"
$G artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1 || \
  $G artifacts repositories create "${REPO}" --location="${REGION}" --repository-format=docker \
    --description="NeuroBridge container images"

step "Building and deploying (first build takes ~5-10 minutes)"
$G run deploy "${SERVICE}" \
  --source=. \
  --region="${REGION}" \
  --service-account="${RUN_SA}" \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},GEMINI_MODEL=${GEMINI_MODEL}" \
  --memory=2Gi --cpu=1 \
  --min-instances=0 --max-instances=2 \
  --concurrency=20 --timeout=3600 \
  --session-affinity \
  --port=8080

URL="$($G run services describe "${SERVICE}" --region="${REGION}" --format='value(status.url)')"

step "Smoke test"
for i in 1 2 3 4 5 6; do
  if [[ "$(curl -s --max-time 30 "${URL}/_stcore/health" || true)" == "ok" ]]; then
    echo "Health check: ok"
    break
  fi
  echo "Waiting for the service to start ($i/6)..."
  sleep 10
done

printf '\n\033[1;32mDeployed: %s\033[0m\n' "${URL}"
echo "Open it in a browser, go to Communicate, replay a trial, confirm, and click 'Suggest phrases'."
echo "The sidebar should show 'Gemini: connected'."
echo "Logs:     gcloud run services logs read ${SERVICE} --project=${PROJECT_ID} --region=${REGION} --limit=50"
echo "Teardown: bash scripts/deploy_cloud_run.sh --teardown"
