# Deploying NeuroBridge to Cloud Run

This puts the Streamlit app on a **public HTTPS URL** in Google Cloud. Gemini runs on
**Vertex AI** in the same project, using the service's own identity (no API key).

> Run this from a machine on a normal network (for example a personal laptop on home Wi-Fi).
> Corporate networks that block Google Cloud APIs or Cloud Shell will make the deploy fail
> with "Request is prohibited by organization's policy".

## What you need

| Item | Notes |
|---|---|
| Google account that owns the project | Default project: `awesome-project-491217` |
| Billing linked to the project | Already done; ₹1,500/month budget alerts at 50/75/90/100% |
| [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) | Includes `gcloud` |
| Git | To clone the repo |
| **Not** needed | Docker, Python - the image is built in the cloud |

## Steps

```bash
# 1. Sign in and clone
gcloud auth login
git clone https://gitlab.com/avinash1216kapil/neurolab.git
cd neurolab
git checkout develop
```

Private repo? Create a GitLab personal access token with `read_repository` and clone with
`https://oauth2:<TOKEN>@gitlab.com/avinash1216kapil/neurolab.git`.

```bash
# 2. Deploy (about 10 minutes the first time)
# macOS / Linux / Git Bash:
bash scripts/deploy_cloud_run.sh

# Windows PowerShell:
powershell -ExecutionPolicy Bypass -File scripts\deploy_cloud_run.ps1
```

The script prints the URL at the end, for example `https://neurobridge-xxxxxxxx-uc.a.run.app`.

Different project or region: `PROJECT_ID=my-project REGION=us-east4 bash scripts/deploy_cloud_run.sh`
(PowerShell: `-ProjectId my-project -Region us-east4`).

## What the script does

1. Checks you are signed in, the repo is complete and billing is on.
2. Enables Cloud Run, Cloud Build, Artifact Registry, Vertex AI, IAM and Logging APIs.
3. Creates a runtime service account `neurobridge-run` with only **Vertex AI User** and **Logs Writer**.
4. Grants **Cloud Run Builder** to the Compute Engine default service account, which Google
   uses to build source deployments.
5. Builds the `Dockerfile` with Cloud Build and deploys it with:
   public access, 2 GiB memory, 0-2 instances (scales to zero when idle), session affinity
   (Streamlit keeps per-user state), 60-minute request timeout (Streamlit uses a WebSocket).
6. Waits for the health check and prints the URL.

Re-running the script is safe and deploys a new revision.

## Check it works

1. Open the URL. The banner should read **MODE: RECORDED EEG PLAYBACK** and the sidebar
   **Gemini: connected**.
2. **Communicate** tab: replay a right-hand trial, **Confirm**, then **Suggest phrases**.
   You should get Gemini suggestions, not the "showing default phrases" warning.
3. Pick a **synthetic** trial condition in the sidebar and replay. It should show
   **Neutral - poor signal quality**.
4. **Model results** tab shows the held-out metrics.

Logs: `gcloud run services logs read neurobridge --project=awesome-project-491217 --region=us-central1 --limit=50`

## Troubleshooting

| Symptom | Fix |
|---|---|
| `prohibited by organization's policy` / `VPC_SERVICE_CONTROLS` | You are on a network that blocks Google Cloud; switch networks. |
| Build fails with a permission error on the Compute Engine service account | Wait 2 minutes for IAM to propagate and re-run. If that account is disabled, enable it under IAM > Service accounts. |
| Sidebar says **Gemini: not configured** | The service is missing `GOOGLE_CLOUD_PROJECT`; re-run the script. |
| Phrases show "Gemini unavailable (ClientError)" | Check `roles/aiplatform.user` on `neurobridge-run@PROJECT.iam.gserviceaccount.com` and that the Vertex AI API is enabled. |
| First page load is slow | Normal after idle - the service starts from zero (~10-20 s). |

## Cost and cleanup

With 0 minimum instances you pay only while someone uses the app; a hackathon demo
should cost a few rupees. Gemini 2.5 Flash calls cost fractions of a rupee each.

Remove everything after judging:

```bash
bash scripts/deploy_cloud_run.sh --teardown
# or
powershell -ExecutionPolicy Bypass -File scripts\deploy_cloud_run.ps1 -Teardown
```

This deletes the service, the container images and the runtime service account. Billing
stays linked; unlink it in the console if you want a hard stop.
