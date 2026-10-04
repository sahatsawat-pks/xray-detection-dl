"""
gcp.py — Google Cloud Platform adapter.

Implements CloudAdapter for GCP services:
  - Cloud Storage (upload / download)
  - Artifact Registry (push_image)
  - Vertex AI (submit_training, register_model, deploy, invoke)
  - Cloud Monitoring (emit_metric)
  - Vertex AI GenAI (generate)
"""

import os
import subprocess

from cloudlayer.base import CloudAdapter


class GCPAdapter(CloudAdapter):
    """GCP implementation of the CloudAdapter interface."""

    def __init__(self):
        self.project_id = os.environ.get("PROJECT_ID", "")
        self.region = os.environ.get("REGION", "asia-southeast1")
        self.blob_uri = os.environ.get("BLOB_URI", "")
        self.registry = os.environ.get("CONTAINER_REGISTRY", "")

    def upload(self, local_path: str, key: str) -> str:
        """Upload a local file to GCS."""
        from google.cloud import storage

        # Parse bucket and prefix from BLOB_URI (gs://bucket/prefix)
        uri = self.blob_uri.rstrip("/")
        parts = uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        blob_key = f"{prefix}/{key}".lstrip("/")

        client = storage.Client(project=self.project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_key)
        blob.upload_from_filename(local_path)

        remote_uri = f"gs://{bucket_name}/{blob_key}"
        print(f"[gcp] Uploaded {local_path} → {remote_uri}")
        return remote_uri

    def download(self, uri: str, local_path: str) -> None:
        """Download a file from GCS."""
        from google.cloud import storage

        # Parse gs://bucket/key
        path = uri.replace("gs://", "")
        bucket_name, blob_key = path.split("/", 1)

        client = storage.Client(project=self.project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_key)

        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"[gcp] Downloaded {uri} → {local_path}")

    def push_image(self, local_tag: str) -> str:
        """Tag and push a Docker image to Artifact Registry. Returns digest-pinned URI."""
        remote_tag = f"{self.registry}/{local_tag}"
        subprocess.run(["docker", "tag", local_tag, remote_tag], check=True)
        subprocess.run(["docker", "push", remote_tag], check=True)

        # Get the digest
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", remote_tag],
            capture_output=True, text=True, check=True,
        )
        digest_uri = result.stdout.strip()
        print(f"[gcp] Pushed {local_tag} → {digest_uri}")
        return digest_uri

    def submit_training(self, image_uri: str, args: dict) -> str:
        """Submit a Vertex AI Custom Training job."""
        raise NotImplementedError(
            "submit_training requires Vertex AI setup. "
            "Implement when cloud training is configured."
        )

    def wait_training(self, job_id: str) -> dict:
        """Wait for a Vertex AI training job."""
        raise NotImplementedError(
            "wait_training requires Vertex AI setup."
        )

    def register_model(self, model_uri: str, name: str) -> str:
        """Register model in Vertex AI Model Registry."""
        raise NotImplementedError(
            "register_model requires Vertex AI setup."
        )

    def deploy(self, model_ref: str, endpoint: str, instance: str) -> str:
        """Deploy to a Vertex AI Endpoint."""
        raise NotImplementedError(
            "deploy requires Vertex AI Endpoint setup."
        )

    def invoke(self, endpoint: str, payload: dict) -> dict:
        """Invoke a Vertex AI Endpoint."""
        raise NotImplementedError(
            "invoke requires a deployed Vertex AI Endpoint."
        )

    def emit_metric(self, name: str, value: float, unit: str) -> None:
        """Emit a custom metric to Cloud Monitoring."""
        from google.cloud import monitoring_v3
        from google.protobuf import timestamp_pb2
        import time

        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{self.project_id}"

        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/{name}"
        series.resource.type = "global"

        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 1e9)

        interval = monitoring_v3.TimeInterval(
            end_time=timestamp_pb2.Timestamp(seconds=seconds, nanos=nanos)
        )
        point = monitoring_v3.Point(
            interval=interval,
            value=monitoring_v3.TypedValue(double_value=value),
        )
        series.points = [point]

        client.create_time_series(name=project_name, time_series=[series])
        print(f"[gcp] Emitted metric {name}={value} {unit}")

    def generate(self, prompt: str, params: dict) -> dict:
        """Call Vertex AI Generative AI."""
        raise NotImplementedError(
            "generate requires Vertex AI GenAI setup."
        )

    def teardown(self, tags: dict) -> list[str]:
        """Delete GCP resources matching the given labels."""
        # For now, log which resources would be deleted
        print(f"[gcp] Teardown requested for labels: {tags}")
        print("[gcp] Manual teardown required — check Cloud Console")
        return []
