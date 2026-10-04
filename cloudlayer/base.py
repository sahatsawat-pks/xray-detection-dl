"""
base.py — CloudAdapter abstract base class.

Defines the provider-neutral interface for all cloud operations.
Exactly one concrete adapter (gcp.py, aws.py, or azure.py) implements this.
Nothing in src/ or tests/ may import from cloudlayer/ directly.
"""

from abc import ABC, abstractmethod


class CloudAdapter(ABC):
    """Provider-neutral interface for cloud MLOps operations."""

    @abstractmethod
    def upload(self, local_path: str, key: str) -> str:
        """Upload a local file to object storage. Returns the remote URI."""
        ...

    @abstractmethod
    def download(self, uri: str, local_path: str) -> None:
        """Download a file from object storage to a local path."""
        ...

    @abstractmethod
    def push_image(self, local_tag: str) -> str:
        """Tag and push a Docker image to the container registry. Returns the pushed URI with digest."""
        ...

    @abstractmethod
    def submit_training(self, image_uri: str, args: dict) -> str:
        """Submit a training job to managed compute. Returns the job ID."""
        ...

    @abstractmethod
    def wait_training(self, job_id: str) -> dict:
        """Wait for a training job to complete. Returns final status and metrics."""
        ...

    @abstractmethod
    def register_model(self, model_uri: str, name: str) -> str:
        """Register a model artifact in the model registry. Returns the model version reference."""
        ...

    @abstractmethod
    def deploy(self, model_ref: str, endpoint: str, instance: str) -> str:
        """Deploy a model to an endpoint. Returns the endpoint URL."""
        ...

    @abstractmethod
    def invoke(self, endpoint: str, payload: dict) -> dict:
        """Send a prediction request to a deployed endpoint. Returns the response."""
        ...

    @abstractmethod
    def emit_metric(self, name: str, value: float, unit: str) -> None:
        """Emit a custom metric to the cloud monitoring service."""
        ...

    @abstractmethod
    def generate(self, prompt: str, params: dict) -> dict:
        """Call a managed LLM endpoint. Returns response text and token counts."""
        ...

    @abstractmethod
    def teardown(self, tags: dict) -> list[str]:
        """Delete all resources matching the given tags. Returns list of deleted resource IDs."""
        ...
