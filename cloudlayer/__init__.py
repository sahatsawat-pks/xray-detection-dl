"""
cloudlayer — Cloud abstraction layer.

Exposes provider-neutral interface CloudAdapter and factory function get_adapter().
Only modules in cloudlayer/ may import vendor SDKs (google.cloud, boto3, etc.).
"""

from cloudlayer.base import CloudAdapter
from cloudlayer.gcp import GCPAdapter


def get_adapter() -> CloudAdapter:
    """Return the concrete CloudAdapter configured for the project."""
    import os
    provider = os.getenv("CLOUD_PROVIDER", "gcp").lower()
    if provider == "gcp":
        return GCPAdapter()
    raise ValueError(f"Unsupported CLOUD_PROVIDER: {provider}")


__all__ = ["CloudAdapter", "GCPAdapter", "get_adapter"]
