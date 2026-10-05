"""
model_loader.py — Singleton model loader for inference.

Loads the EfficientNet-B0 checkpoint ONCE at startup. Provides
get_model() and is_ready() for the /ready endpoint. Handles
GPU/CPU fallback and image preprocessing.
"""

import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ── State ─────────────────────────────────────────────────────────────────────
_state: dict = {
    "model": None,
    "model_name": "",
    "device": "cpu",
    "version": "unknown",
    "loaded_at": None,
}

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60  # Below this → flag as "uncertain"
CLASSES = ["Non-Fractured", "Fractured"]
IMG_SIZE = 224

# ImageNet normalization
_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _get_device() -> str:
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_name: str = "ModelFinal",
    checkpoint_dir: str | None = None,
    version: str = "unknown",
) -> None:
    """
    Load a model checkpoint into memory. Called once at startup.

    Args:
        model_name: Architecture name (ModelBaseline, ModelImproved, ModelFinal).
        checkpoint_dir: Directory containing .pth files. Defaults to project models/.
        version: Version string (MLflow run ID or git SHA) for tracing.
    """
    global _state

    if checkpoint_dir is None:
        checkpoint_dir = str(Path(__file__).resolve().parent.parent / "models")

    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best_vloss.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run 'make train' first."
        )

    device = _get_device()

    # Import model class dynamically
    if model_name == "ModelBaseline":
        from src.models import ModelBaseline
        model = ModelBaseline(num_classes=2)
    elif model_name == "ModelImproved":
        from src.models import ModelImproved
        model = ModelImproved(num_classes=2, freeze_layers=6)
    elif model_name == "ModelFinal":
        from src.models import ModelFinal
        model = ModelFinal(num_classes=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load weights
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Warm-up inference
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        model(dummy)

    _state = {
        "model": model,
        "model_name": model_name,
        "device": device,
        "version": version,
        "loaded_at": time.time(),
    }

    print(f"[model_loader] Loaded {model_name} on {device} (checkpoint: {ckpt_path})")


def get_model() -> torch.nn.Module | None:
    """Return the loaded model, or None if not loaded."""
    return _state["model"]


def get_model_name() -> str:
    """Return the name of the loaded model architecture."""
    return _state["model_name"]


def get_device() -> str:
    """Return the device the model is on."""
    return _state["device"]


def get_version() -> str:
    """Return the model version string."""
    return _state["version"]


def is_ready() -> bool:
    """Return True if the model is loaded and ready to serve."""
    return _state["model"] is not None


def validate_image(image: Image.Image) -> tuple[bool, str]:
    """
    Validate an uploaded image for inference.

    Returns (is_valid, error_message).
    Checks: format, size, entropy (rejects blank images).
    """
    # Check minimum dimensions
    try:
        w, h = image.size
    except Exception:
        return False, "Cannot read image dimensions"
    if w < 10 or h < 10:
        return False, f"Image too small: {w}x{h} (minimum 10x10)"

    # Check for degenerate images (all-black, all-white)
    try:
        arr = np.array(image.convert("L"))  # Grayscale
    except (OSError, Exception) as e:
        return False, f"Image is corrupted or truncated: {e}"
    entropy = -np.sum(
        (counts := np.bincount(arr.ravel(), minlength=256) / arr.size + 1e-10)
        * np.log2(counts)
    )
    if entropy < 1.0:
        return False, f"Image appears blank or degenerate (entropy={entropy:.2f})"

    return True, ""


def preprocess(image: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image for inference. Returns (1, 3, 224, 224) tensor."""
    image = image.convert("RGB")
    tensor = _TRANSFORM(image).unsqueeze(0)
    return tensor.to(_state["device"])


def generate_gradcam(
    image: Image.Image,
    target_class: int = 1,
    alpha: float = 0.40,
) -> str | None:
    """
    Generate a Grad-CAM localization heatmap overlaid on the radiograph.

    Target class 1 corresponds to 'Fractured'.
    Returns base64-encoded JPEG image string, or None if computation fails.
    """
    model = get_model()
    if model is None:
        return None

    # Identify target convolutional layer
    target_layer = None
    if hasattr(model, "model") and hasattr(model.model, "features"):
        target_layer = model.model.features[-1]
    elif hasattr(model, "features"):
        target_layer = model.features[-1]

    if target_layer is None:
        return None

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def f_hook(module, inp, out):
        activations.append(out)

    def b_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h_fwd = target_layer.register_forward_hook(f_hook)
    h_bwd = target_layer.register_full_backward_hook(b_hook)

    try:
        tensor = preprocess(image)
        tensor.requires_grad = True

        logits = model(tensor)
        score = logits[0, target_class]
        model.zero_grad()
        score.backward()

        if not activations or not gradients:
            return None

        act = activations[0].detach()
        grad = gradients[0].detach()

        # Channel weights via Global Average Pooling
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = F.relu(torch.sum(weights * act, dim=1, keepdim=True))

        # Prepare display image (thumbnail to max 512px for crisp rendering and fast payload)
        disp_img = image.convert("RGB")
        disp_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        disp_w, disp_h = disp_img.size

        cam = F.interpolate(cam, size=(disp_h, disp_w), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        # Smooth JET colormap (pure numpy: zero extra overhead)
        r = np.clip(1.5 - np.abs(4.0 * cam_norm - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * cam_norm - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * cam_norm - 1.0), 0.0, 1.0)
        heatmap = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

        orig_np = np.array(disp_img)
        blended = (orig_np * (1.0 - alpha) + heatmap * alpha).astype(np.uint8)

        out_img = Image.fromarray(blended)
        buf = io.BytesIO()
        out_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"[gradcam] Warning: Could not compute Grad-CAM: {e}")
        return None
    finally:
        h_fwd.remove()
        h_bwd.remove()


def predict(image: Image.Image) -> dict:
    """
    Run inference on a single PIL image.

    Returns dict with: label, confidence, fractured_probability,
    inference_time_ms, uncertain, model_version, model_name, heatmap_base64.
    """
    model = get_model()
    if model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    tensor = preprocess(image)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
    elapsed_ms = (time.perf_counter() - start) * 1000

    fractured_prob = probs[0, 1].item()
    predicted_class = 1 if fractured_prob >= 0.5 else 0
    confidence = fractured_prob if predicted_class == 1 else (1 - fractured_prob)

    # Generate Grad-CAM whenever a fracture is detected or suspected
    heatmap_b64 = None
    if predicted_class == 1 or fractured_prob >= 0.25:
        heatmap_b64 = generate_gradcam(image, target_class=1)

    return {
        "label": CLASSES[predicted_class],
        "confidence": round(confidence, 4),
        "fractured_probability": round(fractured_prob, 4),
        "inference_time_ms": round(elapsed_ms, 2),
        "uncertain": confidence < CONFIDENCE_THRESHOLD,
        "model_version": _state["version"],
        "model_name": _state["model_name"],
        "heatmap_base64": heatmap_b64,
    }
