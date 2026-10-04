"""
models.py — Model definitions for Bone Fracture Detection.

Three model iterations:
  - ModelBaseline : Simple 4-block CNN (from scratch)
  - ModelImproved : ResNet-18 fine-tuned (transfer learning)
  - ModelFinal    : EfficientNet-B0 fine-tuned (best performance)

Moved from root model.py to src/ for Layer 1 (provider-neutral) compliance.
"""

import torch.nn as nn
import torchvision.models as models


# ── Model 1: Baseline CNN ─────────────────────────────────────────────────────
class ModelBaseline(nn.Module):
    """
    Simple 4-block CNN trained from scratch.
    Establishes a performance baseline — no pretrained weights.
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224 → 112
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 112 → 56
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 56 → 28
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4: 28 → 4×4 (adaptive)
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Model 2: ResNet-18 Fine-Tuned ─────────────────────────────────────────────
class ModelImproved(nn.Module):
    """
    ResNet-18 pretrained on ImageNet, fine-tuned for fracture detection.
    First 6 layers frozen; deeper layers + custom head are trainable.
    Improvement: better generalisation via transfer learning + class-weighted loss.
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.5,
                 freeze_layers: int = 6):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze early layers
        for layer in list(backbone.children())[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False

        in_feats = backbone.fc.in_features  # 512
        self.features    = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_feats, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Model 3: EfficientNet-B0 Fine-Tuned ──────────────────────────────────────
class ModelFinal(nn.Module):
    """
    EfficientNet-B0 pretrained on ImageNet — compound-scaled, 5.3M params.
    Full fine-tuning with deeper custom head.
    Best F1 + AUC across all three iterations.
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, 256), nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


# ── Helper ────────────────────────────────────────────────────────────────────
def count_params(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
