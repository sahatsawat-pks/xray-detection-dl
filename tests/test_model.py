"""
test_model.py — Model architecture tests (Unit tier).

Verifies that each model instantiates correctly, produces the expected
output shape, and can save/load checkpoints. No cloud SDK imports.
"""

import torch


class TestModelBaseline:
    """Tests for ModelBaseline (simple 4-block CNN)."""

    def test_instantiation(self, model_baseline):
        """Model should instantiate without errors."""
        assert model_baseline is not None

    def test_forward_shape(self, model_baseline, sample_batch, device):
        """Forward pass should produce (batch_size, 2) output."""
        images, _ = sample_batch
        model = model_baseline.to(device)
        output = model(images.to(device))
        assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"

    def test_param_count(self, model_baseline):
        """Baseline should have ~2.5M total params."""
        from src.models import count_params
        params = count_params(model_baseline)
        assert params["total"] > 1_000_000, "Too few parameters"
        assert params["frozen"] == 0, "Baseline should have no frozen params"


class TestModelImproved:
    """Tests for ModelImproved (ResNet-18 fine-tuned)."""

    def test_instantiation(self, model_improved):
        assert model_improved is not None

    def test_forward_shape(self, model_improved, sample_batch, device):
        images, _ = sample_batch
        model = model_improved.to(device)
        output = model(images.to(device))
        assert output.shape == (4, 2)

    def test_frozen_layers(self, model_improved):
        """Should have some frozen parameters from transfer learning."""
        from src.models import count_params
        params = count_params(model_improved)
        assert params["frozen"] > 0, "ResNet-18 should have frozen early layers"

    def test_trainable_params(self, model_improved):
        """Should have trainable params for fine-tuning."""
        from src.models import count_params
        params = count_params(model_improved)
        assert params["trainable"] > 0


class TestModelFinal:
    """Tests for ModelFinal (EfficientNet-B0 fine-tuned)."""

    def test_instantiation(self, model_final):
        assert model_final is not None

    def test_forward_shape(self, model_final, sample_batch, device):
        images, _ = sample_batch
        model = model_final.to(device)
        output = model(images.to(device))
        assert output.shape == (4, 2)

    def test_efficient_params(self, model_final):
        """EfficientNet-B0 should be ~5.3M params — more efficient than ResNet-18."""
        from src.models import count_params
        params = count_params(model_final)
        assert params["total"] < 10_000_000, "EfficientNet-B0 should be < 10M params"

    def test_gradient_flow(self, model_final, sample_batch, device):
        """Gradients should flow through the model during backprop."""
        images, labels = sample_batch
        model = model_final.to(device)
        model.train()
        output = model(images.to(device))
        loss = torch.nn.CrossEntropyLoss()(output, labels.to(device))
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        assert has_grad, "No gradients flowing through the model"


class TestCheckpoint:
    """Tests for model save/load workflow."""

    def test_save_and_load(self, model_final, tmp_path, device):
        """Model should save and reload with identical outputs."""
        from src.models import ModelFinal

        model = model_final.to(device)
        model.eval()

        # Save
        ckpt_path = tmp_path / "test_checkpoint.pth"
        torch.save(model.state_dict(), ckpt_path)

        # Load into fresh model
        loaded = ModelFinal(num_classes=2).to(device)
        loaded.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        loaded.eval()

        # Compare outputs
        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            out_orig = model(test_input)
            out_loaded = loaded(test_input)
        assert torch.allclose(out_orig, out_loaded, atol=1e-6), \
            "Loaded model produces different outputs"
