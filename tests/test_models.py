"""Tests for audio_xai.models."""

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed (install the 'models' extra)")


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------


class TestSpectra:
    def test_import(self) -> None:
        from audio_xai.models import Spectra

        assert Spectra is not None

    def test_default_output_shape(self) -> None:
        from audio_xai.models import Spectra

        model = Spectra()
        x = torch.zeros(2, 1, 128, 128)
        logits = model(x)
        assert logits.shape == (2, 527)

    def test_custom_num_classes(self) -> None:
        from audio_xai.models import Spectra

        model = Spectra(num_classes=10)
        x = torch.zeros(1, 1, 64, 64)
        assert model(x).shape == (1, 10)

    def test_multi_channel_input(self) -> None:
        from audio_xai.models import Spectra

        model = Spectra(in_channels=2)
        x = torch.zeros(1, 2, 64, 64)
        assert model(x).shape == (1, 527)

    def test_pretrained_requires_hf_model_name(self) -> None:
        """Spectra(pretrained=True) without an hf_model_name must raise ValueError."""
        from audio_xai.models import Spectra

        with pytest.raises(ValueError, match="hf_model_name"):
            Spectra(pretrained=True)


# ---------------------------------------------------------------------------
# AudioViT
# ---------------------------------------------------------------------------


class TestAudioViT:
    def test_import(self) -> None:
        from audio_xai.models import AudioViT

        assert AudioViT is not None

    def test_default_output_shape(self) -> None:
        from audio_xai.models import AudioViT

        model = AudioViT()
        x = torch.zeros(2, 1, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 527)

    def test_custom_num_classes(self) -> None:
        from audio_xai.models import AudioViT

        model = AudioViT(num_classes=50)
        x = torch.zeros(1, 1, 224, 224)
        assert model(x).shape == (1, 50)


# ---------------------------------------------------------------------------
# AudioResNet50
# ---------------------------------------------------------------------------


class TestAudioResNet50:
    def test_import(self) -> None:
        from audio_xai.models import AudioResNet50

        assert AudioResNet50 is not None

    def test_default_output_shape(self) -> None:
        from audio_xai.models import AudioResNet50

        model = AudioResNet50()
        x = torch.zeros(2, 1, 128, 128)
        logits = model(x)
        assert logits.shape == (2, 527)

    def test_custom_num_classes(self) -> None:
        from audio_xai.models import AudioResNet50

        model = AudioResNet50(num_classes=10)
        x = torch.zeros(1, 1, 64, 64)
        assert model(x).shape == (1, 10)


# ---------------------------------------------------------------------------
# Package-level export
# ---------------------------------------------------------------------------


def test_models_package_exports() -> None:
    import audio_xai.models as m

    assert hasattr(m, "Spectra")
    assert hasattr(m, "AudioViT")
    assert hasattr(m, "AudioResNet50")
