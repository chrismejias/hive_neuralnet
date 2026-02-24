"""Tests for device detection and GPU utilities."""

import pytest
import torch

from hive_engine.device import get_device, device_summary


class TestGetDevice:
    """Test device auto-detection and explicit selection."""

    def test_auto_returns_valid_device(self):
        """Auto-detect should return a valid torch device."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "mps", "cpu")

    def test_none_same_as_auto(self):
        """None should behave the same as 'auto'."""
        d1 = get_device(None)
        d2 = get_device("auto")
        assert d1.type == d2.type

    def test_cpu_always_works(self):
        """CPU should always be available."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_invalid_cuda_raises(self):
        """Requesting CUDA when unavailable should raise RuntimeError."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, can't test unavailability")
        with pytest.raises(RuntimeError, match="CUDA requested"):
            get_device("cuda")

    def test_cuda_returns_cuda_if_available(self):
        """If CUDA is available, requesting it should succeed."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = get_device("cuda")
        assert device.type == "cuda"

    def test_cuda_with_index(self):
        """Requesting cuda:0 should work if CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = get_device("cuda:0")
        assert device.type == "cuda"
        assert device.index == 0

    def test_mps_returns_mps_if_available(self):
        """If MPS is available, requesting it should succeed."""
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS not available")
        device = get_device("mps")
        assert device.type == "mps"

    def test_auto_prefers_cuda_over_mps(self):
        """Auto-detect should prefer CUDA over MPS."""
        device = get_device("auto")
        if torch.cuda.is_available():
            assert device.type == "cuda"

    def test_tensor_can_be_placed_on_device(self):
        """A tensor should be placeable on the detected device."""
        device = get_device()
        t = torch.zeros(2, 3, device=device)
        assert t.device.type == device.type


class TestDeviceSummary:
    """Test human-readable device summary."""

    def test_cpu_summary(self):
        """CPU summary should be simple string."""
        s = device_summary(torch.device("cpu"))
        assert "cpu" in s

    def test_mps_summary(self):
        """MPS summary should mention Apple Metal."""
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS not available")
        s = device_summary(torch.device("mps"))
        assert "Metal" in s

    def test_cuda_summary(self):
        """CUDA summary should include GPU name and memory."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        s = device_summary(torch.device("cuda"))
        assert "GB" in s


class TestAMPIntegration:
    """Test that mixed precision config propagates correctly."""

    def test_amp_auto_false_on_cpu(self):
        """AMP should auto-disable on CPU."""
        from hive_engine.trainer import Trainer, TrainConfig
        from hive_engine.neural_net import NetConfig
        config = TrainConfig(device="cpu")
        trainer = Trainer(config=config, net_config=NetConfig.small())
        assert trainer.use_amp is False
        assert trainer._grad_scaler is None

    def test_amp_explicit_false(self):
        """use_amp=False should disable AMP even on CUDA."""
        from hive_engine.trainer import Trainer, TrainConfig
        from hive_engine.neural_net import NetConfig
        config = TrainConfig(use_amp=False)
        trainer = Trainer(config=config, net_config=NetConfig.small())
        assert trainer.use_amp is False

    def test_amp_auto_true_on_cuda(self):
        """AMP should auto-enable on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from hive_engine.trainer import Trainer, TrainConfig
        from hive_engine.neural_net import NetConfig
        config = TrainConfig(device="cuda")
        trainer = Trainer(config=config, net_config=NetConfig.small())
        assert trainer.use_amp is True
        assert trainer._grad_scaler is not None
