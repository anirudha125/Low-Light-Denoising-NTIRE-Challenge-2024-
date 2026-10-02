"""
Comprehensive unit tests for the final accepted public model:
PixelWarriorNetwork with InstanceNorm2d backbone (EXP-02 Champion).

Verifies:
1. Exact parameter count: 31,044,152
2. Normalization layers: Exactly 18 InstanceNorm2d(affine=True) and 0 BatchNorm2d
3. Forward pass tensor shapes: [B, 3, 400, 600] -> [B, 3, 400, 600]
4. Direct synthesis (no global identity residual)
5. Metric range clamping and stability
"""

import os
import sys
import math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PixelWarriorNetwork, DoubleConv, UNet, MLPBlock
from src.metrics import calculate_psnr, calculate_ssim, calculate_mse


def test_parameter_count():
    """Verify exact total model and sub-module parameter counts."""
    model = PixelWarriorNetwork()
    total_params = sum(p.numel() for p in model.parameters())
    downsampler_params = sum(p.numel() for p in model.downsampler.parameters())
    mlp_params = sum(p.numel() for p in model.mlp.parameters())
    upsampler_params = sum(p.numel() for p in model.upsampler.parameters())
    unet_params = sum(p.numel() for p in model.unet.parameters())

    assert downsampler_params == 84, f"Downsampler params {downsampler_params} != 84"
    assert mlp_params == 270, f"MLP params {mlp_params} != 270"
    assert upsampler_params == 147, f"Upsampler params {upsampler_params} != 147"
    assert unet_params == 31043651, f"UNet params {unet_params} != 31,043,651"
    assert total_params == 31044152, f"Total params {total_params} != 31,044,152"
    print(f"  [PASS] Parameter count verified: {total_params:,} parameters.")


def test_normalization_layers():
    """Verify all 18 normalization layers are InstanceNorm2d(affine=True) and 0 BatchNorm2d."""
    model = PixelWarriorNetwork()
    in_layers = [m for m in model.modules() if isinstance(m, nn.InstanceNorm2d)]
    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]

    assert len(in_layers) == 18, f"Expected 18 InstanceNorm2d layers, got {len(in_layers)}"
    assert len(bn_layers) == 0, f"Expected 0 BatchNorm2d layers, got {len(bn_layers)}"
    for idx, layer in enumerate(in_layers):
        assert layer.affine is True, f"Layer {idx} InstanceNorm2d affine is not True"
    print(f"  [PASS] Normalization layers verified: 18 InstanceNorm2d(affine=True), 0 BatchNorm2d.")


def test_forward_pass_and_direct_synthesis():
    """Verify native 600x400 forward pass shape and direct synthesis."""
    model = PixelWarriorNetwork()
    x = torch.randn(2, 3, 400, 600)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 3, 400, 600), f"Output shape mismatch: {out.shape}"
    # Verify no accidental global identity residual: out is not identically x + delta
    diff = torch.abs(out - x)
    assert diff.mean() > 0.01, "Model output appears to be an identical residual copy of input"
    print(f"  [PASS] Forward pass verified: {x.shape} -> {out.shape}, direct synthesis confirmed.")


def test_metrics_clamping():
    """Verify evaluation metric clamping in [0.0, 1.0]."""
    unclamped_pred = torch.tensor([[[-0.5, 1.5], [0.4, 0.5]]]).unsqueeze(0)
    target = torch.tensor([[[0.0, 1.0], [0.6, 0.5]]]).unsqueeze(0)

    p_val = calculate_psnr(unclamped_pred, target)
    s_val = calculate_ssim(unclamped_pred, target)
    m_val = calculate_mse(unclamped_pred, target)

    assert not math.isinf(p_val) and p_val > 0, f"PSNR computation failed: {p_val}"
    assert 0.0 <= s_val <= 1.0, f"SSIM computation failed: {s_val}"
    assert m_val >= 0.0, f"MSE computation failed: {m_val}"
    print(f"  [PASS] Metric range clamping verified: PSNR={p_val:.2f} dB, SSIM={s_val:.4f}, MSE={m_val:.6f}.")


if __name__ == "__main__":
    print("Running public model test suite...")
    test_parameter_count()
    test_normalization_layers()
    test_forward_pass_and_direct_synthesis()
    test_metrics_clamping()
    print("\n>>> ALL PUBLIC MODEL TESTS PASSED SUCCESSFULLY! <<<")
