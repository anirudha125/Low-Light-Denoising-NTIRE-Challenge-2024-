import math
import torch
import torch.nn.functional as F

def calculate_mse(output: torch.Tensor, target: torch.Tensor, clamp_range=(0.0, 1.0)) -> float:
    """
    Computes Mean Squared Error with predictions clamped to [0.0, 1.0].
    """
    out_c = torch.clamp(output, clamp_range[0], clamp_range[1])
    tgt_c = torch.clamp(target, clamp_range[0], clamp_range[1])
    return F.mse_loss(out_c, tgt_c).item()


def calculate_psnr(output: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, clamp_range=(0.0, 1.0)) -> float:
    """
    Computes standard Peak Signal-to-Noise Ratio (PSNR) with MAX=1.0.
    Ensures predictions are clamped to the valid range [0, 1] before calculation.
    """
    out_c = torch.clamp(output, clamp_range[0], clamp_range[1])
    tgt_c = torch.clamp(target, clamp_range[0], clamp_range[1])
    mse = F.mse_loss(out_c, tgt_c).item()
    if mse == 0.0:
        return float('inf')
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int, sigma: float = 1.5) -> torch.Tensor:
    _1D_window = _gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim(output: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
                   data_range: float = 1.0, clamp_range=(0.0, 1.0)) -> float:
    """
    Computes Structural Similarity Index (SSIM) on tensors of shape [B, C, H, W] or [C, H, W].
    Clamps values to [0.0, 1.0] and uses standard 11x11 Gaussian window with sigma=1.5.
    """
    out_c = torch.clamp(output, clamp_range[0], clamp_range[1])
    tgt_c = torch.clamp(target, clamp_range[0], clamp_range[1])

    if out_c.dim() == 3:
        out_c = out_c.unsqueeze(0)
    if tgt_c.dim() == 3:
        tgt_c = tgt_c.unsqueeze(0)

    channels = out_c.size(1)
    window = _create_window(window_size, channels, sigma).to(out_c.device).type(out_c.dtype)

    mu1 = F.conv2d(out_c, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(tgt_c, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(out_c * out_c, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(tgt_c * tgt_c, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(out_c * tgt_c, window, padding=window_size // 2, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()
