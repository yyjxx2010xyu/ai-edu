import math
import torch
import torch.nn.functional as F


def psnr(x, y, data_range=1.0):
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return 99.0
    return 10 * math.log10((data_range ** 2) / mse)


def ssim(x, y, data_range=1.0, window_size=11):
    # x, y: (B,C,H,W), range [0,1]
    # Lightweight SSIM approximation (single-scale, Gaussian window)
    try:
        import torch
    except Exception:
        return None
    device = x.device
    channel = x.size(1)

    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(i - window_size // 2) ** 2 / float(2 * sigma ** 2)) for i in range(window_size)])
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window @ _1D_window.t()
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    window = create_window(window_size, channel).to(device=device, dtype=x.dtype)

    mu1 = F.conv2d(x, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(y, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(x * x, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(x * y, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()
