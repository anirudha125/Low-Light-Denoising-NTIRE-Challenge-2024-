import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Double convolution block used in UNet:
    (Conv2d -> InstanceNorm2d -> ReLU -> Conv2d -> InstanceNorm2d -> ReLU)
    Ablated in EXP-02: BatchNorm2d -> InstanceNorm2d(affine=True) to eliminate
    batch-dependent illumination statistics shift in small-batch DDP regimes.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Standard 4-stage UNet backbone.
    Preserved exactly as in the canonical baseline, with InstanceNorm2d normalization.
    """
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear')

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class MLPBlock(nn.Module):
    """
    Pointwise MLP block:
    (1x1 Conv -> GELU -> 1x1 Conv -> Sigmoid + Residual)
    """
    def __init__(self, in_channels, hidden_channels):
        super(MLPBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.gelu(x1)
        x1 = self.conv2(x1)
        x1 = self.sigmoid(x1)
        x = x + x1
        return x


class PixelWarriorNetwork(nn.Module):
    """
    Accepted Public Final Model (EXP-02 Champion):
    - Input: [B, 3, 400, 600]
    - Downsampler: Conv2d(3, 3, kernel_size=3, stride=2, padding=1) -> [B, 3, 200, 300]
    - MLP Stack: 6x MLPBlock(3, 6) -> [B, 3, 200, 300]
    - Upsampler: ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1) -> [B, 3, 400, 600]
    - UNet Backbone: 4-stage UNet with 18 InstanceNorm2d(affine=True) layers -> [B, 3, 400, 600]
    - Direct Synthesis (no global residual connection)
    Total parameters: 31,044,152
    """
    def __init__(self, in_channels=3, mlp_hidden_dim=6, num_mlp_blocks=6, num_out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downsampler = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.mlp = nn.Sequential(*[MLPBlock(in_channels, mlp_hidden_dim) for _ in range(num_mlp_blocks)])
        self.upsampler = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.unet = UNet(in_channels, num_out_channels, features)

    def forward(self, x):
        feat = self.downsampler(x)
        feat = self.mlp(feat)
        feat = self.upsampler(feat)
        out = self.unet(feat)
        return out
