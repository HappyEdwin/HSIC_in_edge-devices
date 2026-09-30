import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class SSResNet(nn.Module):
    """
    Spectral-Spatial ResNet (SS-ResNet) for Hyperspectral Image Classification.
    Architecture 100% compliant with AMD-Xilinx DPUCZDX8G (Kria KV260 DPU ISA):
      - Native 2D Convolutions with fused BatchNorm & ReLU
      - Elementwise Residual Addition
      - Native 2D Average Pooling
      - 1x1 Conv classification head (avoids Linear/GEMM boundary issues)
      - Zero Conv3D, zero LayerNorm, zero GELU, zero Attention -> 1 Unified DPU Kernel.
    """
    def __init__(self, in_bands=30, num_classes=16, patch_size=13):
        super(SSResNet, self).__init__()
        self.in_bands = in_bands
        self.num_classes = num_classes
        self.patch_size = patch_size
        
        # Stem: Joint Spectral-Spatial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_bands, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual Stage 1: (B, 64, 13, 13)
        self.layer1 = ResBlock(64, 64, stride=1)
        
        # Residual Stage 2: (B, 64, 13, 13) -> (B, 128, 7, 7)
        self.layer2 = ResBlock(64, 128, stride=2)
        
        # Residual Stage 3: (B, 128, 7, 7) -> (B, 128, 4, 4)
        self.layer3 = ResBlock(128, 128, stride=2)
        
        # Spatial Global/Average Pooling: 4x4 -> 1x1
        self.pool = nn.AvgPool2d(kernel_size=4, stride=1)
        
        # Classification Head using 1x1 Conv (100% DPU friendly)
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        # x shape: (B, in_bands, patch_size, patch_size)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    model = SSResNet(in_bands=30, num_classes=16, patch_size=13)
    dummy = torch.randn(2, 30, 13, 13)
    out = model(dummy)
    params = sum(p.numel() for p in model.parameters())
    print(f"[SSResNet] Input shape: {dummy.shape} -> Output shape: {out.shape}")
    print(f"[SSResNet] Total parameters: {params:,}")
