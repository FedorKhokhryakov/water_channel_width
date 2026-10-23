"""U-Net для сегментации кадров кормовой камеры (4 класса)."""
import torch
import torch.nn as nn

__all__ = ["DoubleConv", "UNet"]


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net с 5 уровнями (энкодер d1..d5, декодер u4..u1).

    Вход:  (N, in_channels, H, W), H и W кратны 16.
    Выход: (N, n_classes, H, W) — логиты.
    """

    def __init__(self, n_classes=4, in_channels=3, base_c=32):
        super().__init__()
        self.d1 = DoubleConv(in_channels, base_c)
        self.d2 = DoubleConv(base_c, base_c*2)
        self.d3 = DoubleConv(base_c*2, base_c*4)
        self.d4 = DoubleConv(base_c*4, base_c*8)
        self.d5 = DoubleConv(base_c*8, base_c*16)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.u4 = DoubleConv(base_c*24, base_c*8)
        self.u3 = DoubleConv(base_c*12, base_c*4)
        self.u2 = DoubleConv(base_c*6, base_c*2)
        self.u1 = DoubleConv(base_c*3, base_c)

        self.final = nn.Conv2d(base_c, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        x4 = self.d4(self.pool(x3))
        x5 = self.d5(self.pool(x4))

        y4 = self.up(x5)
        y4 = torch.cat([y4, x4], dim=1)
        y4 = self.u4(y4)

        y3 = self.up(y4)
        y3 = torch.cat([y3, x3], dim=1)
        y3 = self.u3(y3)

        y2 = self.up(y3)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.u2(y2)

        y1 = self.up(y2)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.u1(y1)

        out = self.final(y1)
        return out
