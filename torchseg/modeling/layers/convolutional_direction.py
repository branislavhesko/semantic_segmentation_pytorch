from typing import List

import einops
import kornia as K
import torch


class ConvBNRELU(torch.nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )


class SeparableConvolutionalBlock(torch.nn.Module):

    def __init__(self, kernel_size, in_channels, out_channels) -> None:
        super().__init__()
        self.horizontal_conv = ConvBNRELU(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=1, padding=(kernel_size // 2, 0))
        self.vertical_conv = ConvBNRELU(in_channels, out_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size // 2))
        self.conv = ConvBNRELU(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.last_conv = torch.nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.horizontal_conv(x), self.vertical_conv(x), self.conv(x)], dim=1)
        return self.last_conv(x)


class DirectionalBlock(torch.nn.Module):

    def __init__(self, angles: torch.Tensor | List, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.dir_block = SeparableConvolutionalBlock(kernel_size, in_channels, out_channels)
        self.angles = angles
        self.output_conv = torch.nn.Conv2d(out_channels * len(angles), out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dir_features = []
        for angle in self.angles:
            rotated_features = K.geometry.rotate(x, angle)
            rotated_features = self.dir_block(rotated_features)
            dir_features.append(K.geometry.rotate(rotated_features.to(x.dtype), -angle))
        return self.output_conv(torch.cat(dir_features, dim=1))


if __name__ == "__main__":
    from tqdm import tqdm
    block = DirectionalBlock(angles=torch.tensor([0, 45, 90, 135]).float().cuda(),in_channels=64, out_channels=384, kernel_size=11).cuda()
    with torch.no_grad():
        for idx in tqdm(range(100)):
            out = block(torch.rand(2, 64, 64, 64).cuda())
        print(out.shape)