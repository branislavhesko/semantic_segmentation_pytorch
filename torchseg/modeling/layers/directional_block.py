from typing import List

import einops
import kornia as K
import torch

from torchseg.modeling.layers.convolutional_direction import ConvBNRELU


class DirectionalBlock(torch.nn.Module):

    def __init__(self, angles: torch.Tensor | List, in_channels:int, out_channels: int, feature_size: torch.Size | List[int]) -> None:
        super().__init__()
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=192, nhead=8, batch_first=True), num_layers=4
        )
        self.angles = angles
        self.input_conv = torch.nn.Sequential(torch.nn.Conv2d(in_channels, 192, kernel_size=3, padding=1), torch.nn.BatchNorm2d(192), torch.nn.ReLU(inplace=True))
        self.conv_block = torch.nn.Sequential(ConvBNRELU(192, 192, kernel_size=3, padding=1, stride=1), ConvBNRELU(192, 192, kernel_size=3, padding=1, stride=1))
        self.output_conv = torch.nn.Conv2d(192 * (len(angles) + 1), out_channels, kernel_size=3, padding=1)
        self.positional_encoding = torch.nn.parameter.Parameter(torch.randn(1, 192, feature_size[0], feature_size[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dir_features = []
        x = self.input_conv(x)
        self.angles.data = self.angles.data.to(x.dtype)
        self.positional_encoding.data = self.positional_encoding.data.to(x.dtype)
        for angle in self.angles:
            rotated_features = K.geometry.rotate(x + self.positional_encoding.data, angle)
            rotated_features = einops.rearrange(rotated_features, "b c h w -> (b h) w c")
            rotated_features = self.transformer(rotated_features)
            rotated_features = einops.rearrange(rotated_features, "(b h) w c -> b c h w", h=x.shape[2])

            dir_features.append(K.geometry.rotate(rotated_features.to(x.dtype), -angle))
        dir_features.append(self.conv_block(x))
        return self.output_conv(torch.cat(dir_features, dim=1))


if __name__ == "__main__":
    from tqdm import tqdm
    block = DirectionalBlock(angles=torch.tensor([0, 45, 90, 135]).float().cuda(), out_channels=384).cuda()
    with torch.no_grad():
        for idx in tqdm(range(100)):
            out = block(torch.rand(2, 64, 64, 64).cuda())
        print(out.shape)