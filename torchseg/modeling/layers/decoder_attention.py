from turtle import forward
from typing import List
import torch

from torchseg.modeling.layers.convolutional_direction import ConvBNRELU
from torchseg.modeling.layers.directional_block import DirectionalBlock


class SegNextBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.depth_ini = ConvBNRELU(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_square = ConvBNRELU(in_channels=out_channels, out_channels=out_channels, kernel_size=(5, 5), stride=1, padding=2)
        self.conv1 = torch.nn.Sequential(
            ConvBNRELU(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 7), stride=1, padding=(0, 3), groups=out_channels // 2),
            ConvBNRELU(in_channels=out_channels, out_channels=out_channels, kernel_size=(7, 1), stride=1, padding=(3, 0), groups=out_channels // 2),
         )
        self.conv2 = torch.nn.Sequential(
            ConvBNRELU(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 11), stride=1, padding=(0, 5), groups=out_channels // 2),
            ConvBNRELU(in_channels=out_channels, out_channels=out_channels, kernel_size=(11, 1), stride=1, padding=(5, 0), groups=out_channels // 2),
        )
        self.conv3 = torch.nn.Sequential(
            ConvBNRELU(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 21), stride=1, padding=(0, 10), groups=out_channels // 2),
            ConvBNRELU(in_channels=out_channels, out_channels=out_channels, kernel_size=(21, 1), stride=1, padding=(10, 0), groups=out_channels // 2),
        )
        self.depthwise = ConvBNRELU(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, image):
        image = self.depth_ini(image)
        x = self.conv_square(image)
        c1, c2, c3 = (conv(x) for conv in (self.conv1, self.conv2, self.conv3))
        attn = self.depthwise(c1 + c2 + c3)
        return attn * image



class Decoder(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            embed_dim,
            feature_size: torch.Size | List[int],
            angles: torch.Tensor | None = None,
            num_heads: int | None =None,
        ) -> None:
        super().__init__()
        if angles is not None:
            # TODO: add batchnorm + relu?
            self.directional_block = DirectionalBlock(angles, in_channels=in_channels, out_channels=embed_dim, feature_size=feature_size)

        else:

            self.directional_block = SegNextBlock(in_channels=in_channels, out_channels=embed_dim)


        self.decoder_layer = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.directional_block(x)
        return self.decoder_layer(x)


if __name__ == "__main__":
    m = SegNextBlock(128, 128)
    print(m(torch.rand(2, 128, 128, 128)))