import torch

from torchseg.modeling.layers.directional_block import DirectionalBlock


class Decoder(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            embed_dim,
            angles: torch.Tensor | None = None,
            num_heads: int | None =None
        ) -> None:
        super().__init__()
        if angles is not None:
            # TODO: add batchnorm + relu?
            self.directional_block = DirectionalBlock(angles, in_channels=in_channels, out_channels=embed_dim)

        else:
            self.directional_block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(embed_dim),
                torch.nn.ReLU(inplace=True),
            )


        self.decoder_layer = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.directional_block(x)
        return self.decoder_layer(x)


if __name__ == "__main__":
    d = Decoder(64, 64, 384, angles=torch.tensor([0, 45, 90, 135]).float().cuda()).cuda()
    with torch.no_grad():
        print(d(torch.rand(2, 64, 64, 64).cuda()).shape)