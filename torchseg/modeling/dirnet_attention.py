from turtle import forward
import einops
import torch
import torch.nn.functional as F
import timm

from torchseg.modeling.layers.decoder_attention import Decoder


class DirNet(torch.nn.Module):

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnet_b6_ns", pretrained=True, features_only=True)
        self.register_buffer("angles", torch.nn.Parameter(torch.tensor([0, 30, 60, 90, 120, 150]).half(), requires_grad=False))
        self.decoder4 = Decoder(576, 64, 384, angles=self.angles, feature_size=(24, 24))
        self.decoder3 = Decoder(264, 64, 384, angles=self.angles, feature_size=(48, 48))
        self.decoder2 = Decoder(136, 64, 120, angles=self.angles, feature_size=(96, 96))
        self.decoder1 = Decoder(104, 64, 96, angles=None, feature_size=(192, 192))
        self.decoder0 = Decoder(96, 64, 64, angles=None, feature_size=(384, 384))
        self.out_block = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b, c, h, w = image.shape
        feature_list = self.backbone(image)
        features = {}
        for layer_idx, feature in enumerate(feature_list):
            features[f"layer{layer_idx}"] = feature

        decoder4_out = self.decoder4(features["layer4"])
        decoder3_out = self.decoder3(torch.cat([features["layer3"], decoder4_out], dim=1))
        decoder2_out = self.decoder2(torch.cat([features["layer2"], decoder3_out], dim=1))
        decoder1_out = self.decoder1(torch.cat([features["layer1"], decoder2_out], dim=1))
        decoder0_out = self.decoder0(torch.cat([features["layer0"], decoder1_out], dim=1))
        return F.interpolate(self.out_block(decoder0_out), size=(h, w), mode="bilinear", align_corners=False)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    d = DirNet(2).cuda()
    with torch.no_grad():
        out = d(torch.rand(1, 3, 768, 768).cuda()).cpu().numpy()

    plt.imshow(out[0, 0])
    plt.show()