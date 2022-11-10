from turtle import forward
import einops
import torch
import torch.nn.functional as F
import timm

from torchseg.modeling.layers.decoder_attention import Decoder
from torchseg.modeling.segnext import MSCAN
from torchvision.models import efficientnet_v2_m
from torchvision.models.feature_extraction import create_feature_extractor


def get_backbone(backbone_name):
    if backbone_name == "efficientnetv2_m":
        m1 = efficientnet_v2_m(pretrained=True)
        model = create_feature_extractor(
            m1, {
                'features.1.2.add': 'layer0',
                'features.2.4.add': 'layer1',
                'features.3.4.add': 'layer2',
                'features.5.13.add': 'layer3',
                'features.8': 'layer4'
        })
        return model
    if backbone_name == "resnet50":
        model = timm.create_model("resnet50", pretrained=True, features_only=True)
        return model

    if backbone_name == "segnext":
        model = MSCAN()
        return model



class DirNet(torch.nn.Module):
    IMG_SIZE: int = 768

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = get_backbone("resnet50")
        self.angles = torch.nn.Parameter(torch.tensor([0., 30., 60., 90., 120., 150.]), requires_grad=False)
        self.decoder4 = Decoder(512, 256, 192, angles=self.angles, feature_size=(self.IMG_SIZE // 32, self.IMG_SIZE // 32))
        self.decoder3 = Decoder(512, 256, 192, angles=self.angles, feature_size=(self.IMG_SIZE // 16, self.IMG_SIZE // 16))
        self.decoder2 = Decoder(384, 192, 192, angles=self.angles, feature_size=(self.IMG_SIZE // 8, self.IMG_SIZE // 8))
        self.decoder1 = Decoder(256, 96, 96, angles=None, feature_size=(self.IMG_SIZE // 4, self.IMG_SIZE // 4))
        self.decoder0 = Decoder(96, 64, 64, angles=None, feature_size=(self.IMG_SIZE // 2, self.IMG_SIZE // 2))
        self.out_block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b, c, h, w = image.shape
        feature_list = self.backbone(image)
        self.angles.data = self.angles.data.to(image.dtype)
        features = {}
        for layer_idx, feature in enumerate(feature_list):
            features[f"layer{layer_idx + 1}"] = feature

        decoder4_out = self.decoder4(features["layer4"])
        decoder3_out = self.decoder3(torch.cat([features["layer3"], decoder4_out], dim=1))
        decoder2_out = self.decoder2(torch.cat([features["layer2"], decoder3_out], dim=1))
        decoder1_out = self.decoder1(torch.cat([features["layer1"], decoder2_out], dim=1))
        decoder0_out = self.decoder0(decoder1_out)
        return F.interpolate(self.out_block(decoder0_out), size=(h, w), mode="bilinear", align_corners=False)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    d = DirNet(2).cuda()
    with torch.no_grad():
        out = d(torch.rand(1, 3, 768, 768).cuda()).cpu().numpy()

    plt.imshow(out[0, 0])
    plt.show()