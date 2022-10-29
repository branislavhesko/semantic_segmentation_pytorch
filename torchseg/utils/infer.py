import copy
import glob
import os

import cv2
import einops
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from skimage.morphology import skeletonize, label

from torchseg.configuration.config import Config
from torchseg.modeling.combine_net import CombineNet
from torchseg.modeling.dirnet_attention import DirNet
from torchseg.modeling.deeplab_torchvision import DeepLabV3
from torchseg.utils.resize import resize, interp_methods
from torchseg.utils.visualization import visualization_binary, visualization_feature_maps


def dice(pred, gt):
    return 2 * (pred * gt).sum() / (pred.sum() + gt.sum())

def sensitivity(pred, gt):
    return (pred * gt).sum() / gt.sum()


config = Config()
config.device = "cuda"
model = DirNet(num_classes=config.num_classes).to(config.device)
path = "checkpoint_best.pth"
out_path = "best"
os.makedirs(out_path, exist_ok=True)
images_path = "/home/brani/doktorat/semantic_segmentation_pytorch/data/DRIVE/test/images"

model.load_state_dict(torch.load(path))
model.eval()
dices = []
for img_file in tqdm(glob.glob(os.path.join(images_path, "*.tif"))):
    img_orig = cv2.cvtColor(cv2.imread(img_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.
    mask = np.asarray(Image.open(img_file.replace("images", "1st_manual").replace("_test.tif", "_manual1.gif")))
    horig, worig, _ = img_orig.shape
    img = resize(img_orig, out_shape=(config.image_size, config.image_size), interp_method=interp_methods.linear)
    h, w, _ = img.shape
    tiles = einops.rearrange(img, "(tile_h h) (tile_w w) c -> (tile_h tile_w) h w c", tile_h=1, tile_w=1)
    tiles = torch.from_numpy(tiles).float().to(config.device).permute(0, 3, 1, 2)
    tiles = torch.nn.functional.interpolate(tiles, size=(config.image_size, config.image_size), mode="bilinear", align_corners=False)
    with torch.no_grad():
        output = model(tiles).sigmoid()
        #output[:, 1, ...] += 0.04
        prediction = torch.argmax(output, dim=1).cpu().numpy()

    prediction = einops.rearrange(prediction, "(tile_h tile_w) h w -> (tile_h h) (tile_w w)", tile_h=1, tile_w=1)
    # skeleton = skeletonize(prediction2)
    #skeleton = label(skeleton)
    #max_label = np.argmax([np.sum(skeleton == i) for i in range(1, np.max(skeleton) + 1)]) + 1
    #skeleton = skeleton == max_label
    # prediction = cv2.erode(prediction.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(2, 2)), iterations=1)
    prediction = resize(prediction.astype(np.uint8) * 255, out_shape=(horig, worig), interp_method=interp_methods.linear)
    #skeleton = cv2.resize(skeleton.astype(np.uint8), (worig, horig), interpolation=cv2.INTER_LINEAR)

    vis = np.zeros((horig, worig, 3), dtype=np.uint8)
    vis[prediction > 128, 1] = 255
    vis[mask > 128, 2] = 255
    dices.append(dice(prediction > 128, mask > 0))
    print(dices[-1])

    cv2.imwrite(os.path.join(out_path, os.path.basename(img_file).split("_")[0] + ".png"), prediction)

    # cv2.imwrite(os.path.join(out_path, os.path.basename(img_file).split("_")[0] + "_s.png"), skeleton)

print(np.mean(dices))