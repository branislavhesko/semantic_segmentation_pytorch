from cgi import test
import glob
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from torchseg.configuration.config import Config
from torchseg.modeling.dirnet_attention import DirNet


config = Config()
config.device = "cpu"
model = DirNet(num_classes=config.num_classes).to(config.device)
path = "checkpoint.pth"
out_path = "out6"
os.makedirs(out_path, exist_ok=True)
images_path = "/home/brani/doktorat/semantic_segmentation_pytorch/data/DRIVE/test/images"

model.load_state_dict(torch.load(path))
model.eval()

for img_file in tqdm(glob.glob(os.path.join(images_path, "*.tif"))):
    img = cv2.cvtColor(cv2.imread(img_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.
    h, w, _ = img.shape
    img = cv2.resize(img, (config.image_size, config.image_size))
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(config.device)

    with torch.no_grad():
        output = model(img).squeeze(0)
        prediction = torch.argmax(output.sigmoid(), dim=0).cpu().numpy()


    prediction = cv2.resize((prediction * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    # prediction = cv2.dilate(prediction, np.ones((3, 3), np.uint8), iterations=1)
    # prediction = cv2.erode(prediction, np.ones((2, 2), np.uint8), iterations=1)
    cv2.imwrite(os.path.join(out_path, os.path.basename(img_file).split("_")[0] + ".png"), prediction)
