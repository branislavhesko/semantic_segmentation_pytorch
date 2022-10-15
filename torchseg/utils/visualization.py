import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch


def visualization_binary(gt, prediction):
    assert len(gt.shape) == 3, "GT should be in shape batch, height, width"
    visualization = torch.zeros(gt.shape[0], 3, *gt.shape[1:], dtype=torch.float32)
    visualization[:, 1, ...][prediction > 0] = 1
    visualization[:, 0, ...][gt > 0] = 1
    return visualization


def visualization_feature_maps(model_output):
    fig = plt.figure(dpi=200)
    idx = 0
    for i in range(model_output.shape[0]):
        for j in range(model_output.shape[1]):
            idx += 1
            plt.subplot(model_output.shape[0], model_output.shape[1], idx)
            plt.imshow(model_output[i, j, ...].detach().cpu().numpy())
    return fig