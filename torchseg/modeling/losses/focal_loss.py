import torch


class FocalLoss(torch.nn.Module):

    def __init__(self, alfa=2., beta=4.):
        super().__init__()
        self._alfa = alfa
        self._beta = beta

    def forward(self, output, labels):
        output = torch.clamp(output.sigmoid(), 0, 1 - 1e-4)
        labels_layered = torch.zeros_like(output)
        for idx in range(output.shape[1]):
            labels_layered[:, idx, :, :][labels == idx] = 1
        loss_point = torch.mean((1 - output[
            labels_layered == 1.]) ** self._alfa * torch.log(output[labels_layered == 1.] + 1e-5))
        loss_background = torch.mean((1 - labels_layered) ** self._beta * output ** self._alfa * torch.log(1 - output + 1e-5))
        return -1 * (loss_point + loss_background)
