import torch


def iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu() + target_inds.long().sum().data.cpu() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious


def dice(pred, target, num_classes):
    dice = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()  # Cast to long to prevent overflows
        dice.append(float(2 * intersection) / (pred_inds.long().sum().data.cpu() + target_inds.long().sum().data.cpu()))
    return dice


class MetricMaker:

    def __init__(self, classes) -> None:
        self.dices = {cls: [] for cls in classes}
        self.ious = {cls: [] for cls in classes}
        self.classes = classes

    def update(self, pred, target):
        dices = dice(pred, target, len(self.classes))
        ious = iou(pred, target, len(self.classes))
        for idx, cls in enumerate(self.classes):
            self.dices[cls].append(dices[idx])
            self.ious[cls].append(ious[idx])

    @property
    def last_dice(self):
        return {cls: self.dices[cls][-1] for cls in self.classes}

    @property
    def last_iou(self):
        return {cls: self.ious[cls][-1] for cls in self.classes}

    @property
    def mean_dice(self):
        return {cls: torch.tensor(self.dices[cls]).mean().item() for cls in self.classes}

    @property
    def mean_iou(self):
        return {cls: torch.tensor(self.ious[cls]).mean().item() for cls in self.classes}

    def reset(self):
        self.dices = {cls: [] for cls in self.classes}
        self.ious = {cls: [] for cls in self.classes}
