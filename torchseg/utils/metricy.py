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

    def update(self, pred, target, num_classes):
        dices = dice(pred, target, num_classes)
        ious = iou(pred, target, num_classes)
        for cls in range(num_classes):
            self.dices[cls].append(dices[cls])
            self.ious[cls].append(ious[cls])
