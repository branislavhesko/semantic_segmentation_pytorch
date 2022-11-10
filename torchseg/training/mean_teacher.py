from sre_parse import State
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from torchseg.configuration.config import Config, State
from torchseg.data.drive_dataset import get_data_loaders_semi_supervised

from torchseg.modeling.dirnet_attention import DirNet
from torchseg.modeling.ema import apply_ema
from torchseg.modeling.losses.dice import DiceBCELoss
from torchseg.utils.metrics import MetricMaker


class MeanTeacherTrainer:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.teacher = DirNet(num_classes=self.config.num_classes).to(self.config.device)
        self.student = DirNet(num_classes=self.config.num_classes).to(self.config.device)

        for m in [self.student, self.teacher]:
            m.load_state_dict(torch.load("checkpoint_best.pth"))

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay
        )
        self.data_loaders = get_data_loaders_semi_supervised(self.config)
        self.loss = DiceBCELoss()
        self.metrics = {
            State.train: MetricMaker(["background", "vessel"]),
            State.val: MetricMaker(["background", "vessel"]),
        }
        self.writer = SummaryWriter()

    def train(self):
        best_dice = 0

        for epoch in range(self.config.training.num_epochs):
            self.train_epoch(epoch)
            val_dices = self.validate_epoch(epoch)
            if State.test in self.data_loaders:
                self.test_epoch(epoch)
            if val_dices["vessel"] > best_dice:
                best_dice = val_dices["vessel"]
                print("Found new best dice: {}".format(best_dice))
                torch.save(self.teacher.state_dict(), f"checkpoint_best_teacher.pth")

    def train_epoch(self, epoch):
        self.teacher.eval()
        self.student.train()
        loader = tqdm(self.data_loaders[State.train])
        scaler = torch.cuda.amp.GradScaler()

        for idx, data in enumerate(loader):
            image = data[0].to(self.config.device)
            mask = data[1].to(self.config.device)
            student_indices = [idx for idx, m in enumerate(mask) if m.sum() < 0]
            generated_mask = self._generate_mask_by_teacher(image[student_indices])
            mask[student_indices] = generated_mask
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.config.training.amp_enabled):
                student_output = self.student(image)
                loss = self.loss(student_output, mask)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            apply_ema(self.teacher, self.student, self.config.training.ema_decay)


    def _generate_mask_by_teacher(self, image):
        if image.shape[0] == 0:
            return torch.empty(0, 768, 768, dtype=torch.long, device=self.config.device)
        with torch.no_grad():
            mask = self.teacher(image)
        return mask.argmax(dim=1)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        torch.cuda.empty_cache()
        validation_bar = tqdm(self.data_loaders[State.val])
        self.teacher.eval()
        self.metrics[State.val].reset()

        for idx, data in enumerate(validation_bar):
            image, mask = [d.to(self.config.device) for d in data]
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.teacher(image)
                loss = self.loss(outputs, mask)
            prediction = torch.argmax(outputs, dim=1)
            self.metrics[State.val].update(prediction, mask)
            self.writer.add_scalar("val/loss", loss.item(), epoch * len(self.data_loaders[State.val]) + idx)

            if False and idx % 100 == 0:
                self.writer.add_images("val/images", image, epoch * len(self.data_loaders[State.val]) + idx)
                self.writer.add_images("val/visualization", visualization_binary(mask, prediction), epoch * len(self.data_loaders[State.val]) + idx)
                self.writer.add_figure("val/feature_maps", visualization_feature_maps(outputs), epoch * len(self.data_loaders[State.val]) + idx)
            validation_bar.set_description(f"Epoch: {epoch}")
        for cls_, value in self.metrics[State.val].mean_iou.items():
            self.writer.add_scalar(f"VAL_MEAN_IOU/{cls_}", value, epoch)
            print(f"VAL_MEAN_IOU/{cls_}: {value}")

        for cls_, value in self.metrics[State.val].mean_dice.items():
            self.writer.add_scalar(f"VAL_MEAN_DICE/{cls_}", value, epoch)
            print(f"VAL_MEAN_DICE/{cls_}: {value}")
        return self.metrics[State.val].mean_dice

if __name__ == "__main__":
    config = Config()
    trainer = MeanTeacherTrainer(config)
    trainer.train()