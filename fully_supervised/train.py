import configs.fully_supervised as config
from data.img_embeds_dataset import ImageEmbeds
from modelling.sam_mask_generator import OWSamMaskGenerator
from fully_supervised.model import FullySupervisedClassifier

from segment_anything import sam_model_registry
from modelling.criterion import SetCriterion
from modelling.matcher import HungarianMatcher

import argparse
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    # ModelSummary,
)

# from torchmetrics.detection.mean_ap import MeanAveragePrecision


class LitFullySupervisedClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.criterion = Criterion()
        # self.map = MeanAveragePrecision(iou_type="segm")  # , class_metrics=per_class)
        self.criterion = self.set_criterion()

        self.validation_step_gt = []
        self.validation_step_pred = []

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)

        # loss = outputs.loss
        loss = self.criterion(outputs)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss = self.criterion(outputs)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Check if any predictions were made.
        # self.map.update(pred_metric_input, gt_metric_input)

    def on_validation_epoch_end(self):
        pass
        # mAPs = {"val_" + k: v for k, v in self.map.compute().items()}
        # self.print(mAPs)
        # self.log_dict(mAPs, sync_dist=True)
        # self.map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.lr)  # , weight_decay=config.weight_decay)

        return optimizer

    def set_criterion(self):
        eos_coef = 0.1
        weight_dict = {"loss_ce": 1, "loss_bbox": 5}
        weight_dict["loss_giou"] = 2

        losses = ["labels"]  # , "boxes", "cardinality"]

        matcher = HungarianMatcher()
        criterion = SetCriterion(
            self.model.num_classes - 1, matcher, weight_dict=weight_dict, eos_coef=eos_coef, losses=losses
        )
        criterion.to(self.model.device)

        return criterion


def parse_args():
    parser = argparse.ArgumentParser(prog="Supervised Mask2Formers")
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        # required=True,
        help="Directory to store model checkpoints/",
    )
    parser.add_argument(
        "-l",
        "--log-dir",
        # required=True,
        help="Directory to store (Tensorboard) logging.",
    )
    parser.add_argument("-n", "--num-gpus", type=int, help="Number of GPUs to use.")

    args = parser.parse_args()
    return args


def load_data():
    dataset_train = ImageEmbeds(config.data_dir, "train")
    dataset_val = ImageEmbeds(config.data_dir, "val")

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=ImageEmbeds.collate_fn,
        num_workers=config.num_workers,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=ImageEmbeds.collate_fn,
        num_workers=config.num_workers,
    )

    return dataloader_train, dataloader_val


def load_model():
    sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device=config.device)

    mask_generator = OWSamMaskGenerator(sam)

    model = FullySupervisedClassifier(mask_generator, config.num_layers, config.hidden_dim, config.num_classes)

    return model


if __name__ == "__main__":
    args = parse_args()

    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.log_dir = args.log_dir
    if args.num_gpus:
        config.num_devices = args.num_gpus

    pl.seed_everything(config.seed, workers=True)

    dataloader_train, dataloader_val = load_data()

    model = load_model()
    model = LitFullySupervisedClassifier(model)

    # Trainer callbacks.
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        save_top_k=1,
        monitor="val_map",
        mode="max",
        filename="best_model_{epoch}",
    )
    checkpoint_callback = ModelCheckpoint(dirpath=config.checkpoint_dir, every_n_epochs=config.save_every)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # model_summary = ModelSummary()

    trainer = pl.Trainer(
        # fast_dev_run=16,
        # limit_train_batches=0.001,  # FIXME: remove this for actual training!
        # limit_val_batches=0.001,
        default_root_dir=config.checkpoint_dir,
        logger=pl.loggers.tensorboard.TensorBoardLogger(save_dir=config.log_dir),
        accelerator=config.device,
        log_every_n_steps=1,
        devices=config.num_devices,
        enable_checkpointing=True,
        max_epochs=config.epochs,
        # gradient_clip_val=config.clip,
        # gradient_clip_algorithm="value",
        callbacks=[
            best_checkpoint_callback,
            checkpoint_callback,
            lr_monitor,
            # model_summary,
        ],
    )

    trainer.fit(model, dataloader_train, dataloader_val)
    # trainer.validate(model, dataloader_val)
