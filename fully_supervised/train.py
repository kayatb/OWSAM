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
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class LitFullySupervisedClassifier(pl.LightningModule):
    """Lightning module for training the fully supervised classification head."""

    def __init__(self, device):
        super().__init__()
        self.model = self.load_model(device)
        self.criterion = self.set_criterion(device)
        self.map = MeanAveragePrecision(bbox_format="xywh", iou_type="bbox")  # TODO: can also calculate for segm masks.

        self.validation_step_gt = []
        self.validation_step_pred = []

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss = self.criterion(outputs, batch["targets"])

        self.log("train_class_error", loss["class_error"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_ce", loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss = self.criterion(outputs, batch["targets"])

        self.log("val_class_error", loss["class_error"], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_loss_ce", loss["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        pred_metric_input = self.get_map_format(outputs)
        self.map.update(pred_metric_input, batch["targets"])

    def on_validation_epoch_end(self):
        mAPs = {"val_" + k: v for k, v in self.map.compute().items()}
        self.print(mAPs)
        self.log_dict(mAPs, sync_dist=True)
        self.map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.lr)  # , weight_decay=config.weight_decay)

        return optimizer

    def load_model(self, device):
        sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")

        mask_generator = OWSamMaskGenerator(sam)

        model = FullySupervisedClassifier(mask_generator, config.num_layers, config.hidden_dim, config.num_classes)
        sam.to(device=device)  # Only do this after the mask decoder has been changed by the generator!
        model.to(device)

        # self.model.sam_generator.predictor.model.to(device)
        # self.model.sam_generator.predictor.model.prompt_encoder.to(device)
        # self.model.sam_generator.predictor.model.mask_decoder.to(device)

        return model

    def set_criterion(self, device):
        """Use the DETR loss (but only the classification part)."""
        eos_coef = 0.1
        weight_dict = {"loss_ce": 1, "loss_bbox": 5}
        weight_dict["loss_giou"] = 2

        losses = ["labels"]  # , "boxes", "cardinality"]

        matcher = HungarianMatcher()
        criterion = SetCriterion(
            self.model.num_classes - 1, matcher, weight_dict=weight_dict, eos_coef=eos_coef, losses=losses
        )
        criterion.to(device)

        return criterion

    def get_map_format(self, outputs):
        """Convert the data to the format that the metric will accept:
        a list of dictionaries, where each dictionary corresponds to a single image.
        Args:
            outputs: dict containing model outputs with keys `masks`, `pred_logits`, and `pred_boxes`

        Returns:
            pred_conv: list of dicts, where each dict contains `boxes`, `scores` and `labels`
        """
        pred_conv = []

        for i in range(outputs["pred_logits"].shape[0]):  # Loop over the batches
            pred_dict = {
                "boxes": outputs["pred_boxes"][i],
                "scores": outputs["iou_scores"][i],
                "labels": torch.argmax(outputs["pred_logits"][i], dim=1),  # Get labels from the logits
            }
            pred_conv.append(pred_dict)

        return pred_conv


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
    parser.add_argument("-e", "--embed-dir", help="Directory where the pre-extracted image embeddings are saved.")
    parser.add_argument("-n", "--num-gpus", type=int, help="Number of GPUs to use.")

    args = parser.parse_args()
    return args


def load_data():
    dataset_train = ImageEmbeds(config.embeds_train, config.ann_train, config.device)
    dataset_val = ImageEmbeds(config.embeds_val, config.ann_val, config.device)

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


if __name__ == "__main__":
    args = parse_args()

    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.log_dir = args.log_dir
    if args.embed_dir:
        config.embed_dir = args.embed_dir
    if args.num_gpus:
        config.num_devices = args.num_gpus

    pl.seed_everything(config.seed, workers=True)

    dataloader_train, dataloader_val = load_data()

    model = LitFullySupervisedClassifier(config.device)

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
        # fast_dev_run=True,
        # limit_train_batches=0.5,  # FIXME: remove this for actual training!
        # limit_val_batches=0.5,
        default_root_dir=config.checkpoint_dir,
        logger=pl.loggers.tensorboard.TensorBoardLogger(save_dir=config.log_dir),
        accelerator="gpu" if config.device == "cuda" else "cpu",
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
