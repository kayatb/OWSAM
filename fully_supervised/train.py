import configs.fully_supervised as config
from data.datasets.mask_feature_dataset import MaskData
from fully_supervised.model import LinearClassifier
from fully_supervised.coco_eval import CocoEvaluator

from modelling.criterion import SetCriterion
from modelling.matcher import HungarianMatcher

import argparse
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    # LearningRateMonitor,
    # ModelSummary,
)


class LitFullySupervisedClassifier(pl.LightningModule):
    """Lightning module for training the fully supervised classification head."""

    def __init__(self, device, label_map, tune_config=None):
        super().__init__()

        if tune_config:  # If a tuner config is given, override the default config values.
            config.num_layers = tune_config["num_layers"]
            config.hidden_dim = tune_config["hidden_dim"]
            config.lr = tune_config["lr"]

        self.label_map = label_map  # Mapping from continuous ids back to original annotated ones.

        self.model = self.load_model(device)
        self.criterion = self.set_criterion(device)
        self.evaluator = CocoEvaluator(config.ann_val, ["bbox"])
        # self.evaluator.coco_eval["bbox"].params.useCats = 0  # For calculating object vs no-object mAP

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss = self.criterion(outputs, batch["targets"])

        self.log(
            "train_class_error",
            loss["class_error"].item(),
            batch_size=len(batch["boxes"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss_ce",
            loss["loss"].item(),
            batch_size=len(batch["boxes"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss = self.criterion(outputs, batch["targets"])

        self.log(
            "val_class_error",
            loss["class_error"].item(),
            batch_size=len(batch["boxes"]),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_loss_ce",
            loss["loss"].item(),
            batch_size=len(batch["boxes"]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        results = CocoEvaluator.to_coco_format(batch["img_ids"], outputs, self.label_map)
        self.evaluator.update(results)

    def on_validation_epoch_end(self):
        self.evaluator.synchronize_between_processes()
        self.evaluator.accumulate()
        results = self.evaluator.summarize()

        self.log_dict(results["bbox"])

        self.evaluator.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.lr)  # , weight_decay=config.weight_decay)

        return optimizer

    def load_model(self, device):
        model = LinearClassifier(config.num_layers, config.hidden_dim, config.num_classes, pad_num=config.pad_num)
        model.to(device)

        return model

    def set_criterion(self, device):
        """Use the DETR loss (but only the classification part)."""
        # Default DETR values
        eos_coef = 0.05  # Was 0.1
        weight_dict = {"loss_ce": 1, "loss_bbox": 5}
        weight_dict["loss_giou"] = 2

        losses = ["labels"]  # , "boxes", "cardinality"]

        matcher = HungarianMatcher()
        criterion = SetCriterion(
            self.model.num_classes, matcher, weight_dict=weight_dict, eos_coef=eos_coef, losses=losses
        )
        criterion.to(device)

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


def load_data(batch_size=None):
    dataset_train = MaskData(config.masks_train, config.ann_train, config.device, pad_num=config.pad_num)
    dataset_val = MaskData(config.masks_val, config.ann_val, config.device, pad_num=config.pad_num)

    bs = batch_size if batch_size else config.batch_size  # Override config if batch_size is given.

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=bs,
        shuffle=True,
        collate_fn=MaskData.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=bs,
        shuffle=False,
        collate_fn=MaskData.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    return dataloader_train, dataloader_val, dataset_val.continuous_to_cat_id


if __name__ == "__main__":
    args = parse_args()

    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.log_dir = args.log_dir
    if args.num_gpus:
        config.num_devices = args.num_gpus

    pl.seed_everything(config.seed, workers=True)

    dataloader_train, dataloader_val, label_map = load_data()

    model = LitFullySupervisedClassifier(config.device, label_map)

    # Trainer callbacks.
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        save_top_k=1,
        monitor="map",
        mode="max",
        filename="best_model_{epoch}",
    )
    checkpoint_callback = ModelCheckpoint(dirpath=config.checkpoint_dir, every_n_epochs=config.save_every)

    trainer = pl.Trainer(
        # fast_dev_run=3,
        # limit_train_batches=0.5,  # FIXME: remove this for actual training!
        # limit_val_batches=0.5,
        default_root_dir=config.checkpoint_dir,
        logger=pl.loggers.tensorboard.TensorBoardLogger(save_dir=config.log_dir),
        accelerator="gpu" if config.device == "cuda" else "cpu",
        # log_every_n_steps=1,
        devices=config.num_devices,
        enable_checkpointing=True,
        max_epochs=config.epochs,
        # gradient_clip_val=config.clip,
        # gradient_clip_algorithm="value",
        callbacks=[
            best_checkpoint_callback,
            checkpoint_callback,
            # lr_monitor,
            # model_summary,
        ],
        # profiler="simple",
    )

    # trainer.fit(model, dataloader_train, dataloader_val)

    model = LitFullySupervisedClassifier.load_from_checkpoint(
        "checkpoints/10_512_1e-4/best_model_epoch=23.ckpt", device=config.device, label_map=label_map
    )
    trainer.validate(model, dataloader_val)
