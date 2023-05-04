import configs.discovery as config
from data.mask_feature_dataset import MaskData

import argparse
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

# from torchmetrics.detection.mean_ap import MeanAveragePrecision


class LitDiscovery(pl.LightningModule):
    """Lightning module for training the discovery network."""

    def __init__(self, device):
        super().__init__()
        self.model = self.load_model(device)
        # self.criterion = self.set_criterion(device)
        # self.map = MeanAveragePrecision(box_format="xywh", iou_type="bbox")

    def training_step(self, batch, batch_idx):
        # Get outputs from model
        # Get loss
        # Log the loss
        # Return the loss
        pass

    def validation_step(self, batch, batch_idx):
        # Get outputs from model
        # Get loss
        # Log the loss
        # Calculate / update / log evaluation metric
        pass

    def on_validation_epoch_end(self):
        # Calculate and log evaluation metric over whole dataset
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.lr)  # , weight_decay=config.weight_decay)

        return optimizer

    def load_model(self, device):
        # Load model
        # Model to device
        # Return model
        pass

    # def set_criterion(self, device):
    #     """Use the DETR loss (but only the classification part)."""
    #     # Default DETR values
    #     eos_coef = 0.05  # Was 0.1
    #     weight_dict = {"loss_ce": 1, "loss_bbox": 5}
    #     weight_dict["loss_giou"] = 2

    #     losses = ["labels"]  # , "boxes", "cardinality"]

    #     matcher = HungarianMatcher()
    #     criterion = SetCriterion(
    #         self.model.num_classes, matcher, weight_dict=weight_dict, eos_coef=eos_coef, losses=losses
    #     )
    #     criterion.to(device)

    #     return criterion


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
    # Make datasets
    # Make dataloaders
    # Return dataloaders
    pass


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

    model = LitDiscovery(config.device)

    # Trainer callbacks.
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        save_top_k=1,
        monitor="map_all",  # TODO: implement evaluation
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
        ],
    )

    trainer.fit(model, dataloader_train, dataloader_val)

    # model = LitFullySupervisedClassifier.load_from_checkpoint(
    #     "checkpoints/epoch=499-step=500.ckpt", device=config.device
    # )
    # trainer.validate(model, dataloader_val)
