import configs.discovery as config
from data.datasets.mask_feature_dataset import ImageMaskData, DiscoveryImageMaskData
from discovery.discovery_model import DiscoveryModel

import argparse
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import CombinedLoader

# from torchmetrics.detection.mean_ap import MeanAveragePrecision


class LitDiscovery(pl.LightningModule):
    """Lightning module for training the discovery network."""

    def __init__(self, device):
        super().__init__()
        self.model = self.load_model(device)
        # self.criterion = self.set_criterion(device)
        # self.map = MeanAveragePrecision(box_format="xywh", iou_type="bbox")

    def training_step(self, batch, batch_idx):
        # FIXME: padding is weird now, since we don't have a bg class anymore
        loss, supervised_loss, discovery_loss = self.model(batch["labeled"], batch["unlabeled"])
        supervised_loss = {"train_" + k: v for k, v in supervised_loss.items()}
        discovery_loss = {"train_" + k: v for k, v in discovery_loss.items()}

        self.log_dict(
            supervised_loss,
            batch_size=len(batch["labeled"]["boxes"]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "discovery_loss",
            discovery_loss["train_discovery_loss"],
            batch_size=len(batch["labeled"]["boxes"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_total_loss",
            loss,
            batch_size=len(batch["labeled"]["boxes"]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: handle sequential combined dataloading
        # TODO: Calculate / update / log evaluation metric
        loss, supervised_loss, discovery_loss = self.model(batch)

        supervised_loss = {"val_" + k: v for k, v in supervised_loss.items()}
        discovery_loss = {"val_" + k: v for k, v in discovery_loss.items()}

        self.log_dict(
            supervised_loss,
            batch_size=len(batch["labeled"]["boxes"]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log_dict(
            discovery_loss,
            batch_size=len(batch["labeled"]["boxes"]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "val_total_loss",
            loss,
            batch_size=len(batch["labeled"]["boxes"]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_validation_epoch_end(self):
        # TODO: Calculate and log evaluation metric over whole dataset
        pass

    def configure_optimizers(self):
        # TODO: RNCDL uses SGD here.
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.lr)

        return optimizer

    def load_model(self, device):
        model = DiscoveryModel("checkpoints/rpn_TUMlike/best_model_epoch=45.ckpt")
        model.to(device)

        return model


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
    # For using multiple dataloaders, see --> https://lightning.ai/docs/pytorch/latest/data/iterables.html#multiple-dataloaders
    # TODO: return labeled and unlabeled dataloader here.
    dataset_train_labeled = ImageMaskData(
        config.masks_dir, config.ann_train_labeled, config.img_train, config.device, train=True, pad_num=config.pad_num
    )
    dataset_val_labeled = ImageMaskData(
        config.masks_dir, config.ann_val_labeled, config.img_val, config.device, train=False, pad_num=config.pad_num
    )
    dataloader_train_labeled = DataLoader(
        dataset_train_labeled,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=dataset_train_labeled.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    dataloader_val_labeled = DataLoader(
        dataset_val_labeled,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=dataset_val_labeled.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    dataset_train_unlabeled = DiscoveryImageMaskData(
        config.masks_dir,
        config.ann_train_unlabeled,
        config.img_train,
        config.device,
        train=True,
        pad_num=config.pad_num,
        num_views=config.num_views,
    )
    dataset_val_unlabeled = DiscoveryImageMaskData(
        config.masks_dir, config.ann_val_unlabeled, config.img_val, config.device, train=False, pad_num=config.pad_num
    )
    dataloader_train_unlabeled = DataLoader(
        dataset_train_unlabeled,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=dataset_train_unlabeled.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    dataloader_val_unlabeled = DataLoader(
        dataset_val_unlabeled,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=dataset_val_unlabeled.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    dataloader_train = CombinedLoader(
        {"labeled": dataloader_train_labeled, "unlabeled": dataloader_train_unlabeled}, mode="max_size_cycle"
    )
    dataloader_val = CombinedLoader(
        {"labeled": dataloader_val_labeled, "unlabeled": dataloader_val_unlabeled}, mode="sequential"
    )

    return dataloader_train, dataloader_val


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
        gradient_clip_val=config.clip,
        gradient_clip_algorithm=config.clip_type,
        callbacks=[
            best_checkpoint_callback,
            checkpoint_callback,
        ],
    )

    trainer.fit(model, dataloader_train)  # , dataloader_val)  # FIXME: fix validation step and put val dataloader back

    # model = LitFullySupervisedClassifier.load_from_checkpoint(
    #     "checkpoints/epoch=499-step=500.ckpt", device=config.device
    # )
    # trainer.validate(model, dataloader_val)
