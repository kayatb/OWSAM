import configs.discovery as config
from data.mask_feature_dataset import MaskData
from discovery.discovery_network import DiscoveryClassifier

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
        loss = self.model(batch)
        self.log_dict(loss, batch_size=len(batch["masks"]), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log_dict(loss, batch_size=len(batch["masks"]), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # TODO: Calculate / update / log evaluation metric

    def on_validation_epoch_end(self):
        # TODO: Calculate and log evaluation metric over whole dataset
        pass

    def configure_optimizers(self):
        # TODO: RNCDL uses SGD here.
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.lr)

        return optimizer

    def load_model(self, device):
        model = DiscoveryClassifier(
            num_labeled=config.num_labeled,
            num_unlabeled=config.num_unlabeled,
            feat_dim=config.feat_dim,
            hidden_dim=config.hidden_dim,
            proj_dim=config.proj_dim,
            num_views=config.num_views,
            memory_batches=config.memory_batches,
            items_per_batch=config.items_per_batch,
            memory_patience=config.memory_patience,
            num_iters_sk=config.num_iters_sk,
            epsilon_sk=config.epsilon_sk,
            temperature=config.temperature,
            batch_size=config.batch_size,
            num_hidden_layers=config.num_layers,
        )
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
    dataset_train_labeled = MaskData(config.masks_train, config.ann_train, config.device, pad_num=config.pad_num)
    dataset_val_labeled = MaskData(config.masks_val, config.ann_val, config.device, pad_num=config.pad_num)

    dataloader_train_labeled = DataLoader(
        dataset_train_labeled,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=MaskData.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    dataloader_val_labeled = DataLoader(
        dataset_val_labeled,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=MaskData.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    return dataloader_train_labeled, dataloader_val_labeled


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

    trainer.fit(model, dataloader_train, dataloader_val)

    # model = LitFullySupervisedClassifier.load_from_checkpoint(
    #     "checkpoints/epoch=499-step=500.ckpt", device=config.device
    # )
    # trainer.validate(model, dataloader_val)
