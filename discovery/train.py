import configs.discovery as config
from data.datasets.mask_feature_dataset import ImageMaskData, DiscoveryImageMaskData
from discovery.discovery_model import DiscoveryModel
from eval.coco_eval import CocoEvaluator
from eval.lvis_eval import LvisEvaluator

import argparse
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import CombinedLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.warmup_scheduler import WarmUpScheduler


class LitDiscovery(pl.LightningModule):
    """Lightning module for training the discovery network."""

    def __init__(self, device, len_train_data, label_map):
        super().__init__()
        self.len_train_data = len_train_data
        self.model = self.load_model(device)
        self.supervis_evaluator = CocoEvaluator(config.ann_val_labeled, ["bbox"])
        self.discovery_evaluator = LvisEvaluator(
            config.ann_val_unlabeled, ["bbox"], known_class_ids=config.lvis_known_class_ids
        )  # TODO: check if known class IDs are continuous or original IDs
        self.label_map = label_map  # Label mapping for the labelled dataset.

    def training_step(self, batch, batch_idx):
        # FIXME: padding is weird now, since we don't have a bg class anymore
        loss, supervised_loss, discovery_loss, _, _ = self.model(batch["labeled"], batch["unlabeled"])
        supervised_loss = {"train_" + k: v for k, v in supervised_loss.items()}
        del supervised_loss["train_supervised_loss"]  # Equal to CE loss.
        discovery_loss = {"train_" + k: v for k, v in discovery_loss.items()}

        self.log_dict(
            supervised_loss,
            batch_size=len(batch["labeled"]["boxes"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_discovery_loss",
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
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # lightning_optimizer = self.optimizers()  # self = your model
        # for param_group in lightning_optimizer.optimizer.param_groups:
        #     print("lr", param_group["lr"])

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # Validation datasets are processed sequentially instead of in parallel,
        # so we only have a supervised or unsupervised batch at a time.
        # TODO: Calculate / update / log evaluation metric
        if dataloader_idx == 0:  # Labeled dataset
            _, supervised_loss, _, outputs, _ = self.model(supervised_batch=batch, unsupervised_batch=None)
            supervised_loss = {"val_" + k: v for k, v in supervised_loss.items()}
            del supervised_loss["val_supervised_loss"]  # Equal to CE loss.

            self.log_dict(
                supervised_loss,
                batch_size=len(batch["boxes"]),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            results = CocoEvaluator.to_coco_format(batch["img_ids"], outputs, self.label_map, config.num_labeled)
            self.supervis_evaluator.update(results)

        else:  # Unlabeled dataset
            _, _, discovery_loss, _, outputs = self.model(supervised_batch=None, unsupervised_batch=batch)
            discovery_loss = {"val_" + k: v for k, v in discovery_loss.items()}

            self.log(
                "val_discovery_loss",
                discovery_loss["val_discovery_loss"],
                batch_size=len(batch["boxes"]),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            results = LvisEvaluator.to_lvis_format(
                batch["img_ids"], outputs, self.label_map
            )  # TODO: label map for LVIS dataset
            self.discovery_evaluator.update(results)

    def on_validation_epoch_end(self):
        # TODO: Calculate and log evaluation metric over whole dataset
        # TODO: ensure there is a key e.g. "map_all" we can use to checkpoint the best model.
        # Evaluator for labeled dataset
        self.supervis_evaluator.synchronize_between_processes()
        self.supervis_evaluator.accumulate()
        results = self.supervis_evaluator.summarize()

        self.log_dict(results["bbox"])

        self.supervis_evaluator.reset()

        # Evaluator for unlabeled dataset.
        results = self.discovery_evaluator.summarize()
        self.log_dict(results["bbox"])
        self.discovery_evaluator.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer, eta_min=config.end_lr, T_max=config.epochs * self.len_train_data
        )
        warmup_scheduler = WarmUpScheduler(
            optimizer,
            cosine_scheduler,
            config.warmup_steps,
            config.warmup_start_lr,
            self.len_train_data,
        )
        return [optimizer], [{"scheduler": warmup_scheduler, "interval": "step"}]

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
        config.masks_dir,
        config.ann_val_unlabeled,
        config.img_val,
        config.device,
        train=False,
        pad_num=config.pad_num,
        num_views=config.num_views,
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

    return dataloader_train, dataloader_val, dataset_val_labeled.continuous_to_cat_id


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

    model = LitDiscovery(config.device, len(dataloader_train), label_map)

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
        limit_train_batches=0.001,  # FIXME: remove this for actual training!
        limit_val_batches=0.001,
        default_root_dir=config.checkpoint_dir,
        logger=pl.loggers.tensorboard.TensorBoardLogger(save_dir=config.log_dir),
        accelerator="gpu" if config.device == "cuda" else "cpu",
        # log_every_n_steps=1,
        devices=config.num_devices,
        enable_checkpointing=True,
        max_epochs=config.epochs,
        callbacks=[
            best_checkpoint_callback,
            checkpoint_callback,
        ],
    )

    trainer.fit(model, dataloader_train, dataloader_val)  # FIXME: fix validation step and put val dataloader back

    # model = LitFullySupervisedClassifier.load_from_checkpoint(
    #     "checkpoints/epoch=499-step=500.ckpt", device=config.device
    # )
    # trainer.validate(model, dataloader_val)
