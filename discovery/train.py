import configs.discovery as config
from data.datasets.fasterrcnn_data import ImageData
from discovery.discovery_model import DiscoveryModel, ForwardMode
from eval.coco_eval import CocoEvaluator

import argparse
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import CombinedLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.warmup_scheduler import WarmUpScheduler


class LitDiscovery(pl.LightningModule):
    """Lightning module for training the discovery network.
    No discovery validation during training, only supervised validation is done for
    efficiency's sake. Do discovery validation once after training has completed."""

    def __init__(self, device, len_train_data, supervis_label_map):
        super().__init__()
        self.len_train_data = len_train_data
        self.model = self.load_model(device)
        self.supervis_evaluator = CocoEvaluator(config.ann_val_labeled, ["bbox"])

        self.supervis_label_map = supervis_label_map  # Label mapping for the labelled dataset.

    def training_step(self, batch, batch_idx):
        loss, supervised_loss, discovery_loss = self.model(batch["labeled"], batch["unlabeled"], mode=ForwardMode.TRAIN)
        supervised_loss = {"train_" + k: v for k, v in supervised_loss.items()}
        discovery_loss = {"train_" + k: v for k, v in discovery_loss.items()}

        self.log_dict(
            supervised_loss,
            batch_size=len(batch["labeled"]["sam_boxes"]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_discovery_loss",
            discovery_loss["train_discovery_loss"],
            batch_size=len(batch["labeled"]["sam_boxes"]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_total_loss",
            loss,
            batch_size=len(batch["labeled"]["sam_boxes"]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # lightning_optimizer = self.optimizers()  # self = your model
        # for param_group in lightning_optimizer.optimizer.param_groups:
        #     print("lr", param_group["lr"])

        return loss

    def validation_step(self, batch, batch_idx):
        # Only the supervised dataset is evaluated during training. The unsupervised dataset is evaluated afterwards.
        # if dataloader_idx == 0:  # Labeled dataset
        outputs = self.model(supervised_batch=batch, unsupervised_batch=None, mode=ForwardMode.SUPERVISED_VAL)
        # supervised_loss = {"val_" + k: v for k, v in supervised_loss.items()}

        # self.log_dict(
        #     supervised_loss,
        #     batch_size=len(batch["sam_boxes"]),
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )
        results = CocoEvaluator.to_coco_format_fasterrcnn(batch["img_ids"], outputs, self.supervis_label_map)
        self.supervis_evaluator.update(results)

        # else:  # Unlabeled dataset
        #     outputs = self.model(supervised_batch=None, unsupervised_batch=batch)
        #     discovery_loss = {"val_" + k: v for k, v in discovery_loss.items()}

        #     self.log(
        #         "val_discovery_loss",
        #         discovery_loss["val_discovery_loss"],
        #         batch_size=len(batch["sam_boxes"]),
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=True,
        #         logger=True,
        #     )

    def on_validation_epoch_end(self):
        # Evaluation on the supervised dataset.
        # Evaluator for labeled dataset
        self.supervis_evaluator.synchronize_between_processes()
        self.supervis_evaluator.accumulate()
        results = self.supervis_evaluator.summarize()

        self.log_dict(results["bbox"])

        self.supervis_evaluator.reset()

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
        model = DiscoveryModel(config.supervis_ckpt)
        model.to(device)

        return model


def parse_args():
    parser = argparse.ArgumentParser(prog="Supervised Mask2Formers")
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        help="Directory to store model checkpoints/",
    )
    parser.add_argument(
        "-l",
        "--log-dir",
        help="Directory to store (Tensorboard) logging.",
    )
    parser.add_argument("-n", "--num-gpus", type=int, help="Number of GPUs to use.")

    args = parser.parse_args()
    return args


def load_data():
    # For using multiple dataloaders, see --> https://lightning.ai/docs/pytorch/latest/data/iterables.html#multiple-dataloaders
    dataset_train_labeled = ImageData(
        config.masks_dir,
        config.ann_train_labeled,
        config.img_dir,
        config.device,
        offset=1,
    )
    dataset_val_labeled = ImageData(
        config.masks_dir,
        config.ann_val_labeled,
        config.img_dir,
        config.device,
        offset=1,  # Validation does need ID 0 for bg class, since we add a fake one at the postprocessing.
    )
    dataloader_train_labeled = DataLoader(
        dataset_train_labeled,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=ImageData.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    dataloader_val_labeled = DataLoader(
        dataset_val_labeled,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=ImageData.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    dataset_train_unlabeled = ImageData(
        config.masks_dir,
        config.ann_train_unlabeled,
        config.img_dir,
        config.device,
        offset=0,
    )
    # dataset_val_unlabeled = ImageData(
    #     config.masks_dir,
    #     config.ann_val_unlabeled,
    #     config.img_dir,
    #     config.device,
    # )
    dataloader_train_unlabeled = DataLoader(
        dataset_train_unlabeled,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=ImageData.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    # dataloader_val_unlabeled = DataLoader(
    #     dataset_val_unlabeled,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     collate_fn=ImageData.collate_fn,
    #     num_workers=config.num_workers,
    #     persistent_workers=True,
    #     pin_memory=True,
    #     prefetch_factor=3,
    # )

    dataloader_train = CombinedLoader(
        {"labeled": dataloader_train_labeled, "unlabeled": dataloader_train_unlabeled}, mode="max_size_cycle"
    )
    # dataloader_val = CombinedLoader(
    #     {"labeled": dataloader_val_labeled, "unlabeled": dataloader_val_unlabeled}, mode="sequential"
    # )

    return dataloader_train, dataloader_val_labeled, dataset_val_labeled.continuous_to_cat_id


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
    # Save a checkpoint every config.save_every epochs without overwriting the previous checkpoint.
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir, every_n_epochs=config.save_every, save_top_k=-1
    )

    trainer = pl.Trainer(
        # fast_dev_run=3,
        # limit_train_batches=0.001,  # FIXME: remove this for actual training!
        # limit_val_batches=0.001,
        default_root_dir=config.checkpoint_dir,
        logger=pl.loggers.tensorboard.TensorBoardLogger(save_dir=config.log_dir),
        accelerator="gpu" if config.device == "cuda" else "cpu",
        # log_every_n_steps=1,
        devices=config.num_devices,
        enable_checkpointing=True,
        max_epochs=config.epochs,
        callbacks=[
            checkpoint_callback,
        ],
        # Necessary when num_devices > 1, since we don't use the discovery classification head during the initial
        # memory filling and when training on multiple GPUs, pl doesn't like unused params otherwise.
        # strategy=pl.strategies.DDPStrategy(find_unused_parameters=True),
    )

    trainer.fit(model, dataloader_train, dataloader_val)
    # trainer.validate(model, dataloader_val)

    # model = LitFullySupervisedClassifier.load_from_checkpoint(
    #     "checkpoints/epoch=499-step=500.ckpt", device=config.device
    # )
    # trainer.validate(model, dataloader_val)
