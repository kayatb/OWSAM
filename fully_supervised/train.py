from data.datasets.mask_feature_dataset import CropMaskData, CropFeatureMaskData, ImageMaskData
from fully_supervised.model import LinearClassifier, ResNetClassifier, SAMRPN
from eval.coco_eval import CocoEvaluator

from modelling.criterion import SetCriterion
from modelling.matcher import HungarianMatcher
from modelling.mixup import mixup
from utils.misc import get_pad_ids, add_padding
from utils.warmup_scheduler import WarmUpScheduler

import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


class LitFullySupervisedClassifier(pl.LightningModule):
    """Lightning module for training the fully supervised classification head."""

    def __init__(self, device, label_map, len_train_data=None):
        super().__init__()

        self.label_map = label_map  # Mapping from continuous ids back to original annotated ones.
        self.len_train_data = len_train_data  # Length of the train data. Necessary for warm-up scheduler.

        self.model = self.load_model(device)
        self.criterion = self.set_criterion(device)
        self.evaluator = CocoEvaluator(config.ann_val, ["bbox"])
        # self.evaluator.coco_eval["bbox"].params.useCats = 0  # For calculating object vs no-object mAP

    def training_step(self, batch, batch_idx):
        if config.model_type == "mlp" and config.use_mixup and self.training:
            mixed_features, mixed_targets = self.do_mixup(batch)
            batch["crop_features"] = mixed_features
            outputs = self.model(batch)
            loss = self.criterion(outputs, mixed_targets, use_mixup=True, num_masks=batch["num_masks"])

        else:
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch["targets"], use_mixup=False)

            self.log(  # TODO: make this possible with mix-up as well.
                "train_class_error",
                loss["class_error"].item(),
                batch_size=len(batch["boxes"]),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        self.log(
            "train_loss_ce",
            loss["loss"].item(),
            batch_size=len(batch["boxes"]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for param_group in self.optimizers().optimizer.param_groups:
            print("learning_rate:", param_group["lr"])

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
            prog_bar=True,
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
        if config.model_type == "rpn":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
            )

            step_scheduler = MultiStepLR(optimizer, config.milestones, gamma=config.gamma)
            warmup_scheduler = WarmUpScheduler(
                optimizer,
                step_scheduler,
                config.warmup_steps,
                config.warmup_start_lr,
                self.len_train_data,
            )
            return [optimizer], [{"scheduler": warmup_scheduler, "interval": "step"}]
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=config.lr)
            return optimizer

    def load_model(self, device):
        if config.model_type == "mlp":
            model = LinearClassifier(
                config.num_layers, config.hidden_dim, config.num_classes, config.dropout, pad_num=config.pad_num
            )
        elif config.model_type == "resnet":
            model = ResNetClassifier(config.num_classes, config.pad_num)
        elif config.model_type == "rpn":
            model = SAMRPN(config.num_classes, config.feature_extractor_ckpt, pad_num=config.pad_num)
        else:
            raise ValueError(f"Unknown model type `{type}` given.")
        model.to(device)

        return model

    def set_criterion(self, device):
        """Use the DETR loss (but only the classification part)."""
        # Default DETR values
        eos_coef = 0.05  # Was 0.1
        weight_dict = {"loss_ce": 0, "loss_bbox": 5}
        weight_dict["loss_giou"] = 2

        losses = ["labels"]

        matcher = HungarianMatcher()
        criterion = SetCriterion(
            self.model.num_classes,
            matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
            losses=losses,
            use_mixup=config.use_mixup,
            mixup_alpha=config.mixup_alpha,
        )
        criterion.to(device)

        return criterion

    @torch.no_grad()
    def do_mixup(self, batch):
        """Use the Hungarian Matcher used for the criterion to match the input with the targets based on IoU and L2
        bbox distance. Then perform mix-up augmentation."""
        input = {"pred_boxes": batch["boxes"]}  # Format necessary for the matcher.
        src_features = batch["crop_features"]

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.criterion.matcher(input, batch["targets"])
        idx = self.criterion._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(batch["targets"], indices)])

        target_classes = torch.full(
            batch["boxes"].shape[:2],
            self.model.num_classes,
            dtype=torch.int64,
            device=config.device,
        )
        target_classes[idx] = target_classes_o
        # Filter out the padded targets
        mask_ids, _ = get_pad_ids(batch["num_masks"], config.pad_num)
        target_classes = target_classes.flatten(0, 1)[mask_ids]

        # Mix it up!
        mixed_features, mixed_labels = mixup(
            src_features,
            target_classes,
            config.mixup_alpha,
            self.criterion.num_classes + 1,
        )

        # Add padding back to the labels
        mixed_labels = add_padding(
            mixed_labels, batch["num_masks"], config.num_classes, config.pad_num, config.device, mode="targets"
        )

        return mixed_features, mixed_labels


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
    parser.add_argument(
        "-m",
        "--model-type",
        required=True,
        choices=["mlp", "resnet", "rpn"],
        help="Which model to train (i.e. which config to use).",
    )
    parser.add_argument("-n", "--num-gpus", type=int, help="Number of GPUs to use.")

    args = parser.parse_args()
    return args


def load_data():
    if config.model_type == "mlp":
        dataset_train = CropFeatureMaskData(
            config.masks_dir, config.ann_train, config.crop_feat_dir, config.device, pad_num=config.pad_num
        )
        dataset_val = CropFeatureMaskData(
            config.masks_dir, config.ann_val, config.crop_feat_dir, config.device, pad_num=config.pad_num
        )
        collate_fn = CropFeatureMaskData.collate_fn

    elif config.model_type == "resnet":
        dataset_train = CropMaskData(
            config.masks_dir, config.ann_train, config.img_train, config.device, pad_num=config.pad_num
        )
        dataset_val = CropMaskData(
            config.masks_dir, config.ann_val, config.img_val, config.device, pad_num=config.pad_num
        )
        collate_fn = CropMaskData.collate_fn

    elif config.model_type == "rpn":
        dataset_train = ImageMaskData(
            config.masks_dir, config.ann_train, config.img_train, config.device, train=True, pad_num=config.pad_num
        )
        dataset_val = ImageMaskData(
            config.masks_dir, config.ann_val, config.img_val, config.device, train=False, pad_num=config.pad_num
        )
        collate_fn = ImageMaskData.collate_fn

    else:
        raise ValueError(f"Unknown model_type `{config.model_type}` given.")

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    return dataloader_train, dataloader_val, dataset_val.continuous_to_cat_id


if __name__ == "__main__":
    args = parse_args()

    if args.model_type == "mlp":
        import configs.fully_supervised.mlp as config
    elif args.model_type == "resnet":
        import configs.fully_supervised.resnet_crops as config
    elif args.model_type == "rpn":
        import configs.fully_supervised.rpn as config
    else:
        raise ValueError(f"Unkown model_type `{args.model_type}` given as cmd argument.")

    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.log_dir = args.log_dir
    if args.num_gpus:
        config.num_devices = args.num_gpus

    pl.seed_everything(config.seed, workers=True)

    dataloader_train, dataloader_val, label_map = load_data()

    model = LitFullySupervisedClassifier(config.device, label_map, len_train_data=len(dataloader_train))

    # Trainer callbacks.
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        save_top_k=1,
        monitor="map",
        mode="max",
        filename="best_model_{epoch}",
    )
    checkpoint_callback = ModelCheckpoint(dirpath=config.checkpoint_dir, every_n_epochs=config.save_every)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # TODO: could set precision=16 for mixed precision training, as done by RNCDL.
    # https://lightning.ai/docs/pytorch/1.5.7/advanced/mixed_precision.html
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
        # gradient_clip_val=config.clip,
        # gradient_clip_algorithm="value",
        callbacks=[best_checkpoint_callback, checkpoint_callback],
        # profiler="simple",
    )

    trainer.fit(model, dataloader_train, dataloader_val)

    # model = LitFullySupervisedClassifier.load_from_checkpoint(
    #     "checkpoints/dino_features/supervised_coco_dino_3layers_2048dim_1e-3lr-6oT8y-2741952/best_model_epoch=7.ckpt",
    #     device=config.device,
    #     label_map=label_map,
    # )
    # trainer.validate(model, dataloader_val)
