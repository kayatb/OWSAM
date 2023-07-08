import configs.discovery as config
from fully_supervised.model import SAMFasterRCNN
from discovery.discovery_network import DiscoveryClassifier
from discovery.roi_heads_discovery import RoIHeadsDiscovery

import torch
import torch.nn as nn


class ForwardMode:
    """Different modes for the discovery model forward pass."""

    TRAIN = 0
    SUPERVISED_VAL = 1
    UNSUPERVISED_VAL = 2


class DiscoveryModel(nn.Module):
    def __init__(self, supervised_ckpt):
        super().__init__()

        self.supervised_model = self.load_supervised_model(supervised_ckpt)
        self.remove_background_from_supervised_model()
        self.supervised_loss_lambda = config.supervised_loss_lambda

        self.discovery_model = DiscoveryClassifier(
            self.supervised_model.classifier,
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
            sk_mode="lognormal",
        )

    def forward(self, supervised_batch, unsupervised_batch, mode):
        if mode == ForwardMode.TRAIN:
            return self.forward_train(supervised_batch, unsupervised_batch)
        elif mode == ForwardMode.SUPERVISED_VAL:
            return self.forward_supervis_val(supervised_batch)
        elif mode == ForwardMode.UNSUPERVISED_VAL:
            return self.forward_unsupervis_val(unsupervised_batch)
        else:
            raise ValueError(f"Unknown ForwardMode {mode} given.")

    def forward_train(self, supervised_batch, unsupervised_batch):
        # Contains "loss_classifier"
        supervised_loss = self.supervised_model(supervised_batch)
        supervised_loss = {"supervised_" + k: v for k, v in supervised_loss.items()}

        if not self.is_discovery_memory_filled():
            supervised_loss["supervised_loss_classifier"] *= 0

        # Generate features for two different augmented images (so each in feats)
        # Extract features for the ROIs from both augmented images
        # Input to discovery_model should be list with the RoI features for each view.
        box_feats = self.supervised_model.get_box_features(
            unsupervised_batch, num_views=config.num_views, is_discovery_train=True
        )
        # TODO: once batched processing implemented. Split the different views.
        # box_feats = torch.split(box_feats, len(unsupervised_batch["images"]))
        # assert len(box_feats[0]) == len(box_feats[1])

        discovery_loss = self.discovery_model(box_feats, unsupervised_batch["sam_boxes"])
        discovery_loss = {"discovery_" + k: v for k, v in discovery_loss.items()}

        print(discovery_loss.keys())
        loss = (
            supervised_loss["supervised_loss_classifier"] * self.supervised_loss_lambda
            + discovery_loss["discovery_loss"]
            + discovery_loss["discovery_loss_similarity"]
        )

        return loss, supervised_loss, discovery_loss

    def forward_supervis_val(self, batch):
        """Validation forward pass on supervised data."""
        self.supervised_model.eval()

        output = self.supervised_model(batch)
        return output

    def forward_unsupervis_val(self, batch):
        """Validation forward pass on unsupervised data."""
        self.supervised_model.eval()
        self.discovery_model.eval()

        box_feats = self.supervised_model.get_box_features(batch, is_discovery_train=False, num_views=1)
        output = self.discovery_model.forward_heads_single_view(box_feats[0])

        return output

    @torch.no_grad()
    def extract_gt_preds(self, batch, is_supervis):
        """Get the box features for the GT boxes and the SAM boxes in a single pass.
        Also return the post-processed detections for SAM with novel classes, but only for the unsupervised data.
        Used for discovery evaluation after training."""
        self.eval()
        self.supervised_model.eval()
        self.discovery_model.eval()

        original_image_sizes = [img.shape[-2:] for img in batch["images"]]

        images, sam_boxes, targets = self.supervised_model.transform(
            batch["images"], batch["sam_boxes"], batch["targets"]
        )
        target_boxes = [t["boxes"] for t in targets]
        # Get the image features.
        features = self.supervised_model.backbone(images.tensors.to(sam_boxes[0].device))
        boxes_to_extract = [target_boxes] if is_supervis else [target_boxes, sam_boxes]

        # Get box features for both boxes.
        box_features = []
        for boxes in boxes_to_extract:
            proposals, _ = self.supervised_model.rpn(boxes, batch["iou_scores"])
            box_feats = self.supervised_model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
            box_feats = self.supervised_model.roi_heads.box_head(box_feats)

            box_features.append(box_feats)

        # Get the predicted labels for the GT boxes.
        gt_logits = self.discovery_model.forward_heads_single_view(box_features[0])
        gt_preds = torch.argmax(gt_logits, dim=-1)

        sam_outputs = []
        if not is_supervis:
            sam_logits, sam_box_regression = self.supervised_model.roi_heads.box_predictor(box_features[1])
            # Predict known and novel classes
            sam_logits = self.discovery_model.forward_heads_single_view(box_features[1])
            # sam_box_regression = torch.zeros(
            #     sam_logits.shape[0], (sam_logits.shape[1] + 1) * 4, device=sam_logits.device
            # )

            boxes, scores, labels = self.supervised_model.roi_heads.postprocess_detections(
                sam_logits, None, proposals, images.image_sizes
            )
            sam_outputs = []
            num_images = len(boxes)
            for i in range(num_images):
                sam_outputs.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
            sam_outputs = self.supervised_model.transform.postprocess(
                sam_outputs, images.image_sizes, original_image_sizes
            )

        return sam_outputs, gt_preds  # Features for SAM boxes and predictions for GT boxes.

    def load_supervised_model(self, ckpt_path):
        """Load the pre-trained supervised classification head."""
        ckpt_state_dict = torch.load(ckpt_path)["state_dict"]

        # Change the state dict to match the plain Torch module.
        model_state_dict = {}
        for key in ckpt_state_dict.keys():
            if key.startswith("model"):
                model_state_dict[key[len("model.") :]] = ckpt_state_dict[key]

        model = SAMFasterRCNN(
            config.num_labeled + 1,
            config.feature_extractor_ckpt,
            trainable_backbone_layers=0,
            rpn_nms_thresh=config.rpn_nms_thresh,
        )
        model.roi_heads = RoIHeadsDiscovery(
            model.roi_heads.box_roi_pool,
            model.roi_heads.box_head,
            model.roi_heads.box_predictor,
            model.box_fg_iou_thresh,
            model.box_bg_iou_thresh,
            model.box_batch_size_per_image,
            model.box_positive_fraction,
            model.bbox_reg_weights,
            model.roi_heads.score_thresh,
            model.roi_heads.nms_thresh,
            model.roi_heads.detections_per_img,
        )
        model.load_state_dict(model_state_dict)
        model.freeze()

        return model

    def remove_background_from_supervised_model(self):
        """Remove the background class from the supervised classifier to enable discovery of new classes."""
        classifier = self.supervised_model.classifier
        # Background is index 0, so the first weight/bias element.
        classifier.weight = nn.Parameter(classifier.weight[1:])
        classifier.bias = nn.Parameter(classifier.bias[1:])

        self.supervised_model.num_classes -= 1

    def is_discovery_memory_filled(self):
        return self.discovery_model.memory_patience == 0

    def from_checkpoint(self, ckpt_path):
        """Load weights from a checkpoint from PyTorch Lightning."""
        ckpt_state_dict = torch.load(ckpt_path)["state_dict"]

        # Change the state dict to match the plain Torch module.
        model_state_dict = {}
        for key in ckpt_state_dict.keys():
            if key.startswith("model"):
                model_state_dict[key[len("model.") :]] = ckpt_state_dict[key]

        self.load_state_dict(model_state_dict)


if __name__ == "__main__":
    model = DiscoveryModel("checkpoints/faster_rcnn_TUM/best_model_epoch=45.ckpt")
    # print(model.supervised_classifier)
    # print(model.state_dict()["supervised_classifier.classifier.bias"].shape)
